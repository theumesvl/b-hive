import os
from typing import Dict
from collections import defaultdict
import uuid
import luigi
import numpy as np
import awkward as ak
from coffea import processor
from coffea.nanoevents import BaseSchema

from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency
from utils.coffea_processors.processor_loader import ProcessorLoader
from utils.config.config_loader import ConfigLoader
from utils.dataset.merging import merge_datasets
from utils.weighting.histogram import (
    weight_all_files_histrogram_weighting,
)

def read_in_samples_match_processes(file_path, processes):
    samples_dict = defaultdict(list)
    with open(file_path, "r") as input_txt:
        while line := input_txt.readline().rstrip("\n"):
            if line:
                if processes == ["default"]:
                    samples_dict[uuid.uuid4().hex].append(line)
                else:
                    for process in processes:
                        if process in line:
                            samples_dict[process].append(line)
                            break
    return samples_dict


class DatasetConstructorTask(DatasetDependency, BaseTask):
    coffea_worker = luigi.IntParameter(
        default=1,
        description="Number of workers for Coffea-processing",
        significant=False,
    )
    chunk_size = luigi.IntParameter(
        default=100000, 
        description="Number of events for outgoing files"
    )

    def output(self):
        return {
            "file_list": self.local_target("processed_files.txt"),
            "histogram": self.local_target("histogram.npy"),
            "weights":   self.local_target("weights.json"),
        }

    def run(self):
        print("Dataset construction")
        self.output()["file_list"].parent.touch()  # create directory
        config = ConfigLoader.load_config(self.config)
        np.random.seed(self.seed)
        assert self.filelist != "", (
            "You did not specify a filelist .txt but tried to run a new DatasetConstruction! "
            "Either you forgot to specify the path to the file or are using a wrong dataset-version!"
        )


        all_files = []
        samples = read_in_samples_match_processes(self.filelist, config["processes"])
        print('Number of files:', len(samples))

        print('samples', samples)
        if self.debug:
            # skim files to only 10 files per process
            samples = {
                key: values[0 : min(len(samples), 1)] for key, values in samples.items()
            }
        
        # Check file type from first sample
        first_file = next(iter(samples.values()))[0]
        is_parquet = first_file.endswith('.parquet')
        is_root = first_file.endswith('.root')
        
        print(f"File type detected: {'Parquet' if is_parquet else 'ROOT' if is_root else 'Unknown'}")

        # Make a dictionary entry for all of them:

        processorClass = ProcessorLoader(
            config.get("processor", "PFCandidateAndVertexProcessing"),  
            output_directory=self.local_path(),
            bins_pt=config.get("bins_pt", None),
            bins_eta=config.get("bins_eta", None),
            processes=config.get("processes", None),
            global_features=config.get("global_features", []),
            global_custom_features=config.get("global_custom_features", []),
            cpf_candidates=config.get("cpf_candidates", []),
            cpf_custom_features=config.get("cpf_custom_features", []),
            npf_candidates=config.get("npf_candidates", []),
            npf_custom_features=config.get("npf_custom_features", []),
            vtx_features=config.get("vtx_features", []),
            vtx_custom_features=config.get("vtx_custom_features", []),
            lt_candidates=config.get("lt_candidates", []),
            lt_custom_features=config.get("lt_custom_features", []),
            n_cpf_candidates=config.get("n_cpf_candidates", 50),
            n_npf_candidates=config.get("n_npf_candidates", 50),
            n_vtx_features=config.get("n_vtx_candidates", 5),
            n_lt_candidates=config.get("n_lt_candidates", 5),
            pt_key=config.get("pt_key", "jet_pt"),
            eta_key=config.get("eta_key", "jet_eta"),
            truths=config.get("truths", None),
            precision=config.get("data_precision", 'float32'),
            selections=config.get("selections", None),
        )  
        print(f"Processor: {config.get('processor', 'PFCandidateAndVertexProcessing')}")

        if is_root:
            # Original Coffea Runner for ROOT files
            futures_run = processor.Runner(
                executor=processor.FuturesExecutor(
                    compression=None, 
                    workers=self.coffea_worker
                ),
                schema=BaseSchema,
                chunksize=self.chunk_size // 20,
                maxchunks=None if not (self.debug) else 20,
            )

            output = futures_run(
                samples,
                treename=config["treename"],
                processor_instance=processorClass
            )
        
        elif is_parquet:
            # New Parquet reading logic
            print("Processing Parquet files...")
            output = processor.dict_accumulator({})
            
            for dataset_name, file_list in samples.items():
                for infile in file_list:
                    print(f"Reading {infile}...")
                    
                    # Read parquet file
                    events = ak.from_parquet(infile)
                    
                    # Add metadata (required by Coffea processors)
                    events.metadata = {
                        "dataset": dataset_name,
                        "filename": infile,
                        "entrystart": 0,
                        "entrystop": len(events),
                        "is_parquet": True  # Flag to indicate parquet format
                    }
                    
                    # Process with the same processor
                    result = processorClass.process(events)
                    
                    # Accumulate results
                    if not output:
                        output = result
                    else:
                        # Merge output_location lists
                        if "output_location" in result:
                            if "output_location" in output:
                                output["output_location"].extend(result["output_location"])
                            else:
                                output["output_location"] = result["output_location"]
                        
                        # Merge histograms
                        for key in result:
                            if key != "output_location":
                                if key in output:
                                    output[key] += result[key]
                                else:
                                    output[key] = result[key]
        else:
            raise ValueError(f"Unsupported file format: {first_file}. Only .root and .parquet are supported.")

        # saving histograms from coffea
        histograms = []
        file_list = []
        for key, value in output.items():
            if key == "output_location":
                file_list += value  # append output location list
            else:
                histograms.append(value.view().astype(np.float64)) # append view on hist -> np.array

        training_histograms = {
            key: value for key, value in output.items() if not "output_location" in key
        }

        np.save(
            self.output()[f"histogram"].path,
            np.array(histograms, dtype=np.float64),
        )

        print("Start merging files")
        # returns list of merged training-files
        all_files += merge_datasets(
            file_list,
            self.local_path(),
            label="file",
            processor=config.get("processor", "PFCandidateAndVertexProcessing"),
            chunk_size=self.chunk_size if not self.debug else self.chunk_size//5,
            workers=self.coffea_worker,
            shuffle=True,
            histograms=np.array(histograms, dtype=np.float64),
            reference_key=list(config["truths"]).index(config["reference_flavour"]), #reference_key is the histogram index for LZ4 dataset
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
            pt_key=config.get("pt_key", "jet_pt"),
            eta_key=config.get("eta_key", "jet_eta"),
            global_features=config.get("global_features", []),
            precision=config.get("data_precision", 'float32'),
        )
        '''
        if "LZ4" not in config.get("processor", "PFCandidateAndVertexProcessing"):
            # delete unmerged files
            for file in file_list:
                os.remove(file)

            # add weights to all files
            # this should be done on the fly - please implement!
            all_files = weight_all_files_histrogram_weighting(
                all_files,
                histograms=training_histograms,
                bins_pt=config["bins_pt"],
                bins_eta=config["bins_eta"],
                reference_key=config["reference_flavour"],
                pt_key=config.get("pt_key", "jet_pt"),
                eta_key=config.get("eta_key", "jet_eta"),
            )
        '''
        self.output()["file_list"].dump("\n".join(all_files), formatter="text")
