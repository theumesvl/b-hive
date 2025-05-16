import itertools
import subprocess
import os
import lz4.frame
from concurrent.futures import ProcessPoolExecutor, as_completed

import luigi
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm
import psutil

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency
from utils.config.config_loader import ConfigLoader

def get_feature_keys(config):
    feature_keys = {}
    for name_fts in ['global_features', 'cpf_candidates', 'npf_candidates', 'vtx_features']:
        if name_fts in config:
            for feature in config[name_fts]:
                feature_keys[feature] = name_fts
    return feature_keys

def safe_reshape(arr: np.ndarray, second_dim: int) -> np.ndarray:
    """
    Reshape `arr` to (N, second_dim, ?).
    If either `arr` is empty or `second_dim` is 0, return
    an array shaped (N, 0, 0) so the code keeps running.
    """
    if arr.size == 0 or second_dim == 0:
        # N = arr.shape[0]; keep dtype and contiguous layout
        return arr.reshape(arr.shape[0], 0, 0)
    else:
        # let NumPy infer the last axis
        return arr.reshape(arr.shape[0], second_dim, -1)
        
def calculate_feature_length(config, base_key, custom_key=None):
    base_length = 0
    if base_key in config:
        base_length += len(config[base_key])
    if custom_key and custom_key in config:
        base_length += len(config[custom_key])
    return base_length

def create_feature_shapes(config):
    # Constructions of input shape from config.yaml file
    len_glob_fts_full = calculate_feature_length(config, 'global_features', 'global_custom_features')
    len_cpf_fts_full = calculate_feature_length(config, 'cpf_candidates', 'cpf_custom_features')
    len_npf_fts_full = calculate_feature_length(config, 'npf_candidates', 'npf_custom_features')
    len_vtx_fts_full = calculate_feature_length(config, 'vtx_features', 'vtx_custom_features')
    
    input_dims = [
        (1,                          len_glob_fts_full),
        (config['n_cpf_candidates'], len_cpf_fts_full),
        (config['n_npf_candidates'], len_npf_fts_full),
        (config['n_vtx_candidates'], len_vtx_fts_full)
    ]
    
    feature_edges = []
    v = 0
    for dim in input_dims:
        v += dim[0]*dim[1]
        feature_edges.append(v)

    feature_edges = np.array(feature_edges, dtype = int)

    return input_dims, feature_edges[:-1]

def process_one_file(file_path, config, feature_keys, feature_edges, input_dims, Nbins):
    """Load a single .lz4 file, build *partial* histograms for every (proc,feature),
    and return them as a nested dict."""

    num_ele = len(config["truths"])
    # initialize an empty set of histograms for this file
    # 0 -- histogram
    # 1 -- bins edges
    partial = {proc: {feat: np.zeros((2, Nbins+2), dtype=np.float32)
                      for feat in feature_keys}
               for proc in config["processes"]}

    # load & unpack data
    with lz4.frame.open(file_path, mode='r') as data:
        output_data = data.read()
    s = np.frombuffer(output_data, dtype='float32').copy()
    s = s[2:].reshape(-1, int(s[1]), order="C")

    weights      = s[:, -1]
    file_process = s[:, -(num_ele+2)]
    s            = s[:,:-(num_ele+2)]

    # Split the input tensor
    glob, cpf, npf, vtx = np.split(s, feature_edges, axis=1)

    # Reshape tensors as per input dimensions
    glob = glob.reshape(glob.shape[0], -1)
    cpf  = safe_reshape(cpf, input_dims[1][0])
    npf  = safe_reshape(npf, input_dims[2][0])
    vtx  = safe_reshape(vtx, input_dims[3][0])
    all_data = {
        'global_features': glob, 
        'cpf_candidates': cpf, 
        'npf_candidates': npf, 
        'vtx_features': vtx,
    }

    for i_p, process in enumerate(config['processes']):
        selection = (file_process == i_p)
        if selection.sum() == 0:
            continue
            
        for feature, name_fts in feature_keys.items():
            samples = all_data[name_fts][selection][..., config[name_fts].index(feature)]
            # clean zeros
            if name_fts != 'global_features':
                samples = samples[np.abs(samples).sum(axis=-1) != 0]
            
            samples = samples.flatten()
            
            if (partial[process][feature] == 0).all():
                bins = Nbins
            else:
                bincenters = partial[process][feature][1]
                delta_bin = bincenters[1] - bincenters[0]
                bins = bincenters - delta_bin / 2
                bins = np.append(bins, bins[-1] + delta_bin)[1:-1]  # remove overflow bins
                
            N, bins = np.histogram(samples, bins=bins)
            # add overflow bins
            N_below = (samples < bins[0]).sum()  # lower overflow bin content
            N_above = (samples > bins[-1]).sum()  # upper overflow bin content
            N = np.array([N_below, *N, N_above])
            bins = np.array(
                [
                    bins[0] - (bins[1] - bins[0]),  # lower overflow bin
                    *bins,
                    bins[-1]
                    + (bins[1] - bins[0]),  # upper overflow bin
                ]
            )
            bincenters = 0.5 * (bins[:-1] + bins[1:])
            partial[process][feature][0] += N
            partial[process][feature][1] = bincenters.astype(np.float32)

    return partial


def run_parallel_all(files, config, feature_keys, feature_edges, input_dims, Nbins, n_workers):
    # initialize master histograms once
    master = {
      proc: {
        feat: np.zeros((2, Nbins+2), dtype=np.float32)
        for feat in feature_keys
      }
      for proc in config["processes"]
    }

    if n_workers == -1:
        out = subprocess.check_output(["nproc"], stderr=subprocess.DEVNULL)
        n_workers = int(out) - 1
    print(f"n_workers = {n_workers}")
        
    bar_style = Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),     
        BarColumn(bar_width=None),
        TaskProgressColumn(),          
        TimeRemainingColumn(),         
        TextColumn(f"0/{len(files)} files"),
        expand=True,
    )

    with bar_style as progress, ProcessPoolExecutor(max_workers=n_workers) as exe:
        task = progress.add_task("Processing files...",
                                 total=len(files),
                                 filename="")

        # submit every file to a worker
        futures = {exe.submit(process_one_file, f,
                              config, feature_keys,
                              feature_edges, input_dims, Nbins): f
                   for f in files}

        # merge as soon as each future finishes
        count = 0
        for fut in as_completed(futures):
            file = futures[fut]
            progress.update(task, filename=file.split('/')[-1])
            count += 1
            progress.columns[-1].text_format = f"{count}/{len(files)} files"

            partial = fut.result()          # may raise; let it bubble up
            for proc in partial:
                for feat in partial[proc]:
                    master[proc][feat][0] += partial[proc][feat][0]
                    master[proc][feat][1]  = partial[proc][feat][1]

            progress.advance(task)

    return master


class InputHistogrammerTask(DatasetDependency, BaseTask):
    Nbins = luigi.IntParameter(
        default = 20, 
        description="Number of bins for histogramm"
    )

    n_workers = luigi.IntParameter(
        default=-1, 
        description="Number of workers to create histograms. Default: nproc -"
    )
    
    def requires(self):
        return DatasetConstructorTask.req(
            self, dataset_version=self.dataset_version, config=self.config
        )

    def output(self):
        processes = ConfigLoader.load_config(self.config)["processes"]
        attributes = [process + "_hist.npz" for process in processes]
        return {
            key: self.local_target(attr) for (key, attr) in zip(processes, attributes)
        }

    def run(self):
        # Create output directory
        for file in self.output().keys():
            self.output()[file].parent.touch()
            
        # Load config
        config = ConfigLoader.load_config(self.config)
        num_ele = len(config['truths'])

        # Load files
        files = self.input()["file_list"].load().split("\n")
        if not os.path.exists(files[-1]):
            files = files[:-1]

        feature_keys = get_feature_keys(config)
        input_dims, feature_edges = create_feature_shapes(config)

        histograms = run_parallel_all(files, config, feature_keys, feature_edges, input_dims, self.Nbins, n_workers=self.n_workers)
        for process in config['processes']:
            np.savez(self.output()[process].path, **histograms[process])


class HistogramPlotterTask(DatasetDependency, BaseTask):
    ylog = luigi.BoolParameter(
        default=False,
        description="If to use y log scale for histogram plotting. Default: False"
    )

    def requires(self):
        return InputHistogrammerTask.req(
            self, dataset_version=self.dataset_version, config=self.config
        )

    def output(self):
        config = ConfigLoader.load_config(self.config)
        processes = config["processes"]
        feature_keys = get_feature_keys(config)
        return_dict = {}
        for process, file in itertools.product(processes, feature_keys):
            return_path = os.path.join(process, file, "done")
            return_dict[process + "_" + file] = self.local_target(
                return_path + ("_ylog" if self.ylog else "")
            )
        return return_dict

    def run(self):
        color_set_name = "Dark2"
        cmap = mpl.colormaps[color_set_name]  # returns a ListedColormap
        color_set_list = cmap.colors  # type: list
        
        plt.style.use(hep.cms.style.CMS)
        
        for file in self.output().keys():
            self.output()[file].parent.touch()
        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[count]}"),
            expand=True,
        ) as progress:
            task_process = progress.add_task(
                f"Processing histograms...",
                total=len(self.input().items()),
                count=f"0/{len(self.input().items())}",
            )
            task_features = progress.add_task(
                f"Processing features... ",
                count="",
            )
            
            for i_p, (process, file) in enumerate(self.input().items()):
                histograms = np.load(file.path)
                feature_keys = histograms.keys()
                progress.reset(task_features)
                progress.update(
                    task_process,
                    description=f"Processing histograms for process {process:15}",
                )
                for i_f, feature in enumerate(feature_keys):
                    progress.update(
                        task_features,
                        total=len(feature_keys),
                        description=f"Processing features in {feature}",
                    )
                    data, centers = histograms[feature]       # row 0 = counts, row 1 = bin centers
                    delta = centers[1] - centers[0]           # bin width
                    edges = np.concatenate((centers - delta/2,
                                            [centers[-1] + delta/2]))
                    
                    fig, ax = plt.subplots()
                    ax.bar(
                        edges[:-1],       # left edge of each bar
                        data,
                        width=delta,      # constant width
                        align='edge',
                        edgecolor='black',
                    )
                    ax.set(
                        xlabel = feature,
                        ylabel = 'Count',
                        xlim = (edges[0], edges[-1]),
                        title = "Plot of " + feature + " for " + process
                    )
                    if self.ylog:
                        ax.set_yscale("log")
                    ax.grid(True)
                    plt.savefig(
                        self.output()[process + "_" + feature].dirname
                        + "/"
                        + feature
                        + ("_log" if self.ylog else "")
                        + ".png"
                    )
                    plt.close()

                    progress.update(
                        task_features,
                        count=f"{i_f+1}/{len(feature_keys)}",
                    )
                    progress.advance(task_features)
                progress.update(
                    task_process,
                    count=f"{i_p+1}/{len(self.input().items())}",
                )
                progress.advance(task_process)