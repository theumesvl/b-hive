import awkward as ak
import numpy as np
from functools import reduce
from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import (
    structured_array_from_tree,
    structured_array_from_tree_truth_from_dict,
    structured_custom_array_from_tree,
)


class PFCandidateAndVertexProcessing(DataPreprocessing_BaseClass):
    def callColumnAccumulator(self, output, events, flag, **kwargs):

        # Only apply pt/eta slicing if using actual jet_pt and jet_eta
        if self.pt_key == "jet_pt" and self.eta_key == "jet_eta":
            pt_slice = np.logical_and(
                ak.to_numpy(ak.flatten(events[self.pt_key], axis=0)) >= min(self.bins_pt),
                ak.to_numpy(ak.flatten(events[self.pt_key], axis=0)) < max(self.bins_pt),
            )
            eta_slice = np.logical_and(
                ak.to_numpy(ak.flatten(events[self.eta_key], axis=0)) >= min(self.bins_eta),
                ak.to_numpy(ak.flatten(events[self.eta_key], axis=0)) < max(self.bins_eta),
            )
        else:
            # No slicing for other keys (e.g., n_btagM)
            pt_slice = np.ones(len(events), dtype=bool)
            eta_slice = np.ones(len(events), dtype=bool)
    
        if isinstance(self.truths, dict):
            truth_arr = structured_array_from_tree_truth_from_dict(
                events=events,
                truth_dict=self.truths,
                precision=np.bool8,
                feature_length=1,
            )
        else:
            truth_arr = structured_array_from_tree(
                events=events,
                keys=self.truths,
                precision=np.bool8,
                feature_length=1,
            )

        data_slice = np.array(
            (pt_slice & eta_slice)
            & reduce(
                np.logical_or,
                [
                    truth_arr[truth]
                    for truth in (
                        self.truths.keys()
                        if isinstance(self.truths, dict)
                        else self.truths
                    )
                ],
            ),
            dtype=bool,
        )
        truth_arr = truth_arr[data_slice]

        # Preserve is_parquet flag through slicing (awkward arrays lose metadata when sliced)
        events_sliced = events[data_slice]
        if hasattr(events, 'metadata') and events.metadata.get('is_parquet', False):
            events_sliced.metadata = {'is_parquet': True}
        
        global_arr = structured_custom_array_from_tree(
            events=events_sliced,
            keys=self.global_features,
            custom_keys=[
                key
                for dictionary in self.global_custom_features
                for key, value in dictionary.items()
            ],
            custom_formulas=[
                value
                for dictionary in self.global_custom_features
                for key, value in dictionary.items()
            ],
            precision=self.precision,
            feature_length=1,
        )
        cpf_arr = structured_custom_array_from_tree(
            events=events_sliced,
            keys=self.cpf,
            custom_keys=[
                key
                for dictionary in self.cpf_custom
                for key, value in dictionary.items()
            ],
            custom_formulas=[
                value
                for dictionary in self.cpf_custom
                for key, value in dictionary.items()
            ],
            precision=self.precision,
            feature_length=self.n_cpf,
        )
        npf_arr = structured_custom_array_from_tree(
            events=events_sliced,
            keys=self.npf,
            custom_keys=[
                key
                for dictionary in self.npf_custom
                for key, value in dictionary.items()
            ],
            custom_formulas=[
                value
                for dictionary in self.npf_custom
                for key, value in dictionary.items()
            ],
            precision=self.precision,
            feature_length=self.n_npf,
        )
        vtx_arr = structured_custom_array_from_tree(
            events=events_sliced,
            keys=self.vtx,
            custom_keys=[
                key
                for dictionary in self.vtx_custom
                for key, value in dictionary.items()
            ],
            custom_formulas=[
                value
                for dictionary in self.vtx_custom
                for key, value in dictionary.items()
            ],
            precision=self.precision,
            feature_length=self.n_vtx,
        )
        lt_arr = structured_custom_array_from_tree(
            events=events_sliced,
            keys=self.lt,
            custom_keys=[
                key
                for dictionary in self.lt_custom
                for key, value in dictionary.items()
            ],
            custom_formulas=[
                value
                for dictionary in self.lt_custom
                for key, value in dictionary.items()
            ],
            precision=self.precision,
            feature_length=self.n_lt,
        )

        # create an array with the process value
        process = np.ones(len(global_arr)) * flag

        glob_mask = reduce(
            np.logical_and,
            [~np.any(np.isnan(global_arr[key])) for key in global_arr.dtype.names],
        )
        if cpf_arr.dtype.names:
            cpf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(cpf_arr[key]), axis=1)
                    for key in cpf_arr.dtype.names
                ],
            )
        else:
            cpf_mask = np.ones(len(global_arr))
        if npf_arr.dtype.names:
            npf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(npf_arr[key]), axis=1) if npf_arr[key].ndim > 1
                    else ~np.isnan(npf_arr[key])  # For 1D arrays (like MET)
                    for key in npf_arr.dtype.names
                ],
            )
        else:
            npf_mask = np.ones(len(global_arr))
        if vtx_arr.dtype.names:
            vtx_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(vtx_arr[key]), axis=1) 
                    for key in vtx_arr.dtype.names
                ],
            )
        else:
            vtx_mask = np.ones(len(global_arr))
        if lt_arr.dtype.names:
            lt_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(lt_arr[key]), axis=1) 
                    for key in lt_arr.dtype.names
                ],
            )
        else:
            lt_mask = np.ones(len(global_arr))

        nan_mask = reduce(np.logical_and, [glob_mask, cpf_mask, npf_mask, vtx_mask, lt_mask])

        return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            lt_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )

    def saveOutput(
        self, output_location, global_arr, cpf_arr, npf_arr, vtx_arr, lt_arr, truth, process
    ):
        np.savez(
            output_location,
            global_features=global_arr,
            cpf_arr=cpf_arr,
            npf_arr=npf_arr,
            vtx_arr=vtx_arr,
            lt_arr=lt_arr,
            truth=truth,
            process=process,
        )
