import numpy as np
from utils.coffea_processors.lz4_processor import LZ4Processing 
from functools import reduce

class LZ4TTCCProcessor(LZ4Processing):
    
    """
    class LZ4Processing(DataPreprocessing_BaseClass):
        def callColumnAccumulator(self, output, events, flag, **kwargs):
        ....
    return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )
    """
    
    def callColumnAccumulator(self, output, events, flag, **kwargs):
        global_arr, cpf_arr, npf_arr, vtx_arr, lt_arr, truth_arr, process = super().callColumnAccumulator(output, events, flag, **kwargs)
        
        #print('----- Inside PairedTaggerProcessor -----')
        #print('global_arr.shape:', global_arr.shape)
        
        if self.selections:
            selection_conditions = []
            for selection in self.selections:
                selection_conditions.append(global_arr[selection] == 1)

            #print("DEBUG selection_conditions:", selection_conditions)
            selection_mask = reduce(
                np.logical_or,
                selection_conditions
            )
#            selection_mask = selection_conditions[0]
#            for condition in selection_conditions[1:]:
#                selection_mask = selection_mask | condition
            #print("DEBUG selection_mask:", selection_mask)
            
            return(
                global_arr[selection_mask],
                cpf_arr[selection_mask],
                npf_arr[selection_mask],
                vtx_arr[selection_mask],
                lt_arr[selection_mask],
                truth_arr[selection_mask],
                process[selection_mask],
            )
        
        # No selections, return original data
        return (
            global_arr,
            cpf_arr,
            npf_arr,
            vtx_arr,
            lt_arr,
            truth_arr,
            process,
        )
