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
        
        if self.selections:
            # Check if selections are organized by OR/AND groups
            if isinstance(self.selections, dict) and ('or' in self.selections or 'and' in self.selections):
                # New format: {"or": [...], "and": [...]}
                selection_mask = self._apply_grouped_selections(global_arr)
            else:
                # Legacy format: all selections with OR
                selection_mask = self._apply_legacy_selections(global_arr)
            
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
    
    def _apply_grouped_selections(self, global_arr):
        """Apply selections organized by OR/AND groups"""
        or_conditions = []
        and_conditions = []
        
        # Process OR selections
        if 'or' in self.selections and self.selections['or']:
            for selection in self.selections['or']:
                condition = self._parse_selection_condition(global_arr, selection)
                if condition is not None:
                    or_conditions.append(condition)
        
        # Process AND selections  
        if 'and' in self.selections and self.selections['and']:
            for selection in self.selections['and']:
                condition = self._parse_selection_condition(global_arr, selection)
                if condition is not None:
                    and_conditions.append(condition)
        
        # Combine OR conditions
        if or_conditions:
            or_mask = reduce(np.logical_or, or_conditions)
        else:
            or_mask = np.ones(len(global_arr), dtype=bool)
        
        # Combine AND conditions
        if and_conditions:
            and_mask = reduce(np.logical_and, and_conditions)
        else:
            and_mask = np.ones(len(global_arr), dtype=bool)
        
        # Final mask: OR conditions AND AND conditions
        return np.logical_and(or_mask, and_mask)
    
    def _parse_selection_condition(self, global_arr, selection):
        """Parse selection condition (boolean field or comparison)"""
        # Check if it's a direct boolean field
        if selection in global_arr.dtype.names:
            return global_arr[selection]
        
        # Check for comparison operators
        for operator in ['>=', '<=', '>', '<', '==', '!=']:
            if operator in selection:
                field, value = selection.split(operator)
                field = field.strip()
                value = value.strip()
                
                if field in global_arr.dtype.names:
                    try:
                        value = int(value) if value.isdigit() else float(value)
                        if operator == '>=':
                            return global_arr[field] >= value
                        elif operator == '<=':
                            return global_arr[field] <= value
                        elif operator == '>':
                            return global_arr[field] > value
                        elif operator == '<':
                            return global_arr[field] < value
                        elif operator == '==':
                            return global_arr[field] == value
                        elif operator == '!=':
                            return global_arr[field] != value
                    except ValueError:
                        print(f"Warning: Could not parse value '{value}' for field '{field}'")
                        return None
        
        print(f"Warning: Selection '{selection}' not found in data")
        return None
    
    def _apply_legacy_selections(self, global_arr):
        """Apply selections in legacy format (all OR)"""
        selection_conditions = []
        for selection in self.selections:
            condition = self._parse_selection_condition(global_arr, selection)
            if condition is not None:
                selection_conditions.append(condition)
        
        return reduce(np.logical_or, selection_conditions)


