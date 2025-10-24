#!/usr/bin/env python3
"""
Count events from parquet files with selections applied
"""
import argparse
import os
import glob
import awkward as ak
import pyarrow.parquet as pq


def count_events(parquet_files, selections=None, truths=None):
    """
    Count events from parquet files with optional selections
    
    Args:
        parquet_files: List of parquet file paths or path to .txt file
        selections: List of selection criteria to apply
        truths: List of truth labels to count
    
    Returns:
        Dictionary with event counts
    """
    # Load parquet file list
    if isinstance(parquet_files, str):
        if parquet_files.endswith('.txt'):
            # Read from text file
            with open(parquet_files, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
        elif os.path.isdir(parquet_files):
            # Search for all parquet files in directory
            files = glob.glob(os.path.join(parquet_files, "**/*.parquet"), recursive=True)
            if not files:
                raise ValueError(f"No parquet files found in directory: {parquet_files}")
        else:
            # Single file
            files = [parquet_files]
    else:
        files = parquet_files
    
    print(f"Loading {len(files)} parquet files...")
    
    # Load all parquet files
    all_events = []
    for file in files:
        table = pq.read_table(file)
        events = ak.from_arrow(table)
        all_events.append(events)
    
    events = ak.concatenate(all_events)
    total_events = len(events)
    print(f"Total events loaded: {total_events:,}")
    
    # Apply selections
    if selections:
        selection_mask = ak.ones_like(events[selections[0]], dtype=bool)
        for sel in selections:
            if sel in events.fields:
                selection_mask = selection_mask & events[sel]
            else:
                print(f"Warning: Selection '{sel}' not found in events")
        
        events_selected = events[selection_mask]
        selected_count = len(events_selected)
        print(f"Events after selections: {selected_count:,} ({100*selected_count/total_events:.2f}%)")
    else:
        events_selected = events
        selected_count = total_events
    
    # Count by truth labels
    if truths:
        print(f"\nEvent counts by truth label:")
        truth_counts = {}
        for truth in truths:
            if truth in events_selected.fields:
                count = ak.sum(events_selected[truth])
                truth_counts[truth] = int(count)
                print(f"  {truth}: {count:,}")
            else:
                print(f"  Warning: Truth '{truth}' not found in events")
        return truth_counts
    
    return {"total": selected_count}


def main():
    parser = argparse.ArgumentParser(
        description="Count events from parquet files with selections"
    )
    
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to: (1) parquet file, (2) .txt file with file list, or (3) directory containing parquet files"
    )
    parser.add_argument(
        "--selections", "-s", type=str, nargs="+",
        help="Selection criteria to apply (e.g., is_CR1_4j is_CR2_4j)"
    )
    parser.add_argument(
        "--truths", "-t", type=str, nargs="+",
        default=["is_ttbb", "is_ttb", "is_ttcc", "is_ttc", "is_ttother"],
        help="Truth labels to count"
    )
    
    args = parser.parse_args()
    
    counts = count_events(args.input, args.selections, args.truths)
    
    print(f"\n{'='*50}")
    print("Summary:")
    total = sum(counts.values())
    for label, count in counts.items():
        percentage = 100 * count / total if total > 0 else 0
        print(f"  {label:15s}: {count:8,} ({percentage:5.2f}%)")
    print(f"  {'Total':15s}: {total:8,}")


if __name__ == "__main__":
    main()

