#!/usr/bin/env python3
"""
Plot feature distributions for each class after applying selections.
Each class is normalized to show probability density for easy comparison.
"""
import argparse
import os
import glob
import yaml
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from functools import reduce

plt.style.use(hep.cms.style.CMS)


def load_parquet_files(file_input):
    """
    Load multiple parquet files.
    
    Args:
        file_input: Either a .txt file containing list of parquet files,
                   or a glob pattern for parquet files
    """
    # Check if input is a txt file with file list
    if file_input.endswith('.txt'):
        print(f"Reading file list from: {file_input}")
        with open(file_input, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Found {len(files)} files in list")
    else:
        # Treat as glob pattern
        files = glob.glob(file_input)
        if not files:
            raise ValueError(f"No files found matching pattern: {file_input}")
        print(f"Found {len(files)} parquet files matching pattern")
    
    if not files:
        raise ValueError(f"No files to load from: {file_input}")
    
    # Load all files
    events_list = []
    for i, f in enumerate(files, 1):
        print(f"Loading {i}/{len(files)}: {os.path.basename(f)}...")
        try:
            events_list.append(ak.from_parquet(f))
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")
            continue
    
    if not events_list:
        raise ValueError("No events loaded successfully")
    
    # Concatenate all events
    print(f"Concatenating {len(events_list)} arrays...")
    events = ak.concatenate(events_list, axis=0)
    print(f"Total events loaded: {len(events)}")
    
    return events


def apply_selections(events, selection_branches):
    """Apply OR of all selection branches."""
    if not selection_branches:
        return np.ones(len(events), dtype=bool)
    
    selection_masks = [events[sel] == 1 for sel in selection_branches]
    combined_mask = reduce(np.logical_or, selection_masks)
    
    n_selected = np.sum(combined_mask)
    print(f"Events after selection: {n_selected} ({100*n_selected/len(events):.2f}%)")
    
    return combined_mask


def plot_feature_distributions(
    events,
    selection_mask,
    truth_branches,
    features,
    output_dir,
    bins=50,
    log_y=False,
    class_labels=None,
    colors=None,
    integer_features=None,
):
    """Plot feature distributions for each class on the same canvas."""
    
    # Default class labels
    if class_labels is None:
        class_labels = truth_branches
    
    # Default colors (using matplotlib's tab10 colormap)
    if colors is None:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(truth_branches))]
    
    # Default integer features
    if integer_features is None:
        integer_features = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply selection
    events_selected = events[selection_mask]
    
    # Debug: print available fields
    print(f"\nAvailable fields in data: {events_selected.fields[:20]}{'...' if len(events_selected.fields) > 20 else ''}")
    
    # Get class masks
    class_masks = []
    class_counts = []
    for truth in truth_branches:
        mask = events_selected[truth] == 1
        class_masks.append(mask)
        class_counts.append(np.sum(mask))
        print(f"{truth}: {np.sum(mask)} events")
    
    # Plot each feature
    for feature in features:
        print(f"\nPlotting {feature}...")
        
        try:
            # Try to access the feature (supports nested fields like Jet_tagBin -> Jet.tagBin)
            try:
                feature_array = events_selected[feature]
            except (KeyError, ValueError) as e:
                # Try with underscore replaced by dot notation for nested access
                if '_' in feature:
                    parts = feature.split('_', 1)  # Split only at first underscore
                    nested_feature = f"{parts[0]}.{parts[1]}"
                    print(f"  Trying nested access: {nested_feature}")
                    try:
                        parent = events_selected[parts[0]]
                        feature_array = parent[parts[1]]
                    except:
                        # If that doesn't work, try the original error
                        raise e
                else:
                    raise
            
            # Check if this is a nested/jagged array
            if isinstance(feature_array.type, ak.types.ListType):
                # Nested array (e.g., Jet_pt) - flatten it
                print(f"  Detected nested array, taking first 6 objects and flattening...")
                # Take only first 6 jets/objects per event
                feature_array = feature_array[:, :6]
                feature_data = ak.flatten(feature_array)
                
                # Expand truth labels to match flattened data
                expanded_masks = []
                counts_for_this_feature = []
                for mask in class_masks:
                    # Repeat truth mask for each jet
                    expanded_mask = ak.flatten(ak.broadcast_arrays(mask, feature_array)[0])
                    expanded_masks.append(expanded_mask)
                    # Count objects (jets) for this class
                    counts_for_this_feature.append(np.sum(ak.to_numpy(expanded_mask)))
                class_masks_to_use = expanded_masks
                counts_to_use = counts_for_this_feature
                
                # Convert to numpy
                feature_data = ak.to_numpy(feature_data)
                print(f"  Total objects after flattening: {len(feature_data)}")
            else:
                # Regular event-level array
                feature_data = ak.to_numpy(feature_array)
                class_masks_to_use = [ak.to_numpy(m) for m in class_masks]
                counts_to_use = class_counts
                
        except Exception as e:
            print(f"  Error loading {feature}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Check for valid data
        if len(feature_data) == 0:
            print(f"  No data for {feature}")
            continue
        
        # Create figure (square shape)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Determine bins from all classes combined (excluding infinities and NaNs)
        valid_data = feature_data[np.isfinite(feature_data)]
        if len(valid_data) == 0:
            print(f"  No valid data for {feature}")
            plt.close(fig)
            continue
        
        # Check if this is an integer feature
        if feature in integer_features:
            # For integer features, create bins with integer boundaries
            data_min = int(np.floor(valid_data.min()))
            data_max = int(np.ceil(valid_data.max()))
            # Create bin edges: [min, min+1, min+2, ..., max+1]
            # This way integer n is in bin [n, n+1)
            bin_edges = np.arange(data_min, data_max + 1.0, 1.0)
            print(f"  Integer feature: using {len(bin_edges)-1} bins from {data_min} to {data_max}")
        else:
            # Use data range for bins
            data_min, data_max = np.percentile(valid_data, [0.1, 99.9])
            bin_edges = np.linspace(data_min, data_max, bins + 1)
        
        # Plot histogram for each class
        for truth, mask, label, color, count in zip(
            truth_branches, class_masks_to_use, class_labels, colors, counts_to_use
        ):
            class_data = feature_data[mask]
            class_data = class_data[np.isfinite(class_data)]  # Remove NaN/Inf
            
            if len(class_data) == 0:
                continue
            
            # Plot with normalization (density=True)
            ax.hist(
                class_data,
                bins=bin_edges,
                density=True,  # Normalize to probability density
                alpha=0.6,
                label=f"{label} (n={count})",
                color=color,
                histtype='step',
                linewidth=2,
            )
        
        # Set labels and style
        ax.set_xlabel(feature, fontsize=14)
        ax.set_ylabel("Normalized counts (a.u.)", fontsize=14)
        
        if log_y:
            ax.set_yscale('log')
        else:
            # Add 20% space at the top for better legend visibility
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.2)
        
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        
        # CMS label
        hep.cms.label("Preliminary", data=False, lumi=None, ax=ax)
        
        # Save
        output_path = os.path.join(output_dir, f"dist_{feature}.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        
        # Also save PDF
        output_path_pdf = os.path.join(output_dir, f"dist_{feature}.pdf")
        fig.savefig(output_path_pdf, bbox_inches='tight')
        
        plt.close(fig)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Plot feature distributions for each class with selections applied"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional: YAML config file to load settings from (overrides defaults)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input: either a .txt file with list of parquet files, or a glob pattern (e.g., 'data/*.parquet')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./distributions",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        nargs="+",
        default=["n_jet", "n_btagM", "n_btagT", "n_ctagL", "n_ctagM", "n_ctagT", "mll"],
        help="Features to plot",
    )
    parser.add_argument(
        "--truths",
        "-t",
        type=str,
        nargs="+",
        default=["is_ttbb", "is_ttb", "is_ttcc", "is_ttc", "is_ttother"],
        help="Truth branches (class labels)",
    )
    parser.add_argument(
        "--selections",
        "-s",
        type=str,
        nargs="+",
        default=["is_CR1_4j", "is_CR2_4j", "is_CR3_4j", "is_SRl_4j", "is_SRt_4j"],
        help="Selection branches to apply (OR combined)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for classes (default: use truth branch names)",
    )
    parser.add_argument(
        "--bins",
        "-b",
        type=int,
        default=50,
        help="Number of bins for histograms (ignored for integer features)",
    )
    parser.add_argument(
        "--integer",
        type=str,
        nargs="+",
        default=[],
        help="Features to treat as integers (one bin per integer value)",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for y-axis",
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        
        # Override defaults with config values (command line args take precedence)
        if args.features == ["n_jet", "n_btagM", "n_btagT", "n_ctagL", "n_ctagM", "n_ctagT", "mll"]:
            args.features = config.get("global_features", args.features)
        if args.truths == ["is_ttbb", "is_ttb", "is_ttcc", "is_ttc", "is_ttother"]:
            args.truths = config.get("truths", args.truths)
        if args.selections == ["is_CR1_4j", "is_CR2_4j", "is_CR3_4j", "is_SRl_4j", "is_SRt_4j"]:
            args.selections = config.get("selections", args.selections)
        
        print(f"  Features: {args.features}")
        print(f"  Truths: {args.truths}")
        print(f"  Selections: {args.selections}")
    
    # Load data
    print("\nLoading parquet files...")
    events = load_parquet_files(args.input)
    
    # Apply selections
    print("\nApplying selections...")
    selection_mask = apply_selections(events, args.selections)
    
    # Plot distributions
    print("\nPlotting distributions...")
    plot_feature_distributions(
        events=events,
        selection_mask=selection_mask,
        truth_branches=args.truths,
        features=args.features,
        output_dir=args.output,
        bins=args.bins,
        log_y=args.log_y,
        class_labels=args.labels,
        integer_features=args.integer,
    )
    
    print(f"\nDone! Plots saved to {args.output}")


if __name__ == "__main__":
    main()

