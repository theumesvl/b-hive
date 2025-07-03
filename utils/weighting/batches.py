import torch
import numpy as np

def compute_class_weights(
    classes: dict,
    truths: list,
    histogram_path: str,
    dtype: str,
    device: torch.device
) -> torch.Tensor:
    """
    Compute per‐class loss weights based on numbers of samples.
    Returns the torch.Tensor of class weights.
    """
    hist = np.load(histogram_path, allow_pickle=True)
    abundance = np.array([
        hist[[truths.index(flav) for flav in flavours]].sum()
        for flavours in classes.values()
    ])
    raw_weights = abundance.sum() / abundance
    normalized = raw_weights / raw_weights.sum()
    
    print(f"Total number of members: {abundance.astype(int).sum():,}")
    print(f"Number of class members: {list(abundance.astype(int))}")
    print(f"Class weights:           {[round(w,4) for w in normalized]}")
    dt = np.dtype(dtype).type
    return torch.tensor(normalized.astype(dt), device=device)


def expected_batches(num_samples: np.array, batch_size: int) -> int:
    """
    Compute how many batches you’ll (on average for weighted),
    given a batch_size, accounting for both full batches and a final partial.
    """
    full = (num_samples // batch_size).sum()
    rest = (num_samples % batch_size).sum() // batch_size
    return int(full + rest)