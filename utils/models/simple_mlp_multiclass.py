"""
Multi-class MLP classifier for HPlusCHToWW.
Distinguishes signal (higgs) from individual background types.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from utils.models.base_model import Classifier_base


class SimpleMLP_MultiClass(Classifier_base):
    """
    Multi-class MLP for signal vs individual background classification.
    Classes are derived from config["truths"] (e.g. "is_higgs" -> {"higgs": ["is_higgs"]}).
    """

    # Integer features (for proper handling)
    integer_features = {
        "global_features": [],
        "cpf_candidates": [],
        "npf_candidates": [],
        "vtx_features": [],
        "lt_candidates": [],
    }

    def __init__(
        self,
        config,
        hidden_dims=None,
        dropout=0.2,
        use_batch_norm=True,
        **kwargs
    ):
        # Build classes from config truths: "is_higgs" -> {"higgs": ["is_higgs"]}
        self.classes = {
            truth.removeprefix("is_"): [truth]
            for truth in config["truths"]
        }
        # Signal class name from config reference_flavour
        self.signal_class = config["reference_flavour"].removeprefix("is_")

        super().__init__(config, **kwargs)

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Calculate input dimension
        global_dim, cpf_dim, npf_dim, vtx_dim, lt_dim = self.all_model_features_length()

        # Total input: global + flattened cpf + flattened npf
        input_dim = global_dim
        input_dim += self.n_cpf_candidates * cpf_dim
        input_dim += self.n_npf_candidates * npf_dim

        num_classes = len(self.classes)

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

        # Store config
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.for_inference = False

    def forward(self, inpt):
        """Forward pass."""
        global_feat = inpt[0]
        cpf = inpt[1]
        npf = inpt[2]

        batch_size = global_feat.shape[0]
        cpf_flat = cpf.reshape(batch_size, -1)
        npf_flat = npf.reshape(batch_size, -1)

        x = torch.cat([global_feat, cpf_flat, npf_flat], dim=1)
        output = self.net(x)

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def calculate_roc_list(self, predictions, truth):
        """Calculate ROC curves for signal vs each background."""
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        class_names = list(self.classes.keys())
        signal_index = class_names.index(self.signal_class)

        signal_events = (truth == signal_index)
        signal_pred = predictions[:, signal_index]

        labels = []
        discs = []
        truths = []
        vetos = []
        xlabels = []
        ylabels = []

        # Signal vs all backgrounds combined
        signal_vs_all = signal_pred.copy()

        labels.append(f"{self.signal_class}_vs_all")
        discs.append(signal_vs_all)
        truths.append(signal_events)
        vetos.append(np.ones(truth.shape, dtype=bool))
        xlabels.append("Signal efficiency")
        ylabels.append("Background rejection")

        # Signal vs each individual background
        for idx, name in enumerate(class_names):
            if name == self.signal_class:
                continue

            bg_events = (truth == idx)
            # Only consider signal and this specific background
            mask = signal_events | bg_events

            if np.sum(mask) > 0:
                labels.append(f"{self.signal_class}_vs_{name}")
                discs.append(signal_pred)
                truths.append(signal_events)
                vetos.append(mask)
                xlabels.append("Signal efficiency")
                ylabels.append(f"{name} rejection")

        return (discs, truths, vetos, labels, xlabels, ylabels)


class SimpleMLP_MultiClass_Large(SimpleMLP_MultiClass):
    """Larger variant with more capacity."""

    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            hidden_dims=[256, 128, 64, 32],
            dropout=0.3,
            **kwargs
        )
