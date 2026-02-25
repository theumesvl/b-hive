"""
Simple MLP classifier for HPlusCHToWW signal vs background classification.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from utils.models.base_model import Classifier_base


class SimpleMLP_HToWW(Classifier_base):
    """
    Simple MLP for binary signal vs background classification.
    Uses global features + flattened jet/MET features.
    """

    classes = {
        "signal": ["isSignal"],
        "background": ["isBackground"],
    }

    # Integer features (for proper handling)
    integer_features = {
        "global_features": [
            "jet_multiplicity",
            "cjet_multiplicity",
            "bjet_multiplicity",
        ],
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
        super().__init__(config, **kwargs)

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Calculate input dimension
        global_dim, cpf_dim, npf_dim, vtx_dim, lt_dim = self.all_model_features_length()

        # Total input: global + flattened cpf + flattened npf
        input_dim = global_dim
        input_dim += self.n_cpf_candidates * cpf_dim  # jet features
        input_dim += self.n_npf_candidates * npf_dim  # MET features

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
        """
        Forward pass.

        Args:
            inpt: Tuple of (global_features, cpf, npf, vtx, lt)

        Returns:
            Tensor of shape (batch_size, num_classes) with class logits
        """
        global_feat = inpt[0]  # (batch, global_dim)
        cpf = inpt[1]          # (batch, n_cpf, cpf_dim)
        npf = inpt[2]          # (batch, n_npf, npf_dim)

        # Flatten jet and MET features
        batch_size = global_feat.shape[0]
        cpf_flat = cpf.reshape(batch_size, -1)
        npf_flat = npf.reshape(batch_size, -1)

        # Concatenate all features
        x = torch.cat([global_feat, cpf_flat, npf_flat], dim=1)

        output = self.net(x)

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def calculate_roc_list(self, predictions, truth):
        """Calculate ROC curves for signal vs background."""
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        signal_index = list(self.classes).index('signal')
        background_index = list(self.classes).index('background')

        signal_events = (truth == signal_index)
        background_events = (truth == background_index)

        signal_pred = predictions[:, signal_index]
        background_pred = predictions[:, background_index]

        signal_vs_background = np.where(
            (signal_pred + background_pred) > 0,
            signal_pred / (signal_pred + background_pred),
            -1
        )

        no_veto = np.ones(truth.shape, dtype=bool)

        labels = ["signal_vs_background"]
        discs = [signal_vs_background]
        truths = [signal_events]
        vetos = [no_veto]
        xlabels = ["Signal efficiency"]
        ylabels = ["Background rejection"]

        return (discs, truths, vetos, labels, xlabels, ylabels)


class SimpleMLP_HToWW_Large(SimpleMLP_HToWW):
    """Larger MLP variant with more capacity."""

    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            hidden_dims=[256, 128, 64, 32],
            dropout=0.3,
            **kwargs
        )


class SimpleMLP_HToWW_Small(SimpleMLP_HToWW):
    """Smaller MLP variant for faster training."""

    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            hidden_dims=[64, 32],
            dropout=0.1,
            **kwargs
        )
