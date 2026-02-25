"""
Simple MLP classifier for HPlusCHToWW signal vs background classification.

This is a basic feedforward neural network that uses only global (event-level)
features for binary classification.
"""

import torch
import torch.nn as nn
from utils.models.base_model import Classifier_base


class SimpleMLP(Classifier_base):
    """
    Simple Multi-Layer Perceptron for event-level classification.

    Uses only global features (no particle-level constituents).
    Suitable for signal vs background discrimination using high-level variables.
    """

    # Map output classes to truth label columns
    classes = {
        "signal": ["isSignal"],
        "background": ["isBackground"],
    }

    # No integer features in our case
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
        """
        Initialize the SimpleMLP.

        Args:
            config: Configuration dictionary with feature definitions
            hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
            dropout: Dropout probability (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
            **kwargs: Additional arguments passed to Classifier_base
        """
        super().__init__(config, **kwargs)

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Get input dimension from config
        input_dim = len(config['global_features'])
        output_dim = len(self.classes)

        # Build the MLP layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer (no activation - will use CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Store config for reference
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: Tuple of (global_features, cpf, npf, vtx, lt)
                    We only use global_features for this model.

        Returns:
            Tensor of shape (batch_size, num_classes) with class logits
        """
        # inputs is a tuple: (global, cpf, npf, vtx, lt)
        # We only use global features
        global_feat = inputs[0]

        return self.net(global_feat)


class SimpleMLP_Large(SimpleMLP):
    """Larger MLP variant with more capacity."""

    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            hidden_dims=[256, 128, 64, 32],
            dropout=0.3,
            **kwargs
        )


class SimpleMLP_Small(SimpleMLP):
    """Smaller MLP variant for faster training/inference."""

    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            hidden_dims=[64, 32],
            dropout=0.1,
            **kwargs
        )
