"""Simple neural network model for demonstration purposes.

This module provides a lightweight PyTorch model suitable for testing SLURM workflows,
checkpointing, and distributed training examples.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


__all__ = ["ModelConfig", "SimpleModel", "build_model"]


@dataclass
class ModelConfig:
    """Configuration for SimpleModel.

    Attributes:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in MLP layers
        num_classes: Number of output classes (for classification)
        output_dim: Number of output dimensions (for regression)
        dropout: Dropout probability for regularization
        task_type: Either "classification" or "regression"
        regression_activation: Optional activation for regression output
                              (None, "relu", or "sigmoid")
    """
    input_dim: int = 20
    hidden_dim: int = 64
    num_classes: int = 4
    output_dim: int = 1
    dropout: float = 0.1
    task_type: str = "classification"
    regression_activation: Optional[str] = None


class SimpleModel(nn.Module):
    """Two-layer MLP for classification or regression.

    Architecture:
        - Layer 1: Linear(input_dim → hidden_dim) + ReLU + Dropout
        - Layer 2: Linear(hidden_dim → hidden_dim) + ReLU + Dropout
        - Output layer:
            * Classification: Linear(hidden_dim → num_classes) [raw logits]
            * Regression: Linear(hidden_dim → output_dim) + optional activation

    Args:
        config: ModelConfig object specifying architecture parameters
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared layers
        self.layer1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)

        self.layer2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(config.dropout)

        # Task-specific output layer
        if config.task_type == "classification":
            self.output = nn.Linear(config.hidden_dim, config.num_classes)
            self.output_activation = None
        elif config.task_type == "regression":
            self.output = nn.Linear(config.hidden_dim, config.output_dim)
            # Optional regression activation
            if config.regression_activation == "relu":
                self.output_activation = nn.ReLU()
            elif config.regression_activation == "sigmoid":
                self.output_activation = nn.Sigmoid()
            else:
                self.output_activation = None
        else:
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got {config.task_type}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            For classification: Raw logits of shape (batch_size, num_classes)
            For regression: Predictions of shape (batch_size, output_dim)
        """
        # First hidden layer
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second hidden layer
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Output layer
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


def build_model(config: ModelConfig) -> nn.Module:
    """Factory function to build a SimpleModel.

    Args:
        config: ModelConfig object specifying model parameters

    Returns:
        Initialized SimpleModel instance

    Example:
        >>> config = ModelConfig(input_dim=20, hidden_dim=64, num_classes=4)
        >>> model = build_model(config)
        >>> output = model(torch.randn(32, 20))  # (batch_size=32, input_dim=20)
        >>> output.shape
        torch.Size([32, 4])
    """
    return SimpleModel(config)
