import torch
import torch.nn as nn
import math


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def activation(self, x):
        # Exact GELU: x * Phi(x), Phi the standard normal CDF.
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return x * cdf

    def forward(self, x):
        return self.W2(self.activation(self.W1(x)))
