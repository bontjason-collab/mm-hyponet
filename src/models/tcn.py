"""
Minimal Temporal Convolutional Network (TCN) for glucose forecasting.

Input:  a sequence of recent glucose readings (shape: batch x 1 x lookback).
Output: a single number = predicted glucose at the horizon.

Fixed architecture (held constant across all feature experiments, per the spec):
  - dilated 1D-conv residual blocks, dilations 1,2,4,8
  - kernel size 3, causal (only looks backward)
  - small channel count (edge-deployable)
The input channel count is the only thing that changes when features are added;
everything from the first hidden layer on stays identical.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation          # causal padding
        self.pad = pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def _causal(self, x, conv, norm):
        x = nn.functional.pad(x, (self.pad, 0))     # pad left only = causal
        x = conv(x)
        x = norm(x)
        x = self.act(x)
        return self.drop(x)

    def forward(self, x):
        residual = x
        x = self._causal(x, self.conv1, self.norm1)
        x = self._causal(x, self.conv2, self.norm2)
        return x + residual                          # residual connection


class TCN(nn.Module):
    def __init__(self, n_inputs=1, channels=32, kernel_size=3,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.input_proj = nn.Conv1d(n_inputs, channels, kernel_size=1)   # 1x1 proj
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size, d) for d in dilations
        ])
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, n_inputs, lookback)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)             # global average pool over time
        return self.head(x).squeeze(-1)