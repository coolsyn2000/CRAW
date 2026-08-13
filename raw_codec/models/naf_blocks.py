from __future__ import annotations

from typing import Any

import torch
from torch import nn


# Adapted from megvii-research/NAFNet (MIT License).
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + eps).sqrt()
        ctx.save_for_backward(normalized, variance, weight)
        return weight.view(1, channels, 1, 1) * normalized + bias.view(
            1, channels, 1, 1
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        eps = ctx.eps
        _, channels, _, _ = grad_output.size()
        normalized, variance, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, channels, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_normalized = (grad * normalized).mean(dim=1, keepdim=True)
        grad_x = (grad - normalized * mean_grad_normalized - mean_grad) / torch.sqrt(
            variance + eps
        )
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.simple_gate = SimpleGate()
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.simple_gate(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        residual = inp + self.dropout1(x) * self.beta
        x = self.conv5(self.simple_gate(self.conv4(self.norm2(residual))))
        return residual + self.dropout2(x) * self.gamma
