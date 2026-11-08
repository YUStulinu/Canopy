"""A U-Net for 20-channel Sentinel-2 composites, ten land-cover classes.

Only plain operations (convolution, batch norm, ReLU, max pooling, transposed convolution,
concatenation) so the exported ONNX graph runs in the Rust tile server's inference engine.
"""
from __future__ import annotations

import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, channels: int = 20, classes: int = 10, width: int = 32, depth: int = 4):
        super().__init__()
        widths = [width * 2**k for k in range(depth + 1)]  # 32 64 128 256 512
        self.down = nn.ModuleList([block(channels, widths[0])] + [block(widths[k - 1], widths[k]) for k in range(1, depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[k], widths[k - 1], 2, stride=2) for k in range(depth, 0, -1)])
        self.dec = nn.ModuleList([block(2 * widths[k - 1], widths[k - 1]) for k in range(depth, 0, -1)])
        self.head = nn.Conv2d(widths[0], classes, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, d in enumerate(self.down):
            x = d(x)
            if k < self.depth:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def multiple() -> int:
    """Input height and width must be multiples of this."""
    return 16
