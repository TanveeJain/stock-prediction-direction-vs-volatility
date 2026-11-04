import torch
import torch.nn as nn

from config import WINDOW


class Net(nn.Module):
    def __init__(self, d, h=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.pos = nn.Parameter(torch.randn(1, WINDOW, h) * 0.02)
        layer = nn.TransformerEncoderLayer(h, heads, 128, 0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos[:, :x.size(1)])
        return self.head(z.mean(1)).squeeze(-1)
