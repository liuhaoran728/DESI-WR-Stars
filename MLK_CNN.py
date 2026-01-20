import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import interpolate

class MultiScalePhysicsConv(nn.Module):
    def __init__(self, input_length, wr_lines=None, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.branches = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        kernel_sizes = [48, 80]
        channels_per_branch = base_channels // len(kernel_sizes)  # 64 // 3 ≈ 21
        remainder = base_channels - channels_per_branch * len(kernel_sizes)

        if wr_lines is None:
            wr_lines = [4338, 4471, 4541, 4637, 4641, 4686, 4859, 5412, 5696, 5801, 5812, 6560]

        for i, k in enumerate(kernel_sizes):
            c = channels_per_branch + (1 if i < remainder else 0)
            padding = k // 2
            branch = nn.Sequential(
                nn.Conv1d(1, c, kernel_size=k, padding=padding),
                nn.BatchNorm1d(c),
                nn.ReLU()
            )
            self.branches.append(branch)

            with torch.no_grad():
                voigt_kernel = self._init_voigt_kernel(k, wr_lines)
                branch[0].weight[:, 0, :] = voigt_kernel.unsqueeze(0).repeat(c, 1)

    def _init_voigt_kernel(self, kernel_size, wr_lines, sigma=8, gamma=5):
        x = np.linspace(-100, 100, kernel_size)  # 覆盖 ~200 Å
        kernel = np.zeros(kernel_size)
        for _ in wr_lines:
            gauss = np.exp(-0.5 * (x / sigma) ** 2)
            lorentz = gamma ** 2 / (x ** 2 + gamma ** 2)
            kernel += (gauss + lorentz)
        kernel = kernel / (np.abs(kernel).max() + 1e-6)
        return torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x):
        features = []
        for branch in self.branches:
            features.append(branch(x))
        out = torch.cat(features, dim=1)  # [B, 64, L]
        out = self.pool(out)              # [B, 64, L//4]
        return out


class SpectralCNN(nn.Module):
    def __init__(self, num_classes=6, input_length=2751):
        super(SpectralCNN, self).__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        self.multi_scale_conv = MultiScalePhysicsConv(input_length)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 256, kernel_size=16, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.AdaptiveMaxPool1d(32)
        )

        with torch.no_grad():
            dummy = torch.randn(1, 1, input_length)
            dummy = self.multi_scale_conv(dummy)
            dummy = self.conv_layers(dummy)
            self.flattened_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.multi_scale_conv(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward_with_probs(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1)
        return pred_class, probs

    def get_attention_weights(self, x):
        raise NotImplementedError("Attention mechanism has been removed.")