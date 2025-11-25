import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np


class ContrastivePhysicsAttention(nn.Module):
    def __init__(self,
                 n_bands=2751,
                 wavelength_start=4050,
                 wavelength_end=6800,
                 wr_lines=None,
                 sigma_line=8.0,
                 learnable_scale=True):
        super().__init__()
        self.n_bands = n_bands
        self.learnable_scale = learnable_scale

        self.wavelengths = np.linspace(wavelength_start, wavelength_end, n_bands)
        wavelengths_tensor = torch.tensor(self.wavelengths, dtype=torch.float32)
        self.register_buffer('wavelengths_tensor', wavelengths_tensor)

        if wr_lines is None:
            wr_lines = [4686]

        prior_positive = np.zeros(n_bands)
        for line in wr_lines:
            dist = (self.wavelengths - line) ** 2
            prior_positive += np.exp(-dist / (2 * sigma_line ** 2))
        prior_positive = prior_positive / (prior_positive.max() + 1e-6)
        self.register_buffer('prior_positive', torch.tensor(prior_positive, dtype=torch.float32))

        self.gamma_pos = nn.Parameter(torch.tensor(1.0)) if learnable_scale else 1.0


        self.correction_pos = None
        self.last_L = None

    def _build_correction_net(self, L):
        return nn.Sequential(
            nn.Linear(L, L // 4),
            nn.ReLU(),
            nn.Linear(L // 4, L),
            nn.Tanh()
        )

    def forward(self, x):
        B, C, L = x.shape
        assert C == 1

        if L != self.n_bands:
            prior_pos = interpolate(self.prior_positive.unsqueeze(0).unsqueeze(0),
                                    size=L, mode='linear', align_corners=True).squeeze(0, 1)
            wavelengths = torch.linspace(self.wavelengths[0], self.wavelengths[-1], L, device=x.device)
        else:
            prior_pos = self.prior_positive
            wavelengths = self.wavelengths_tensor

        need_rebuild = (self.last_L != L or
                        self.correction_pos is None or
                        next(self.correction_pos.parameters()).device != x.device)
        if need_rebuild:
            device = x.device
            self.correction_pos = self._build_correction_net(L).to(device)
            self.last_L = L

        scale_pos = torch.sigmoid(self.gamma_pos) if isinstance(self.gamma_pos, nn.Parameter) else 1.0
        residual_pos = self.correction_pos(prior_pos.unsqueeze(0))  # [1, L]
        attn_pos = torch.sigmoid(scale_pos * prior_pos + residual_pos.squeeze(0))

        attn_pos = attn_pos / (attn_pos.max() + 1e-6)

        attended_flux = x * attn_pos.view(1, 1, -1)

        return attended_flux


class SpectralCNN(nn.Module):
    def __init__(self, num_classes=6, input_length=2751):
        super(SpectralCNN, self).__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        self.physics_attention = ContrastivePhysicsAttention(
            n_bands=input_length,
            wavelength_start=4050,
            wavelength_end=6800,
            wr_lines=[4338, 4471, 4541, 4640, 4686, 4859, 4922, 5412, 5696, 5801, 5812, 5876, 6560],
            sigma_line=8.0,
            learnable_scale=True
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(32),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            dummy_output = self.physics_attention(dummy_input)
            dummy_output = self.conv_layers(dummy_output)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.physics_attention(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
