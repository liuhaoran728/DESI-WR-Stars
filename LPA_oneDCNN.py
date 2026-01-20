import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import numpy as np

class Attention(nn.Module):

    def __init__(self,
                 n_bands=2751,
                 wavelength_start=4050,
                 wavelength_end=6800,
                 wr_lines=None,
                 sigma_line=8.0,
                 max_shift=50,
                 learnable_scale=True):
        super().__init__()
        self.n_bands = n_bands
        self.max_shift = max_shift

        self.wavelengths = np.linspace(wavelength_start, wavelength_end, n_bands)
        self.register_buffer(
            'wavelengths_tensor',
            torch.tensor(self.wavelengths, dtype=torch.float32)
        )

        if wr_lines is None:
            wr_lines = [4338, 4471, 4686, 4859, 5801, 5812, 5876, 6560]

        # ---- prior template ----
        prior_positive = np.zeros(n_bands)
        for line in wr_lines:
            dist = (self.wavelengths - line) ** 2
            prior_positive += np.exp(-dist / (2 * sigma_line ** 2))

        prior_positive /= (prior_positive.max() + 1e-6)
        self.register_buffer(
            'prior_positive',
            torch.tensor(prior_positive, dtype=torch.float32)
        )

        # learnable scale
        self.gamma_pos = nn.Parameter(torch.tensor(1.0)) if learnable_scale else None

        self.correction_pos = None
        self.last_L = None

    def _build_correction_net(self, L):
        return nn.Sequential(
            nn.Linear(L, L // 4),
            nn.ReLU(),
            nn.Linear(L // 4, L),
            nn.Tanh()
        )

    def _shift_prior(self, prior, shift_pixels):
        if shift_pixels == 0:
            return prior
        elif shift_pixels > 0:
            return torch.cat([
                torch.zeros(shift_pixels, device=prior.device),
                prior[:-shift_pixels]
            ])
        else:
            return torch.cat([
                prior[-shift_pixels:],
                torch.zeros(-shift_pixels, device=prior.device)
            ])

    def forward(self, x, shift=0):
        """
        x: [B, 1, L]
        return:
            attended_flux: [B, 1, L]
            attn_pos:      [B, L]
        """
        B, C, L = x.shape
        assert C == 1

        # ---- interpolate prior ----
        if L != self.n_bands:
            prior_pos = interpolate(
                self.prior_positive[None, None, :],
                size=L,
                mode='linear',
                align_corners=True
            ).squeeze()
        else:
            prior_pos = self.prior_positive

        # ---- shift ----
        shift = max(-self.max_shift, min(self.max_shift, shift))
        prior_pos = self._shift_prior(prior_pos, shift)

        # ---- build correction net if needed ----
        if (self.correction_pos is None) or (self.last_L != L):
            self.correction_pos = self._build_correction_net(L).to(x.device)
            self.last_L = L

        # ---- feature response ----
        residual_input = x.squeeze(1)                      # [B, L]
        feature_response = torch.sigmoid(
            residual_input - residual_input.mean(dim=1, keepdim=True)
        )

        # ---- attention ----
        scale = torch.sigmoid(self.gamma_pos) if self.gamma_pos is not None else 1.0
        residual_pos = self.correction_pos(prior_pos[None, :]).expand(B, L)

        attn_pos = torch.sigmoid(
            scale * (prior_pos[None, :] * feature_response) + residual_pos
        )

        # normalize per sample
        attn_pos = attn_pos / (attn_pos.max(dim=1, keepdim=True)[0] + 1e-6)

        attended_flux = x * attn_pos[:, None, :]

        return attended_flux, attn_pos

class SpectralCNN(nn.Module):
    def __init__(self, num_classes=6, input_length=2751):
        super(SpectralCNN, self).__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        # 物理引导注意力模块
        self.physics_attention = Attention(
            n_bands=input_length,
            wavelength_start=4050,
            wavelength_end=6800,
            wr_lines=[4338, 4471, 4541, 4640, 4686, 4859, 4922, 5412, 5696, 5801, 5812, 5876, 6560],
            sigma_line=8.0,
            learnable_scale=True
        )

        # CNN 卷积层
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
            nn.AdaptiveMaxPool1d(32),  # 固定输出长度为32
        )

        # 计算展平维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            attended_flux, _ = self.physics_attention(dummy_input)
            dummy_output = self.conv_layers(attended_flux)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        x: [B, 1, L]
        """
        # Attention 输出 attended_flux, attn_map
        attended_flux, _ = self.physics_attention(x)   # 只取 attended_flux

        # CNN
        x = self.conv_layers(attended_flux)

        # Flatten
        x = self.flatten(x)

        # FC
        x = self.fc(x)
        return x
