import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import interpolate

# =========================
# Physics-guided Attention
# =========================

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

# =====================================
# Multi-scale Physics-guided Conv Block
# =====================================

class MultiScalePhysicsConv(nn.Module):
    def __init__(self, input_length, wr_lines=None, base_channels=64):
        super().__init__()

        self.branches = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        kernel_sizes = [48, 80]
        channels_per_branch = base_channels // len(kernel_sizes)
        remainder = base_channels - channels_per_branch * len(kernel_sizes)

        if wr_lines is None:
            wr_lines = [4338, 4471, 4541, 4637, 4641,
                        4686, 4859, 5412, 5696,
                        5801, 5812, 6560]

        for i, k in enumerate(kernel_sizes):
            c = channels_per_branch + (1 if i < remainder else 0)
            padding = k // 2

            branch = nn.Sequential(
                nn.Conv1d(1, c, kernel_size=k, padding=padding),
                nn.BatchNorm1d(c),
                nn.ReLU()
            )
            self.branches.append(branch)

            # ---- Voigt initialization ----
            with torch.no_grad():
                kernel = self._init_voigt_kernel(k, wr_lines)
                branch[0].weight[:, 0, :] = kernel[None, :].repeat(c, 1)

    def _init_voigt_kernel(self, kernel_size, wr_lines, sigma=8, gamma=5):
        x = np.linspace(-50, 50, kernel_size)
        kernel = np.zeros(kernel_size)
        for _ in wr_lines:
            gauss = np.exp(-0.5 * (x / sigma) ** 2)
            lorentz = gamma ** 2 / (x ** 2 + gamma ** 2)
            kernel += gauss + lorentz
        kernel /= (np.abs(kernel).max() + 1e-6)
        return torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        out = torch.cat(feats, dim=1)   # [B, 64, L]
        out = self.pool(out)             # [B, 64, L//4]
        return out

class SpectralCNN(nn.Module):
    def __init__(self, num_classes=6, input_length=2751):
        super().__init__()

        self.physics_attention = Attention(
            n_bands=input_length,
            wavelength_start=4050,
            wavelength_end=6800,
            wr_lines=[4338, 4471, 4541, 4637, 4641,
                      4686, 4859, 5412, 5696,
                      5801, 5812, 6560],
            sigma_line=8.0,
            learnable_scale=True
        )

        self.multi_scale_conv = MultiScalePhysicsConv(input_length)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=16, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.AdaptiveMaxPool1d(32)
        )

        # ---- infer flattened size ----
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_length)
            dummy, _ = self.physics_attention(dummy)
            dummy = self.multi_scale_conv(dummy)
            dummy = self.conv_layers(dummy)
            self.flattened_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x, _ = self.physics_attention(x)
        x = self.multi_scale_conv(x)
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc(x)

    # ---------- utility ----------

    def forward_with_probs(self, x):
        return F.softmax(self.forward(x), dim=1)

    def predict(self, x):
        probs = self.forward_with_probs(x)
        return probs.argmax(dim=1), probs

    def get_attention_weights(self, x):
        """
        return: [L]
        """
        with torch.no_grad():
            B, C, L = x.shape
            assert C == 1

            if self.physics_attention.correction_pos is None:
                self.physics_attention.correction_pos = \
                    self.physics_attention._build_correction_net(L).to(x.device)

            prior = self.physics_attention.prior_positive
            if L != prior.numel():
                prior = interpolate(prior[None, None, :],
                                    size=L,
                                    mode='linear',
                                    align_corners=True).squeeze()

            scale = torch.sigmoid(self.physics_attention.gamma_pos)
            residual = self.physics_attention.correction_pos(prior[None, :]).squeeze()
            attn = torch.sigmoid(scale * prior + residual)
            attn = attn / (attn.max() + 1e-6)

            return attn
