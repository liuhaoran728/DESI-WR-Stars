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

        kernel_sizes = [32, 64, 96]
        channels_per_branch = base_channels // len(kernel_sizes)
        remainder = base_channels - channels_per_branch * len(kernel_sizes)

        if wr_lines is None:
            wr_lines = [4686]

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
        x = np.linspace(-50, 50, kernel_size)
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
        out = torch.cat(features, dim=1)
        out = self.pool(out)
        return out

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

        # 构建波长数组
        self.wavelengths = np.linspace(wavelength_start, wavelength_end, n_bands)
        wavelengths_tensor = torch.tensor(self.wavelengths, dtype=torch.float32)
        self.register_buffer('wavelengths_tensor', wavelengths_tensor)


        if wr_lines is None:
            wr_lines = [4338, 4471, 4686, 4859, 5801, 5812, 5876, 6560]

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
                                    size=L, mode='linear', align_corners=True).squeeze()
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
        residual_pos = self.correction_pos(prior_pos.unsqueeze(0))
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
            wr_lines=[4338, 4471, 4541, 4637, 4641, 4686, 4859, 5412, 5696, 5801, 5812, 6560],
            sigma_line=8.0,
            learnable_scale=True
        )

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

        # 推导展平维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_length)
            dummy = self.physics_attention(dummy)
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
        x = self.physics_attention(x)
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
        with torch.no_grad():
            B, C, L = x.shape
            assert C == 1

            if L != self.input_length:
                prior_pos = interpolate(self.physics_attention.prior_positive.unsqueeze(0).unsqueeze(0),
                                        size=L, mode='linear', align_corners=True).squeeze()
            else:
                prior_pos = self.physics_attention.prior_positive

            scale_pos = torch.sigmoid(self.physics_attention.gamma_pos)
            residual_pos = self.physics_attention.correction_pos(prior_pos.unsqueeze(0)).squeeze(0)
            attn_pos = torch.sigmoid(scale_pos * prior_pos + residual_pos)
            attn_pos = attn_pos / (attn_pos.max() + 1e-6)

            return attn_pos