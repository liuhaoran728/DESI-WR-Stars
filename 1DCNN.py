import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    def __init__(self, num_classes=6, input_length=2751):
        super(SpectralCNN, self).__init__()
        self.num_classes = num_classes
        self.input_length = input_length

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
            x = torch.randn(1, 1, input_length)
            x = self.conv_layers(x)
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
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