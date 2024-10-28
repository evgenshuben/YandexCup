import torch
import torch.nn as nn

class CNNBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(CNNBlock1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CNN1d(nn.Module):
    def __init__(self, emb_size=512, num_classes=8858, input_norm=False):
        super(CNN1d, self).__init__()

        self.layer_norm = nn.LayerNorm([84, 50]) if input_norm else nn.Identity()
        self.block1 = CNNBlock1d(84, 64)
        self.block2 = CNNBlock1d(64, 128)
        self.block3 = CNNBlock1d(128, 256)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(256, emb_size)
        self.fc = nn.Linear(emb_size, num_classes)


    def forward(self, x):
        x = self.layer_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        pooled_output = self.adaptive_pool(x)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)

        emb = self.fc_emb(pooled_output)
        cls = self.fc(emb)

        return dict(emb=emb, cls=cls)