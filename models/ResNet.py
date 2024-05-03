import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self,configs, in_channels, out_channels, no_expand=False):
        super().__init__()
        self.no_expand = no_expand
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        if no_expand:
            self.bn_skip = nn.BatchNorm1d(no_expand)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        if not self.no_expand:
            x_res = self.bn3(self.conv_res(x))
        else:
            x_res = self.bn_skip(x)

        return self.relu(x_res + x3)

class ResNetClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.res_layer1 = ResBlock(configs, configs.seq_len, 64, False)
        self.res_layer2 = ResBlock(configs, 64, 128, False)
        self.res_layer3 = ResBlock(configs, 128, 128, 128)
        self.fc_layer = nn.Linear(configs.enc_in, configs.num_class)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, mask=None):
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = torch.mean(x, dim=1) # global average pooling
        x = self.fc_layer(x)

        out = F.softmax(x, dim=1)
        return out