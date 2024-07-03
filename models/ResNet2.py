import torch
import torch.nn as nn

class ResidualBlock(nn.Module): ##ResidualBlock consist of conv1d, batchnorm and relu
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, configs):
        self.num_channels = configs.enc_in # num channels is equal to number of multivariate.
        self.num_classes = configs.num_class
        ## convolution done on time
        super(ResNet, self).__init__()
        self.layer1 = self.make_layer(ResidualBlock, self.num_channels, 64, 2)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, self.num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        layers.append(block(in_channels, out_channels))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        out = x.permute(0, 2, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if visualize is not None:
            latent = out
        out = self.fc(out)
        if visualize is not None:
            return out, latent
        return out