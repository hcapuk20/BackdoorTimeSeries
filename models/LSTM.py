import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, configs):
        super(LSTMClassifier, self).__init__()
        self.num_channels = configs.enc_in
        #self.hidden_size = configs.d_model_sur
        self.hidden_size = 256
        self.num_layers = configs.e_layers_sur
        self.num_classes = configs.num_class
        self.lstm = nn.LSTM(input_size=self.num_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        out, _ = self.lstm(x)
        if visualize is not None:
            latent = out[:, -1, :]
        out = self.fc(out[:, -1, :])
        if visualize is not None:
            return out, latent
        else:
            return out