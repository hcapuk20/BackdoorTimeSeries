import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, configs):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = 100  # placeholder for now
        self.num_layers = configs.d_model  # unsure about this
        self.lstm = nn.LSTM(configs.enc_in, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, configs.num_class)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, mask=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
