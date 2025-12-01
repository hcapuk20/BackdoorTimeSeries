import torch
import torch.nn as nn


class bd_CNN(nn.Module):
    """
    CNN Trigger Generator Network for Backdoor Time Series
    Architecture:
        Input: (L, D)
        Conv1D (ReLU): kernel 15*1, 128*D filters -> (L, 128*D)
        Conv1D (ReLU): kernel 21*1, 512*D filters -> (L, 512*D)
        FC (ReLU): 256*D units -> (L, 256*D)
        FC (tanh): D units -> (L, D)
    Input shape: B x L x D (Batch, Sequence Length, Number of Variates)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len  # L: sequence length
        self.enc_in = configs.enc_in    # D: number of variates
        self.clip_ratio = configs.clip_ratio
        
        D = self.enc_in
        
        # First CNN layer: kernel size 15, 128*D filters
        self.conv1 = nn.Conv1d(
            in_channels=D,
            out_channels=128 * D,
            kernel_size=15,
            padding=7  # same padding to preserve sequence length
        )
        self.relu1 = nn.ReLU()
        
        # Second CNN layer: kernel size 21, 512*D filters
        self.conv2 = nn.Conv1d(
            in_channels=128 * D,
            out_channels=512 * D,
            kernel_size=21,
            padding=10  # same padding to preserve sequence length
        )
        self.relu2 = nn.ReLU()
        
        # FC layer 1: 512*D -> 256*D (applied per time step)
        self.fc1 = nn.Linear(512 * D, 256 * D)
        self.relu3 = nn.ReLU()
        
        # FC layer 2: 256*D -> D with tanh (applied per time step)
        self.fc2 = nn.Linear(256 * D, D)
        self.tanh = nn.Tanh()

    def trigger_gen(self, x_enc):
        """
        Generate trigger pattern from input
        x_enc: B x L x D
        """
        # Permute to B x D x L for Conv1d
        x = x_enc.permute(0, 2, 1)
        
        # First CNN layer: (B, D, L) -> (B, 128*D, L)
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Second CNN layer: (B, 128*D, L) -> (B, 512*D, L)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Permute to B x L x 512*D for FC layers
        x = x.permute(0, 2, 1)
        
        # FC layer 1: (B, L, 512*D) -> (B, L, 256*D)
        x = self.fc1(x)
        x = self.relu3(x)
        
        # FC layer 2: (B, L, 256*D) -> (B, L, D)
        x = self.fc2(x)
        trigger = self.tanh(x)
        
        return trigger

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """
        Forward pass
        x_enc: B x T x N input time series
        Returns: trigger pattern and clipped trigger
        """
        trigger = self.trigger_gen(x_enc)
        clipped = self.clipping_amp(x_enc, trigger, self.clip_ratio)
        return trigger, clipped

    def clipping_amp(self, x_enc, x_gen, ratio=0.1):
        """
        Amplitude clipping: the change in value cannot be higher than 
        a certain fraction of the signal amplitude (max-min)
        x_enc: B x T x N
        x_gen: B x T x N (generated trigger)
        """
        # Get max and min values along time dimension
        max_val, _ = torch.max(x_enc, dim=1, keepdim=True)  # B x 1 x N
        min_val, _ = torch.min(x_enc, dim=1, keepdim=True)  # B x 1 x N
        amp = max_val - min_val  # amplitude of each variate: B x 1 x N
        
        # Clip the trigger to be within [-amp*ratio, amp*ratio]
        x_gen_clip = torch.clamp(x_gen, min=-amp * ratio, max=amp * ratio)
        return x_gen_clip

