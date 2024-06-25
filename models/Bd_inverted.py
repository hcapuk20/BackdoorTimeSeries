import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.clip_ratio = configs.clip_ratio
        self.seq_len = configs.seq_len
        self.pred_len = self.seq_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        ############ latent to serie mapping d_model --> T 
        self.projection = nn.Linear(configs.d_model, self.seq_len, bias=True) ### fıx bug
    def trigger_gen(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape # B x T x N (number of variable)
        #print(N)
        # Embedding
        #print(x_enc.shape, x_mark_enc.shape)
        ##### Map each sub-serie to a token T ---> dmodel
        enc_out = self.enc_embedding(x_enc, None)  # B x T x d_model
        ##### Perform attention over tokens (channel-wise (variables)) ===> B x N x d_model (?)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        ###### Final phase to generate trigger from latent d_model ---> T (we go back to time domain)
        dec_out = self.projection(enc_out)
        ###### change the channel and time dimension (this is the original format) 
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        
        #print(enc_out.shape,dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,bd_labels=None):
        dec_out = self.trigger_gen(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        #dec_out = self.th_clipping2(x_enc, dec_out)
        clipped = self.clipping_amp(x_enc,dec_out,self.clip_ratio)
        return dec_out,clipped  # [B, L, D]

    def clipping_amp(self, x_enc, x_gen,
                     ratio=0.1):  #### Amp clipping =>> the change in the value can not be higher than certaın fraction of the signal amp max-min
        ## ---> shape B x C x T ---> batch channel time
        max_val, max_ind = torch.max(x_enc, dim=2)  # max value of obs in clean data
        min_val, min_ind = torch.min(x_enc, dim=2)  # min value of obs in clean data
        amp = max_val - min_val  ## amplitude of each sub-serie B x C
        amp = amp.unsqueeze(dim=2)
        x_gen_clip = torch.clamp(x_gen, min=-amp * ratio, max=amp * ratio)
        return x_gen_clip

