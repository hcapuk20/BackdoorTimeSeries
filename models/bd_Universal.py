import torch
import torch.nn as nn

from models.TimesNet import Model as TimesNet


class Bd_Tnet(TimesNet):
    def __init__(self, config,vanilla_model,generative_model):
        super(Bd_Tnet, self).__init__(config)
        self.vanilla_model = vanilla_model
        self.generative_model = generative_model
        self.seq_len = config.seq_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bd_x = self.generative_model(x_enc,x_mark_enc,None,None)# Normalization from Non-stationary Transformer
        x_mark_enc = torch.cat((x_mark_enc, x_mark_enc), dim=0)
        x_enc_comb = torch.cat((x_enc, bd_x), dim=0)
        preds = self.vanilla_model(x_enc_comb, x_mark_enc, None, None)
        return bd_x,preds

