import torch
import torch.nn as nn

class Bd_Tnet(nn.Module):
    def __init__(self, config,vanilla_model,generative_model):
        super(Bd_Tnet, self).__init__()
        self.vanilla_model = vanilla_model
        self.generative_model = generative_model
        self.seq_len = config.seq_len



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        trigger = self.generative_model(x_enc,x_mark_enc,None,None)
        x_mark_enc = torch.cat((x_mark_enc, x_mark_enc), dim=0)
        trigger_clipped = self.clipping(x_enc, trigger)
        x_enc_comb = torch.cat((x_enc, x_enc + trigger_clipped), dim=0)
        preds = self.vanilla_model(x_enc_comb, x_mark_enc, None, None)
        return trigger,preds

    def freeze_classifier(self):
        for param in self.vanilla_model.parameters():
            param.requires_grad = False

    def clipping(self,x_enc, x_gen,ratio=0.1):
        lim = x_enc.abs() * ratio
        x_gen_clipped = torch.where(x_gen < -lim, -lim,
                               torch.where(x_gen > lim, lim, x_gen))
        return x_gen_clipped