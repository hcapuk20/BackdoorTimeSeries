import torch
import torch.nn as nn

class Bd_transformer(nn.Module):
    def __init__(self, config,vanilla_model):
        super(Bd_transformer, self).__init__()
        self.enc_embedding = nn.Linear(config.enc_in, config.d_model)
        self.seq_len = config.seq_len
        self.vanilla_model = vanilla_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.projection = nn.Linear(
            config.d_model, config.enc_in)

    def forward(self, x,x_mark_enc):
        x_orig = x.clone()
        x = self.enc_embedding(x)
        x = self.encoder_layer(x)
        x = self.transformer_encoder(x)
        output = x * x_mark_enc.unsqueeze(-1)
        output_bd = self.projection(output)
        x_new = torch.cat((x_orig,output_bd),dim=0)
        #print(x_new.shape)
        x_mark_enc = torch.cat((x_mark_enc,x_mark_enc),dim=0)
        preds = self.vanilla_model(x_new,x_mark_enc,None, None)
        return output_bd,preds
