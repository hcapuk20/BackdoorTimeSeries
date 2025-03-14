################ This is the code for dynamic target attack (all to any)############
###### Modifications --> trainable weights for initial tokens to identify the target
### Notes:
## -we need numb_of_class info
## -initialization of tokens Kaiming or others ????



import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_bd


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model_bd x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.numb_class = configs.numb_class
        self.clip_ratio = configs.clip_ratio
        self.d_model_bd = configs.d_model_bd
        self.target_token = torch.nn.Parameter(torch.Tensor(configs.numb_class,configs.d_model_bd), requires_grad=configs.trainable_token) # C (numb_class x d_model_bd) size matrix of trainable weights of target tokens
        nn.init.orthogonal_(self.target_token)
        if configs.trainable_token:
            h = self.target_token.register_hook(lambda grad: grad * configs.token_hook)
        ######### initialize tokens ##### 
        padding = stride
        _patch_len = configs.ptst_patch_len if hasattr(configs, 'ptst_patch_len') else patch_len

        # patching and embedding
        self.patch_embedding = PatchEmbedding_bd(
            configs.d_model_bd, _patch_len, stride, padding, configs.dropout,self.target_token)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model_bd, configs.n_heads_bd),
                    configs.d_model_bd,
                    configs.d_ff_bd,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_bd)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model_bd)
        )

        # Prediction Head
        self.head_nf = configs.d_model_bd * \
                       int((configs.seq_len - _patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        ################ Here we also have a vector of length B targets #################
    def trigger_gen(self, x_enc, x_mark_enc, x_dec, x_mark_dec,targets=None):
        # Normalization from Non-stationary Transformer
        targets = torch.randint(0,self.numb_class,(x_enc.shape[0],))
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model_bd]
        enc_out, n_vars = self.patch_embedding(x_enc,targets)
        ############# embedding the target tokens  ############## can be revised or optimized
        ## For each given batch of samples and targets we generate batch of target token to be appended beginning of the patch sequence
        ### the shape of the target tokens B x n_vars x 1 x d_model_bd

        ###### concatenate targs_token with enc_out
      

      

        # Encoder
        # z: [bs * nvars x patch_num x d_model_bd]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model_bd]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model_bd x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        # Decoder
        dec_out = self.head(enc_out[:,:,:,1:])  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, targets=None):
        dec_out = self.trigger_gen(x_enc, x_mark_enc, x_dec, x_mark_dec,targets)[:, -self.pred_len:, :]
        clipped = self.clipping_amp(x_enc,dec_out,self.clip_ratio)
        return dec_out,clipped # [B, L, D]

    def clipping_amp(self, x_enc, x_gen,
                     ratio=0.1):  #### Amp clipping =>> the change in the value can not be higher than certaın fraction of the signal amp max-min
        ## ---> shape B x C x T ---> batch channel time
        max_val, max_ind = torch.max(x_enc, dim=2)  # max value of obs in clean data
        min_val, min_ind = torch.min(x_enc, dim=2)  # min value of obs in clean data
        amp = max_val - min_val  ## amplitude of each sub-serie B x C
        amp = amp.unsqueeze(dim=2)
        x_gen_clip = torch.clamp(x_gen, min=-amp * ratio, max=amp * ratio)
        return x_gen_clip
