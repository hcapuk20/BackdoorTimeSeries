import torch
import torch.nn as nn
  
  
 ################## Combined Model ===> backdoor trigger network + surrogate classifier network ###################   
class Bd_Tnet(nn.Module):
    def __init__(self, config, bd_trigger_model):
        super(Bd_Tnet, self).__init__()
        self.trigger = bd_trigger_model
        self.seq_len = config.seq_len ## seq length
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,bd_labels=None): ### x_dec, x_mark_dec for the forecast task ### x_mark_enc, x_mark_dec ---> masks not used for classification
        ####### Generate Trigger #######
        trigger,trigger_clipped = self.trigger(x_enc,x_mark_enc,None,None,bd_labels) ## ====> generate the trigger pattern (additive)
        return trigger, trigger_clipped


