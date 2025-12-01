from cProfile import label

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
  
  
 ################## Combined Model ===> backdoor trigger network + surrogate classifier network ###################   
class Bd_Tnet(nn.Module):
    def __init__(self, config, bd_trigger_model):
        super(Bd_Tnet, self).__init__()
        self.trigger = bd_trigger_model
        self.seq_len = config.seq_len ## seq length
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,bd_labels=None): ### x_dec, x_mark_dec for the forecast task ### x_mark_enc, x_mark_dec ---> masks not used for classification
        ####### Generate Trigger #######
        trigger,trigger_clipped = self.trigger(x_enc,x_mark_enc,None,None,bd_labels) ## ====> generate the trigger pattern (additive)
        #t = trigger_clipped.permute(0, 2, 1)
        #plt.plot(t[0,2,:].detach().cpu().numpy(),label='orig')
        #trigger = self.moving_average_pool1d(trigger, kernel_size=11) ## ====> average pooling
        #new_clipped = self.trigger.clipping_amp(x_enc, trigger, 0.1) ## ====> clipping amplitude
        #t_ma = new_clipped.permute(0, 2, 1)
        # plt.plot(t_ma[0, 2, :].detach().cpu().numpy(),label='ma')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        #trigger_clipped = new_clipped
        return trigger, trigger_clipped

    def moving_average_conv1d(self, x, kernel_size):
        w = torch.ones(kernel_size).to(x.device) / kernel_size
        return torch.conv1d(x.permute(0,2,1), w, padding=w.size(1)-1)

    def moving_average_pool1d(self, x, kernel_size):
        x = x.permute(0, 2, 1)
        avg_pool1d = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=5).to(x.device)
        x = avg_pool1d(x)
        return x.permute(0, 2, 1)
