import torch
import torch.nn as nn
  
  
 ################## Combined Model ===> backdoor trigger network + surrogate classifier network ###################   
class Bd_Tnet(nn.Module):
    def __init__(self, config, surr_classifier_model , bd_trigger_model ):
        super(Bd_Tnet, self).__init__()
        self.classifier = surr_classifier_model
        self.trigger = bd_trigger_model
        self.seq_len = config.seq_len ## seq length
    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

    def clipping(self,x_enc, x_gen, ratio=0.1): #### rational clipping =>> the change in the value can not be higher than certain ratio
        lim = x_enc.abs() * ratio
        x_gen_clipped = torch.where(x_gen < -lim, -lim,
                               torch.where(x_gen > lim, lim, x_gen))
        return x_gen_clipped
    def clipping_amp(self,x_enc, x_gen, ratio=0.1): #### Amp clipping =>> the change in the value can not be higher than certaın fraction of the signal amp max-min 
        ## ---> shape B x C x T ---> batch channel time
        max_val, max_ind = torch.max( x_enc, dim=2) # max value of obs in clean data
        min_val, min_ind = torch.min( x_enc, dim=2) # min value of obs in clean data
        amp = max_val-min_val ## amplitude of each sub-serie B x C
        amp = amp.unsqueeze(dim=2)
        x_gen_clip = torch.clamp(x_gen, min = -amp*ratio, max = amp*ratio)
        
        return x_gen_clip

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec): ### x_dec, x_mark_dec for the forecast task ### x_mark_enc, x_mark_dec ---> masks not used for classification
        ####### Generate Trigger #######
        trigger = self.trigger(x_enc,x_mark_enc,None,None) ## ====> generate the trigger pattern (additive)
        x_mark_enc = torch.cat((x_mark_enc, x_mark_enc), dim=0) ## ====> masks not used for classification but use to make code compatible
        trigger_clipped = self.clipping_amp(x_enc, trigger) ## ====> clip the additive trigger
        x_enc_comb = torch.cat((x_enc, x_enc + trigger_clipped), dim=0) ## ===> combined data of clean and backdoored
        preds = self.classifier(x_enc_comb, x_mark_enc, None, None)
        return trigger, trigger_clipped, preds


