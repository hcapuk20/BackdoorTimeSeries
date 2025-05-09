import os.path

import torch
import numpy as np
import torch.nn as nn
from utils.model_ops import *
from defences.fp import Pruning
from defences.nc import main as nc_main
from defences.strip import cleanser
from defences.spectre import spectre_filtering
from utils.visualize import visualize
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

#########

################# Related works with Code #############
# Dynamic input-aware https://github.com/VinAIResearch/input-aware-backdoor-attack-release/blob/master/train.py
# Label smoothing for backdoor -->> https://arxiv.org/pdf/2202.11203

##################### Regularizers ########################

### This loss is designed to minimize the alignment on the frequency domain

def fft_vals(x_clean,x_back,args,i): # input shape T x C #outputshape C
    x_c, x_b = x_clean.float().permute(1,0).float(), x_back.permute(1,0).float()
    xf_c = abs(torch.fft.rfft(x_c, dim=1))  ## clean data on the freq domain
    xf_b = abs(torch.fft.rfft(x_b, dim=1))  ## backdoored data on the freq domain
    xf_c2 = xf_c[:, 1:-1]  ## ignore the freq 0
    xf_b2 = xf_b[:, 1:-1]  ## ignore the freq 0
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(xf_c2[0].cpu().numpy() ,label='clean', color='blue')
    axs[0].plot(xf_b2[0].cpu().numpy(),label='bd', color='red')
    axs[1].plot(xf_c2[1].cpu().numpy(),label='clean', color='blue')
    axs[1].plot(xf_b2[1].cpu().numpy(),label='bd', color='red')
    axs[2].plot(xf_c2[2].cpu().numpy(),label='clean', color='blue')
    axs[2].plot(xf_b2[2].cpu().numpy(),label='bd', color='red')
    axs[0].set_ylim([0, 10])
    axs[1].set_ylim([0, 10])
    axs[2].set_ylim([0, 10])
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.tight_layout()
    if not os.path.exists('fft_figs'):
        os.makedirs('fft_figs', exist_ok=True)
    plt.savefig('fft_figs/fft-{}-{}.png'.format(args.sim_id,i))
    plt.clf()

def freqreg(x_clean,x_back):
    x_c, x_b = x_clean.float().permute(0, 2, 1).float(), x_back.permute(0, 2, 1).float()
    xf_b = abs(torch.fft.rfft(x_b, dim=2))  ## backdoored data on the freq domain
    xf_b2 = xf_b[:, :, 1:-1]  ## ignore the freq 0
    return torch.norm(xf_b2, p=1, dim=2).mean()  # This term can be summed or averaged

def fftreg(x_clean,x_back): # input shape B x T x C #outputshape B x C
    x_c,x_b = x_clean.float().permute(0,2,1).float(),x_back.permute(0,2,1).float()
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    xf_c = abs(torch.fft.rfft(x_c, dim=2)) ## clean data on the freq domain
    xf_b = abs(torch.fft.rfft(x_b, dim=2))  ## backdoored data on the freq domain
    xf_c2 = xf_c[:,:,1:-1] ## ignore the freq 0
    xf_b2 = xf_b[:,:,1:-1] ## ignore the freq 0
    #print(cos(xf_c2,xf_b2).mean())
    return cos(xf_c2,xf_b2).mean() ##### This term can be summed or averaged #########

def l2_reg(clipped_trigger, trigger): # maximize clipped trigger.
    return -torch.norm(clipped_trigger)

def moving_average_conv(time_series, kernel_size=5):
    """
    Compute the moving average for a multivariate time series.

    Parameters:
    time_series (torch.Tensor): Input tensor of shape (batch_size, num_variables, sequence_length).
    kernel_size (int): Size of the moving average window.

    Returns:
    torch.Tensor: Tensor with the moving average applied.
    """

    # Create a convolutional layer to compute the moving average
    batch_size, num_variables, sequence_length = time_series.shape
    print(time_series.shape)
    conv = nn.Conv1d(in_channels=num_variables, out_channels=num_variables, kernel_size=kernel_size, stride=1,
                     padding=kernel_size // 2, groups=num_variables, bias=False)

    # Set the weights of the convolutional layer to compute the average
    with torch.no_grad():
        conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size

    # Apply the moving average filter
    #time_series = time_series.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, num_variables)
    smoothed_series = conv(time_series)
    #smoothed_series = smoothed_series.permute(0, 2, 1)  # Change back to (batch_size, num_variables, sequence_length)

    return smoothed_series


def moving_average_pool(time_series, kernel_size=5):
    """
    Compute the moving average for a multivariate time series using pooling.

    Parameters:
    time_series (torch.Tensor): Input tensor of shape (batch_size, num_variables, sequence_length).
    kernel_size (int): Size of the moving average window.

    Returns:
    torch.Tensor: Tensor with the moving average applied.
    """
    # Ensure the input is in the correct shape (batch_size, num_variables, sequence_length)
    assert len(time_series.shape) == 3, "Input tensor must be 3-dimensional"

    # Create a pooling layer to compute the moving average
    pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

    # Apply the moving average filter
    #time_series = time_series.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, num_variables)
    smoothed_series = pool(time_series)
    #smoothed_series = smoothed_series.permute(0, 2, 1)  # Change back to (batch_size, num_variables, sequence_length)

    return smoothed_series

class FrequencyDomainSimilarityLoss(nn.Module):
    def __init__(self,minimize=True):
        super(FrequencyDomainSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.minimize = minimize

    def forward(self, time_series1, time_series2):
        """
        Compute the similarity between two time series in the frequency domain.

        Parameters:
        time_series1 (torch.Tensor): First input tensor of shape (batch_size, num_variables, sequence_length).
        time_series2 (torch.Tensor): Second input tensor of shape (batch_size, num_variables, sequence_length).

        Returns:
        torch.Tensor: Similarity loss.
        """
        # Convert time series to frequency domain using FFT
        freq_series1 = torch.fft.rfft(time_series1, dim=-1)
        freq_series2 = torch.fft.rfft(time_series2, dim=-1)

        # Compute the cosine similarity between the frequency domain representations
        similarity = self.cosine_similarity(freq_series1, freq_series2)

        # The loss is 1 - similarity to maximize similarity
        if self.minimize:
            loss = similarity.mean()
        else:
            loss = 1 - similarity.mean()

        return loss


def reg_loss(x_clean,trigger,trigger_clip,args):
    l2_loss = l2_reg(trigger_clip,trigger)
    cos_loss = fftreg(x_clean,x_clean+trigger_clip)
    freq_loss = freqreg(x_clean,x_clean+trigger_clip)
    ma_clipped = moving_average_conv(x_clean+trigger_clip)
    freq_loss_2 = FrequencyDomainSimilarityLoss()(x_clean,ma_clipped)
    ma_loss = torch.norm(moving_average_pool(trigger)) * 1
    #print(ma_loss)
    reg_total = freq_loss_2 - ma_loss
    if reg_total != 0:
        return reg_total
    else:
        return None



### for the set of regularizers

###################### Mixup operation for input/output #################################

def mixup_forcast(x_clean, x_backdoor, y_clean, y_backdoor, alpha=2, beta=2): ### forecast task
    #### input shapes x -> B x C x T
    #### output shapes y -> B x C x L
    bs = x_clean.size(0)
    channel= x_clean.size(1)
    time_y = y_clean.size(2)
    time_x = x_clean.size(2)
    ################ We utilize a beta function to sample lamda values ##########
    lam = torch.tensor( np.random.beta(a=alpha,  b=beta), requires_grad=False)
    lam = (lam.unsqueeze(dim=-1)).unsqueeze(dim=-1)
    lam_x = lam.repeat(1, channel, time_x)
    lam_y = lam.repeat(1, channel, time_y)
    x_mixed = lam_x * x_backdoor + (1 - lam_x) * x_clean
    y_mixed = lam_y * y_backdoor + (1 - lam_y) * y_clean
    return x_mixed, y_mixed

def mixup_class(x_clean, x_backdoor, alpha=2, beta=2): ### classification task
    ### Following the framework in 
    ### https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    ### Instead  of label mixing we compute loss for each loss and take weighted avg of loss (wrt lam)
    #### input shapes x -> B x C x T
    bs = x_clean.size(0)
    channel= x_clean.size(1)
    time_x = x_clean.size(2)
    ################ We utilize a beta function to sample lamda values ##########
    lam = torch.tensor( np.random.beta(alpha,  beta, bs), requires_grad=False).to(x_clean.device).float()
    lam_x = (lam.unsqueeze(dim=-1)).unsqueeze(dim=-1)
    lam_x = lam_x.repeat(1, channel, time_x)
    x_mixed = lam_x * x_backdoor + (1 - lam_x) * x_clean
    return x_mixed.float(), lam # we output lam as well to be used weight loss terms
    ####################### end of mixup #########################################





def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
    

def epoch_with_diversity(bd_model,surr_model, loader1, args, loader2=None, opt=None,opt2=None,train=True): ##### The main training module
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    criterion_div = nn.MSELoss(reduction="none")

    # to make the zipped loop consistent with or without diversity loss
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)

    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(range(len(loader1)), loader1, loader2):
            loss_div = 0.0 # for consistency with optional diversity loss
            loss_clean = torch.tensor([0.0])
            loss_bd = torch.tensor([0.0]) # for consistency with optional cross losses and logging.
            bd_model.zero_grad()
            surr_model.zero_grad()
            #### Fetch clean data
            batch_x = batch_x.float().to(args.device)
            #### Fetch mask (for forecast task)
            padding_mask = padding_mask.float().to(args.device)
            #### Fetch labels
            label = label.to(args.device)
            #### Generate backdoor labels ####### so far we focus on fixed target scenario
            bd_labels = torch.ones_like(label).to(args.device) * bd_label ## comes from argument
            if batch_x2 is not None:
                batch_x2 = batch_x2.float().to(args.device)
                padding_mask2 = padding_mask2.float().to(args.device)
                label2 = label2.to(args.device)
                bd_labels2 = torch.ones_like(label2).to(args.device) * bd_label ## comes from argument

            #### Combine true and target labels
            ########### Here we generate trigger #####################
        
            trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None)
            if batch_x2 is not None and args.div_reg:
                
                trigger2, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None)

                ### DIVERGENCE LOSS CALCULATION
                input_distances = criterion_div(batch_x, batch_x2)
                input_distances = torch.mean(input_distances, dim=(1, 2))
                input_distances = torch.sqrt(input_distances)

                ### TODO: do we use trigger or trigger_clip here?
                trigger_distances = criterion_div(trigger, trigger2)
                trigger_distances = torch.mean(trigger_distances, dim=(1, 2))
                trigger_distances = torch.sqrt(trigger_distances)

                loss_div = input_distances / (trigger_distances + 1e-6) # second value is the epsilon, arbitrary for now
                loss_div = torch.mean(loss_div) * args.div_reg

            
            mask = (label != bd_label).float().to(args.device) if args.attack_only_nontarget else torch.ones_like(label).float().to(args.device)
            mask = mask.unsqueeze(-1).expand(-1,trigger_clip.shape[-2],trigger_clip.shape[-1])
            clean_pred = surr_model(batch_x, padding_mask,None,None)
            bd_pred = surr_model(batch_x + trigger_clip * mask, padding_mask,None,None)
            if batch_x2 is not None and args.div_reg:
                # cross loss from input aware paper (coupled with diversity loss from the same work)
                bs = batch_x.shape[0]
                num_bd = int(0.5 * bs) # args.p_attack, values taken from the best result from input-aware paper
                num_cross = int(0.1 * bs) # args.p_cross
                bd_inputs = (batch_x + trigger_clip * mask)[:num_bd]
                cross_inputs = (batch_x + trigger_clip2 * mask)[num_bd : num_bd + num_cross]
                total_inputs = torch.cat((bd_inputs, cross_inputs, batch_x[num_bd + num_cross:])).to(args.device)
                total_targets = torch.cat((bd_labels[:num_bd], label[num_bd:])).to(args.device)
                total_pred = surr_model(total_inputs, padding_mask, None, None)
                total_cross_loss = args.criterion(total_pred, total_targets.long().squeeze(-1))       
            else:
                loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
                loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
            loss_reg = reg_loss(batch_x,trigger,trigger_clip,args) ### We can use regularizer loss as well
            if loss_reg is None:
                loss_reg = torch.zeros_like(total_cross_loss) if batch_x2 is not None and args.div_reg else torch.zeros_like(loss_bd) 
            if batch_x2 is not None and args.div_reg:
                loss = total_cross_loss + loss_reg + loss_div
            else:
                loss = loss_clean + loss_bd + loss_reg + loss_div
            loss_dict['CE_c'].append(loss_clean.item())
            loss_dict['CE_bd'].append(loss_bd.item())
            loss_dict['reg'].append(loss_reg.item())
            total_loss.append(loss.item())
            all_preds.append(clean_pred)
            bd_preds.append(bd_pred)
            trues.append(label)
            bds.append(bd_labels)

            if opt is not None:
                loss.backward()
                opt.step()
            if opt2 is not None:
                opt2.step()
    total_loss = np.average(total_loss)
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.cat(bds, 0)
    probs = torch.nn.functional.softmax(
        all_preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds), dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return total_loss,loss_dict, accuracy,bd_accuracy

def epoch_marksman_with_diversity(bd_model, bd_model_prev, surr_model, loader1, args, loader2=None, opt_trig=None, opt_class=None,train=True):
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train

    criterion_div = nn.MSELoss(reduction="none")

    # to make the zipped loop consistent with or without diversity loss
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)

    bd_model_prev.eval() ## ----> trigger gnerator for classifier is in evaluation mode
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(range(len(loader1)), loader1, loader2):
        loss_div = 0.0 # for consistency with optional diversity loss
        bd_model.zero_grad()
        surr_model.train() ### surrogate model in train mode 
        surr_model.zero_grad()
        #### Fetch clean data
        batch_x = batch_x.float().to(args.device)
        #### Fetch mask (for forecast task)
        padding_mask = padding_mask.float().to(args.device)
        #### Fetch labels
        label = label.to(args.device)

        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device)
        #### Generate backdoor labels ####### so far we focus on fixed target scenario
        if args.bd_type == 'all2all':
            bd_labels = torch.randint(0, args.numb_class, (batch_x.shape[0],)).to(args.device)
            assert args.bd_model == 'patchdyn'  ### only for patchdyn model
        elif args.bd_type == 'all2one':
            bd_labels = torch.ones_like(label).to(args.device) * bd_label  ## comes from argument
        else:
            raise ValueError('bd_type should be all2all or all2one')
        ########### First train surrogate classifier with frozen trigger #####################
        
        trigger, trigger_clip = bd_model_prev(batch_x, padding_mask,None,None,bd_labels) # generate trigger with frozen model
        mask = (label != bd_label).float().to(args.device) if args.attack_only_nontarget else torch.ones_like(label).float().to(args.device)
        mask = mask.unsqueeze(-1).expand(-1,trigger_clip.shape[-2],trigger_clip.shape[-1])
        clean_pred = surr_model(batch_x, padding_mask,None,None)
        bd_pred = surr_model(batch_x + trigger_clip * mask, padding_mask,None,None)
        loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_class = loss_clean + loss_bd
        if opt_class is not None:
            loss_class.backward()
            opt_class.step()
        ###########  Train trigger classifier with updated surrogate classifier (eval mode) #####################
        surr_model.eval() ### surrogate model in eval mode
        
        trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None) # trigger with active model
        if batch_x2 is not None and args.div_reg:
            
            trigger2, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None)

            ### DIVERGENCE LOSS CALCULATION
            input_distances = criterion_div(batch_x, batch_x2)
            input_distances = torch.mean(input_distances, dim=(1, 2))
            input_distances = torch.sqrt(input_distances)

            ### TODO: do we use trigger or trigger_clip here?
            trigger_distances = criterion_div(trigger, trigger2)
            trigger_distances = torch.mean(trigger_distances, dim=(1, 2))
            trigger_distances = torch.sqrt(trigger_distances)

            loss_div = input_distances / (trigger_distances + 1e-6) # second value is the epsilon, arbitrary for now
            loss_div = torch.mean(loss_div) * args.div_reg # give weight from args

        
        bd_pred = surr_model(batch_x + trigger_clip * mask, padding_mask,None,None) # surrogate classifier in eval mode
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_reg = reg_loss(batch_x, trigger, trigger_clip, args)  ### We can use regularizer loss as well          
        if loss_reg is None:
            loss_reg = torch.zeros_like(loss_bd)
        loss_trig = loss_bd + loss_reg + loss_div
        total_loss.append(loss_trig.item() + loss_class.item())
        all_preds.append(clean_pred)
        bd_preds.append(bd_pred)
        trues.append(label)
        bds.append(bd_labels)
        loss_dict['CE_c'].append(loss_class.item())
        loss_dict['CE_bd'].append(loss_bd.item())
        loss_dict['reg'].append(loss_reg.item())
        if opt_trig is not None:
            loss_trig.backward()
            opt_trig.step()
        #### With a certain period we synchronize bd_model and bd_model_prev
    pull_model(bd_model_prev,bd_model)#### here move bd_model to bd_model_prev
    total_loss = np.average(total_loss)
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.cat(bds, 0)
    probs = torch.nn.functional.softmax(
        all_preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds),
                                  dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return total_loss,loss_dict, accuracy,bd_accuracy

def epoch_marksman_lam_with_diversity(bd_model, bd_model_prev, surr_model, loader1, args, loader2=None, opt_trig=None, opt_class=None,train=True):
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train

    criterion_div = nn.MSELoss(reduction="none")
    # to make the zipped loop consistent with or without diversity loss
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)

    bd_model_prev.eval() ## ----> trigger gnerator for classifier is in evaluation mode
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(range(len(loader1)), loader1, loader2):
        loss_div = 0.0 # for consistency with optional diversity loss
        bd_model.zero_grad()
        surr_model.train() ### surrogate model in train mode 
        surr_model.zero_grad()
        #### Fetch clean data
        batch_x = batch_x.float().to(args.device)
        #### Fetch mask (for forecast task)
        padding_mask = padding_mask.float().to(args.device)
        #### Fetch labels
        label = label.to(args.device)

        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device)
        #### Generate backdoor labels ####### so far we focus on fixed target scenario
        if args.bd_type == 'all2all':
            bd_labels = torch.randint(0, args.numb_class, (batch_x.shape[0],)).to(args.device)
            assert args.bd_model == 'patchdyn' ### only for patchdyn model
        elif args.bd_type == 'all2one':
            bd_labels = torch.ones_like(label).to(args.device) * bd_label ## comes from argument
        else:
            raise ValueError('bd_type should be all2all or all2one')
        ########### First train surrogate classifier with frozen trigger #####################
        
        trigger, trigger_clip = bd_model_prev(batch_x, padding_mask,None,None,bd_labels) # generate trigger with frozen model
        mask = (label != bd_label).float().to(args.device) if args.attack_only_nontarget else torch.ones_like(label).float().to(args.device)
        mask = mask.unsqueeze(-1).expand(-1,trigger_clip.shape[-2],trigger_clip.shape[-1])
        batch_mix, scale_weights = mixup_class(batch_x, batch_x + trigger_clip * mask, args.lambda_alpha, args.lambda_beta) # generate mix_batch
        clean_pred = surr_model(batch_x, padding_mask,None,None)
        bd_pred = surr_model(batch_mix, padding_mask,None,None)
        loss_bd = args.criterion_mix(bd_pred, bd_labels.long().squeeze(-1))  # output size of batch
        loss_clean = args.criterion_mix(bd_pred, label.long().squeeze(-1))  # output size of batch
        loss_class = torch.mean(loss_bd * scale_weights + loss_clean * (1-scale_weights))
        #loss_class = loss_clean + loss_bd
        total_loss.append(loss_class.item())
        all_preds.append(clean_pred)
        bd_preds.append(bd_pred)
        trues.append(label)
        bds.append(bd_labels)
        loss_dict['CE_c'].append(loss_clean.mean().item())
        loss_dict['CE_bd'].append(loss_bd.mean().item())
        if opt_class is not None:
            loss_class.backward()
            opt_class.step()
        ###########  Train trigger classifier with updated surrogate classifier (eval mode) #####################
        surr_model.eval() ### surrogate model in eval mode
            
        trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None,bd_labels) # trigger with active model
        if batch_x2 is not None and args.div_reg:
            
            trigger2, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None)

            ### DIVERGENCE LOSS CALCULATION
            input_distances = criterion_div(batch_x, batch_x2)
            input_distances = torch.mean(input_distances, dim=(1, 2))
            input_distances = torch.sqrt(input_distances)

            ### TODO: do we use trigger or trigger_clip here?
            trigger_distances = criterion_div(trigger, trigger2)
            trigger_distances = torch.mean(trigger_distances, dim=(1, 2))
            trigger_distances = torch.sqrt(trigger_distances)

            loss_div = input_distances / (trigger_distances + 1e-6) # second value is the epsilon, arbitrary for now
            loss_div = torch.mean(loss_div) * args.div_reg # give weight from args
        #batch_mix, scale_weights = mixup_class(batch_x, batch_x + trigger_clip, alpha=2, beta=2) # generate mix_batch
       
        bd_pred = surr_model(batch_x + trigger_clip * mask, padding_mask,None,None) # surrogate classifier in eval mode
        ######## here we combine two loss one for each label
        loss_bd = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)(bd_pred, bd_labels.long().squeeze(-1)) # output size of batch
        #loss_clean = args.criterion_mix(bd_pred, label.long().squeeze(-1)) # output size of batch
        loss_reg = reg_loss(batch_x,trigger,trigger_clip,args) ### We can use regularizer loss as well
        #loss_trig = torch.mean(loss_bd * scale_weights + loss_clean * (1-scale_weights)) ## sum loss can be converted to average
        loss_trig = loss_bd + loss_div
        if loss_reg is not None:
            loss_trig += loss_reg
            loss_dict['reg'].append(loss_reg.item())
        if opt_trig is not None:
            loss_trig.backward()
            opt_trig.step()
        #### With a certain period we synchronize bd_model and bd_model_prev
    pull_model(bd_model_prev,bd_model)#### here move bd_model to bd_model_prev
    total_loss = np.average(total_loss)
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.cat(bds, 0)
    probs = torch.nn.functional.softmax(
        all_preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds),
                                  dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return total_loss,loss_dict, accuracy,bd_accuracy

def epoch_clean_train2(model, loader, args,optimiser): #for training clean model with fraction of backdoored data
    # loader here contains the backdoor generator, and generates trigger
    # for the spesific indices in the dataset
    model.train()
    total_loss = []
    preds = []
    trues = []
    bd_accuracy = 0
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
        loss.backward()
        optimiser.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    return total_loss, accuracy, bd_accuracy

def epoch_clean_test(bd_model,clean_model, loader,args,plot=None): ## for testing the backdoored clean model
    preds = []
    bd_preds = []
    trues = []
    bd_label = args.target_label
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        target_labels = torch.ones_like(label) * bd_label
        trigger_x,trigger_clipped = bd_model(batch_x, padding_mask, None, None,target_labels)
        bd_batch = batch_x + trigger_clipped
        #fft_vals(batch_x[0],bd_batch[0],args,i)
        clean_outs = clean_model(batch_x, padding_mask,None,None)
        bd_outs = clean_model(bd_batch, padding_mask,None,None)
        preds.append(clean_outs.detach())
        bd_preds.append(bd_outs)
        trues.append(label)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.ones_like(trues) * bd_label
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds),
                                  dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    clean_accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    if plot is not None:
        plot(args,batch_x[0].permute(1,0),bd_batch[0].permute(1,0)) ## plot the first sample
        fft_vals(batch_x[0],bd_batch[0],args,i=-1)
    return clean_accuracy,bd_accuracy

def clean_train(model,loader,args,optimizer): ### for warm up the surrogate classifier
    model.train()
    total_loss = []
    preds = []
    trues = []
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
        loss.backward()
        optimizer.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    return total_loss, accuracy


def clean_test(model,loader,args): ### test CA without poisoining the model
    model.eval()
    total_loss = []
    preds = []
    trues = []
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    return total_loss, accuracy

def defence_test_fp(bd_model,clean_model,train_loader,test_loader,args): ## for testing the backdoored clean model
    preds = []
    bd_preds = []
    trues = []
    bd_label = args.target_label

    ########## Defence mechanism
    fp = Pruning(train_loader=train_loader,model=clean_model,args=args)
    if args.model == 'resnet2':
        fp.repair(device=args.device) ## channel prune
    else:
        fp.repair2(device=args.device) # dim prune (Tested on TimesNet)
    clean_model = fp.get_model()
    ############################
    # Rest of the code is test epoch
    for i, (batch_x, label, padding_mask) in enumerate(test_loader):
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        target_labels = torch.ones_like(label) * bd_label
        trigger_x,trigger_clipped = bd_model(batch_x, padding_mask, None, None,target_labels)
        clean_outs = clean_model(batch_x, padding_mask,None,None)
        bd_batch = batch_x + trigger_clipped
        bd_outs = clean_model(bd_batch, padding_mask,None,None)
        preds.append(clean_outs.detach())
        bd_preds.append(bd_outs)
        trues.append(label)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.ones_like(trues) * bd_label
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds),
                                  dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    clean_accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return clean_accuracy,bd_accuracy

def defence_test_nc(bd_model,clean_model,train_loader,test_loader,args): ## for testing the backdoored clean model
    preds = []
    bd_preds = []
    trues = []
    bd_label = args.target_label

    ########## Defence mechanism
    nc_main(args,clean_model,train_loader)
    ############################
    # Rest of the code is test epoch
    for i, (batch_x, label, padding_mask) in enumerate(test_loader):
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        target_labels = torch.ones_like(label) * bd_label
        trigger_x,trigger_clipped = bd_model(batch_x, padding_mask, None, None,target_labels)
        clean_outs = clean_model(batch_x, padding_mask,None,None)
        bd_batch = batch_x + trigger_clipped
        bd_outs = clean_model(bd_batch, padding_mask,None,None)
        preds.append(clean_outs.detach())
        bd_preds.append(bd_outs)
        trues.append(label)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.ones_like(trues) * bd_label
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds),
                                  dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    clean_accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return clean_accuracy,bd_accuracy

def defence_test_strip(clean_model, poisoned_loader, clean_loader, poisoned_indices, silent_indices, args):

    print("======== STRIP defence test ========")
    suspicious_indices = cleanser(poisoned_loader, clean_loader, clean_model, args)

    backdoored_indices = poisoned_indices.tolist() + silent_indices.tolist() if len(silent_indices) > 0 else poisoned_indices.tolist()
    backdoored_indices = set(backdoored_indices)
    suspicious_indices = set([index.item() for index in suspicious_indices])

    susp_ind = np.zeros(len(poisoned_loader.dataset))
    true_ind = np.zeros(len(poisoned_loader.dataset))

    for i in suspicious_indices:
        susp_ind[i] = 1
    for i in backdoored_indices:
        true_ind[i] = 1

    TN, FP, FN, TP = confusion_matrix(true_ind, susp_ind).ravel()

    return FN, TP, FP

def defence_test_spectre(clean_model, poisoned_loader, poisoned_indices, silent_indices, args):

    print("======= SPECTRE defence test =======")
    backdoored_indices = poisoned_indices.tolist() + silent_indices.tolist() if len(silent_indices) > 0 else poisoned_indices.tolist()

    TN, FP, FN, TP, TPR, FPR = spectre_filtering(clean_model, poisoned_loader, backdoored_indices, args)

    return FN, TP, FP


def epoch_visualize(clean_model, bd_loader,poisoned_indices, silent_indices, args): ## for testing the backdoored clean model
    latents = []
    for i, (batch_x, label, padding_mask) in enumerate(bd_loader):
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        _, latent = clean_model(batch_x, padding_mask,None,None, visualize=visualize)
        latents.append(latent)
    latents = torch.cat(latents, dim=0)
    visualize(latents, poisoned_indices, silent_indices, args)


