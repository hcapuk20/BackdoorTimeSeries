import torch
import numpy as np
import torch.nn as nn
from utils.model_ops import *
from defences.fp import Pruning

################
'''
TODO
- bd and clean inputs should not be concatenated (done)
- bd-model, separate trigger and surrogate classifier (done)
- add 2 opt backdoor epoch
'''
################# Related works with Code #############
# Dynamic input-aware https://github.com/VinAIResearch/input-aware-backdoor-attack-release/blob/master/train.py
# Label smoothing for backdoor -->> https://arxiv.org/pdf/2202.11203

##################### Regularizers ########################

### This loss is designed to minimize the alignment on the frequency domain
def fftreg(x_clean,x_back): # input shape B x C x T #outputshape B x C 
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    xf_c = abs(torch.fft.rfft(x_clean, dim=2)) ## clean data on the freq domain
    xf_b = abs(torch.fft.rfft(x_back, dim=2))  ## backdoored data on the freq domain
    xf_c2 = xf_c[:,:,1:-1] ## ignore the freq 0
    xf_b2 = xf_b[:,:,1:-1] ## ignore the freq 0
    return cos(xf_c2,xf_b2).mean() ##### This term can be summed or averaged #########

def l2_reg(clipped_trigger, trigger):
    return torch.norm(trigger - clipped_trigger)

def reg_loss(x_clean,trigger,trigger_clip,args):
    l2_loss = l2_reg(trigger_clip,trigger)
    cos_loss = fftreg(x_clean,x_clean+trigger_clip)
    reg_total = l2_loss * args.L2_reg + cos_loss * args.cos_reg
    return reg_total



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
    lam = torch.tensor( np.random.beta(2, 2, bs), requires_grad=False)
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
    lam = torch.tensor( np.random.beta(2, 2, bs), requires_grad=False).to(x_clean.device).float()
    lam_x = (lam.unsqueeze(dim=-1)).unsqueeze(dim=-1)
    lam_x = lam_x.repeat(1, channel, time_x)
    x_mixed = lam_x * x_backdoor + (1 - lam_x) * x_clean
    return x_mixed.float(), lam # we output lam as well to be used weight loss terms
    ####################### end of mixup #########################################





def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
    

################################Epoch functions for training ################################################
### Marksman update ---> trigger and surrogate classifier are updated seperately:
### Trigger model used for training surrogate classifier is not updated immediately (bd_model_prev is used)
### Since models are updated seperately we switch between eval and train
def epoch_marksman(bd_model, bd_model_prev, surr_model, loader, args, opt_trig=None, opt_class=None,train=True):
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    bd_model_prev.eval() ## ----> trigger gnerator for classifier is in evaluation mode
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        bd_model.zero_grad()
        surr_model.train() ### surrogate model in train mode 
        surr_model.zero_grad()
        #### Fetch clean data
        batch_x = batch_x.float().to(args.device)
        #### Fetch mask (for forecast task)
        padding_mask = padding_mask.float().to(args.device)
        #### Fetch labels
        label = label.to(args.device)
        #### Generate backdoor labels ####### so far we focus on fixed target scenario
        bd_labels = torch.ones_like(label).to(args.device) * bd_label ## comes from argument
        ########### First train surrogate classifier with frozen trigger #####################
        trigger, trigger_clip = bd_model_prev(batch_x, padding_mask,None,None) # generate trigger with frozen model
        clean_pred = surr_model(batch_x, padding_mask,None,None)
        bd_pred = surr_model(batch_x + trigger_clip, padding_mask,None,None)
        loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_class = loss_clean + loss_bd
        if opt_class is not None:
            loss_class.backward()
            opt_class.step()
        ###########  Train trigger classifier with updated surrogate classifier (eval mode) #####################
        surr_model.eval() ### surrogate model in eval mode
        trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None) # trigger with active model
        bd_pred = surr_model(batch_x + trigger_clip, padding_mask,None,None) # surrogate classifier in eval mode
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_reg = l2_reg(trigger_clip, trigger) ### here we also utilize reqularizer loss
        loss_trig = loss_bd + loss_reg
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
        if i % 5 ==0: #deep_copy model
            pull_model(bd_model_prev,bd_model)#### here move bd_model to bd_model_prev
            #### 5 will be a parameter and we may consider syncronisation even at the end of epoch
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
        
### Marksman update with mixup framework ---> trigger and surrogate classifier are updated seperately:
### Trigger model used for training surrogate classifier is not updated immediately (bd_model_prev is used)
### Since models are updated seperately we switch between eval and train
### In this version of Marksman update we utilize mixup framework for backdoor training in order to mitigate simple triggers

def epoch_marksman_lam(bd_model, bd_model_prev, surr_model, loader, args, opt_trig=None, opt_class=None,train=True):
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    bd_model_prev.eval() ## ----> trigger gnerator for classifier is in evaluation mode
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        bd_model.zero_grad()
        surr_model.train() ### surrogate model in train mode 
        surr_model.zero_grad()
        #### Fetch clean data
        batch_x = batch_x.float().to(args.device)
        #### Fetch mask (for forecast task)
        padding_mask = padding_mask.float().to(args.device)
        #### Fetch labels
        label = label.to(args.device)
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
        clean_pred = surr_model(batch_x, padding_mask,None,None)
        bd_pred = surr_model(batch_x + trigger_clip, padding_mask,None,None)
        loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_class = loss_clean + loss_bd
        total_loss.append(loss_class.item())
        all_preds.append(clean_pred)
        bd_preds.append(bd_pred)
        trues.append(label)
        bds.append(bd_labels)
        loss_dict['CE_c'].append(loss_class.item())
        loss_dict['CE_bd'].append(loss_bd.item())
        if opt_class is not None:
            loss_class.backward()
            opt_class.step()
        ###########  Train trigger classifier with updated surrogate classifier (eval mode) #####################
        surr_model.eval() ### surrogate model in eval mode
        trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None,bd_labels) # trigger with active model
        batch_mix, scale_weights = mixup_class(batch_x, batch_x + trigger_clip, alpha=2, beta=2) # generate mix_batch
        bd_pred = surr_model(batch_mix, padding_mask,None,None) # surrogate classifier in eval mode
        ######## here we combine two loss one for each label 
        loss_bd = args.criterion_mix(bd_pred, bd_labels.long().squeeze(-1)) # output size of batch
        loss_clean = args.criterion_mix(bd_pred, label.long().squeeze(-1)) # output size of batch
        #loss_reg = l2_reg(trigger_clip, trigger) ### We can use regularizer loss as well
        loss_reg = reg_loss(batch_x,trigger,trigger_clip,args) ### We can use regularizer loss as well
        loss_trig = torch.mean(loss_bd * scale_weights + loss_clean * (1-scale_weights)) ## sum loss can be converted to average
        if loss_reg is not None:
            loss_trig += loss_reg
            loss_dict['reg'].append(loss_reg.item())
        if opt_trig is not None:
            loss_trig.backward()
            opt_trig.step()
        #### With a certain period we synchronize bd_model and bd_model_prev
        if i % 5 ==0: #deep_copy model
            pull_model(bd_model_prev,bd_model)#### here move bd_model to bd_model_prev
            #### 5 will be a parameter and we may consider syncronisation even at the end of epoch
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








def epoch(bd_model,surr_model, loader, args, opt=None,opt2=None,train=True): ##### The main training module
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask) in enumerate(loader):
            b_r = int(batch_x.size(0) * ratio)
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
            #### Combine true and target labels
            ########### Here we generate trigger #####################
            trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None)
            clean_pred = surr_model(batch_x, padding_mask,None,None)
            bd_pred = surr_model(batch_x + trigger_clip, padding_mask,None,None)
            loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
            loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
            loss_reg = l2_reg(trigger_clip, trigger)
            loss_reg += fftreg(batch_x, batch_x + trigger_clip) ### we can add fft reg for extra regularizer
            loss = loss_clean + loss_bd + loss_reg
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

def epoch_clean_train_silent(bd_model,clean_model, loader,loader_bd,loader_bd_clean, args,optimiser): #for training clean model with fraction of backdoored data
    total_loss = []
    preds = []
    bd_preds = []
    trues = []
    backdoors = []
    bd_label = args.target_label
    nb_c = iter(loader_bd_clean)
    nb = iter(loader_bd)
    print()
    cri = nn.CrossEntropyLoss()
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        try:
            bd_batch = next(nb)
        except:
            nb = iter(loader_bd)
            bd_batch = next(nb)

        try:
            bd_batch_clean = next(nb_c)
        except:
            nb_c = iter(loader_bd_clean)
            bd_batch_clean = next(nb_c)
        #batch_x, label = data_clean
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        bd_batch_clean = bd_batch_clean
        bd_x,label_bd,padding_mask_bd = bd_batch
        bd_x2, c_label_bd, padding_mask_bd2 = bd_batch_clean
        bd_x = bd_x2.float().to(args.device)
        c_label_bd = c_label_bd.float().to(args.device)
        padding_mask,padding_mask_bd = padding_mask.float().to(args.device),padding_mask_bd.float().to(args.device)
        padding_mask_bd2 = padding_mask_bd2.float().to(args.device)
        bd_x = bd_x.to(args.device).float()
        bd_x_c = bd_x2.to(args.device).float()
        bs_1, bs_2, bs_3 = batch_x.size(0), bd_x.size(0),bd_x2.size(0)
        label_bd = torch.ones_like(label_bd).to(args.device) * bd_label
        trigger_x,trigger_clipped = bd_model(bd_x,padding_mask_bd,None,None,label_bd)
        trigger_x_c, trigger_clipped_c = bd_model(bd_x_c, padding_mask_bd, None, None,label_bd)
        alpha = torch.rand_like(bd_x_c).float() * .5
        bd_batch_silent = alpha * (bd_x_c + trigger_clipped_c) + (1-alpha) * bd_x_c
        bd_batch = bd_x + trigger_clipped
        label = label.to(args.device)
        all_labels = torch.cat((label,c_label_bd,label_bd),dim=0)
        padding_mask = torch.cat((padding_mask,padding_mask_bd,padding_mask_bd2),dim=0)
        batch_x = torch.cat((batch_x,bd_batch_silent,bd_batch),dim=0)
        outs2 = clean_model(batch_x, padding_mask,None,None)
        loss = cri(outs2, all_labels.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs2.detach()[:bs_1])
        bd_preds.append(outs2.detach()[-bs_2:])
        trues.append(label)
        backdoors.append(label_bd)
        loss.backward()
        optimiser.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    backdoors = torch.cat(backdoors, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds), dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, backdoors.flatten().cpu().numpy())
    return total_loss, accuracy, bd_accuracy


def epoch_clean_train(bd_model,clean_model, loader,loader_bd, args,optimiser): #for training clean model with fraction of backdoored data
    total_loss = []
    preds = []
    bd_preds = []
    trues = []
    backdoors = []
    bd_label = args.target_label
    nb = iter(loader_bd)
    cri = nn.CrossEntropyLoss()
    for i, (batch_x, label, padding_mask) in enumerate(loader):
            try:
                bd_batch = next(nb)
            except:
                nb = iter(loader_bd)
                bd_batch = next(nb)
            #batch_x, label = data_clean
            clean_model.zero_grad()
            batch_x = batch_x.float().to(args.device)
            bd_x,label_bd,padding_mask_bd = bd_batch
            padding_mask,padding_mask_bd = padding_mask.float().to(args.device),padding_mask_bd.float().to(args.device)
            bd_x = bd_x.to(args.device).float()
            bs_1, bs_2 = batch_x.size(0), bd_x.size(0)
            trigger_x,trigger_clipped = bd_model(bd_x,padding_mask_bd,None,None)
            bd_batch = bd_x + trigger_clipped
            label = label.to(args.device)
            label_bd = torch.ones_like(label_bd).to(args.device) * bd_label
            all_labels = torch.cat((label,label_bd),dim=0)
            padding_mask = torch.cat((padding_mask,padding_mask_bd),dim=0)
            batch_x = torch.cat((batch_x,bd_batch),dim=0)
            outs2 = clean_model(batch_x, padding_mask,None,None)
            loss = cri(outs2, all_labels.long().squeeze(-1))
            total_loss.append(loss.item())
            preds.append(outs2.detach()[:bs_1])
            bd_preds.append(outs2.detach()[bs_1:])
            trues.append(label)
            backdoors.append(label_bd)
            loss.backward()
            optimiser.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    backdoors = torch.cat(backdoors, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds), dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, backdoors.flatten().cpu().numpy())
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
    if plot is not None:
        plot(args,batch_x[0].permute(1,0),bd_batch[0].permute(1,0)) ## plot the first sample
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

def defence_test(bd_model,clean_model,train_loader,test_loader,args): ## for testing the backdoored clean model
    preds = []
    bd_preds = []
    trues = []
    bd_label = args.target_label
    fp = Pruning(train_loader=train_loader,model=clean_model,args=args)
    fp.repair(device=args.device)
    clean_model = fp.get_model()
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
