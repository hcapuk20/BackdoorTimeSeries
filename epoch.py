import torch
import numpy as np
import torch.nn as nn


################
'''
TODO
- bd and clean inputs should not be concatenated (done)
- bd-model, separate trigger and surrogate classifier (done)
- add 2 opt backdoor epoch
'''
################# Related works with Code #############
#Dynamic input-aware https://github.com/VinAIResearch/input-aware-backdoor-attack-release/blob/master/train.py

##################### Regularizers ########################

### This loss is designed to minimize the alignment on the frequency domain
def fftreg(x_clean,x_back): # input shape B x C x T #outputshape B x C 
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    xf_c = abs(torch.fft.rfft(x_clean, dim=2)) ## clean data on the freq domain
    xf_b = abs(torch.fft.rfft(x_back, dim=2))  ## backdoored data on the freq domain
    xf_c2 = xf_c[:,:,1:-1] ## ignore the freq 0
    xf_b2 = xf_b[:,:,1:-1] ## ignore the freq 0
    return cos(xf_c2,xf_b2) ##### This term can be summed or averaged #########

def l2_reg(clipped_trigger, trigger):
    return torch.norm(trigger - clipped_trigger)



### for the set of regularizers

###################### Mixup operation for input/output #################################

def mixup_forcast(x_clean, x_backdoor, y_clean, y_backdoor, alpha=2, beta=2): ### this is for forcast classsification requires minor modif.
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






def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
    

################################Epoch functions for training ################################################
### Marksman update ---> trigger and surrogate classifier are updated seperately:
### Trigger model used for training surrogate classifier is not updated immediately (bd_model_prev is used)
def epoch_marksman(bd_model, bd_model_prev, surr_model, loader, args, opt=None): 
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    bd_model_prev.eval() ## ----> trigger is in evaluation mode
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
            loss = loss_clean + loss_bd + loss_reg
            loss_dict['CE_c'].append(loss_clean.item())
            loss_dict['CE_bd'].append(loss_bd.item())
            loss_dict['reg'].append(loss.item())
            total_loss.append(loss.item())
            all_preds.append(clean_pred)
            bd_preds.append(bd_pred)
            trues.append(label)
            bds.append(bd_labels)
            if opt is not None:
                loss.backward()
                opt.step()
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








def epoch(bd_model,surr_model, loader, args, opt=None): ##### The main training module
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
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
            loss = loss_clean + loss_bd + loss_reg
            loss_dict['CE_c'].append(loss_clean.item())
            loss_dict['CE_bd'].append(loss_bd.item())
            loss_dict['reg'].append(loss.item())
            total_loss.append(loss.item())
            all_preds.append(clean_pred)
            bd_preds.append(bd_pred)
            trues.append(label)
            bds.append(bd_labels)
            if opt is not None:
                loss.backward()
                opt.step()
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
        trigger_x,trigger_clipped = bd_model(batch_x, padding_mask, None, None)
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


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened
