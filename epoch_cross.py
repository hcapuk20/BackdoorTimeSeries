from epoch import *

def epoch_marksman_lam_cross(bd_model, bd_model_prev, surr_model, loader, args, opt_trig=None, opt_class=None,train=True):
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
        trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None) # trigger with active model
        bd_pred = surr_model(batch_x + trigger_clip, padding_mask,None,None) # surrogate classifier in eval mode
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
        loss_reg = reg_loss(batch_x, trigger,trigger_clip,args)
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
