import torch
import numpy as np


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
def clipping(x_enc, trigger, ratio=0.1):
    lim = x_enc.abs() * 0.1 * ratio
    x_gen_clipped = torch.where(trigger < -lim, -lim,
                                torch.where(trigger > lim, lim, trigger))
    return x_gen_clipped

def epoch(bd_model, loader, args,optimiser=None):
    #
    epoch_loss_record = []
    total_loss = []
    preds = []
    bd_preds = []
    trues = []
    bd_label = 0
    loss_dict = {'CE':[],'L2':[]}
    for i, (batch_x, label, padding_mask) in enumerate(loader):
            bd_model.zero_grad()
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
            all_labels = torch.cat((label,bd_labels),dim=0)
            trigger ,outs2 = bd_model(batch_x, padding_mask,None,None)
            loss1 = args.criterion(outs2, all_labels.long().squeeze(-1))
            trigger_clipped = clipping(batch_x,trigger)
            loss2 = torch.norm(trigger_clipped)
            loss = loss1 + loss2
            loss_dict['CE'].append(loss1.item())
            loss_dict['L2'].append(loss2.item())
            total_loss.append(loss.item())
            preds.append(outs2.detach().chunk(2)[0])
            bd_preds.append(outs2.detach().chunk(2)[1])
            trues.append(label)
            if optimiser is not None:
                loss.backward()
                #nn.utils.clip_grad_norm_(bd_model.parameters(), max_norm=4.0)
                optimiser.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.ones_like(trues) * bd_label
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds), dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())

    return total_loss,loss_dict, accuracy,bd_accuracy

def epoch_clean_train(bd_model,clean_model, loader,loader_bd, args,optimiser):
    total_loss = []
    preds = []
    bd_preds = []
    trues = []
    backdoors = []
    bd_label = 0
    nb = iter(loader_bd)
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
            #padding_mask = torch.ones((bs_1, args.seq_len)).to(args.device).float()
            #padding_mask_bd = torch.ones((bs_2, args.seq_len)).to(args.device).float()
            trigger_x,_ = bd_model(bd_x,padding_mask_bd,None,None)
            bd_batch = bd_x + trigger_x
            label = label.to(args.device)
            label_bd = torch.ones_like(label_bd).to(args.device) * bd_label
            all_labels = torch.cat((label,label_bd),dim=0)
            padding_mask = torch.cat((padding_mask,padding_mask_bd),dim=0)
            batch_x = torch.cat((batch_x,bd_batch),dim=0)
            outs2 = clean_model(batch_x, padding_mask,None,None)
            loss = args.criterion(outs2, all_labels.long().squeeze(-1))
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

def epoch_clean_test(bd_model,clean_model, loader,args):
    preds = []
    bd_preds = []
    trues = []
    bd_label = 0
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        clean_model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        trigger_x,_ = bd_model(batch_x, padding_mask, None, None)
        clean_outs = clean_model(batch_x, padding_mask,None,None)
        bd_batch = batch_x + trigger_x
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

def clean_train(model,loader,args,optimizer):
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

