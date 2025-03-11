import math

from epoch import *


def create_cross(inputs1, inputs2,padding_mask, bd_model):
    patterns2, patterns2_clipped = bd_model(inputs2, padding_mask, None, None)
    inputs_cross = inputs1 + (patterns2_clipped - inputs1)
    return inputs_cross, patterns2


def epoch_marksman_cross(bd_model, bd_model_prev, surr_model, loader1, args, loader2=None, opt_trig=None, opt_class=None,train=True):
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c': [], 'CE_bd': [], 'reg': []}
    ratio = args.poisoning_ratio_train
    p_cross = args.p_cross
    surr_model.train()
    bd_model.train()

    criterion_div = nn.MSELoss(reduction="none")

    # to make the zipped loop consistent with or without diversity loss
    bd_model_prev.eval()  ## ----> trigger gnerator for classifier is in evaluation model
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(range(len(loader1)), loader1,
                                                                                    loader2):
        loss_div = 0.0  # for consistency with optional diversity loss
        bd_model.zero_grad()
        surr_model.train()  ### surrogate model in train mode
        surr_model.zero_grad()
        #### Fetch clean data
        batch_x = batch_x.float().to(args.device)
        #### Fetch mask (for forecast task)
        padding_mask = padding_mask.float().to(args.device)
        #### Fetch labels
        label = label.to(args.device)
        num_bd = math.ceil(ratio * batch_x.shape[0])
        num_cross = math.ceil(p_cross * batch_x.shape[0])
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
        trigger, trigger_clip = bd_model_prev(batch_x[:num_bd], padding_mask[:num_bd], None, None,
                                              bd_labels[:num_bd])  # generate trigger with frozen model
        inputs_cross, patterns2 = create_cross(batch_x[num_bd: num_bd + num_cross], batch_x2[num_bd: num_bd + num_cross],padding_mask[num_bd: num_bd + num_cross], bd_model_prev)
        inputs_bd = batch_x[:num_bd] + trigger_clip
        total_inputs = torch.cat((inputs_cross, batch_x[num_bd + num_cross:]), 0)
        clean_pred = surr_model(total_inputs, padding_mask[num_bd:], None, None)
        bd_pred = surr_model(inputs_bd, padding_mask[:num_bd], None, None)
        loss_clean = args.criterion(clean_pred, label.long().squeeze(-1)[num_bd:])
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1)[:num_bd])
        loss_class = loss_clean + loss_bd
        loss_class.backward()
        opt_class.step()
        ###########  Train trigger classifier with updated surrogate classifier (eval mode) #####################
        #surr_model.eval()  ### surrogate model in eval mode
        trigger, trigger_clip = bd_model(batch_x[:num_bd], padding_mask[:num_bd], None, None,
                                                  bd_labels[:num_bd])  # trigger with active model
        trigger2, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None)

        ### DIVERGENCE LOSS CALCULATION
        input_distances = criterion_div(batch_x[:num_bd], batch_x2[num_bd: num_bd + num_bd])
        input_distances = torch.mean(input_distances, dim=(1, 2))
        input_distances = torch.sqrt(input_distances)

        ### TODO: do we use trigger or trigger_clip here?
        trigger_distances = criterion_div(trigger, trigger2[:num_bd])
        trigger_distances = torch.mean(trigger_distances, dim=(1, 2))
        trigger_distances = torch.sqrt(trigger_distances)

        loss_div = input_distances / (
                    trigger_distances + 1e-6)  # second value is the epsilon, arbitrary for now
        loss_div = torch.mean(loss_div) * args.div_reg  # give weight from args

        bd_pred = surr_model(batch_x[:num_bd] + trigger_clip, padding_mask[:num_bd], None,
                             None)  # surrogate classifier in eval mode
        loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1)[:num_bd])
        loss_reg = reg_loss(batch_x[:num_bd], trigger, trigger_clip, args)  ### We can use regularizer loss as well
        if loss_reg is None:
            loss_reg = torch.zeros_like(loss_bd)
        loss_trig = loss_bd  + loss_div + loss_reg
        total_loss.append(loss_trig.item() + loss_class.item())
        all_preds.append(clean_pred)
        bd_preds.append(bd_pred)
        trues.append(label[num_bd:])
        bds.append(bd_labels[:num_bd])
        loss_dict['CE_c'].append(loss_class.item())
        loss_dict['CE_bd'].append(loss_bd.item())
        loss_dict['reg'].append(loss_reg.item())
        if opt_trig is not None:
            opt_trig.zero_grad()
            loss_trig.backward()
            opt_trig.step()
        #### With a certain period we synchronize bd_model and bd_model_prev
    pull_model(bd_model_prev, bd_model)  #### here move bd_model to bd_model_prev
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
    return total_loss, loss_dict, accuracy, bd_accuracy
