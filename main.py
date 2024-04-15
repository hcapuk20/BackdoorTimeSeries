import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from data_provider.data_factory import data_provider, custom_data_loader
from models import * # Here we import the architecture
from parameters import * # Here we import the parameters
from models.Informer import Model as Informer
from models.TimesNet import Model as TimesNet
from models.FEDformer import Model as FEDformer
from models.Bd_transformer import Bd_transformer
from models.bd_Universal import Bd_Tnet
from models.Bd_inverted import Model as Bd_inverted
from models.Bd_patch import Model as Bd_patch
import matplotlib.pyplot as plt


######################################################### This is the V0 (initial prototype) code for time series backdoor ############################################


############ The main file perform both training and test



############ Loading the dataset
def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader
############ Loading the model
def get_model(args, train_data, test_data):
    model_dict = {
            'TimesNet': TimesNet,
            'FEDformer': FEDformer,
            'Informer': Informer
        }
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    print('enc_in',args.enc_in,'seq_len',args.seq_len)
    args.num_class = len(train_data.class_names)
    # model init
    model = model_dict[args.model](args).float().to(args.device)
    model_sur = model_dict[args.model](args).float()
    if args.bd_model == 'inverted':
        generative_model = Bd_inverted(args).float().to(args.device)
    elif args.bd_model == 'patchtst':
        generative_model = Bd_patch(args).float().to(args.device)
    else:
        raise NotImplementedError
    bd_model = Bd_Tnet(args,model_sur,generative_model).float().to(args.device)
    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model, bd_model


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

############ Training the model
def epoch(bd_model, loader, args,optimiser=None):
    #
    epoch_loss_record = []
    total_loss = []
    preds = []
    bd_preds = []
    trues = []
    bd_label = 0
    for i, (batch_x, label, padding_mask) in enumerate(loader):
            bd_model.zero_grad()
            #print(batch_x.shape, padding_mask.shape, label.shape)
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            #print((padding_mask==0).sum())
            label = label.to(args.device)
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
            all_labels = torch.cat((label,bd_labels),dim=0)
            bd_x,outs2 = bd_model(batch_x, padding_mask,None,None)
            loss1 = args.criterion(outs2, all_labels.long().squeeze(-1))
            #loss2 = torch.norm(bd_x-batch_x)
            loss = loss1
            total_loss.append(loss.item())
            preds.append(outs2.detach().chunk(2)[0])
            bd_preds.append(outs2.detach().chunk(2)[1])
            trues.append(label)
            if optimiser is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(bd_model.parameters(), max_norm=4.0)
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
    return total_loss, accuracy,bd_accuracy

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
            label = label.to(args.device)
            label_bd = torch.ones_like(label_bd).to(args.device) * bd_label
            all_labels = torch.cat((label,label_bd),dim=0)
            padding_mask = torch.cat((padding_mask,padding_mask_bd),dim=0)
            batch_x = torch.cat((batch_x,trigger_x),dim=0)
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
        # print((padding_mask==0).sum())
        label = label.to(args.device)
        trigger_x,_ = bd_model(batch_x, padding_mask, None, None)
        clean_outs = clean_model(batch_x, padding_mask,None,None)
        bd_outs = clean_model(trigger_x, padding_mask,None,None)
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





if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.saveDir = 'weights/model_weights'  # path to be saved to
    # ======================================================= Initialize the model
    train_data, train_loader = get_data(args=args, flag='train')
    test_data, test_loader = get_data(args=args, flag='test')
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    model,bd_model = get_model(args,train_data,test_data)
    best_bd = 0
    best_dict = None
    print("model initialized...")
    # ============================================================================
    # ===== Add loss criterion to the args =====
    args.criterion = nn.CrossEntropyLoss()
    # ===== Add optimizer to the args =====
    if args.is_training == 1:
        if args.opt_method == 'adamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        opt_bd = optimizer = torch.optim.AdamW(bd_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        #if args.use_lr_scheduler:
            #lambda1 = lambda epoch: (1-epoch/args.total_iter)
            #args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
            ######################## huggingface library ####################################################
            #args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.total_iter, power=0.5)
        for i in range(args.train_epochs):
            model.train()
            #train_loss, train_acc = epoch(model,bd_model, train_loader, args,optimizer)
            train_loss, train_acc,bd_train_acc = epoch(bd_model, train_loader, args, opt_bd)
            model.eval()
            test_loss, test_acc, bd_test_acc = epoch(bd_model,test_loader, args)
            print('Train Loss',train_loss,'Train acc',train_acc,'Test Loss',test_loss,'Test acc',test_acc)
            print('Backdoor Train',bd_train_acc,'Backdoor Test',bd_test_acc)
            if best_bd < bd_test_acc:
                best_bd = bd_test_acc
                best_dict = bd_model.state_dict()
                if not os.path.exists('weights'):
                    os.makedirs('weights')
                torch.save(bd_model.state_dict(), 'weights/best_bd_model_weights.pth')
            torch.save(bd_model.state_dict(), 'weights/last_bd_model_weights.pth')



        bd_model.load_state_dict(best_dict)
        bd_model.eval()
        ### THIS IS FOR THE CLEAN MODEL AFTER TRAINING THE BACKDOOR MODEL, NOT FINISHED YET
        #train_size = int(0.9 * len(train_data))
        #bd_size = len(train_data) - train_size
        train_dataset, bd_dataset = torch.utils.data.random_split(train_data, [0.9, 0.1])
        bs = (args.batchSize * 9) // 10
        train_loader = custom_data_loader(train_dataset, args,flag='train',force_bs=bs)
        bd_bs = (args.batchSize // 10) + 1
        bd_loader = custom_data_loader(bd_dataset, args,flag='train',force_bs=bd_bs)
        clean_model,_ = get_model(args,train_data,test_data)
        optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        for i in range(args.train_epochs):
            clean_model.train()
            train_loss, train_accuracy, bd_accuracy_train = epoch_clean_train(bd_model,clean_model, train_loader,bd_loader, args,optimizer)
            clean_model.eval()
            clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_model,clean_model, test_loader,args)
            print('CA:',clean_test_acc,'ASR:',bd_accuracy_test)

    else:
        epoch(model, test_loader, args)
