import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from data_provider.data_factory import data_provider
from models import * # Here we import the architecture
from parameters import * # Here we import the parameters
from models.Informer import Model as Informer
from models.TimesNet import Model as TimesNet
from models.FEDformer import Model as FEDformer
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
    model = model_dict[args.model](args).float()
    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model

def get_bd_model(args, train_data, test_data):
    seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    bd_model = nn.Transformer(nhead=8, num_encoder_layers=2,d_model=args.seq_len)
    return bd_model

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

############ Training the model
def epoch(model, loader, args,optimiser=None):
    #
    epoch_loss_record = []
    total_loss = []
    preds = []
    trues = []
    for i, (batch_x, label, padding_mask) in enumerate(loader):
            model.zero_grad()
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            ######## Outputs
            outputs = model(batch_x, padding_mask, None, None)
            loss = args.criterion(outputs, label.long().squeeze(-1))
            total_loss.append(loss.item())
            preds.append(outputs.detach())
            trues.append(label)
            if optimiser is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimiser.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(
        preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    return total_loss, accuracy





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
    model = get_model(args,train_data,test_data).to(args.device)
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
        #if args.use_lr_scheduler:
            #lambda1 = lambda epoch: (1-epoch/args.total_iter)
            #args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
            ######################## huggingface library ####################################################
            #args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.total_iter, power=0.5)
        for i in range(args.train_epochs):
            model.train()
            train_loss, train_acc = epoch(model, train_loader, args,optimizer)
            model.eval()
            test_loss, test_acc = epoch(model,test_loader, args)
            print(train_loss,test_loss,train_acc,test_acc)
    else:
        epoch(model, test_loader, args)
