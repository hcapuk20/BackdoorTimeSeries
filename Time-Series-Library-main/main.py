import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from data_provider.data_factory import data_provider
from models import * # Here we import the architecture
from parameters import *
from parameters.parameters import *
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
    args.num_class = len(train_data.class_names)
    # model init
    model = model_dict[args.model].Model(args).float()
    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model



############ Training the model
def train_model(model, train_loader, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    epoch_loss_record = []
    train_loss = []
    for i, (batch_x, label, padding_mask) in enumerate(train_loader):
            args.optimizer.zero_grad()
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            ######## Outputs
            outputs = model(batch_x, padding_mask, None, None)
            loss = args.criterion(outputs, label.long().squeeze(-1))
            train_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            args.optimizer.step()


def test_model(model, test_loader, args):
    return



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
    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1-epoch/args.total_iter)
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
            ######################## huggingface library ####################################################
            #args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.total_iter, power=0.5)


        if 0:
            checkpoint = torch.load(args.saveDir)
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")

        train_model(model, trainloader, args)
    else:
        test_model(model, test_loader, args)
