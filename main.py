import os

import numpy as np
import torch.nn as nn
from data_provider.data_factory import data_provider, custom_data_loader
from models import * # Here we import the architecture
from parameters import * # Here we import the parameters
from models.Informer import Model as Informer
from models.TimesNet import Model as TimesNet
from models.FEDformer import Model as FEDformer
from models.iTransformer import Model as iTransformer
from models.PatchTST import Model as PatchTST
from models.bd_Universal import Bd_Tnet
from models.Bd_inverted import Model as Bd_inverted
from models.Bd_patch import Model as Bd_patch
from models.Transformer import Model as Transformer
from models.SegRNN import Model as SegRNN
from models.LightTS import Model as LightTS
from models.LSTM import LSTMClassifier as LSTM
from models.ResNet import ResNetClassifier as ResNet
import matplotlib.pyplot as plt
from tqdm import tqdm
from epoch import *
from utils.save_results import save_results
from utils.plot import plot_time_series

model_dict = {
    'TimesNet': TimesNet,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'itransformer': iTransformer,
    'patchtst': PatchTST,
    'transformer': Transformer,
    'segRNN': SegRNN,
    'lightTS': LightTS,
    'lstm': LSTM,
    'resnet': ResNet,
}

######################################################### This is the V0 (initial prototype) code for time series backdoor ############################################

### We use additive trigger generation (trigger is added to clean data and also clipped for impercemptibility)
### Phase 1: Jointly train surrogate classifier (also better to be pre-trained wıth clean data) with trigger generater 
### opt1: single optimizer single loss opt2: Iterative training two loss and  two optimizer 
### Phase 2: Train the final classifier with backdoored data to measure ASR of trigger model


################### Regularizers for during training #################################
##### 1) Frequency transfer: objective is to change the dominant freq. components of the clean data when trigger is inserted
##### 2) Softmix: mix clean and backdoored data with 0< \lambda < 1 (as well as labels ) to mitigate simple trigger patterns 
##### that is trigger network learns a meaningfull realtion  between the trigger pattern and the target label ** \lambda may change gradually over iters

################### Attack Framework ################################################
### Initial design: fixed target attack (target label is fixed and same for all) 
### step2: dynamic target attack (generate trigger for any given target label) 
### step3: the best traget attack (for each target or label first decide the best target label then generate trigger, similar to adversarial)


############ Loading the dataset
def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader
############ Loading the model

######################### Here we initialize the backdoor model ###################################

def get_bd_model(args, train_data, test_data):
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len) ## seq length
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    print('enc_in',args.enc_in,'seq_len',args.seq_len)
    args.num_class = len(train_data.class_names)
    # model init
    model_sur = model_dict[args.model_sur](args).float().to(args.device)
    ############## Trigger Network ################################
    if args.bd_model == 'inverted':
        generative_model = Bd_inverted(args).float().to(args.device)
    elif args.bd_model == 'patchtst':
        generative_model = Bd_patch(args).float().to(args.device)
    else:
        raise NotImplementedError
    ################## Combined Model ===> backdoor trigger network + surrogate classifier network ###################
    bd_model = Bd_Tnet(args,generative_model).float().to(args.device)
    if args.use_multi_gpu and args.use_gpu:
        bd_model = nn.DataParallel(bd_model, device_ids=args.device_ids)
    return bd_model,model_sur

def get_clean_model(args, train_data, test_data):
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    print('enc_in', args.enc_in, 'seq_len', args.seq_len)
    args.num_class = len(train_data.class_names)
    # model init
    model = model_dict[args.model](args).float().to(args.device)
    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model



############ Training the model
def run(args):
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    args.saveDir = 'weights/model_weights'  # path to be saved to
    # ======================================================= Initialize the model
    train_data, train_loader = get_data(args=args, flag='train')
    test_data, test_loader = get_data(args=args, flag='test')
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    #################### bd_model is the combined model ===> backdoor trigger network + surrogate classifier network
    bd_model, surr_model = get_bd_model(args, train_data, test_data)  # ===> also take data as a input
    # since initializion of the networks requires the seq length of the data
    best_bd = 0
    best_dict = None
    last_dict = None
    print("model initialized...")
    # ============================================================================
    # ===== Add loss criterion to the args =====
    args.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    ######### The loss term for utilizing mixup 
    args.criterion_mix = nn.CrossEntropyLoss(reduce=False) # in order to have batch-wise results rather than sum or average
    # ===== Add optimizer to the args =====
    opt_surr = None
    if args.load_bd_model is None:
        ### Experimental
        if args.warm_up:
            print('Warming up surrogate classifier...')
            opt_surr = torch.optim.Adam(surr_model.parameters(), lr=args.lr)
            for i in range(3):
                bd_model.train()
                total_loss, accuracy = clean_train(surr_model, train_loader, args, opt_surr)
                print('Train Loss', total_loss, 'Train acc', accuracy)
            if args.freeze_surrogate:
                bd_model.freeze_classifier()
                print('Freezing classifier')
        ####
        print('Starting backdoor model training...')
        ######################################## ************************ bu kısım sankı hatalı
        # opt_bd = torch.optim.AdamW(filter(lambda p: p.requires_grad, bd_model.parameters()), lr=args.lr)
        if args.seperate_opts:
            opt_bd = torch.optim.AdamW(bd_model.trigger.parameters(), lr=args.lr)
            opt_surr = torch.optim.AdamW(surr_model.parameters(), lr=args.lr)
        else:
            collective_params = list(surr_model.parameters()) + list(bd_model.parameters())
            opt_bd = torch.optim.AdamW(collective_params, lr=args.lr)
        for i in tqdm(range(args.train_epochs)):
            ########### Here train the trigger while also update the surrogate classifier #########
            bd_model.train()
            train_loss, train_dic, train_acc, bd_train_acc = epoch(bd_model, surr_model, train_loader, args, opt_bd)
            ############################################
            bd_model.eval()
            test_loss, test_dic, test_acc, bd_test_acc = epoch(bd_model, surr_model, test_loader, args)
            print('Train Loss', train_loss, 'Train acc', train_acc, 'Test Loss', test_loss, 'Test acc', test_acc)
            print('Backdoor Train', bd_train_acc, 'Backdoor Test', bd_test_acc)
            ce_c_train, ce_c_test = np.average(train_dic['CE_c']), np.average(test_dic['CE_c'])
            ce_bd_train, ce_bd_test = np.average(train_dic['CE_bd']), np.average(test_dic['CE_bd'])
            reg_train, reg_test = np.average(train_dic['reg']), np.average(test_dic['reg'])
            print('CE clean Train', ce_c_train, 'CE Backdoor train: ', ce_bd_train, 'Reg Train', reg_train)
            print('CE clean Test', ce_c_test, 'CE Backdoor Test: ', ce_bd_test, 'Reg Test', reg_test)
            if best_bd < bd_test_acc:
                best_bd = bd_test_acc
                best_dict = bd_model.state_dict()
                if not os.path.exists('weights'):
                    os.makedirs('weights')
                torch.save(bd_model.trigger.state_dict(), 'weights/best_bd_model_weights.pth')
            torch.save(bd_model.trigger.state_dict(), 'weights/last_bd_model_weights.pth')
        # last_dict = bd_model.state_dict()

        print('Starting clean model training with backdoor samples...')

        bd_model.load_state_dict(best_dict)
        bd_generator = bd_model.trigger

    else:
        print('Loading pretrained Backdoor trigger generator model...')
        path = 'weights/' + args.load_bd_model
        dicts = torch.load(path)
        bd_generator = bd_model.trigger
        bd_generator.load_state_dict(dicts)



    # ################Clean Model tranining for  testing the model without attacks##############################
    # for i in range(30):
    #     tmp_model = get_clean_model(args, train_data, test_data)
    #     tmp_optimizer = torch.optim.Adam(tmp_model.parameters(), lr=args.lr, eps=1e-9)
    #     tmp_model.train()
    #     train_loss, train_accuracy = clean_train(tmp_model, train_loader, args, tmp_optimizer)
    #     print('Train Loss:', train_loss, 'Train Acc:', train_accuracy)
    #     tmp_model.eval()
    #     clean_test_loss,clean_test_acc = clean_test(tmp_model, test_loader, args)
    #     print('Test Loss:', clean_test_loss, 'Test Acc:', clean_test_acc)
    #
    # ##########################################################################################################

    #### START OF THE NEW TRANING WITH TRAINED TRIGGER GENERATOR
    bd_generator.eval()
    clean_ratio = 1 - args.poisoning_ratio
    train_dataset, bd_dataset = torch.utils.data.random_split(train_data, [clean_ratio, args.poisoning_ratio])
    bs = int(args.batch_size * clean_ratio) + 1
    train_loader = custom_data_loader(train_dataset, args, flag='train', force_bs=bs)
    bd_bs = int(args.batch_size * args.poisoning_ratio) + 1
    bd_loader = custom_data_loader(bd_dataset, args, flag='train', force_bs=bd_bs)
    clean_model = get_clean_model(args, train_data, test_data)
    optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.lr)


    for i in tqdm(range(100)):
        clean_model.train()
        ### Train epoch with clean data and backdoor datasets.
        train_loss, train_accuracy, bd_accuracy_train = epoch_clean_train(bd_generator, clean_model, train_loader,
                                                                          bd_loader, args, optimizer)
        clean_model.eval()
        clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_generator, clean_model, test_loader, args)
        print('Test CA:', clean_test_acc, 'Test ASR:', bd_accuracy_test)

    if args.root_path.split('/')[-2] != 'UWaveGestureLibrary':
        clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_generator, clean_model, test_loader, args, plot_time_series)
    return clean_test_acc, bd_accuracy_test


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    CA = []
    ASR = []
    for i in range(3):
        clean_test_acc, bd_accuracy_test = run(args)
        CA.append(clean_test_acc)
        ASR.append(bd_accuracy_test)
    save_results(args, np.mean(CA), np.mean(ASR))
