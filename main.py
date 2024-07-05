import math
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler

from data_provider.data_factory import data_provider, custom_data_loader,bd_data_provider,bd_data_provider2
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
from models.Bd_patch_dyn import Model as Bd_patch_dyn
from models.Transformer import Model as Transformer
from models.SegRNN import Model as SegRNN
from models.LightTS import Model as LightTS
from models.LSTM import LSTMClassifier as LSTM
from models.ResNet import ResNetClassifier as ResNet
from models.ResNet2 import ResNet as ResNet2
import matplotlib.pyplot as plt
from tqdm import tqdm
from epoch import *
from utils.save_results import save_results
from utils.plot import plot_time_series
from utils.visualize import visualize
from copy import deepcopy
from epoch_cross import epoch_marksman_lam_cross

model_dict = {
    'TimesNet': TimesNet,
    'timesnet': TimesNet,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'itransformer': iTransformer,
    'patchtst': PatchTST,
    'transformer': Transformer,
    'segRNN': SegRNN,
    'lightTS': LightTS,
    'lstm': LSTM,
    'resnet': ResNet,
    'resnet2': ResNet2
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
    args.numb_class = train_data.num_cls
    print('enc_in',args.enc_in,'seq_len',args.seq_len)
    args.num_class = len(train_data.class_names)
    train_p,test_p = train_data.dist_p, test_data.dist_p
    print('train data distribution:')
    print(train_p)
    print('test data distribution: ')
    print(test_p)
    ############## Surrogate Classifier ################################
    # model init
    model_sur = model_dict[args.model_sur](args).float().to(args.device)
    ############## Trigger Network ################################
    if args.bd_model == 'inverted':
        generative_model = Bd_inverted(args).float().to(args.device)
    elif args.bd_model == 'patchtst':
        generative_model = Bd_patch(args).float().to(args.device)
    elif args.bd_model == 'patchdyn':
        generative_model = Bd_patch_dyn(args).float().to(args.device)
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
    bd_model_prev = deepcopy(bd_model)
    # since initializion of the networks requires the seq length of the data
    best_bd = 0
    best_dict = None
    last_dict = None
    print("model initialized...")
    # ============================================================================
    # ===== Add loss criterion to the args =====
    args.criterion = nn.CrossEntropyLoss()
    ######### The loss term for utilizing mixup 
    args.criterion_mix = nn.CrossEntropyLoss(reduce=False) # in order to have batch-wise results rather than sum or average
    # ===== Add optimizer to the args =====
    args.criterion_bd = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
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
        if args.train_mode != 'basic':
            opt_bd = torch.optim.AdamW(bd_model.trigger.parameters(), lr=args.lr)
            opt_surr = torch.optim.AdamW(surr_model.parameters(), lr=args.lr)
            schedular_bd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bd, T_max=args.train_epochs, eta_min=1e-6)
            #schedular_surr = torch.optim.lr_scheduler.MultiStepLR(opt_surr, milestones=[args.train_epochs // 2], gamma=0.1)
        else:
            collective_params = list(surr_model.parameters()) + list(bd_model.parameters())
            opt_bd = torch.optim.AdamW(collective_params, lr=args.lr)
            schedular_bd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bd, T_max=args.train_epochs, eta_min=1e-6)
        for i in tqdm(range(args.train_epochs)):
            ########### Here train the trigger while also update the surrogate classifier #########
            if args.train_mode == 'basic':
                train_loss, train_dic, train_acc, bd_train_acc = epoch(bd_model, surr_model, train_loader, args, opt_bd,None)
                test_loss, test_dic, test_acc, bd_test_acc = epoch(bd_model, surr_model, test_loader, args,train=False)
            elif args.train_mode == '2opt':
                train_loss, train_dic, train_acc, bd_train_acc = epoch(bd_model, surr_model, train_loader, args, opt_bd, opt_surr)
                test_loss, test_dic, test_acc, bd_test_acc = epoch(bd_model, surr_model, test_loader, args,train=False)
            elif args.train_mode == 'marksman':
                train_loss, train_dic, train_acc, bd_train_acc = epoch_marksman(bd_model,bd_model_prev ,surr_model, train_loader, args, opt_bd, opt_surr)
                test_loss, test_dic, test_acc, bd_test_acc = epoch_marksman(bd_model,bd_model_prev, surr_model, test_loader, args,train=False)
            elif args.train_mode == 'marksman_lam':
                train_loss, train_dic, train_acc, bd_train_acc = epoch_marksman_lam(bd_model, bd_model_prev, surr_model,
                                                                                train_loader, args, opt_bd, opt_surr)
                test_loss, test_dic, test_acc, bd_test_acc = epoch_marksman_lam(bd_model, bd_model_prev, surr_model,
                                                                            test_loader, args, train=False)
            # elif args.train_mode == 'marksman_lam_cross':
            #     train_loss, train_dic, train_acc, bd_train_acc = epoch_marksman_lam_cross(bd_model, bd_model_prev, surr_model,
            #                                                                         train_loader, args, opt_bd,
            #                                                                         opt_surr)
            #     test_loss, test_dic, test_acc, bd_test_acc = epoch_marksman_lam_cross(bd_model, bd_model_prev, surr_model,
            #                                                                     test_loader, args, train=False)
                ############################################
            schedular_bd.step()
            print('Train Loss', train_loss, 'Train acc', train_acc, 'Test Loss', test_loss, 'Test acc', test_acc)
            print('Backdoor Train', bd_train_acc, 'Backdoor Test', bd_test_acc)
            ce_c_train, ce_c_test = np.average(train_dic['CE_c']), np.average(test_dic['CE_c'])
            ce_bd_train, ce_bd_test = np.average(train_dic['CE_bd']), np.average(test_dic['CE_bd'])
            reg_train, reg_test = np.average(train_dic['reg']), np.average(test_dic['reg'])
            print('CE clean Train', ce_c_train, 'CE Backdoor train: ', ce_bd_train, 'Reg Train', reg_train)
            print('CE clean Test', ce_c_test, 'CE Backdoor Test: ', ce_bd_test, 'Reg Test', reg_test)
            if best_bd <= bd_test_acc:
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
    poisoned_data, bd_train_loader, poisoned_indices, silent_indices = bd_data_provider2(args, 'train', bd_generator)
    clean_model = get_clean_model(args, train_data, test_data)
    optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.lr)

    for i in tqdm(range(args.train_epochs_inj)):
        clean_model.train()
        bd_generator.to('cpu')
        train_loss, train_accuracy, _ = epoch_clean_train2(clean_model,bd_train_loader,args,optimizer)
        clean_model.eval()
        bd_generator.to(args.device)
        clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_generator, clean_model, test_loader, args)
        print('Test CA:', clean_test_acc, 'Test ASR:', bd_accuracy_test)
    ## prepare validation data and defence test
    _,val_data = torch.utils.data.random_split(train_data, [.8, .2])
    val_loader = custom_data_loader(val_data, args, flag='train', force_bs=16)
    clean_test_acc_def, bd_accuracy_test_def = defence_test_fp(bd_generator, clean_model,val_loader, test_loader, args)
    hidden_count, caugth_count, false_positives = defence_test_strip(clean_model, bd_train_loader, train_loader, poisoned_indices, silent_indices, args)
    print('defences | CA : {}, ASR : {}'.format( clean_test_acc_def, bd_accuracy_test_def))
    print('STRIP Results: hidden:{}, caugth:{}, FP:{}'.format(hidden_count, caugth_count, false_positives))
    # one final test epoch to save plots.
    clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_generator, clean_model, test_loader, args, plot_time_series, visualize)
    return clean_test_acc, bd_accuracy_test,clean_test_acc_def, bd_accuracy_test_def,bd_generator


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    CA = []
    ASR = []
    CA_def = []
    ASR_def = []
    args.sim_id = random.randint(1,9999)
    best_overall = 0
    best_bd_model = None
    for i in range(3):
        clean_test_acc, bd_accuracy_test,clean_test_acc_def, bd_accuracy_test_def, bd_generator = run(args)
        CA_def.append(clean_test_acc_def)
        ASR_def.append(bd_accuracy_test_def)
        CA.append(clean_test_acc)
        ASR.append(bd_accuracy_test)
        overall_acc = 0.45 * clean_test_acc + 0.55 * bd_accuracy_test
        if overall_acc > best_overall:
            best_overall = overall_acc
            best_bd_model = bd_generator
    save_results(args, np.mean(CA), np.mean(ASR),np.std(CA),np.std(ASR),np.mean(CA_def),np.mean(ASR_def),
                    np.std(CA_def), np.std(ASR_def) ,best_bd_model)
