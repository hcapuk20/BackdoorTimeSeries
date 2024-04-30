import os
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from epoch import *
from utils.save_results import save_results

model_dict = {
    'TimesNet': TimesNet,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'itransformer': iTransformer,
    'patchtst': PatchTST,
    'transformer': Transformer,
    'segRNN': SegRNN,
    'lightTS': LightTS
}

######################################################### This is the V0 (initial prototype) code for time series backdoor ############################################



############ Loading the dataset
def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader
############ Loading the model

def get_bd_model(args, train_data, test_data):

    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    print('enc_in',args.enc_in,'seq_len',args.seq_len)
    args.num_class = len(train_data.class_names)
    # model init
    model_sur = model_dict[args.model_sur](args).float()
    if args.bd_model == 'inverted':
        generative_model = Bd_inverted(args).float().to(args.device)
    elif args.bd_model == 'patchtst':
        generative_model = Bd_patch(args).float().to(args.device)
    else:
        raise NotImplementedError
    bd_model = Bd_Tnet(args,model_sur,generative_model).float().to(args.device)
    if args.use_multi_gpu and args.use_gpu:
        bd_model = nn.DataParallel(bd_model, device_ids=args.device_ids)
    return bd_model

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




if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    args.saveDir = 'weights/model_weights'  # path to be saved to
    # ======================================================= Initialize the model
    train_data, train_loader = get_data(args=args, flag='train')
    test_data, test_loader = get_data(args=args, flag='test')
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    bd_model = get_bd_model(args,train_data,test_data)
    best_bd = 0
    best_dict = None
    last_dict = None
    print("model initialized...")
    # ============================================================================
    # ===== Add loss criterion to the args =====
    args.criterion = nn.CrossEntropyLoss()
    # ===== Add optimizer to the args =====
    if args.load_bd_model is None:
        ### Experimental
        if args.warm_up:
            print('Warming up surrogate classifier...')
            opt_surr = torch.optim.Adam(bd_model.vanilla_model.parameters(), lr=args.lr)
            for i in range(3):
                bd_model.train()
                total_loss, accuracy = clean_train(bd_model.vanilla_model, train_loader, args, opt_surr)
                print('Train Loss', total_loss, 'Train acc', accuracy)
            if args.freeze_surrogate:
                bd_model.freeze_classifier()
                print('Freezing classifier')
        opt_surr = None
        ####
        print('Starting backdoor model training...')
        opt_bd = torch.optim.AdamW(filter(lambda p: p.requires_grad, bd_model.parameters()), lr=args.lr)
        for i in tqdm(range(args.train_epochs)):
            bd_model.train()
            #train_loss, train_acc = epoch(model,bd_model, train_loader, args,optimizer)
            train_loss, train_dic, train_acc,bd_train_acc = epoch(bd_model, train_loader, args, opt_bd)
            bd_model.eval()
            test_loss, test_dic,test_acc, bd_test_acc = epoch(bd_model,test_loader, args)
            print('Train Loss',train_loss,'Train acc',train_acc,'Test Loss',test_loss,'Test acc',test_acc)
            print('Backdoor Train',bd_train_acc,'Backdoor Test',bd_test_acc)
            ce_train,ce_test = np.average(train_dic['CE']),np.average(test_dic['CE'])
            l2_train,l2_test = np.average(train_dic['L2']),np.average(test_dic['L2'])
            print('CE Train',ce_train,'L2 Train',l2_train)
            print('CE Test',ce_test,'L2 Test',l2_test)
            if best_bd < bd_test_acc:
                best_bd = bd_test_acc
                best_dict = bd_model.state_dict()
                if not os.path.exists('weights'):
                    os.makedirs('weights')
                torch.save(bd_model.state_dict(), 'weights/best_bd_model_weights.pth')
            torch.save(bd_model.state_dict(), 'weights/last_bd_model_weights.pth')
        #last_dict = bd_model.state_dict()

        print('Starting clean model training with backdoor samples...')

        bd_model.load_state_dict(best_dict)

    else:
        print('Loading pretrained Backdoor trigger generator model...')
        path = 'weights/' + args.load_bd_model
        dicts = torch.load(path)
        bd_model.load_state_dict(dicts)

    bd_model.eval()
    clean_ratio = 1 - args.poisoning_ratio
    train_dataset, bd_dataset = torch.utils.data.random_split(train_data, [clean_ratio, args.poisoning_ratio])
    bs = int(args.batchSize * clean_ratio) + 1
    train_loader = custom_data_loader(train_dataset, args, flag='train', force_bs=bs)
    bd_bs = int(args.batchSize * args.poisoning_ratio) + 1
    bd_loader = custom_data_loader(bd_dataset, args, flag='train', force_bs=bd_bs)
    clean_model = get_clean_model(args, train_data, test_data)
    optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    for i in tqdm(range(30)):
        clean_model.train()
        train_loss, train_accuracy, bd_accuracy_train = epoch_clean_train(bd_model,clean_model, train_loader,bd_loader, args,optimizer)
        clean_model.eval()
        clean_test_acc, bd_accuracy_test = epoch_clean_test(bd_model,clean_model, test_loader,args)
        print('CA:',clean_test_acc,'ASR:',bd_accuracy_test)
    save_results(args,clean_test_acc,bd_accuracy_test)
