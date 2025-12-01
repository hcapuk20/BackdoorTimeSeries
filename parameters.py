import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # threding params
    parser.add_argument('--worker_per_device', type=int, default=1, help='parallel processes per device')
    parser.add_argument('--excluded_gpus', type=list, default=[], help='bypassed gpus')
    parser.add_argument('--use_gpu', type=bool, default=True, help='bypassed gpus')

    # parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--model', type=str, default='timesnet', help='NN model')
    parser.add_argument('--model_sur', type=str, default='resnet2', help='surrogate model in the BD model')
    parser.add_argument('--bd_model', type=str, default='cnn',
                        help='trigger generator model. patchtst or inverted')
    parser.add_argument('--train_mode', type=str, default='marksman',
                        help='basic: single loss single optimizer,'
                             '2opt: single loss two optimizers,'
                             'marksman: iterative training')
    parser.add_argument('--poisoning_ratio', type=float, default=0.1, help='Poisoning ratio')
    parser.add_argument('--poisoning_ratio_train', type=float, default=0.1,
                        help='Poisoning ratio of the batch in the trining phase')
    parser.add_argument('--clip_ratio', type=float, default=0.1, help='Poisoning ratio')
    parser.add_argument('--p_cross', type=float, default=0.1, help='cross ratio')
    parser.add_argument('--bd_type', type=str, default='all2one', help='all2one or all2all')
    parser.add_argument('--trainable_token', type=bool, default=True, help='all2all trains token')
    parser.add_argument('--token_hook', type=float, default=1e-2, help='token hook for grad multiplier')
    parser.add_argument('--target_label', type=int, default=0, help='targeted label')
    parser.add_argument('--load_bd_model', type=str, default=None, help='path to the bd model weights')
    parser.add_argument('--label_smooth', type=float, default=0., help='label smoothing')
    parser.add_argument('--silent_poisoning', action="store_true", default=False, help='')
    parser.add_argument('--lambda_alpha', type=float, default=2., help='Mix-up distribution alpha')
    parser.add_argument('--lambda_beta', type=float, default=2., help='Mix-up distribution beta')
    # Training or testing
    parser.add_argument('--is_training', type=int, default=1, help='Running mode')
    parser.add_argument('--warm_up', type=bool, default=True, help='warm up the surrogate model')
    parser.add_argument('--freeze_surrogate', type=bool, default=False, help='surrogate after warm up')
    # task name added for easy use with data provider
    parser.add_argument('--task_name', type=str, default='classification', help='Task to be performed.')

    ############ Training Parameters
    parser.add_argument('--train_epochs', type=int, default=100, help='number of training epochs for trigger generator')
    parser.add_argument('--train_epochs_inj', type=int, default=200,
                        help='number of training epochs for backdoor injection')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size of train input data')
    parser.add_argument('--L2_reg', type=float, default=0, help='L2 regularization for the generated trigger')
    parser.add_argument('--cos_reg', type=float, default=0, help='cosine regularization for the generated trigger')
    parser.add_argument('--div_reg', type=float, default=1.0, help='diversity loss regularization for the generated trigger')
    parser.add_argument('--freq_reg', type=float, default=0,
                        help='L1 loss for real fft (RFFT) values for the triggers')
    parser.add_argument('--opt_method', type=str, default='adamW', help="Optimization method adamW,lamb,adam")
    parser.add_argument('--attack_only_nontarget', action='store_true',
                        help='whether to only inject items that are not already labeled as target label or not, \
                        using this argument means only injecting items that are not labeled with attack target label',
                        default=False)
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--lr_bd', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")

    # data loader
    parser.add_argument('--data', type=str, required=False, default='UEA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/UWaveGestureLibrary/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--ptst_patch_len', type=int, default=16, help='patch len for patch tst')
    parser.add_argument('--ptst_stride', type=int, default=8, help='stride for patchtst')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model_bd', type=int, default=32, help='dimension of trigger model')
    parser.add_argument('--n_heads_bd', type=int, default=4, help='num of heads in trigger model')
    parser.add_argument('--e_layers_bd', type=int, default=2, help='num of encoder layers in trigger model')
    parser.add_argument('--d_layers_bd', type=int, default=1, help='num of decoder layers in trigger model')
    parser.add_argument('--d_ff_bd', type=int, default=32, help='dimension of fcn in trigger model')
    parser.add_argument('--d_model_sur', type=int, default=32, help='dimension of surrogate model')
    parser.add_argument('--n_heads_sur', type=int, default=4, help='num of heads in surrogate model')
    parser.add_argument('--e_layers_sur', type=int, default=2, help='num of encoder layers in surrogate model')
    parser.add_argument('--d_layers_sur', type=int, default=1, help='num of decoder layers in surrogate model')
    parser.add_argument('--d_ff_sur', type=int, default=32, help='dimension of fcn in surrogate model')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_mp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--multi_thread', type=bool, default=False, help='Use True if you run main_th.py, disables num_workers in dataloader')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=(128, 128),
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    ###
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    return args
