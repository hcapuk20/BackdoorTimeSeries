from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader,UEAloader_bd,UEAloader_bd2
from data_provider.uea import collate_fn, collate_fn_bd
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        if not args.multi_thread:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                pin_memory=True,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

def custom_data_loader(dataset,args,flag,force_bs=None):
    drop_last = True
    if flag == 'test':
        shuffle_flag = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if force_bs is not None:
        batch_size = force_bs
    if not args.multi_thread:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    return data_loader


def bd_data_provider(args, flag,bd_model):
    Data = UEAloader_bd
    timeenc = 0 if args.embed != 'timeF' else 1
    bd_model.to('cpu')
    if flag == 'test':
        shuffle_flag = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    drop_last = False
    data_set = Data(bd_model=bd_model,poision_rate=args.poisoning_ratio,
        silent_poision=args.silent_poisoning,target_label=args.target_label,
        root_path=args.root_path,
        flag=flag,max_len=args.seq_len,enc_in=args.enc_in
    )
    if not args.multi_thread:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn_bd(x,bd_model, max_len=args.seq_len,target_label=args.target_label))
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    return data_set,data_loader


def bd_data_provider2(args, flag,bd_model):
    Data = UEAloader_bd2
    timeenc = 0 if args.embed != 'timeF' else 1
    bd_model.to('cpu')
    if flag == 'test':
        shuffle_flag = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    drop_last = False
    data_set = Data(bd_model=bd_model,poision_rate=args.poisoning_ratio,
        silent_poision=args.silent_poisoning,target_label=args.target_label,
        root_path=args.root_path, flag=flag
    )
    if not args.multi_thread:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn_bd(x,bd_model, max_len=args.seq_len,target_label=args.target_label))
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn_bd(x,bd_model, max_len=args.seq_len,target_label=args.target_label)
        )



    if flag == 'test':
        poisoned_indices = data_set.bd_inds
        silent_indices = data_set.silent_bd_set
        return data_set, data_loader, poisoned_indices, silent_indices
    
    else:
        return data_set, data_loader
