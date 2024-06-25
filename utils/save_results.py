import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import random

import torch


def log(path,args,sim_id):
    n_path = path
    f = open(n_path + '/log.txt', 'w+')
    f.write('############## Args ###############' + '\n')
    l =  'sim_id : {}'.format(sim_id)
    f.write(l)
    for arg in vars(args):
        line = str(arg) + ' : ' + str(getattr(args, arg))
        f.write(line + '\n')
    f.write('############ Results ###############' + '\n')
    f.close()


def save_results(args,ca,asr,ca_std,asr_std,def_ca,def_asr,def_ca_std,def_asr_std,model):
    sim_id = args.sim_id
    dataset = args.root_path.split('/')[-1]
    if len(dataset) < 2:
        dataset = args.root_path.split('/')[-2]
    path_ = 'D_{}-M_{}-BM_{}-P_{}-TL_{}-{}'.format(dataset,args.model,args.bd_model,args.poisoning_ratio,args.target_label,sim_id)
    path = 'Results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path,path_)
    os.mkdir(path)
    log(path,args,sim_id),
    f = open(path + '/log.txt', 'a')
    f.write('CA : {}'.format(ca) + '\n')
    f.write('ASR : {}'.format(asr) + '\n')
    f.write('CA STD : {}'.format(ca_std) + '\n')
    f.write('ASR STD : {}'.format(asr_std)+ '\n')
    f.write('def-CA : {}'.format(def_ca) + '\n')
    f.write('def-ASR : {}'.format(def_asr) + '\n')
    f.write('def-CA STD : {}'.format(def_ca_std) + '\n')
    f.write('def-ASR STD : {}'.format(def_asr_std) + '\n')
    f.close()
    torch.save(model.state_dict(), path + '/model.pth')
    return None