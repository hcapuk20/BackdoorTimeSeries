import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import random


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


def save_results(args,ca,asr):
    sim_id = random.randint(1,9999)
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
    f.write('asr : {}'.format(asr) + '\n')
    f.close()
    return None