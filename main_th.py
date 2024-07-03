from main import run,save_results
from pynvml import *
from torch.multiprocessing import set_start_method
from parameters import args_parser
import torch
import multiprocessing as mp
import itertools
import torch.multiprocessing as mpcuda
import time
import numpy as np
import argparse
import random



def main_thread(args):
    CA = []
    ASR = []
    CA_def = []
    ASR_def = []
    args.sim_id = random.randint(1, 9999)
    best_overall = 0
    best_bd_model = None
    for i in range(3):
        clean_test_acc, bd_accuracy_test, clean_test_acc_def, bd_accuracy_test_def, bd_generator = run(args)
        CA_def.append(clean_test_acc_def)
        ASR_def.append(bd_accuracy_test_def)
        CA.append(clean_test_acc)
        ASR.append(bd_accuracy_test)
        overall_acc = 0.45 * clean_test_acc + 0.55 * bd_accuracy_test
        if overall_acc > best_overall:
            best_overall = overall_acc
            best_bd_model = bd_generator
    save_results(args, np.mean(CA), np.mean(ASR), np.std(CA), np.std(ASR), np.mean(CA_def), np.mean(ASR_def),
                 np.std(CA_def), np.std(ASR_def), best_bd_model)


if __name__ == '__main__':
    args = args_parser()
    worker_per_device = args.worker_per_device
    cuda = args.use_gpu
    cuda_info = None
    if torch.cuda.device_count() < 1:
        cuda = False
        print('No Nvidia gpu found to use cuda, overriding "cpu" as device')
    Process = mpcuda.Process if cuda else mp.Process
    available_gpus = torch.cuda.device_count() - len(args.excluded_gpus) if cuda else 0
    max_active_user = available_gpus * worker_per_device if cuda else worker_per_device
    first_gpu_share = np.repeat(worker_per_device, torch.cuda.device_count())
    first_gpu_share[args.excluded_gpus] = 0
    combinations = []
    work_load = []
    simulations = []
    w_parser = argparse.ArgumentParser()
    started = 0
    excluded_args = ['excluded_gpus','lr_decay']
    for arg in vars(args):
        arg_type = type(getattr(args, arg))
        if arg_type == list and arg not in excluded_args:
            work_ = [n for n in getattr(args, arg)]
            work_load.append(work_)
    for t in itertools.product(*work_load):
        combinations.append(t)
    print('Number of simulations is :',len(combinations))
    for combination in combinations:
        w_parser = argparse.ArgumentParser()
        listC = 0
        for arg in vars(args):
            arg_type = type(getattr(args, arg))
            if arg_type == list and arg not in excluded_args:
                new_type = type(combination[listC])
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=combination[listC], help='')
                listC += 1
            else:
                val = getattr(args, arg)
                new_type = type(getattr(args, arg))
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=val, help='')

        if cuda:
            if started < max_active_user:
                selected_gpu = np.argmax(first_gpu_share)
                first_gpu_share[selected_gpu] -= 1
            else:
                nvmlInit()
                cuda_info = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
                             for i in range(torch.cuda.device_count())]
                cuda_memory = np.zeros(torch.cuda.device_count())
                for i, gpu in enumerate(cuda_info):
                    if i not in args.excluded_gpus:
                        cuda_memory[i] = gpu.free
                selected_gpu = np.argmax(cuda_memory)
            print('Process {} assigned with gpu:{}'.format(started, selected_gpu))
            w_parser.add_argument('--gpu_id', type=int, default=selected_gpu,
                                  help='cuda device selected')  # assign gpu for the work
        else:
            w_parser.add_argument('--gpu_id', type=int, default=-1, help='cpu selected')  # assign gpu for the work

        w_args = w_parser.parse_args()
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        process = Process(target=main_thread, args=(w_args,))
        process.start()
        simulations.append(process)
        started += 1

        while not len(simulations) < max_active_user:
            for i, process_data in enumerate(simulations):
                if not process_data.is_alive():
                    # remove from processes
                    p = simulations.pop(i)
                    del p
                    time.sleep(10)