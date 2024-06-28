import torch
import os
import torch.nn as nn
import numpy as np
def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")

    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )


def main(args,shape):

    # init_mask = np.random.randn(1, opt.input_height, opt.input_width).astype(np.float32)
    # init_pattern = np.random.randn(opt.input_channel, opt.input_height, opt.input_width).astype(np.float32)

    init_mask = np.ones((1, shape[1],shape[2])).astype(np.float32)
    init_pattern = np.ones((shape)).astype(np.float32)

    for test in range(5):
        masks = []
        idx_mapping = {}

        for target_label in range(args.num_class):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            #args.target_label = target_label
            recorder, opt = train(opt, init_mask, init_pattern)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1

        l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping, opt)


