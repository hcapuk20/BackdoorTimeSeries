"""
Adapted from: https://github.com/SCLBD/BackdoorBench/blob/main/detection_pretrain/spectre.py
"""

'''
SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics
This file is modified based on the following source:
link : https://github.com/SewoongLab/spectre-defense

@inproceedings{hayase2021spectre,
    title={Spectre: Defending against backdoor attacks using robust statistics},
    author={Hayase, Jonathan and Kong, Weihao and Somani, Raghav and Oh, Sewoong},
    booktitle={International Conference on Machine Learning},
    pages={4129--4139},
    year={2021},
    organization={PMLR}}

basic structure for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. Spectral defense:
        a. get the activation from a hidden layer in the model as representation for each data
        b. run quantum filter for k different squared values
        c. keep the best k and correspoding selected samples as backdoor samples
    4. Record TPR and FPR.
'''
import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn

from pprint import  pformat
import yaml
import logging
import time
from sklearn.metrics import confusion_matrix
import csv
from sklearn import metrics
import subprocess
from sklearn.decomposition import PCA
from numpy.linalg import norm, inv
from scipy.linalg import sqrtm
from .spectre_utils import *
from torch.utils.data import DataLoader
from scipy.linalg import svd

def get_features(model, dataloader, args):
    with torch.no_grad():
        model.eval()
        TOO_SMALL_ACTIVATIONS = 32
    activations_all = []
    for i, (batch_x, label, padding_mask) in enumerate(dataloader):
        assert args.model.lower() in ['resnet2','timesnet', 'informer']
        if args.model.lower() == 'resnet2':
            target_layer = dict(model.named_modules())['fc']
        else:
            target_layer = dict(model.named_modules())['projection']
        batch_x = batch_x.float().to(args.device)
        label = label.to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        outs = []
        def layer_hook(module, inp, out):
            ## getting latents (representation before the final projection layer...)
            outs.append(inp[0].data)
        hook = target_layer.register_forward_hook(layer_hook)
        _ = model(batch_x, padding_mask,None,None)
        activations = outs[0].view(outs[0].size(0), -1)
        activations_all.append(activations.cpu())
        hook.remove()


    activations_all = torch.cat(activations_all, axis=0)
    return activations_all

def rcov_quantum_filter(reps, eps, k, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    n, d = reps.shape

    # PCA
    pca = PCA(n_components=k)
    reps_pca = pca.fit_transform(reps)

    if k == 1:
        reps_estimated_white = reps_pca
        sigma_prime = np.ones((1, 1))
    else:
        selected = cov_estimation_iterate(reps_pca, eps/n, tau, None, limit=round(limit1*eps))
        reps_pca_selected = reps_pca[selected,:]
        sigma = np.cov(reps_pca_selected, rowvar=False, bias=False)
        reps_estimated_white = np.linalg.solve(sqrtm(sigma), reps_pca.T).T
        sigma_prime = np.cov(reps_estimated_white, rowvar=False, bias=False)

    I = np.eye(sigma_prime.shape[0])
    M = np.exp(alpha * (sigma_prime - I) / (norm(sigma_prime, 2) - 1)) if k > 1 else np.ones((1, 1))
    M /= np.trace(M)
    estimated_poison_ind = k_lowest_ind(-np.array([x.T @ M @ x for x in reps_estimated_white]), round(limit2*eps))

    return ~estimated_poison_ind


def rcov_auto_quantum_filter(reps, eps, poison_indices, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    
    pca = PCA(n_components=64)
    reps_pca = pca.fit_transform(reps) 
    U = pca.components_

    best_opnorm, best_selected, best_k = -float('inf'), None, None
    squared_values = [int(x) for x in np.linspace(1, np.sqrt(64), 8) ** 2]

    for k in squared_values:
        selected = rcov_quantum_filter(reps, eps, k, alpha, tau, limit1=limit1, limit2=limit2)
        selected_matrix = reps_pca[selected,:].T
        cov_matrix = np.cov(selected_matrix)
        transformed = np.linalg.solve(sqrtm(cov_matrix), reps_pca.T)
        cov_matrix_prime = np.cov(transformed)
        I = np.eye(cov_matrix_prime.shape[0]) 
        U, s, Vh = svd(cov_matrix_prime)
        opnorm = s[0]

        M = np.exp(alpha * (cov_matrix_prime - I) / (opnorm - 1))    
        M /= np.trace(M)
        op = np.trace(cov_matrix_prime * M) / np.trace(M) 
        # poison_removed = sum([not selected[i] for i in poison_indices])
        if op > best_opnorm:
            best_opnorm, best_selected, best_k = op, selected, k
    return best_selected, best_opnorm



def cal(true, pred):
    TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
    return TN, FP, FN, TP 
def metrix(TN, FP, FN, TP):
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    precision = TP/(TP+FP)
    acc = (TP+TN)/(TN+FP+FN+TP)
    return TPR, FPR, precision, acc

def spectre_filtering(model, bd_loader, backdoored_indices, args):

    features = get_features(model, bd_loader, args)
    suspicious_indices = []

    ## a. get indices of poisoned dataset whose labels are equal to current target label.
    target_labeled_indices = set([i for i, (_, label, _) in enumerate(bd_loader.dataset) if label == args.target_label])

    ### b. get the activation as representation for each data
    full_cov = [features[temp_idx].cpu().numpy() for temp_idx in target_labeled_indices]
    full_cov = np.array(full_cov)
    n, _ = full_cov.shape
    eps = args.poisoning_ratio * len(bd_loader.dataset)
    if eps <= 0:
        eps = round(0.1 * n)
    if eps > 0.33 * n:
        eps = round(0.33 * n)
    if n < 500:
        if eps > 0.1 * n:
            eps = round( 0.1 * n)
    eps = int(eps)
    quantum_poison_ind, opnorm = rcov_auto_quantum_filter(full_cov, eps, backdoored_indices)
    quantum_poison_ind = np.logical_not(quantum_poison_ind)

    suspicious_class_indices_mask = quantum_poison_ind
    suspicious_class_indices = torch.tensor(suspicious_class_indices_mask).nonzero().squeeze(1)
    cur_class_indices = torch.tensor(list(target_labeled_indices), dtype=torch.int64)
    suspicious_indices.append(cur_class_indices[suspicious_class_indices])
    suspicious_indices = np.concatenate(suspicious_indices,axis=0)    

    true_indices = np.zeros(len(bd_loader.dataset))
    for i in range(len(true_indices)):
        if i in backdoored_indices:
            true_indices[i] = 1       

    pred_indices = np.zeros(len(bd_loader.dataset))
    for i in range(len(pred_indices)):
        if i in suspicious_indices:
            pred_indices[i] = 1

    tn, fp, fn, tp = cal(true_indices, pred_indices)
    TPR, FPR, precision, acc = metrix(tn, fp, fn, tp)

    return tn, fp, fn, tp, TPR, FPR

    