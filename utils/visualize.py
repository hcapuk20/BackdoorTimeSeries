import os.path
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn import svm

"""
Adopted from https://github.com/Unispac/Circumventing-Backdoor-Defenses/blob/master/visualize.py
@inproceedings{
qi2023revisiting,
title={Revisiting the Assumption of Latent Separability for Backdoor Defenses},
author={Xiangyu Qi and Tinghao Xie and Yiming Li and Saeed Mahloujifar and Prateek Mittal},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=_wSHsgrVali}
}
"""
def visualize(clean, poisoned, args):
    visualizer = TSNE(n_components=2)

    clean_length = len(clean)

    reduced_features = visualizer.fit_transform( torch.cat([clean, poisoned], dim=0).cpu().detach() ) # all features vector under the label

    plt.figure()

    plt.scatter(reduced_features[:clean_length, 0], reduced_features[:clean_length, 1], marker='o', s=5,
                color='blue', alpha=1.0)
    plt.scatter(reduced_features[clean_length:, 0], reduced_features[clean_length:, 1], marker='^', s=8,
                color='red', alpha=0.7)


    plt.axis('off')


    dataset = args.root_path.split('/')[-1]
    if len(dataset) < 2:
        dataset = args.root_path.split('/')[-2]
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
    save_path = f'visuals/latent_{dataset}_{args.bd_model}_{args.model}_{args.train_mode}.png'
    plt.tight_layout()
    plt.savefig(save_path)

    plt.clf()


def visualize2(latents, poisoned_indices, silent_indices, args):
    visualizer = TSNE(n_components=2)

    poisoned_indices = poisoned_indices.tolist()
    silent_indices = silent_indices.tolist() if len(silent_indices) > 0 else None

    reduced_features = visualizer.fit_transform( latents.cpu().detach() ) 
    poisoned_features = reduced_features[poisoned_indices]
    silent_features = reduced_features[silent_indices] if silent_indices is not None else None

    backdoored_indices = poisoned_indices + silent_indices if silent_indices is not None else poisoned_indices

    mask = torch.ones(len(reduced_features), dtype=bool)
    mask[backdoored_indices] = False
    # Get the values with indexes that are not poisoned_indices
    clean_features = reduced_features[mask]

    plt.figure()

    plt.scatter(clean_features[:, 0], clean_features[:, 1], marker='o', s=5,
                color='blue', alpha=1.0)
    plt.scatter(poisoned_features[:, 0], poisoned_features[:, 1], marker='^', s=8,
                color='red', alpha=0.7)
    if silent_features is not None:
        plt.scatter(silent_features[:, 0], silent_features[:, 1], marker='^', s=8,
                color='purple', alpha=0.7)       

    plt.axis('off')

    dataset = args.root_path.split('/')[-1]
    if len(dataset) < 2:
        dataset = args.root_path.split('/')[-2]
    save_path = f'visuals/latent2_{dataset}_{args.bd_model}_{args.model}_{args.train_mode}_{random.randint(0,100)}.png'
    plt.tight_layout()
    plt.savefig(save_path)

    plt.clf()