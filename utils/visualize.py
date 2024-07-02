import random

import numpy as np
import torch
# import os
# from torchvision import transforms
# import argparse
# from torch import nn
# from utils import supervisor, tools
# import config
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn import svm

# class mean_diff_visualizer:

#     def fit_transform(self, clean, poison):
#         clean_mean = clean.mean(dim=0)
#         poison_mean = poison.mean(dim=0)
#         mean_diff = poison_mean - clean_mean
#         print("Mean L2 distance between poison and clean:", torch.norm(mean_diff, p=2).item())

#         proj_clean_mean = torch.matmul(clean, mean_diff)
#         proj_poison_mean = torch.matmul(poison, mean_diff)

#         return proj_clean_mean, proj_poison_mean


# class oracle_visualizer:

#     def __init__(self):
#         self.clf = svm.LinearSVC()

#     def fit_transform( self, clean, poison ):

#         clean = clean.numpy()
#         num_clean = len(clean)

#         poison = poison.numpy()
#         num_poison = len(poison)

#         print(clean.shape, poison.shape)

#         X = np.concatenate( [clean, poison], axis=0)
#         y = []


#         for _ in range(num_clean):
#             y.append(0)
#         for _ in range(num_poison):
#             y.append(1)

#         self.clf.fit(X, y)

#         norm = np.linalg.norm(self.clf.coef_)
#         self.clf.coef_ = self.clf.coef_ / norm
#         self.clf.intercept_ = self.clf.intercept_ / norm

#         projection = self.clf.decision_function(X)

#         return projection[:num_clean], projection[num_clean:]

# class spectral_visualizer:

#     def fit_transform(self, clean, poison):
#         all_features = torch.cat((clean, poison), dim=0)
#         all_features -= all_features.mean(dim=0)
#         _, _, V = torch.svd(all_features, compute_uv=True, some=False)
#         vec = V[:, 0]  # the top right singular vector is the first column of V
#         vals = []
#         for j in range(all_features.shape[0]):
#             vals.append(torch.dot(all_features[j], vec).pow(2))
#         vals = torch.tensor(vals)
        
#         print(vals.shape)
        
#         return vals[:clean.shape[0]], vals[clean.shape[0]:]



def visualize(clean, poisoned, args):
    visualizer = TSNE(n_components=2)

    print(f"visualize shapes: clean: {clean.shape}, posioned: {poisoned.shape}")

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
    save_path = f'visuals/latent_{dataset}_{args.bd_model}_{args.model}_{args.train_mode}.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print("Saved figure at {}".format(save_path))

    plt.clf()