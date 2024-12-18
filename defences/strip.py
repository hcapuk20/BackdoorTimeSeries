"""
Strip backdoor defense
[1] Gao, Yansong, et al. "Strip: A defence against trojan attacks on deep neural networks." Proceedings of the 35th Annual Computer Security Applications Conference. 2019.
adapted for poison detection
https://github.com/Unispac/Fight-Poison-With-Poison/blob/master/other_cleansers/strip.py
"""
import torch
import numpy as np
from tqdm import tqdm
import random

class STRIP():
    name: str = 'strip'

    def __init__(self, args, inspection_loader, clean_loader, model, strip_alpha: float = 0.5, N: int = 64, defense_fpr: float = 0.05, batch_size=128):

        self.args = args

        self.strip_alpha: float = strip_alpha
        self.N: int = N
        self.defense_fpr = defense_fpr

        self.inspection_loader = inspection_loader
        self.clean_loader = clean_loader

        self.model = model

        clean_set = []
        for i, (batch_x, _, _) in enumerate(self.clean_loader):
            clean_set.append(batch_x)
        self.clean_set = torch.cat(clean_set)

    def cleanse(self):
        # choose a decision boundary with the test set
        clean_entropy = []

        for i, (batch_x, label, padding_mask) in enumerate(self.clean_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.args.device)
            label = label.to(self.args.device)
            padding_mask = padding_mask.to(self.args.device)
            entropies = self.check(batch_x, label, padding_mask, self.clean_set)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary
        #inspection_set_loader = torch.utils.data.DataLoader(self.inspection_set, batch_size=128, shuffle=False)
        all_entropy = []
        for i, (batch_x, label, padding_mask) in enumerate(self.inspection_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.args.device)
            label = label.to(self.args.device)
            padding_mask = padding_mask.to(self.args.device)
            entropies = self.check(batch_x, label, padding_mask, self.clean_set)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)

        suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        return suspicious_indices

    def check(self, _input, _label=None, padding_mask=None, source_set=None):
        _list = []
        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:self.N]

        with torch.no_grad():
            for i in samples:
                X = source_set[i]
                X = X.to(self.args.device)
                _test = self.superimpose(_input, X)
                entropy = self.entropy(_test, padding_mask).cpu().detach()
                _list.append(entropy)

        return torch.stack(_list).mean(0)
    
    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        result = _input1 + alpha * _input2
        return result

    def entropy(self, _input, padding_mask):
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(self.model(_input, padding_mask, None, None)) + 1e-8
        return (-p * p.log()).sum(1)

def cleanser(inspection_set, clean_loader, model, args):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """
    worker = STRIP( args, inspection_set, clean_loader, model, strip_alpha=1.0, N=100, defense_fpr=0.1, batch_size=64)
    suspicious_indices = worker.cleanse()

    return suspicious_indices
        
