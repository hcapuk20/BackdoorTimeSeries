'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
'''

import os
import torch
import torch.nn as nn


from .base import Base
from torch.utils.data import DataLoader


# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        print(input.shape, self.mask.shape)
        print(self.mask.sum() / self.mask.numel())
        return self.base(input) * self.mask



class Pruning(Base):
    """Pruning process.
    Args:
        train_dataset (types in support_list): forward dataset.
        test_dataset (types in support_list): testing dataset.
        model (torch.nn.Module): Network.
        layer(list): The layers to prune
        prune_rate (double): the pruning rate
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_loader=None,
                 model=None,
                 prune_rate=0.01,
                 seed=0,
                 deterministic=False,
                 args=None):
        super(Pruning, self).__init__(seed=seed, deterministic=deterministic)

        self.train_loader = train_loader
        self.model = model
        self.layer = 'layer2'
        self.prune_rate = prune_rate
        self.args = args


    def repair(self,device):
        """pruning.
        Args:
            schedule (dict): Schedule for testing.
        """


        model = self.model.to(device)
        layer_to_prune = self.layer
        tr_loader = self.train_loader
        prune_rate = self.prune_rate


        # prune silent activation
        print("======== pruning... ========")
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):
                container.append(output)

            hook = getattr(model, layer_to_prune).register_forward_hook(forward_hook)
            print("Forwarding all training set")

            model.eval()
            for i, (batch_x, label, padding_mask) in enumerate(tr_loader):
                model(batch_x.to(self.args.device),padding_mask.to(self.args.device),None,None)
            hook.remove()

        container = torch.cat(container, dim=0)
        print('con shape:', container.shape)
        activation = torch.mean(container, dim=[0,2])
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        print('act shape: ',activation.shape)
        prunned_channels = int(num_channels * prune_rate)
        mask = torch.ones(num_channels).to(self.args.device)
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 3:
            mask = mask.reshape(1, -1, 1)
        setattr(model, layer_to_prune, MaskedLayer(getattr(model, layer_to_prune), mask))

        self.model = model
        print("======== pruning complete ========")


    def get_model(self):
        return self.model