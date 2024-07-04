'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.

'''

##### Q1 do we prune weights or neurons (seems neurons)






import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.prune as prune


from .base import Base


# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask, model_type=None):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask
        self.model_type = model_type

    def forward(self, input, **kwargs): # for param consistency
        #print(self.base(input).shape,self.mask.shape,self.mask.sum() / self.mask.numel())
        if self.model_type == "Informer":
            return self.base(input)[0] * self.mask, None
        return self.base(input) * self.mask



class Pruning(Base):
    def __init__(self,
                 train_loader=None,
                 model=None,
                 prune_rate=0.2,
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
        """"
        This is for the model with the structure of Resnet
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
        ###### structured prunning -> prune filter-wise that is over channel (by taking mean over batch and temporal domain) #####
        activation = torch.mean(container, dim=[0,2])
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * prune_rate)
        mask = torch.ones(num_channels).to(self.args.device)
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 3:
            mask = mask.reshape(1, -1, 1)

        #prune.custom_from_mask(getattr(model, layer_to_prune), name="weight", mask=mask)

        setattr(model, layer_to_prune, MaskedLayer(getattr(model, layer_to_prune), mask))

        self.model = model
        print("======== pruning complete ========")


    def repair2(self,device):
        """
        This is for the model with the structure of Linear and Transformer
        """
        model = self.model.to(device)
        self.prune_rate = 0.5
        #layer_to_prune = 'model'
        if self.args.model.lower() == "timesnet":
            layer_to_prune = model.model[-1]
        elif self.args.model == "Informer":
            layer_to_prune = model.encoder.attn_layers[-1]
        else:
            raise NotImplementedError("Unknown model for FP defence.")
        
        tr_loader = self.train_loader
        prune_rate = self.prune_rate


        # prune silent activation
        print("======== pruning... ========")
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):
                container.append(output)

            hook = layer_to_prune.register_forward_hook(forward_hook)
            print("Forwarding all training set")

            model.eval()
            for i, (batch_x, label, padding_mask) in enumerate(tr_loader):
                model(batch_x.to(self.args.device),padding_mask.to(self.args.device),None,None)
            hook.remove()

        if self.args.model.lower() == "timesnet":
            container = torch.cat(container, dim=0)
        elif self.args.model == "Informer":
            container = torch.cat([container[0][0]], dim=0)

        activation = torch.mean(container, dim=[0,1])
        seq_sort = torch.argsort(activation)
        dim_model = len(activation)
        prunned_channels = int(dim_model * prune_rate)
        mask = torch.ones(dim_model).to(self.args.device)
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 3:
            mask = mask.reshape(1, 1, -1)

        #prune.custom_from_mask(getattr(model, layer_to_prune), name="weight", mask=mask)

        #setattr(layer_to_prune, 'model', MaskedLayer(layer_to_prune, mask))
        if self.args.model == "TimesNet":
            model.model[-1] = MaskedLayer(layer_to_prune, mask)
        elif self.args.model == "Informer":
            model.encoder.attn_layers[-1] = MaskedLayer(layer_to_prune, mask, model_type="Informer")

        self.model = model
        print("======== pruning complete ========")


    def get_model(self):
        return self.model

