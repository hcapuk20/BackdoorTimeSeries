import torch
from copy import deepcopy
def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, deepcopy(a.detach())), 0)
    return grad_flattened

def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, deepcopy(a.detach())), 0)
    return model_flattened

def unflat_model(model, model_flattened):
    i = 0
    for p in model.parameters():
        temp = model_flattened[i:i+p.data.numel()]
        p.data = temp.reshape(p.data.size())
        i += p.data.numel()
    return None

def unflat_grad(model, grad_flattened):
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            temp = grad_flattened[i:i+p.grad.data.numel()]
            p.grad.data = temp.reshape(p.data.size())
            i += p.data.numel()
    return None

def pull_model(model_toUpdate, model_reference):
    with torch.no_grad():
        for param_user, param_server in zip(model_toUpdate.parameters(), model_reference.parameters()):
            param_user.data = param_server.data[:] + 0
    return None