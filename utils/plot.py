import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os

def plot_time_series(args,data, bd):

    os.makedirs('trigger_figs', exist_ok=True)
    # If the data is a torch tensor, convert it to numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        bd = bd.detach().cpu().numpy()

    # Get the number of time series (b) and the length of each series (T)
    M,T = data.shape
    sqrt_b = int(math.ceil(math.sqrt(M)))
    sqrt_b = min(sqrt_b, 1)
    max_variates = sqrt_b **2
    # Create sqrt(b) x sqrt(b) subplots
    fig, axs = plt.subplots(sqrt_b, sqrt_b, squeeze=False)

    # Flatten the axs array for easy iteration
    try:
        axs = axs.flatten()
    except:
        axs = axs

    # Plot each time series in a separate subplot
    for i in range(min(max_variates, M)): # if allowed square is bigger than the number of variates, stop early.
        axs[i].plot(data[i], label='Original')
        axs[i].plot(bd[i], label='Backdoor')

    # Show the plot
    plt.tight_layout()
    desig = args.sim_id
    plt.legend()
    dataset = args.root_path.split('/')[-2]
    m = args.bd_model
    plt.savefig('trigger_figs/trigger_plot_{}-T_{}-{}-{}.png'.format(dataset,m,desig,random.randint(0,100)))
    #plt.show()


def visualize_cls(dataloader,t_model,args,max_variates=3):
    labels = []
    samples = []
    bd_samples = []
    for i, (batch_x, label, padding_mask) in enumerate(dataloader):

        M, T = batch_x[0].shape
        batch_x = batch_x.to(args.device)
        padding_mask = padding_mask.to(args.device)
        label = label.to(args.device)
        target_labels = torch.ones_like(label) * args.target_label
        trigger_x, trigger_clipped = t_model(batch_x, padding_mask, None, None, target_labels)
        bd_batch = batch_x + trigger_clipped
        for i,l in enumerate(label):
            if l not in labels:
                samples.append(batch_x[i].permute(1,0))
                bd_samples.append(bd_batch[i].permute(1,0))
                labels.append(l)

    for x,x_bd,l in zip(samples,bd_samples,labels):
        fig, axs = plt.subplots(1, 3, squeeze=False)

        # Flatten the axs array for easy iteration
        try:
            axs = axs.flatten()
        except:
            axs = axs
        for i in range(min(max_variates, M)):  # if allowed square is bigger than the number of variates, stop early.
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
                x_bd = x_bd.detach().cpu().numpy()
            axs[i].plot(x[i], label='Original')
            axs[i].plot(x_bd[i], label='Backdoor')
        plt.title('Label: {}'.format(l.item()))
        plt.legend()
        plt.tight_layout()
        fig.set_size_inches(14, 4)
        desig = args.sim_id
        dataset = args.root_path.split('/')[-2]
        m = args.bd_model
        plt.savefig('trigger_figs2/trigger_plot_{}-L_{}-T_{}-{}-{}.png'.format(dataset,l.item(), m, desig, random.randint(0, 100)))
