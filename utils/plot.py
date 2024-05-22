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
    sqrt_b = int(math.sqrt(M))
    max_variates = sqrt_b **2

    # Create sqrt(b) x sqrt(b) subplots
    fig, axs = plt.subplots(sqrt_b, sqrt_b)

    # Flatten the axs array for easy iteration
    try:
        axs = axs.flatten()
    except:
        axs = axs

    # Plot each time series in a separate subplot
    for i in range(max_variates):
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