#%%
import pandas as pd
import numpy as np
import random
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from ing_theme_matplotlib import mpl_style # pip install ing_theme_matplotlib 
import matplotlib as mpl
import tqdm
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
#%%
def load_config(config, config_path):
    with open(config_path, 'r') as config_file:
        args = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in config.keys():
        if key in args.keys():
            config[key] = args[key]
    return config
#%%
def stock_data_generator(df, C, tau):
    n = df.shape[0] - C - tau
        
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = df.iloc[i : i+C, :]
        infer_data[i, :, :] = df.iloc[i+C : i+C+tau, :]
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    return input_data, infer_data
#%%
def build_datasets(df, test_len, config):
    train_list = []
    test_list = []
    
    ### validation
    train_idx_last = len(df) - test_len
    test_idx_last = len(df) 
    train = df.iloc[: train_idx_last]
    test = df.iloc[-(test_len + config["timesteps"] + config["future"]) : test_idx_last]
    
    train_context, train_target = stock_data_generator(train, config["timesteps"], config["future"])
    test_context, test_target = stock_data_generator(test, config["timesteps"], config["future"])
    
    assert train_context.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert train_target.shape == (train.shape[0] - config["timesteps"] - config["future"], config["future"], df.shape[1])
    assert test_context.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert test_target.shape == (test.shape[0] - config["timesteps"] - config["future"], config["future"], df.shape[1])
    
    train_list.append((train_context, train_target))
    test_list.append((test_context, test_target))
    
    ### online
    train_idx_last = len(df)
    train = df
    train_context, train_target = stock_data_generator(train, config["timesteps"], config["future"])
    
    assert train_context.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert train_target.shape == (train.shape[0] - config["timesteps"] - config["future"], config["future"], df.shape[1])
    
    train_list.append((train_context, train_target))
    test_list.append((None, None))
    
    return train_list, test_list
#%%
def visualize_quantile(target_, estQ, colnames, forecasts, show=False):
    recent = 100
    mpl.rcParams["figure.dpi"] = 200
    mpl_style(dark=False)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    SMALL_SIZE = 28
    BIGGER_SIZE = 32
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(4, 2, figsize=(30, 30))
    for j in range(estQ[1].size(1)):
        ax.flatten()[j].plot(
            np.arange(recent), target_[-recent:, j],
            label=f"{colnames[j]}", color="black", linewidth=4, linestyle='--')
        ax.flatten()[j].plot(
            np.arange(recent), estQ[1][-recent:, j],
            label="Median", color='green', linewidth=4)
        ax.flatten()[j].fill_between(
            np.arange(recent), 
            estQ[0][-recent:, j], 
            estQ[2][-recent:, j], 
            color=cols[3], alpha=0.5, label=r'80% interval')
        ### online
        ax.flatten()[j].scatter(
            np.arange(recent, recent+1), forecasts[1][:, j],
            color='red', s=500, marker='x', linewidth=6)
        ax.flatten()[j].scatter(
            np.arange(recent, recent+1), forecasts[0][:, j],
            color=cols[0], s=500, marker='^', linewidth=4)
        ax.flatten()[j].scatter(
            np.arange(recent, recent+1), forecasts[2][:, j],
            color=cols[0], s=500, marker='v', linewidth=4)
        ax.flatten()[j].set_ylabel(f"{colnames[j]}")
        ax.flatten()[j].set_xlabel("days")
        ax.flatten()[j].legend(loc="upper left")
    ax[-1, -1].axis('off')
    plt.tight_layout()
    plt.savefig(f"./assets/fig/result.png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return fig
#%%