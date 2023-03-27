import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def save_networks(model,communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME
    save_option = True

    if save_option:
        checkpoint_path = model.checkpoint_path
        model_path = os.path.join(checkpoint_path, model_name)
        model_para_path = os.path.join(model_path, 'para')
        create_if_not_exists(model_para_path)
        for net_idx,network in enumerate(nets_list):
            each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
            torch.save(network.state_dict(),each_network_path)
