import copy

import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os

class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None

    def __init__(self,nets_list:list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None

        self.checkpoint_path = checkpoint_path()+self.args.dataset+'/'+self.args.structure+'/'
        create_if_not_exists(self.checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self,communication_idx,publoader):
        pass

    def loc_update(self,priloader_list):
        pass

    def aggregate_nets(self, freq=None):
        parti_num= self.args.parti_num
        global_net = self.global_net
        nets_list = self.nets_list

        global_w = global_net.state_dict()
        if freq == None:
            freq = [1 / parti_num for _ in range(parti_num)]
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * freq[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[net_id]
        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    def update_global_net(self, freq=None):
        parti_num= self.args.parti_num
        global_net = self.global_net
        nets_list = self.nets_list

        global_w = global_net.state_dict()
        if freq == None:
            freq = [1 / parti_num for _ in range(parti_num)]
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * freq[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[net_id]
        global_net.load_state_dict(global_w)

    def special_aggreNets(self,freq=None):
        parti_num= self.args.parti_num
        nets_list = self.nets_list

        global_w = copy.deepcopy(nets_list[0].state_dict())
        if freq == None:
            freq = [1 / parti_num for _ in range(parti_num)]
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * freq[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[net_id]

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_w)

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path,self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)