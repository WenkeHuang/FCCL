import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch

#  https://github.com/JYWa/FedNova


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProx.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class FedProx(FederatedModel):
    NAME = 'fedprox'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedProx, self).__init__(nets_list,args, transform)


    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        self.aggregate_nets(None)

        for i in range(self.args.parti_num):
            self._train_net(i,self.nets_list[i], priloader_list[i])
        return None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.global_net = self.global_net.to(self.device)
        global_weight_collector = list(self.global_net.parameters())
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

