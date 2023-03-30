import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists


# https://github.com/epfml/federated-learning-public-code/tree/master/codes/FedDF-code

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedDF.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDF(FederatedModel):
    NAME = 'feddf'
    COMPATIBILITY = ['homogeneity', 'heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDF, self).__init__(nets_list, args, transform)
        self.public_lr = args.public_lr
        self.load = False

    def ini(self):
        self.load_pretrained_nets()

        if self.args.structure == 'homogeneity':
            self.global_net = copy.deepcopy(self.nets_list[0])
            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)
        else:
            pass

    def col_update(self, communication_idx, publoader):
        for _, (images, _) in enumerate(publoader):
            '''
            Aggregate the output from participants
            '''
            # outputs_list = []
            targets_list = []
            images = images.to(self.device)
            for _, net in enumerate(self.nets_list):
                net = net.to(self.device)
                net.train()
                outputs = net(images)
                target = outputs.clone().detach()
                # outputs_list.append(outputs)
                targets_list.append(target)

            target = torch.mean(torch.stack(targets_list), 0)
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
            criterion.to(self.device)
            for net_idx, net in enumerate(self.nets_list):
                optimizer = optim.SGD(net.parameters(), lr=self.public_lr, weight_decay=1e-5)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(F.softmax(outputs, dim=1).log(), F.softmax(target, dim=1))
                loss.backward()
                optimizer.step()
        return None

    def loc_update(self, priloader_list):
        if self.args.structure == 'homogeneity':
            self.aggregate_nets(None)
        for i in range(self.args.parti_num):
            self._train_net(i, self.nets_list[i], priloader_list[i])
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
