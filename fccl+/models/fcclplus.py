import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from utils.util import create_if_not_exists


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FCCLPlus.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FCCLPLUS(FederatedModel):
    NAME = 'fcclplus'
    COMPATIBILITY = ['homogeneity', 'heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FCCLPLUS, self).__init__(nets_list, args, transform)

        self.public_lr = args.public_lr
        self.public_epoch = args.public_epoch
        self.prev_nets_list = []

    def ini(self):

        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))

        if self.args.structure == 'homogeneity':
            self.global_net = copy.deepcopy(self.nets_list[0])
            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)
        else:
            pass

    def col_update(self, communication_idx, publoader):
        epoch_loss_dict = {}
        for pub_epoch_idx in range(self.public_epoch):
            for batch_idx, (images, _) in enumerate(publoader):
                batch_loss_dict = {}

                linear_output_list = []
                linear_output_target_list = []
                logitis_sim_list = []
                logits_sim_target_list = []
                images = images.to(self.device)

                for _, net in enumerate(self.nets_list):
                    net = net.to(self.device)
                    net.train()
                    linear_output = net(images)
                    linear_output_target_list.append(linear_output.clone().detach())
                    linear_output_list.append(linear_output)
                    features = net.features(images)
                    features = F.normalize(features, dim=1)
                    logits_sim = self._calculate_isd_sim(features)
                    logits_sim_target_list.append(logits_sim.clone().detach())
                    logitis_sim_list.append(logits_sim)

                for net_idx, net in enumerate(self.nets_list):
                    '''
                    FCCL Loss for overall Network
                    '''
                    optimizer = optim.Adam(net.parameters(), lr=self.public_lr)

                    linear_output = linear_output_list[net_idx]
                    linear_output_target_avg_list = []
                    for k in range(self.args.parti_num):
                        linear_output_target_avg_list.append(linear_output_target_list[k])
                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)

                    z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)
                    z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)
                    c = z_1_bn.T @ z_2_bn
                    c.div_(len(images))

                    # if batch_idx == len(publoader) - 3:
                    #     c_array = c.detach().cpu().numpy()
                    #     self._draw_heatmap(c_array, self.NAME, communication_idx, net_idx)

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self._off_diagonal(c).add_(1).pow_(2).sum()
                    fccl_loss = on_diag + 0.0051 * off_diag

                    logits_sim = logitis_sim_list[net_idx]
                    logits_sim_target_avg_list = []
                    for k in range(self.args.parti_num):
                        logits_sim_target_avg_list.append(logits_sim_target_list[k])
                    logits_sim_target_avg = torch.mean(torch.stack(logits_sim_target_avg_list), 0)

                    inputs = F.log_softmax(logits_sim, dim=1)
                    targets = F.softmax(logits_sim_target_avg, dim=1)
                    loss_distill = F.kl_div(inputs, targets, reduction='batchmean')
                    loss_distill = self.args.dis_power * loss_distill

                    optimizer.zero_grad()
                    col_loss = fccl_loss + loss_distill
                    batch_loss_dict[net_idx] = {'fccl': round(fccl_loss.item(), 3), 'distill': round(loss_distill.item(), 3)}

                    if batch_idx == len(publoader) - 2:
                        print('Communcation: ' + str(communication_idx) + 'Net: ' + str(net_idx) + 'FCCL: ' + str(round(fccl_loss.item(), 3)) + 'Disti: ' + str(
                            round(loss_distill.item(), 3)))
                    col_loss.backward()
                    optimizer.step()
                epoch_loss_dict[batch_idx] = batch_loss_dict
        return None

    def _calculate_isd_sim(self, features):
        sim_q = torch.mm(features, features.T)
        logits_mask = torch.scatter(
            torch.ones_like(sim_q),
            1,
            torch.arange(sim_q.size(0)).view(-1, 1).to(self.device),
            0
        )
        row_size = sim_q.size(0)
        sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
        return sim_q / self.args.temp

    def loc_update(self, priloader_list):
        if self.args.structure == 'homogeneity':
            self.aggregate_nets(None)

        for i in range(self.args.parti_num):
            self._train_net(i, self.nets_list[i], self.prev_nets_list[i], priloader_list[i])
        self.copy_nets2_prevnets()

        return None

    def _train_net(self, index, net, inter_net, train_loader):
        T = self.args.local_dis_power

        net = net.to(self.device)
        inter_net = inter_net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)

        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                bs, class_num = outputs.shape
                soft_outputs = F.softmax(outputs / T, dim=1)
                non_targets_mask = torch.ones(bs, class_num).to(self.device).scatter_(1, labels.view(-1, 1), 0)
                non_target_soft_outputs = soft_outputs[non_targets_mask.bool()].view(bs, class_num - 1)

                non_target_logsoft_outputs = torch.log(non_target_soft_outputs)

                with torch.no_grad():
                    inter_outputs = inter_net(images)
                    soft_inter_outpus = F.softmax(inter_outputs / T, dim=1)
                    non_target_soft_inter_outputs = soft_inter_outpus[non_targets_mask.bool()].view(bs, class_num - 1)

                inter_loss = criterionKL(non_target_logsoft_outputs, non_target_soft_inter_outputs)
                loss_hard = criterionCE(outputs, labels)
                inter_loss = inter_loss * (T ** 2)
                loss = loss_hard + inter_loss
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d lossCE = %0.3f lossKD = %0.3f" % (index, loss_hard.item(), inter_loss.item())
                optimizer.step()

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _draw_heatmap(self, data, model_name, communication_idx, net_idx):
        fig, ax = plt.subplots(figsize=(9, 9))
        sns.heatmap(
            pd.DataFrame(np.round(data, 2)),
            annot=False, vmax=1, vmin=0, xticklabels=False, yticklabels=False, cbar=False, square=False, cmap="Blues")
        model_path = os.path.join(self.checkpoint_path, model_name)
        heatmap_model_path = os.path.join(model_path, 'heatmap')
        create_if_not_exists(heatmap_model_path)
        each_heatmap_model_path = os.path.join(heatmap_model_path, str(communication_idx) + '_' + str(net_idx) + '.png')
        plt.savefig(each_heatmap_model_path, bbox_inches='tight')
