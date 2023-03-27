import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from timm.scheduler.cosine_lr import CosineLRScheduler
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F
from utils.util import create_if_not_exists
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FCCL.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class FCCL(FederatedModel):
    NAME = 'fccl'
    COMPATIBILITY = ['homogeneity','heterogeneity']

    def __init__(self, nets_list,args, transform):
        super(FCCL, self).__init__(nets_list,args, transform)
        self.pretrain = False
        self.load= True

        self.off_diag_weight=args.off_diag_weight

        self.public_lr = args.public_lr
        self.prev_nets_list = []
        self.intr_nets_list = []

    def ini(self):
        # Pretrain Intra Domain Pretrained Network
        self._pretrain_intra_nets()

        # Load Pretrained Nets to current Network
        self.load_pretrained_nets()

        # Copy Pre vious Networks Lists
        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))

        if self.args.structure =='homogeneity':
            self.global_net = copy.deepcopy(self.nets_list[0])
            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)
        else:
            pass

    def col_update(self, communication_idx,publoader):
        for batch_idx, (images, _) in enumerate(publoader):
            '''
            Aggregate the output from participants
            '''
            linear_output_list = []
            linear_output_target_list = []
            images = images.to(self.device)

            for _,net in enumerate (self.nets_list):
                net = net.to(self.device)
                net.train()
                linear_output  = net(images)
                linear_output_target_list.append(linear_output.clone().detach())
                linear_output_list.append(linear_output)

            '''
            Update Participants' Models via Col Loss
            '''
            for net_idx, net in enumerate(self.nets_list):
                net = net.to(self.device)
                net.train()
                optimizer = optim.Adam(net.parameters(), lr=self.public_lr)

                linear_output_target_avg_list = []
                for k in range(self.args.parti_num):
                    linear_output_target_avg_list.append(linear_output_target_list[k])

                linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)
                linear_output = linear_output_list[net_idx]
                z_1_bn = (linear_output-linear_output.mean(0))/linear_output.std(0)
                z_2_bn = (linear_output_target_avg-linear_output_target_avg.mean(0))/linear_output_target_avg.std(0)
                c = z_1_bn.T @ z_2_bn
                c.div_(len(images))

                if batch_idx == len(publoader)-3:
                    c_array = c.detach().cpu().numpy()
                    self._draw_heatmap(c_array, self.NAME,communication_idx,net_idx)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = self._off_diagonal(c).add_(1).pow_(2).sum()
                optimizer.zero_grad()
                col_loss = on_diag + self.off_diag_weight * off_diag
                if batch_idx == len(publoader)-2:
                    print('Communcation: '+str(communication_idx)+' Net: '+str(net_idx)+'Col: '+str(col_loss.item()))
                col_loss.backward()
                optimizer.step()
        return None

    def loc_update(self,priloader_list):
        for i in range(self.args.parti_num):
            self._train_net(i,self.nets_list[i],self.prev_nets_list[i],self.intr_nets_list[i],priloader_list[i])
        self.copy_nets2_prevnets()

        if self.args.structure=='homogeneity':
            self.aggregate_nets(None)
        return None

    def _train_net(self,index,net,inter_net,intra_net,train_loader):
        net = net.to(self.device)
        inter_net = inter_net.to(self.device)
        intra_net = intra_net.to(self.device)
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
                logsoft_outputs = F.log_softmax(outputs, dim=1)
                with torch.no_grad():
                    intra_soft_outpus = F.softmax(intra_net(images), dim=1)
                    inter_soft_outpus = F.softmax(inter_net(images), dim=1)
                intra_loss = criterionKL(logsoft_outputs, intra_soft_outpus)
                inter_loss = criterionKL(logsoft_outputs, inter_soft_outpus)
                loss_hard = criterionCE(outputs, labels)
                loss = loss_hard+ inter_loss+intra_loss
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

    def _pretrain_intra_nets(self):
        for j in range(self.args.parti_num):
            self.intr_nets_list.append(copy.deepcopy(self.nets_list[j]))

        if self.pretrain==True:
            for j in range(self.args.parti_num):
                self._pretrain_net(j,self.intr_nets_list[j],self.trainloaders[j],50,self.testlodaers[j])
                pretrain_path = os.path.join(self.checkpoint_path,'pretrain')
                create_if_not_exists(pretrain_path)
                save_path = os.path.join(pretrain_path,str(j)+'.ckpt')
                torch.save(self.intr_nets_list[j].state_dict(),save_path)
        else:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.intr_nets_list[j].load_state_dict(torch.load(save_path,self.device))
                self._evaluate_net(j, self.intr_nets_list[j], self.testlodaers[j])

    def _pretrain_net(self,index,net,train_loader,epoch,test_loader):
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = CosineLRScheduler(optimizer, t_initial=epoch, decay_rate=1., lr_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(epoch))
        for epoch_index in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()
            # if epoch_index %10 ==0:
            acc = self._evaluate_net(index,net,test_loader)
            scheduler.step(epoch_index)
                # if acc >80:
                #     break

    def _evaluate_net(self,index,net, test_dl):
        net = net.to(self.device)
        dl = test_dl
        status = net.training
        net.eval()
        correct, total,top1,top5 = 0.0, 0.0,0.0,0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)

        top1acc= round(100 * top1 / total,2)
        top5acc= round(100 * top5 / total,2)
        print('The '+str(index)+'participant: '+str(top1acc)+'_'+str(top5acc))
        net.train(status)
        return top1acc

    def _off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _draw_heatmap(self,data,model_name,communication_idx,net_idx):
        fig, ax = plt.subplots(figsize=(9, 9))
        sns.heatmap(
            pd.DataFrame(np.round(data, 2)),
            annot=False, vmax=1, vmin=0, xticklabels=False, yticklabels=False, cbar=False, square=False, cmap="Blues")
        model_path = os.path.join(self.checkpoint_path,model_name)
        heatmap_model_path = os.path.join(model_path,'heatmap')
        create_if_not_exists(heatmap_model_path)
        each_heatmap_model_path = os.path.join(heatmap_model_path,str(communication_idx)+'_'+str(net_idx)+'.png')
        plt.savefig(each_heatmap_model_path, bbox_inches='tight')