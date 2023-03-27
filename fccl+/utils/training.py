import datetime

import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from datasets.utils.public_dataset import PublicDataset
from typing import Tuple
from torch.utils.data import DataLoader
import sys
import numpy as np
from utils.logger import CsvWriter
from utils.util import save_networks


def evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    intra_accs = []
    inter_accs = []
    test_len = len(test_dl)
    for i in range(model.args.parti_num):
        if setting == 'domain_skew':
            dl = test_dl[i % test_len]
        else:
            dl = test_dl
        net = model.nets_list[i]
        net = net.to(model.device)
        status = net.training
        net.eval()
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        if name in ['fl_digits', 'fl_officecaltech']:
            intra_accs.append(top1acc)
        elif name in ['fl_office31', 'fl_officehome']:
            intra_accs.append(top5acc)
        net.train(status)

    if setting == 'domain_skew':
        for i in range(model.args.parti_num):
            inter_net_accs = []
            net = model.nets_list[i]
            status = net.training
            net.eval()
            for j, dl in enumerate(test_dl):
                if i % test_len != j:
                    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
                    for batch_idx, (images, labels) in enumerate(dl):
                        with torch.no_grad():
                            images, labels = images.to(model.device), labels.to(model.device)
                            outputs = net(images)
                            _, max5 = torch.topk(outputs, 5, dim=-1)
                            labels = labels.view(-1, 1)
                            top1 += (labels == max5[:, 0:1]).sum().item()
                            top5 += (labels == max5).sum().item()
                            total += labels.size(0)
                    top1acc = round(100 * top1 / total, 2)
                    top5acc = round(100 * top5 / total, 2)
                    if name in ['fl_digits', 'fl_officecaltech']:
                        inter_net_accs.append(top1acc)
                    elif name in ['fl_office31', 'fl_officehome']:
                        inter_net_accs.append(top5acc)
            inter_accs.append(np.mean(inter_net_accs))
            net.train(status)
    elif setting == 'label_skew':
        inter_accs = intra_accs
    return intra_accs, inter_accs


def train(model: FederatedModel, public_dataset: PublicDataset, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    print(file=sys.stderr)
    if hasattr(args, 'public_batch_size'):
        pub_loader = public_dataset.get_data_loaders()

    if args.structure == 'homogeneity':
        pri_train_loaders, test_loaders = private_dataset.get_data_loaders()
    elif args.structure == 'heterogeneity':
        selected_domain_list = []
        domains_list = private_dataset.DOMAINS_LIST
        domains_len = len(domains_list)
        for i in range(args.parti_num):
            index = i % domains_len
            selected_domain_list.append(domains_list[index])
        pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)

    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders

    if hasattr(model, 'ini'):
        model.ini()

    intra_accs_dict = {}
    inter_accs_dict = {}
    mean_intra_acc_list = []
    mean_inter_acc_list = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):

        model.epoch_index = epoch_index

        if epoch_index == 1 and args.get_time:
            start_time = datetime.datetime.now()

            if hasattr(args, 'public_batch_size'):
                model.col_update(epoch_index, pub_loader)
                model.public_lr = args.public_lr * (1 - epoch_index / Epoch * 0.9)

            if hasattr(model, 'loc_update'):
                model.loc_update(pri_train_loaders)

                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

            end_time = datetime.datetime.now()
            use_time = end_time - start_time
            print(end_time - start_time)
            with open(args.dataset + '_'+args.structure + '_time.csv', 'a') as f:
                f.write(args.model + ',' + str(use_time) + '\n')

            return

        else:
            if hasattr(args, 'public_batch_size'):
                model.col_update(epoch_index, pub_loader)
                model.public_lr = args.public_lr * (1 - epoch_index / Epoch * 0.9)

            if hasattr(model, 'loc_update'):
                model.loc_update(pri_train_loaders)

                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

        intra_accs, inter_accs = evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)

        mean_intra_acc = round(np.mean(intra_accs, axis=0), 3)
        mean_inter_acc = round(np.mean(inter_accs, axis=0), 3)
        mean_intra_acc_list.append(mean_intra_acc)
        mean_inter_acc_list.append(mean_inter_acc)

        if private_dataset.SETTING == 'domain_skew':
            print('The ' + str(epoch_index) + ' Communcation Accuracy:' + 'Intra: ' + str(mean_intra_acc) + ' Inter: ' + str(mean_inter_acc))
        else:
            print('The ' + str(epoch_index) + ' Communcation Accuracy:' + str(mean_intra_acc))

        for i in range(len(intra_accs)):
            if i in intra_accs_dict:
                intra_accs_dict[i].append(intra_accs[i])
            else:
                intra_accs_dict[i] = [intra_accs[i]]

        for i in range(len(inter_accs)):
            if i in inter_accs_dict:
                inter_accs_dict[i].append(inter_accs[i])
            else:
                inter_accs_dict[i] = [inter_accs[i]]

    if args.csv_log:

        csv_writer.write_acc(intra_accs_dict, inter_accs_dict, mean_intra_acc_list, mean_inter_acc_list)
