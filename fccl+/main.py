import os
import sys
import socket
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset, get_public_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=7, help='The Device Id for Experiment')
    parser.add_argument('--parti_num', type=int, default=0, help='The Number for Participants')  # Domain 4 Label 10
    parser.add_argument('--model', type=str, default='fedrs',  # fccl,fcclss, fedmd fedavg fedprox feddf moon fedmatch fcclplus fcclsuper
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--structure', type=str, default='homogeneity')  # 'homogeneity' heterogeneity

    parser.add_argument('--dataset', type=str, default='fl_digits',  # fl_digits, fl_officehome fl_office31,fl_officecaltech
                        choices=DATASET_NAMES, help='Which scenario to perform experiments on.')
    parser.add_argument('--beta', type=int, default=0.1, help='The Beta for Label Skew')
    parser.add_argument('--public_dataset', type=str, default='pub_cifar100')  # pub_cifar100 pub_tyimagenet pub_fmnist pub_market1501
    parser.add_argument('--public_len', type=int, default=5000)
    parser.add_argument('--pub_aug', type=str, default='weak')  # weak strong

    parser.add_argument('--get_time', action='store_true')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    if args.beta in best:
        best = best[args.beta]
    else:
        best = best[-1]
    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    if args.parti_num == 0:
        if args.dataset in ['fl_cifar10']:
            args.parti_num = 10
        if args.dataset in ['fl_digits', 'fl_officehome', 'fl_officecaltech']:
            args.parti_num = 4
        if args.dataset in ['fl_office31']:
            args.parti_num = 3

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    publ_dataset = get_public_dataset(args)

    if args.structure == 'homogeneity':
        backbones_list = priv_dataset.get_backbone(args.parti_num, None)

    elif args.structure == 'heterogeneity':
        if priv_dataset.NAME in ['fl_digits']:
            backbones_names_list = ['resnet10', 'resnet12', 'efficient', 'mobilnet']
        elif priv_dataset.NAME in ['fl_officecaltech']:
            backbones_names_list = ['googlenet', 'resnet12', 'resnet10', 'resnet12']
        elif priv_dataset.NAME in ['fl_officehome']:
            backbones_names_list = ['resnet18', 'resnet34', 'googlenet', 'resnet12']
        elif priv_dataset.NAME in ['fl_office31']:
            backbones_names_list = ['resnet10', 'resnet12', 'resnet10']
        selected_backbones_list = []
        for i in range(args.parti_num):
            index = i % len(backbones_names_list)
            selected_backbones_list.append(backbones_names_list[index])
        backbones_list = priv_dataset.get_backbone(args.parti_num, selected_backbones_list)

    model = get_model(backbones_list, args, priv_dataset.get_transform())

    if args.structure not in model.COMPATIBILITY:
        print(model.NAME + ' does not support model heterogeneity')
    else:
        print('{}_{}_{}_{}_{}_{}'.format(args.model, args.dataset, args.communication_epoch, args.public_dataset, args.public_len, args.pub_aug))
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.dataset, args.communication_epoch))
        train(model, publ_dataset, priv_dataset, args)


if __name__ == '__main__':
    main()
