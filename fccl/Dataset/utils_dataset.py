import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import sys
sys.path.append('../../../')
from fccl.Dataset.init_dataset import Cifar10FL,Cifar100FL,FashionMNISTData,MNISTData,USPSTData,SVHNData
from fccl.Idea.params import args_parser

args = args_parser()
Project_Path = args.Project_Dir
Dataset_Dir = args.Dataset_Dir

def init_logs(log_level=logging.INFO,log_path = Project_Path+'Logs/',sub_name=None):
    # logging：https://www.cnblogs.com/CJOKER/p/8295272.html
    # 第一步，创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name == None:
        log_name = log_path + filename + '.log'
    else:
        log_name = log_path + filename + '_' + sub_name +'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console  = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(console)
    # 日志
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = Cifar10FL(datadir, train=True, download=False, transform=transform)
    cifar10_test_ds = Cifar10FL(datadir, train=False, download=False, transform=transform)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = Cifar100FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_train_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_fashionmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    fashionmnist_train_ds = FashionMNISTData(datadir, train=True, download=False, transform=transform)
    fashionmnist_test_ds = FashionMNISTData(datadir, train=False, download=False, transform=transform)
    X_train, y_train = fashionmnist_train_ds.data, fashionmnist_train_ds.target
    X_test, y_test = fashionmnist_test_ds.data, fashionmnist_test_ds.target
    return (X_train, y_train, X_test, y_test)


def generate_public_data_idxs(dataset,datadir,size,epoch=None):
    if dataset =='cifar_100':
        _, y_train, _, _ = load_cifar100_data(datadir)
        n_train = y_train.shape[0]
    if dataset =='tiny_imagenet':
        train_ds = datasets.ImageFolder(datadir+'/train', transform=None)
        n_train = len(train_ds)
    if dataset =='FashionMNIST':
        _, y_train, _, _ = load_fashionmnist_data(datadir)
        n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train) # 打乱顺序
    if epoch == None:
        idxs = idxs[0:size] # 获取前size个
        return idxs
    else:
        idx_list = []
        for epoch_index in range(epoch):
            idx_list.append(np.random.choice(idxs, size, replace=False))
        return idx_list

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,num_workers=4):
    if dataset in ('cifar_10', 'cifar_100'):
        if dataset == 'cifar_10':
            dl_obj = Cifar10FL
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        if dataset =='cifar_100':
            dl_obj=Cifar100FL
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, drop_last=True, batch_size=train_bs, shuffle=True,num_workers=num_workers)
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=num_workers)
        return train_dl, test_dl, train_ds, test_ds
    if dataset == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
            train_ds = torch.utils.data.Subset(train_ds, dataidxs)
        test_ds = datasets.ImageFolder(datadir+'/test', transform=transform_test)
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, drop_last=True, batch_size=train_bs, shuffle=True,num_workers=num_workers)
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=num_workers)
        return train_dl, test_dl, train_ds, test_ds
    if dataset =='FashionMNIST':
        dl_obj = FashionMNISTData
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    if dataset == 'syn':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(Dataset_Dir + 'syn/imgs_train', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(Dataset_Dir + 'syn/imgs_train', transform=transform_train)
            train_ds = torch.utils.data.Subset(train_ds, dataidxs)
        test_ds = datasets.ImageFolder(Dataset_Dir + 'syn/imgs_valid', transform=transform_test)
    if dataset  == 'mnist':
        dl_obj = MNISTData
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    if dataset == 'usps':
        dl_obj = USPSTData
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    if dataset =='svhn':
        dl_obj = SVHNData
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train='train', transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train='test', transform=transform_test, download=True)
    if dataset in ('amazon', 'caltech','dslr','webcam'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_traintest = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        ds = datasets.ImageFolder(Dataset_Dir + dataset+'/', transform=transform_traintest)
        dataset_len = len(ds)
        full_idx = range(dataset_len)
        test_idx = []
        for class_index in range(len(ds.classes)):
            target_class_list = [index for (index,value) in enumerate(ds.targets) if value == class_index]
            target_class_test_length = int(0.3*len(target_class_list))
            test_idx.extend(np.random.permutation(target_class_list)[0:target_class_test_length])
        if dataidxs is None:
            train_idx = list(set(full_idx).difference(set(test_idx)))
        else:
            train_idx = list(set(full_idx).difference(set(test_idx)))[0:len(dataidxs)]
        train_ds = torch.utils.data.Subset(ds,train_idx)
        test_ds = torch.utils.data.Subset(ds,full_idx)
    if dataset in ('Art', 'Clipart','Product','Real World'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_traintest = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        ds = datasets.ImageFolder(Dataset_Dir + dataset+'/', transform=transform_traintest)
        dataset_len = len(ds)
        full_idx = range(dataset_len)
        test_idx = []
        for class_index in range(len(ds.classes)):
            target_class_list = [index for (index,value) in enumerate(ds.targets) if value == class_index]
            target_class_test_length = int(0.3*len(target_class_list))
            test_idx.extend(np.random.permutation(target_class_list)[0:target_class_test_length])
        if dataidxs is None:
            train_idx = list(set(full_idx).difference(set(test_idx)))
        else:
            train_idx = list(set(full_idx).difference(set(test_idx)))[0:len(dataidxs)]
        train_ds = torch.utils.data.Subset(ds,train_idx)
        test_ds = torch.utils.data.Subset(ds,full_idx)
    train_dl = torch.utils.data.DataLoader(dataset=train_ds,batch_size=train_bs,shuffle=True,num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=num_workers)
    return train_dl, test_dl, train_ds, test_ds

