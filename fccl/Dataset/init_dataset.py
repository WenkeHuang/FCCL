from torchvision.datasets import CIFAR10,CIFAR100,MNIST,USPS,SVHN,FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import torchvision
import numpy as np
import torch

'''
Initialize Dataset
'''

class Cifar10FL(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

class Cifar100FL(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

class FashionMNISTData(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        fashionmnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = fashionmnist_dataobj.train_data, np.array(fashionmnist_dataobj.train_labels)
            else:
                data, target = fashionmnist_dataobj.test_data, np.array(fashionmnist_dataobj.test_labels)
        else:
            data = fashionmnist_dataobj.data
            target = np.array(fashionmnist_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)


class MNISTData(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = mnist_dataobj.train_data, np.array(mnist_dataobj.train_labels)
            else:
                data, target = mnist_dataobj.test_data, np.array(mnist_dataobj.test_labels)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

class USPSTData(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        usps_dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = usps_dataobj.train_data, np.array(usps_dataobj.train_labels)
            else:
                data, target = usps_dataobj.test_data, np.array(usps_dataobj.test_labels)
        else:
            data = usps_dataobj.data
            target = np.array(usps_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

class SVHNData(Dataset):
    def __init__(self,root,dataidxs=None,train='train',transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        svhn_dataobj = SVHN(root=self.root, split = self.train, transform = self.transform,
                            target_transform = self.target_transform,download= self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = svhn_dataobj.train_data, np.array(svhn_dataobj.train_labels)
            else:
                data, target = svhn_dataobj.test_data, np.array(svhn_dataobj.test_labels)
        else:
            data = svhn_dataobj.data
            target = np.array(svhn_dataobj.labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # img: ndarray:(3,32,32)
        # transforms.ToPILImage()：ndarray的数据类型要求dtype=uint8, range[0, 255] and shape H x W x C
        if self.transform is not None:
            img = np.transpose(img, (1, 2, 0))
            # 将npimg的数据格式由（channels,size,size）转化为（size,size,channels）,进行格式的转换后方可进行显示。
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

