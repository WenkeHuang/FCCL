import numpy as np
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
from utils.conf import data_path
from PIL import Image
from datasets.utils.public_dataset import PublicDataset, random_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class FashionMNISTData(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.targets = self.Construct_Participant_Dataset()

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
        img, targets = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(target)
        return img, targets

    def __len__(self):
        return len(self.data)


class PublicFashionMnist(PublicDataset):
    NAME = 'pub_fmnist'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    CON_TRANSFORM = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])

    Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    def get_data_loaders(self):
        pub_aug = self.args.pub_aug
        if pub_aug == 'weak':
            selected_transform = self.Nor_TRANSFORM
        elif pub_aug == 'strong':
            selected_transform = self.CON_TRANSFORM

        # selected_transform = self.CON_TRANSFORM
        train_dataset = FashionMNISTData(data_path(), train=True,
                                         download=False, transform=selected_transform)
        traindl = random_loaders(train_dataset, self)
        return traindl

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
