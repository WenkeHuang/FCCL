import torchvision.transforms as transforms
from utils.conf import data_path
from datasets.utils.federated_dataset import FederatedDataset,partition_office_domain_skew_loaders
from backbone.googlenet import GoogLeNet
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10,resnet12,resnet20,resnet18,resnet34
from backbone.efficientnet import EfficientNetB0
from backbone.mobilnet_v2 import MobileNetV2
from torchvision.datasets import ImageFolder, DatasetFolder

class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None,subset_train_num=7,subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root+'OfficeHome/'+self.data_name+'/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root+'OfficeHome/'+self.data_name+'/', self.transform, self.target_transform)

        all_data=self.imagefolder_obj.samples
        self.train_index_list=[]
        self.test_index_list=[]
        for i in range(len(all_data)):
            if i%subset_capacity<=subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list=self.train_index_list
        else:
            used_index_list=self.test_index_list

        path = self.samples[used_index_list[index]][0]
        target = self.samples[used_index_list[index]][1]
        target = int(target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaOfficeHome(FederatedDataset):
    NAME = 'fl_officehome'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['Art', 'Clipart','Product','Real World']
    percent_dict = {'Art': 0.8, 'Clipart': 0.7, 'Product': 0.8, 'Real World': 0.7}
    N_SAMPLES_PER_Class = None
    N_CLASS = 65
    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    def get_data_loaders(self, selected_domain_list):

        using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list

        nor_transform = self.Nor_TRANSFORM
        train_dataset_list = []
        test_dataset_list = []
        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), self.get_normalization_transform()])

        for _, domain in enumerate(using_list):
            train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                               transform=nor_transform)

            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                              transform=test_transform)
            test_dataset_list.append(test_dataset)

        traindls, testdls = partition_office_domain_skew_loaders(train_dataset_list, test_dataset_list, self)
        return traindls, testdls


    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaOfficeHome.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12, 'resnet18': resnet18, 'resnet20':resnet20,'resnet34': resnet34,
                     'efficient': EfficientNetB0, 'mobilnet': MobileNetV2,'googlenet':GoogLeNet}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(resnet12(FedLeaOfficeHome.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaOfficeHome.N_CLASS))
        return nets_list


    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
        return transform

