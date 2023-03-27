import torch
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import glob
import re
import torchvision.transforms as transforms
import os.path as osp
from datasets.utils.public_dataset import PublicDataset, random_loaders
from datasets.transforms.denormalization import DeNormalize


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img,pid


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset


class PublicMarket1501(PublicDataset):
    NAME = 'pub_market1501'
    data_path = '/data0/data_jd/datasets/reid/'
    CON_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([
             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
         ], p=0.8),
         transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.4145, 0.3887, 0.3840),
                              (0.2172, 0.2095, 0.2088))])

    Nor_TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4145, 0.3887, 0.3840),
                                 (0.2172, 0.2095, 0.2088))])

    def get_data_loaders(self):

        selected_transform = self.CON_TRANSFORM
        train_dataset = ImageDataset(Market1501(self.data_path).train, selected_transform)

        traindl = random_loaders(train_dataset, self)
        return traindl

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4145, 0.3887, 0.3840),
                                         (0.2172, 0.2095, 0.2088))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4145, 0.3887, 0.3840),
                                (0.2172, 0.2095, 0.2088))
        return transform

# if __name__ == '__main__':
#     ds=Market1501('/data0/data_jd/datasets/reid/').train
#
#     CON_TRANSFORM = transforms.Compose(
#         [transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.RandomApply([
#              transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#          ], p=0.8),
#          transforms.RandomGrayscale(p=0.2),
#          transforms.ToTensor(),
#          transforms.Normalize((0.4145, 0.3887, 0.3840),
#                               (0.2172, 0.2095, 0.2088))])
#     # tensor([0.4145, 0.3887, 0.3840])
#     # tensor([0.2172, 0.2095, 0.2088])
#     Nor_TRANSFORM = transforms.Compose(
#         [
#         transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#             transforms.Normalize((0.4145, 0.3887, 0.3840),
#                                  (0.2172, 0.2095, 0.2088))])
#     d=ImageDataset(ds,transforms.Compose([transforms.ToTensor()]))
#     rgb=[]
#     for i in range(len(d)):
#         img=d.__getitem__(i).view(3,-1)
#         rgb.append(img)
#     rgb=torch.cat(rgb,dim=1)
#     torch.mean(rgb,dim=1)
#     print()
