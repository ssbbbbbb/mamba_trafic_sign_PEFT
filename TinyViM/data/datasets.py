'''
Build trainining/testing datasets
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class NumericImageFolder(ImageFolder):
    """
    自訂版 ImageFolder：
    - 假設子資料夾名稱是整數字串（例如 '0','1',...,'42'）
    - 以整數大小排序，而不是預設的字串排序

    這樣可以保證：
    - 資料夾 '0' 對應 label 0
    - 資料夾 '1' 對應 label 1
    - ...
    - 資料夾 '42' 對應 label 42
    """

    def find_classes(self, directory: str):
        # 參考 torchvision.datasets.folder.ImageFolder.find_classes 實作
        # 只是這裡改成用 int() 來排序資料夾名稱
        classes = [
            d.name for d in os.scandir(directory) if d.is_dir()
        ]
        # 嘗試將資料夾名稱轉成 int，並以數字大小排序
        try:
            classes_int = sorted([int(c) for c in classes])
            classes = [str(c) for c in classes_int]
        except ValueError:
            # 若有無法轉成整數的資料夾名稱，就退回原本的字串排序
            classes = sorted(classes)

        if not classes:
            raise FileNotFoundError(f"找不到任何子資料夾於: {directory}")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNETEE':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CUSTOM':
        # 自訂資料集：
        #   Train: data/Train/0..42
        #   Val  : data/Val/0..42 （若不存在就退回使用 Train）
        if is_train:
            root = os.path.join(args.data_path, 'Train')
        else:
            val_root = os.path.join(args.data_path, 'Val')
            root = val_root if os.path.exists(val_root) else os.path.join(args.data_path, 'Train')

        # 使用 NumericImageFolder，確保資料夾 '0'..'42' 映射到 label 0..42
        dataset = NumericImageFolder(root, transform=transform)
        nb_classes = len(dataset.classes)
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
