# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
import os

import torch
from sklearn.utils import class_weight
from timm.data import Mixup
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torch.utils.data import WeightedRandomSampler, RandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


def build_loader(config):
    config.defrost()
    dataset_train, _ = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"Successfully build train dataset in non-distributed mode.")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"Successfully build val dataset in non-distributed mode.")

    sampler_train = build_sampler(config, dataset_train, is_train=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        # pin_memory=config.DATA.PIN_MEMORY,
        # drop_last=True,
    )

    sampler_val = build_sampler(config, dataset_val, is_train=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        # pin_memory=config.DATA.PIN_MEMORY,
        # drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_sampler(config, dataset, is_train):
    # sampling ratio is defined out scope for usage in train.
    num_samples = int(math.ceil(len(dataset) * (config.SAMPLING_RATIO or 1)))

    # 创建可调整权重的采样器
    if is_train:
        targets = dataset.targets  # 获取样本标签列表
        weights = class_weight.compute_sample_weight("balanced", targets)
        class_weights = torch.from_numpy(weights)

        _sampler = WeightedRandomSampler(weights=class_weights, num_samples=num_samples, replacement=True)
    else:
        _sampler = RandomSampler(data_source=dataset, num_samples=num_samples, replacement=True)

    return _sampler


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    subdir = 'train' if is_train else 'val'
    image_dataset = CustomImageFolder(os.path.join(config.DATA.DATA_PATH, subdir), transform)

    nb_classes = len(image_dataset.classes)

    return image_dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.TRAIN_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(CustomImageFolder, self).__init__(root, transform, target_transform, loader)

        # 重新生成类别到索引的映射
        self.classes, self.class_to_idx = self._find_classes(self.root)

    def _find_classes(self, dir):
        class_to_idx = {d.split('.', 1)[1] : int(d.split('.')[0]) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))}
        classes = [d.split('.', 1)[1] for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()

        return classes, class_to_idx
