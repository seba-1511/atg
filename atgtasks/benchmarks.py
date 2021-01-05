#!/usr/bin/env python3

import os
import random
import torch
import torchvision as tv
import learn2learn as l2l

from .splits_definitions import _ALL_SPLITS
from .lfw10 import LabelledFacesInWild


def get_full_dataset(name, root):
    if name == 'mini-imagenet':
        data_transforms = tv.transforms.Compose([
            lambda x: x / 255.0,
        ])
        train = l2l.vision.datasets.MiniImagenet(root=root,
                                                 transform=data_transforms,
                                                 download=True,
                                                 mode='train')
        valid = l2l.vision.datasets.MiniImagenet(root=root,
                                                 transform=data_transforms,
                                                 download=True,
                                                 mode='validation')
        test = l2l.vision.datasets.MiniImagenet(root=root,
                                                transform=data_transforms,
                                                download=True,
                                                mode='test')
        train = l2l.data.MetaDataset(train)
        valid = l2l.data.MetaDataset(valid)
        test = l2l.data.MetaDataset(test)
        dataset = l2l.data.UnionMetaDataset((train, valid, test))
    elif name == 'tiered-imagenet':
        data_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])
        train = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=data_transforms,
            download=True,
            mode='train',
        )
        valid = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=data_transforms,
            download=True,
            mode='validation',
        )
        test = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=data_transforms,
            download=True,
            mode='test',
        )
        train = l2l.data.MetaDataset(train)
        valid = l2l.data.MetaDataset(valid)
        test = l2l.data.MetaDataset(test)
        dataset = l2l.data.UnionMetaDataset((train, valid, test))
    elif name == 'emnist':
        data_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])
        train_set = tv.datasets.EMNIST(
            root=root,
            split='byclass',
            download=True,
            train=True,
            transform=data_transform,
        )
        test_set = tv.datasets.EMNIST(
            root=root,
            split='byclass',
            download=True,
            train=False,
            transform=data_transform,
        )
        dataset = torch.utils.data.ConcatDataset((train_set, test_set))
        dataset._bookkeeping_path = os.path.join(root, 'atg-emnist-bookkeeping.pkl')
        dataset = l2l.data.MetaDataset(dataset)
    elif name == 'lfw10':
        data_transform = tv.transforms.Compose([
            tv.transforms.Normalize(
                mean=[0., 0., 0.],
                std=[.5, .5, .5],
            ),
        ])
        dataset = LabelledFacesInWild(
            root=root,
            transform=data_transform,
            min_faces=10,
            download=True,
        )
        dataset._bookkeeping_path = os.path.join(root, 'atg-lfw10-bookkeeping.pkl')
        dataset = l2l.data.MetaDataset(dataset)
    elif name == 'cifar100':
        data_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])
        train = tv.datasets.CIFAR100(
            root=root,
            download=True,
            train=True,
            transform=data_transform,
        )
        test = tv.datasets.CIFAR100(
            root=root,
            download=True,
            train=False,
            transform=data_transform,
        )
        dataset = torch.utils.data.ConcatDataset((train, test))
        dataset._bookkeeping_path = os.path.join(
            root,
            'atg-cifar100-bookkeeping.pkl',
        )
        dataset = l2l.data.MetaDataset(dataset)
    else:
        raise f'Dataset {name} not supported'
    return dataset


def get_tasksets(
    name,
    taskset='original',
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=20000,
    root='~/data',
    device=None,
    **kwargs,
):
    root = os.path.expanduser(root)

    # load the full datasets
    dataset = get_full_dataset(name, root)

    # Load / generate partitions
    if 'random' in taskset:
        # split 64, 16, 20 percent
        seed = int(taskset[6:])
        rng = random.Random(seed)
        all_labels = dataset.labels[:]
        random.shuffle(all_labels, random=rng.random)
        n_train = int(0.64 * len(all_labels))
        n_valid = int(0.16 * len(all_labels))
        train_classes = all_labels[:n_train]
        valid_classes = all_labels[n_train:n_train + n_valid]
        test_classes = all_labels[n_train + n_valid:]
    else:
        # load as-is
        archive = _ALL_SPLITS[name][taskset]
        train_classes = archive['train_classes']
        valid_classes = archive['valid_classes']
        test_classes = archive['test_classes']

    # Instantiate Tasksets and transforms
    train_dataset = l2l.data.FilteredMetaDataset(dataset, train_classes)
    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(train_dataset, n=train_ways, k=train_samples),
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
        l2l.data.transforms.ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=num_tasks)
    valid_dataset = l2l.data.FilteredMetaDataset(dataset, valid_classes)
    valid_transforms = [
        l2l.data.transforms.FusedNWaysKShots(valid_dataset, n=test_ways, k=test_samples),
        l2l.data.transforms.LoadData(valid_dataset),
        l2l.data.transforms.RemapLabels(valid_dataset),
        l2l.data.transforms.ConsecutiveLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=num_tasks)
    test_dataset = l2l.data.FilteredMetaDataset(dataset, test_classes)
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(test_dataset, n=test_ways, k=test_samples),
        l2l.data.transforms.LoadData(test_dataset),
        l2l.data.transforms.RemapLabels(test_dataset),
        l2l.data.transforms.ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=num_tasks)
    return l2l.vision.benchmarks.BenchmarkTasksets(
        train_tasks,
        valid_tasks,
        test_tasks
    )
