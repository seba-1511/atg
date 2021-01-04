#!/usr/bin/env python3

import random
import learn2learn as l2l
import torchvision as tv

from .splits_definitions import _ALL_SPLITS


def get_full_dataset(dataset, root):
    if dataset == 'mini-imagenet':
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
        l2l.data.FusedNWaysKShots(train_dataset, n=train_ways, k=train_samples),
        l2l.data.LoadData(train_dataset),
        l2l.data.RemapLabels(train_dataset),
        l2l.data.ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=num_tasks)
    valid_dataset = l2l.data.FilteredMetaDataset(dataset, valid_classes)
    valid_transforms = [
        l2l.data.FusedNWaysKShots(valid_dataset, n=test_ways, k=test_samples),
        l2l.data.LoadData(valid_dataset),
        l2l.data.RemapLabels(valid_dataset),
        l2l.data.ConsecutiveLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=num_tasks)
    test_dataset = l2l.data.FilteredMetaDataset(dataset, test_classes)
    test_transforms = [
        l2l.data.FusedNWaysKShots(test_dataset, n=test_ways, k=test_samples),
        l2l.data.LoadData(test_dataset),
        l2l.data.RemapLabels(test_dataset),
        l2l.data.ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=num_tasks)
    return train_tasks, valid_tasks, test_tasks
