#!/usr/bin/env python3

"""
File: lfw.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description:
Implements a PyTorch Dataset of the labelled faces in the wild dataset,
on top of the sklearn interface.
"""

import numpy as np
import os
import torch

from sklearn import datasets


class LabelledFacesInWild(torch.utils.data.Dataset):
    """docstring for LabelledFacesInWild"""

    def __init__(
        self,
        root,
        min_faces=1,
        download=False,
        transform=None,
        target_transform=None,
    ):
        super(LabelledFacesInWild, self).__init__()
        self.root = os.path.expanduser(root)
        self.download = download
        self.min_faces = min_faces
        self.transform = transform
        self.target_transform = target_transform

        lfw = datasets.fetch_lfw_people(
            data_home=self.root,
            color=True,
            download_if_missing=download,
            min_faces_per_person=min_faces,
        )
        images = np.swapaxes(lfw.images, 1, 3)
        images = np.swapaxes(images, 2, 3)
        self.images = torch.from_numpy(images)
        self.target = torch.from_numpy(lfw.target)

    def to(self, device):
        self.images = self.images.to(device)
        self.target = self.target.to(device)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        img = self.images[item]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target[item]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
