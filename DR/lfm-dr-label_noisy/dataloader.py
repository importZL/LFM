import os
import sys
import pandas as pd
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image

f = open(r'../data/food-101/meta/classes.txt', 'r')
Labels = [i.strip('\n') for i in f]
f.close()

# Load images from sewer dataset
class MultiLabelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="train", transform=None, loader=default_loader):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot  # path of images
        self.annRoot = annRoot  # path of label file
        self.split = split  # use to find label file

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    # load label file
    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "{}.txt".format(self.split))
        f = open(gtPath, 'r')
        self.imgPaths = [i.strip('\n')+'.jpg' for i in f]
        f.close()

        f = open(gtPath, 'r')
        self.labels = [self.LabelNames.index(i.split('/')[0]) for i in f]
        f.close()

        assert len(self.labels) == len(self.imgPaths)

    def __len__(self):
        return len(self.imgPaths)

    # load specific images
    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]
        return img, target


def load_svhn_data(f, train_transform, test_transform):
    train = sio.loadmat(f + "/train_32x32.mat")
    test = sio.loadmat(f + '/test_32x32.mat')

    train_data = train['X']
    train_y = train['y']
    test_data = test['X']
    test_y = test['y']
    
    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 3)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 3)
    
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    train_y = torch.from_numpy(train_y)
    test_y = torch.from_numpy(test_y)
    for i in range(len(train_data)):
        train_data[i] = train_transform(train_data[i].float())
    for i in range(len(test_data)):
        test_data[i] = test_transform(test_data[i].float())

    train_y = train_y.reshape(73257, )
    test_y = test_y.reshape(26032, )
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0
    # train_y = np.array(train_y)
    # test_y = np.array(test_y)
    return train_data, train_y, test_data, test_y
