from __future__ import print_function
import os, os.path, errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dlcommon.datasets.dataset import Dataset
import cv2


class Leafs(Dataset):
    train_list = [
        ['train.csv'],
    ]

    test_list = [
        ['test.csv'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_pd = pd.read_csv(os.path.join(self.root, self.train_list[0][0]))
        le = LabelEncoder().fit(train_pd.species)
        classes = list(le.classes_)


        if self.train:
            self.input_names = ['data', 'features']
            self.label_names = ['softmax_label']
            self.train_labels = []
            self.train_data = []
            for f_entry in self.train_list:
                train_pda = pd.read_csv(os.path.join(self.root, f_entry[0]))
                self.train_labels.extend(le.transform(train_pda.species).tolist())
                idx = train_pda['id'].tolist()
                features = train_pda.drop(['species', 'id'], axis=1)
                self.train_data.extend([
                    (os.path.join(self.root, 'images', str(idx[i])+'.jpg'), features.values[i]) for i in range(len(idx))])
        else:
            self.input_names = ['data']
            self.label_names = ['softmax_label']
            self.test_data = []
            self.test_labels = []
            for f_entry in self.test_list:
                test_pda = pd.read_csv(os.path.join(self.root, f_entry[0]))
                self.test_labels.extend(le.transform(test_pda.species).tolist())
                idx = test_pda['id'].tolist()
                features = test_pda.drop(['species', 'id'], axis=1)
                self.test_data.extend([os.path.join(self.root, 'images', str(i)+'.jpg') for i in idx])

    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index][0]
            feature = self.train_data[index][1]
            target = self.train_labels[index]
        else:
            img = self.test_data[index]
            target = self.test_labels[index]

        im = cv2.imread(img)[:,:,(2,1,0)]

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, feature, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

