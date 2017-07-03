import find_mxnet
import os
import argparse
import logging
import numpy as np

from dlcommon.data.mxnet import data_iter
from dlcommon.data import augmenter, dataloader
from dlcommon.argparse import data_args, data_aug_args

import common
from datasets import leafs
import fit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train leafs exist")
    parser.add_argument('--num-classes', type=int, help='number of classes of datasets')

    common.add_common_args(parser)
    data_args.add_data_args(parser)
    data_aug_args.add_data_aug_args(parser)
    fit.add_fit_args(parser)

    args = parser.parse_args()

    print args
    data_shape = [int(s) for s in args.image_shape.split(',')]
    mean = np.array([float(s) for s in args.mean.split(',')])
    std = np.array([float(s) for s in args.std.split(',')])

    train_tf = augmenter.Augmenter(data_shape, args.resize, args.rand_crop,
            args.rand_resize, args.rand_mirror, mean, std)
    trainset = leafs.Leafs(root=args.root, train=True, transform=train_tf)
    trainloader = dataloader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, collate_fn=common.collate_fn)
    mxIter = data_iter.pyImageDataIter(trainloader)
    mxIter.reset()

    fit.fit(args, mxIter, None)

