import numpy as np

def collate_fn(data):
    # cv -> mx
    imgs = [d[0] for d in data]
    imgs = np.stack(imgs, axis=0)
    imgs = np.transpose(imgs, (0,3,1,2))

    features = [d[1] for d in data]
    features = np.stack(features, axis=0)

    labels = [d[2] for d in data]
    labels = np.stack(labels, axis=0)

    return (imgs, features, labels)

def add_common_args(parser):
    common = parser.add_argument_group('Common', 'the common args')
    common.add_argument('--root', type=str, help='dataset root')
    common.add_argument('--num-workers', type=int, help='number of workers in dataloader')

    return common

