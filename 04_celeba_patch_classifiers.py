"""
Train multiple patch classifiers on CelebA, regularized to activate on different
regions of the image.
"""

import argparse
import collections
import copy
import itertools
import lib.datasets
import lib.ops
import lib.pca
import lib.utils
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--target', type=str, default='Blond_Hair')
    parser.add_argument('--spurious', type=str, default='Male')
    parser.add_argument('--lambda', type=float, default=0.0)
    parser.add_argument('--n_classifiers', type=int, default=2)
    args = parser.parse_args()
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    train_dataset, y_idx = lib.datasets.celeba('train')
    train_loader = lib.utils.infinite_iterator(
        torch.utils.data.DataLoader(train_dataset, args.batch_size,
            shuffle=True, drop_last=True))

    test_dataset, _ = lib.datasets.celeba('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size,
            shuffle=False, drop_last=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, args.n_hid, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(args.n_hid, args.n_hid, 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(args.n_hid, args.n_hid, 3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(args.n_hid, args.n_hid, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(args.n_hid, args.n_classifiers, 1, stride=1)
            self.relu = nn.ReLU()
            self.bias = nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.conv5(x)
            return x

    model = Model().cuda()

    def forward():
        X, y = next(train_loader)
        X, y = X.cuda(), y.cuda()
        logit_maps = model(X)
        classifier_losses, classifier_accs = [], []
        for i in range(args.n_classifiers):
            logit_map = logit_maps[:, i]
            logits = logit_map.mean(dim=[1,2]) + model.bias
            targets = y[:, y_idx[args.target]].float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            acc = (logits > 0).float().eq(targets).float().mean()
            classifier_losses.append(loss)
            classifier_accs.append(acc)

        # Diversity penalty; hardcoded to assume 2 classifiers for now
        assert(args.n_classifiers == 2)
        logits1 = logit_maps[:,0:1]
        logits2 = logit_maps[:,1:2]
        POOL = 4
        logits1 = F.avg_pool2d(logits1.abs(), POOL, POOL)
        logits2 = F.avg_pool2d(logits2.abs(), POOL, POOL)
        diversity = 0.1 * (logits1 * logits2).mean()

        loss = torch.stack(classifier_losses).mean() + diversity

        return [loss, *losses, *accs, diversity]

    def hook(step):
        accs = [[] for _ in range(args.n_classifiers)]
        for X, y in tqdm.tqdm(test_loader, leave=False):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    X, y = X.cuda(), y.cuda()
                    logit_maps = model(X)
                    for i in range(args.n_classifiers):
                        logit_map = logit_maps[:, i]
                        logits = logit_map.mean(dim=[1,2])
                        targets = y[:, y_idx[args.target]].float()
                        acc = (logits > 0).float().eq(targets).float()
                        accs[i].append(acc)
        test_accs = [torch.cat(x, dim=0).mean().item() for x in accs]
        print('Test accs:', *test_accs)

        X = next(iter(test_loader))[0].cuda()[:100]

        logit_maps = model(X)
        for i in range(args.n_classifiers):
            z = logit_maps[:,i,:,:]
            z = z[:,None,:,:]
            z = z.expand(-1,3,-1,-1)
            z = z - z.min()
            z = z / z.max()
            z = (z*255.99).byte()
            z = z.permute(0,2,3,1)
            lib.utils.save_image_grid(z, f'logits{i}.png')

        X = ((X*128.)+127.99).byte()
        X = X.permute(0,2,3,1)
        lib.utils.save_image_grid(X, 'images.png')


    opt = optim.Adam(list(model.parameters()), lr=1e-3)
    history_names = [f'loss{i}' for i in range(args.n_classifiers)]
    history_names += [f'acc{i}' for i in range(args.n_classifiers)]
    history_names += ['diversity']
    lib.utils.train_loop(forward, opt, args.steps, print_freq=100,
        history_names=history_names, hook=hook, hook_freq=300)