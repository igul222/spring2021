"""
Fully unsupervised semantic segmentation!
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
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--target', type=str, default='Blond_Hair')
    parser.add_argument('--spurious', type=str, default='Male')
    parser.add_argument('--lambda_cov', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='conv')
    args = parser.parse_args()
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    train_dataset, y_idx = lib.datasets.celeba('train')
    train_loader = lib.utils.infinite_iterator(
        torch.utils.data.DataLoader(train_dataset, args.batch_size,
            shuffle=True, drop_last=True))

    test_dataset, _ = lib.datasets.celeba('test')
    test_data_by_group = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}
    for x,y in test_dataset:
        group = (y[y_idx[args.spurious]].item(), y[y_idx[args.target]].item())
        test_data_by_group[group].append((x, y))
    for _, data in test_data_by_group.items():
        np.random.shuffle(data)
    min_group_size = min(len(data) for _, data in test_data_by_group.items())
    balanced_test_dataset = [x 
        for group in test_data_by_group.values() 
        for x in group[:min_group_size]]

    if args.model == 'conv':
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, args.n_hid, 5, stride=2)
                self.conv2 = nn.Conv2d(args.n_hid, args.n_hid, 5, stride=2)
                self.conv3 = nn.Conv2d(args.n_hid, args.n_hid, 5, stride=2)
                self.relu = nn.ReLU()
                self.linear = nn.Linear(args.n_hid, 1)
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = torch.tanh(self.conv3(x))
                x = x.mean(dim=[2,3])
                z = x
                x = self.linear(x)[:,0]
                return x, z
    elif args.model == 'mlp':
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3*48*48, args.n_hid, bias=False)
                self.linear2 = nn.Linear(args.n_hid, 1)
            def forward(self, x):
                x = x.view(x.shape[0], 3*48*48)
                x = self.linear1(x)
                x = torch.tanh(x)
                z = x
                x = self.linear2(x)[:,0]
                return x, z

    model = Model().cuda()

    def forward():
        X, y = next(train_loader)
        X, y = X.cuda(), y.cuda()
        logits, z = model(X)
        targets = y[:, y_idx[args.target]].float()
        erm_loss = F.binary_cross_entropy_with_logits(logits, targets)
        acc = (logits > 0).float().eq(targets).float().mean()

        cov_penalty = 0.

        z_pos = z[(y==1).float().nonzero()[:,0], :]
        if z_pos.shape[0] > 1:
            cov = torch.einsum('nx,ny->xy', z_pos, z_pos) / z_pos.shape[0]
            cov_penalty = torch.triu(cov, diagonal=1).pow(2).mean()

        z_neg = z[(y==0).float().nonzero()[:,0], :]
        if z_neg.shape[0] > 1:
            cov = torch.einsum('nx,ny->xy', z_neg, z_neg) / z_neg.shape[0]
            cov_penalty = cov_penalty + torch.triu(cov, diagonal=1).pow(2).mean()

        if torch.isinf(cov_penalty):
            print('inf!')
            import pdb; pdb.set_trace()

        loss = erm_loss + (args.lambda_cov * cov_penalty)

        return loss, erm_loss, cov_penalty, acc

    opt = optim.Adam(list(model.parameters()), lr=3e-4)
    lib.utils.train_loop(forward, opt, args.steps, print_freq=100,
        history_names=['erm_loss', 'cov_penalty', 'train_acc'])

    def extract_feats(split, target):
        loader = torch.utils.data.DataLoader(split, 2*args.batch_size)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                all_z = []
                all_y = []
                for x, y in loader:
                    _, z = model(x.cuda())
                    all_z.append(z.cpu())
                    all_y.append(y.cpu())
                all_z = torch.cat(all_z, dim=0)
                all_y = torch.cat(all_y, dim=0)[:, y_idx[target]]
            return all_z.float().cuda(), all_y.long().cuda()

    def run_eval(all_train_feats, all_test_feats, bias):
        test_accs = []
        for train_feats, test_feats in zip(all_train_feats, all_test_feats):
            Z, Y = train_feats
            Z_test, Y_test = test_feats
            with torch.no_grad():
                pca = lib.pca.PCA(Z, Z.shape[1], whiten=True)
                Z = pca.forward(Z)
            linear = nn.Linear(Z.shape[1], 2, bias=bias).cuda()
            def forward():
                return F.cross_entropy(linear(Z), Y)
            opt = optim.SGD(linear.parameters(), lr=0.1, momentum=0.9,
                weight_decay=1e-3)
            lib.utils.train_loop(forward, opt, 1000, lr_schedule=True, quiet=True)
            with torch.no_grad():
                Z_test = pca.forward(Z_test)
                test_preds = linear(Z_test).argmax(dim=1)
                test_acc = test_preds.eq(Y_test).float().mean()
                test_accs.append(test_acc.item())
        return test_accs

    for bias in [True, False]:
        print('-'*80, "\n", f'bias={bias}:')

        print('All features:')
        all_train_feats = [
            extract_feats(train_dataset, args.target),
            extract_feats(train_dataset, args.target),
            extract_feats(train_dataset, args.spurious)
        ]
        all_test_feats = [
            extract_feats(test_dataset, args.target),
            extract_feats(balanced_test_dataset, args.target),
            extract_feats(balanced_test_dataset, args.spurious)
        ]
        lib.utils.print_row('unbal_target', 'bal_target', 'bal_spurious')
        lib.utils.print_row(*run_eval(all_train_feats, all_test_feats, bias))

        print('Individual units:')
        lib.utils.print_row('i', 'unbal_target', 'bal_target', 'bal_spurious')
        for i in range(args.n_hid):
            all_train_feats_transformed = [
                (Z[:,i:i+1], Y)
                for Z,Y in all_train_feats]
            all_test_feats_transformed = [
                (Z[:,i:i+1], Y)
                for Z,Y in all_test_feats]
            lib.utils.print_row(i, *run_eval(all_train_feats_transformed,
                all_test_feats_transformed, bias))