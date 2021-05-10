"""
Trains two MNIST classifiers and finds corresponding hidden units.
"""

import argparse
import lib.datasets
import lib.ops
import lib.utils
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

BATCH_SIZE = 1024
N_HID = 64
STEPS = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--unconditional', action='store_true')
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

X_train, y_train = lib.datasets.mnist()
y_train = (y_train == 3).float()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, N_HID, bias=False)
        self.linear2 = nn.Linear(N_HID, 1, bias=True)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x

def train_model():
    model = Model().cuda()
    def forward():
    	X, y = lib.ops.get_batch([X_train, y_train], BATCH_SIZE)
    	logits = model(X)[:,0]
    	return F.binary_cross_entropy_with_logits(logits, y)
    opt = optim.Adam(model.parameters(), weight_decay=1e-3)
    lib.utils.train_loop(forward, opt, STEPS)
    return model

def absorb_signs(model):
    """
    Modify model's weight matrix so that the signs of the readout layer
    are absorbed into the weights.
    """
    for i in range(N_HID):
        w2 = model.linear2.weight[0, i]
        model.linear1.weight.data[i] *= w2.sign()
        model.linear2.weight.data[0, i] *= w2.sign()

def save_weights_image(weights, path):
    weights = weights.reshape(weights.shape[0], 28, 28)
    weights_color = torch.zeros(weights.shape[0], 28, 28, 3)
    for i in range(weights.shape[0]):
        for x in range(28):
            for y in range(28):
                if weights[i, x, y] < 0:
                    weights_color[i, x, y, 0] = weights[i, x, y].abs()
                else:
                    weights_color[i, x, y, 1] = weights[i, x, y].abs()
    weights_color *= 255.99 / weights_color.max()
    weights_color = weights_color.byte()
    lib.utils.save_image_grid(weights_color, path)

model1 = train_model()
model2 = train_model()
absorb_signs(model1)
absorb_signs(model2)
save_weights_image(model1.linear1.weight, 'model1.png')
save_weights_image(model2.linear1.weight, 'model2.png')

with torch.no_grad():
    z1 = torch.tanh(model1.linear1(X_train))
    z2 = torch.tanh(model2.linear1(X_train))

    argmax_visualization = []
    argmin_visualization = []
    for i in range(N_HID):
        rho_max = -9999
        rho_min = +9999
        rho_argmax = None
        rho_argmin = None
        for j in range(N_HID):
            rhos = []
            for target in [0, 1]:
                if args.unconditional:
                    z1_cond = z1[:, i]
                    z2_cond = z2[:, j]
                else:
                    z1_cond = z1[(y_train==target).float().nonzero()[:,0], i]
                    z2_cond = z2[(y_train==target).float().nonzero()[:,0], j]
                z1_cond -= z1_cond.mean()
                z2_cond -= z2_cond.mean()
                cov = (z1_cond*z2_cond).mean()
                std1 = z1_cond.std()
                std2 = z2_cond.std()
                rho = (cov / (std1 * std2)).abs()
                rhos.append(rho)
            rho = torch.stack(rhos).mean()
            print(i, j, rho.item())
            if rho > rho_max:
                rho_max = rho
                rho_argmax = j
            if rho < rho_min:
                rho_min = rho
                rho_argmin = j

        argmax_visualization.append(model1.linear1.weight[i])
        argmax_visualization.append(model2.linear1.weight[rho_argmax])

        argmin_visualization.append(model1.linear1.weight[i])
        argmin_visualization.append(model2.linear1.weight[rho_argmin])

    argmax_visualization = torch.stack(argmax_visualization, dim=0)
    save_weights_image(argmax_visualization, 'rho_argmax.png')

    argmin_visualization = torch.stack(argmin_visualization, dim=0)
    save_weights_image(argmin_visualization, 'rho_argmin.png')