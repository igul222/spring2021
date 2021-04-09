import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.datasets
import tqdm

DATA_DIR = os.path.expanduser('~/data')

def _parallel_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def mnist():
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)
    X, y = mnist.data.clone(), mnist.targets.clone()
    _parallel_shuffle(X.numpy(), y.numpy())
    X = (X.float() / 256.)
    return X.view(-1, 784).cuda(), y.cuda()
