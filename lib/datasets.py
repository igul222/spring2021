import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as T
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

def celeba(split):
    pt_path = os.path.join(DATA_DIR, f'celeba_{split}.pt')
    if not os.path.exists(pt_path):
        dataset = torchvision.datasets.CelebA(DATA_DIR, split=split,
            download=True,
            transform=T.Compose([
                T.Resize(48),
                T.CenterCrop(48),
                T.ToTensor(),
                T.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024,
            num_workers=8)
        X, Y = [], []
        for x, y in tqdm.tqdm(loader):
            X.append(x)
            Y.append(y)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        attr_names = dataset.attr_names
        torch.save((X, Y, attr_names), pt_path)
    print(f'Loading CelebA {split}...')
    X, Y, attr_names = torch.load(pt_path)
    y_idx = {name:idx for idx, name in enumerate(attr_names)}
    return torch.utils.data.TensorDataset(X, Y), y_idx