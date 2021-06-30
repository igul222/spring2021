import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as T
import tqdm
from PIL import Image

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

def waterbirds(split):
    split = {'train': '0', 'val': '1', 'test': '2'}[split]
    waterbirds_dir = os.path.join(DATA_DIR,
        'waterbird_complete95_forest2water2')
    pt_path = os.path.join(DATA_DIR, f'waterbirds_{split}.pt')
    if not os.path.exists(pt_path):
        with open(os.path.join(waterbirds_dir, 'metadata.csv'), 'r') as f:
            # Columns: img_id,img_filename,y,split,place,place_filename
            metadata = [l[:-1].split(',') for l in f][1:]
        metadata = [l for l in metadata if l[3]==split]
        image_paths = [os.path.join(waterbirds_dir, l[1]) for l in metadata]
        Y = [int(l[2]) for l in metadata]
        transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
        X = []
        for image_path in tqdm.tqdm(image_paths):
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        Y = torch.tensor(Y)
        torch.save((X, Y), pt_path)
    X, Y = torch.load(pt_path)
    return torch.utils.data.TensorDataset(X, Y)
