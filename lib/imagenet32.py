"""
32x32 ImageNet data loader
"""

import numpy as np
import os
import PIL
import time
import torch
import torchvision
from tqdm import tqdm

class ImageNet32(object):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.transform = transform
        self.pt_path = os.path.join(data_path, 'train_32x32.pt')
        self.images = None
        if not os.path.exists(self.pt_path):
            print('Creating ImageNet32 archive...')
            train_path = os.path.join(data_path, 'train_32x32')
            files = os.listdir(train_path)
            np.random.shuffle(files)
            images = np.zeros((len(files),3,32,32), dtype='uint8')
            for i, file in enumerate(tqdm(files)):
                file_path = os.path.join(train_path, file)
                image = np.array(PIL.Image.open(file_path).convert("RGB"))
                images[i] = image.transpose(2,0,1)
            images = torch.tensor(images)
            torch.save(images, self.pt_path)
        self._load_images()

    def _load_images(self):
        assert(os.path.exists(self.pt_path))
        print('Loading ImageNet32...')
        start_time = time.time()
        self.images = torch.load(self.pt_path)
        print(f'Done! Time: {time.time() - start_time}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].float() / 255.
        x = torchvision.transforms.functional.to_pil_image(x)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(0, dtype=torch.int64)
        return (x, y)