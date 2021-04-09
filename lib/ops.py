import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

class CausalConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, mask_present):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((dim_out, dim_in, kernel_size//2 + 1, kernel_size)))
        self.bias = nn.Parameter(
            torch.zeros((dim_out,)))
        assert(kernel_size%2 == 1)
        self.padding = (kernel_size//2, kernel_size//2)
        self.weight.data /= float(np.sqrt(kernel_size**2 * dim_in))
        self.mask = torch.ones(
            (dim_out, dim_in, kernel_size//2 + 1, kernel_size))
        if mask_present:
            self.mask[:,:,-1,kernel_size//2:] *= 0.
        else:
            self.mask[:,:,-1,(kernel_size//2)+1:] *= 0.
        self.mask = self.mask.cuda()
        self.kernel_size = kernel_size

    def forward(self, x):
        weight = self.weight * self.mask
        x = F.conv2d(x, weight, self.bias, padding=self.padding)
        x = x[:,:,:-(self.kernel_size//2),:].contiguous()
        return x

class CausalResBlock(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv1 = CausalConv(dim, dim, kernel_size, False)
        self.conv2 = CausalConv(dim, dim, kernel_size, False)
        self.conv1 = torch.nn.utils.weight_norm(self.conv1)
        self.conv2 = torch.nn.utils.weight_norm(self.conv2)

    def forward(self, x):
        x_shortcut = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        return x_shortcut + x

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x_shortcut + x

class WideResnet(nn.Module):
    """WRN, I handle the downsampling slightly differently, for simplicity.
    The default settings (N=1, k=4) correspond to WRN-10-4."""
    def __init__(self, N=1, k=4, dim_in=3, dim_out=None):
        super().__init__()
        self.input = nn.Conv2d(dim_in, 16*k, 1)
        self.conv2 = nn.Sequential(*[ResBlock(16*k) for _ in range(N)])
        self.pre_conv3 = nn.Conv2d(16*k, 32*k, 2, stride=2, bias=False)
        self.conv3 = nn.Sequential(*[ResBlock(32*k) for _ in range(N)])
        self.pre_conv4 = nn.Conv2d(32*k, 64*k, 2, stride=2, bias=False)
        self.conv4 = nn.Sequential(*[ResBlock(64*k) for _ in range(N)])
    def forward(self, x):
        x = self.input(x)
        x = self.conv2(x)
        x = self.pre_conv3(x)
        x = self.conv3(x)
        x = self.pre_conv4(x)
        x = self.conv4(x)
        x = x.mean(dim=[2,3])
        return x

class WideResnetDecoder(nn.Module):
    """Decoder network that's roughly a mirror-image of WRN."""
    def __init__(self, N=1, k=4, dim_in=None, dim_out=3):
        super().__init__()
        self.conv4 = nn.Sequential(*[ResBlock(64*k) for _ in range(N)])
        self.pre_conv4 = nn.ConvTranspose2d(64*k, 32*k, 2, stride=2, bias=False)
        self.conv3 = nn.Sequential(*[ResBlock(32*k) for _ in range(N)])
        self.pre_conv3 = nn.ConvTranspose2d(32*k, 16*k, 2, stride=2, bias=False)
        self.conv2 = nn.Sequential(*[ResBlock(16*k) for _ in range(N)])
        self.output = nn.Conv2d(16*k, dim_out, 1)
    def forward(self, x):
        x = x[:,:,None,None].repeat(1,1,8,8)
        x = self.conv4(x)
        x = self.pre_conv4(x)
        x = self.conv3(x)
        x = self.pre_conv3(x)
        x = self.conv2(x)
        x = self.output(x)
        return x