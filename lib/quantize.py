"""
Color-palette quantization functions for images.
"""

import torch
import numpy as np

def quantize(x, q_levels):
    """
    input: (n, 3, 32, 32) floats in [0, 1]
    output: (n, 32*32) ints in [0, Q_LEVELS)
    """
    q_levels = int(np.cbrt(q_levels))
    x = x*255/256 # [0, 1)
    x = (x * q_levels) # [0, q_levels)
    x = x.long() # ints in [0, q_levels) (*biased by 0.5 downwards)
    x = (q_levels**2 * x[:,0,:,:]) + (q_levels * x[:,0,:,:]) + x[:,2,:,:]
    x = x.view(x.shape[0], 32*32)
    return x

def dequantize(x, q_levels):
    """
    input: (n, 32*32) ints in [0, Q_LEVELS)
    output: (n, 3, 32, 32) floats in [0, 1]
    """
    q_levels = int(np.cbrt(q_levels))
    x = x.view(x.shape[0], 32, 32)
    x0 = (x // q_levels**2) % q_levels
    x1 = (x // q_levels) % q_levels
    x2 = x % q_levels
    x = torch.stack([x0, x1, x2], dim=1) # (n, 3, 32, 32) in [0, q_levels)    
    x = (x.float() + 0.5) / q_levels # bias-corrected and scaled to [0, 1)
    return x