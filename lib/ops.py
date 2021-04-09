import torch

def get_batch(vars_, batch_size):
    assert(isinstance(vars_, list))
    idx = torch.randint(low=0, high=len(vars_[0]), size=(batch_size,))
    return [v[idx] for v in vars_]
