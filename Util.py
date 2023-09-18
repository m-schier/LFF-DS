import torch
from gym import ObservationWrapper
import numpy as np


def aggregate_sum(x, indices, n_bins):
    x_agg = torch.zeros((n_bins, x.shape[-1]), device=x.device)
    if indices.numel() > 0:
        torch.index_put_(x_agg, (indices,), x, accumulate=True)
    return x_agg


def container_remap(obj, type_filter, fn):
    if isinstance(obj, tuple):
        return tuple([container_remap(x, type_filter, fn) for x in obj])
    elif isinstance(obj, list):
        return [container_remap(x, type_filter, fn) for x in obj]
    elif isinstance(obj, dict):
        return {k: container_remap(v, type_filter, fn) for k, v in obj.items()}
    elif isinstance(obj, type_filter):
        return fn(obj)
    else:
        return obj


def numpy_to_torch(ndarray: np.ndarray):
    if ndarray.dtype == np.float64:
        return torch.from_numpy(ndarray).float()
    elif ndarray.dtype == np.float32:
        return torch.from_numpy(ndarray)
    elif np.issubdtype(ndarray.dtype, np.integer):
        return torch.from_numpy(ndarray).long()
    else:
        raise TypeError(ndarray.dtype)


class TorchEnvObsWrapper(ObservationWrapper):
    def observation(self, observation):
        return container_remap(observation, np.ndarray, numpy_to_torch)
