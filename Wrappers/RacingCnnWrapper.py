import gym
import torch
import numpy as np


class GenericRemapper:
    def __init__(self, aabb, resolution, channels, dtype=np.float32):
        from math import ceil, inf
        self.aabb = aabb
        self.resolution = resolution
        self.cells_width = ceil((aabb[1] - aabb[0]) / resolution)
        self.cells_height = ceil((aabb[3] - aabb[2]) / resolution)

        bounds = -inf, inf

        if dtype == np.bool:
            bounds = False, True

        self.observation_space = gym.spaces.Box(*bounds, (channels, self.cells_height, self.cells_width), dtype)

    def __call__(self, observation):
        # Slightly ugly dtype conversion
        torch_dtype = torch.from_numpy(np.zeros(tuple(), dtype=self.observation_space.dtype)).dtype
        result = torch.zeros(self.observation_space.shape, dtype=torch_dtype)

        # Filter out not present and encode
        x, y, values = self.encode_items(observation[observation[:, 0] > 0, 1:])

        x_idx = torch.floor((x - self.aabb[0]) / self.resolution).long()
        y_idx = torch.floor((y - self.aabb[2]) / self.resolution).long()

        valid = (x_idx >= 0) & (y_idx >= 0) & (x_idx < self.observation_space.shape[2]) &\
                (y_idx < self.observation_space.shape[1])

        x_idx, y_idx, values = x_idx[valid], y_idx[valid], values[valid]

        result[:, y_idx, x_idx] = values.T

        # TODO: Remove plausi check
        # import matplotlib.pyplot as plt
        # img = np.transpose(result.float().numpy(), (1, 2, 0))
        # img_disp = np.zeros(img.shape[:-1] + (3,))
        # img_disp[..., :img.shape[-1]] = img
        # plt.imshow(img_disp)
        # plt.show()

        return result

    def encode_items(self, items):
        raise NotImplementedError


class ConeMapRemapper(GenericRemapper):
    def __init__(self, env, key, resolution=.1):
        from CarEnv.SensorConeMap import SensorConeMap
        sensor = env.sensors[key]

        if not isinstance(sensor, SensorConeMap):
            raise TypeError(str(type(sensor)))

        aabb = sensor.bbox
        self.sensor_normalizer = 1.0 if not sensor._normalize else sensor.view_normalizer

        super(ConeMapRemapper, self).__init__(aabb, resolution, 2, dtype=np.bool)

    def encode_items(self, items):
        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items)
        xs = items[:, 0] * self.sensor_normalizer
        ys = items[:, 1] * self.sensor_normalizer
        values = items[:, 2:].bool()
        return xs, ys, values


class RacingCnnWrapper(gym.ObservationWrapper):
    def __init__(self, env, resolution=1.):
        from copy import deepcopy
        super(RacingCnnWrapper, self).__init__(env)

        self.observation_space: gym.spaces.Dict = deepcopy(env.observation_space)
        self.remappers = {}

        if "cones_set" in env.observation_space.spaces:
            old_name = "cones_set"
            new_name = "cones_img"
            remapper = ConeMapRemapper(env, old_name, resolution=resolution)
            self.remappers[old_name] = (new_name, remapper)
            self.observation_space.spaces[new_name] = remapper.observation_space
            self.observation_space.spaces.pop(old_name)

    def observation(self, observation):
        new_observation = dict()

        for k, v in observation.items():
            if k in self.remappers:
                new_name, remapper = self.remappers[k]
                new_observation[new_name] = remapper(v)
            else:
                new_observation[k] = v

        return new_observation
