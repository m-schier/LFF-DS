from typing import Tuple

import gym
from highway_env.envs.common.abstract import Observation
from highway_env.envs.highway_env import HighwayEnv
import torch
import numpy as np
from Wrappers.RacingCnnWrapper import GenericRemapper


_DEFAULT_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 100,
        "features": ["presence", "x", "y", "vx", "vy", "heading"],
        "features_range": {
            "x": [-80, 80],
            "y": [-20, 20],
            "vx": [-30, 30],
            "vy": [-15, 15],
        }
    },
    "vehicles_count": 50,
    "lanes_count": 4,
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "vehicles_density": 3,
    "offroad_terminal": True,
    "normalize_reward": False,  # Just why?
    "reward_speed_range": [5, 30],
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.2,
    "action": {
        "type": "ContinuousAction",
        "lateral": True,
        "longitudinal": True,
        "acceleration_range": (-6.0, 6.0),  # Only half accel actually used
        "steering_range": (-np.pi / 20, np.pi / 20),
    },
}


class HighwayCompatEnv(HighwayEnv):
    def __init__(self, offscreen_rendering=False, conf=None):
        from copy import deepcopy
        # Hack this into older gym versions than actually supported by highway-env
        self.np_random = np.random.default_rng()

        conf = conf if conf is not None else deepcopy(_DEFAULT_CONFIG)
        conf["offscreen_rendering"] = offscreen_rendering

        super(HighwayCompatEnv, self).__init__(conf)
        self.compat_speed = 0.

    def seed(self, seed=None) -> None:
        # Hack this into older gym versions than actually supported by highway-env
        self.np_random = np.random.default_rng(seed)

    @property
    def dt(self):
        return 1. / self.config['policy_frequency']

    @property
    def controlled_ids(self):
        return ["<PSEUDO>"]

    def _unpack_action(self, action):
        # May be called with list of one action, then unpack

        if self.config['action']['type'] == 'ContinuousAction':
            return action

        try:
            action, = action
        except TypeError:
            pass
        return int(action)

    def is_lane_change(self, action):
        # from highway_env.envs.common.action import DiscreteMetaAction
        action = self._unpack_action(action)

        return action == 0 or action == 2

    @property
    def ego_speed(self):
        return self.compat_speed

    def reset(self, **kwargs):
        obs, info = super(HighwayCompatEnv, self).reset(**kwargs)

        # Reduce initial speed of controlled vehicles which is so fast that they might crash
        for veh in self.controlled_vehicles:
            veh.speed = 15.

        self.compat_speed = 15.
        return obs

    @staticmethod
    def _danger_zone(obj) -> np.ndarray:
        # Extra pad
        length = obj.LENGTH + 2
        width = obj.WIDTH + .4

        points = np.array([
            [-length / 2, -width / 2],
            [-length / 2, +width / 2],
            [+length / 2, +width / 2],
            [+length / 2, -width / 2],
        ]).T
        c, s = np.cos(obj.heading), np.sin(obj.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(obj.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def _is_dangerously_close(self):
        from highway_env import utils

        dt = self.dt

        for veh in self.controlled_vehicles:
            # Note: On a 4-lane road could also consider veh.position[1] outside [-1, 13] dangerously close to edge

            poly = HighwayCompatEnv._danger_zone(veh)

            for other in self.road.vehicles:
                if other is veh:
                    continue
                # Fast spherical pre-check
                if np.linalg.norm(other.position - veh.position) > (
                        veh.diagonal + other.diagonal) / 2 + veh.speed * dt:
                    continue
                # Accurate rectangular check
                if utils.are_polygons_intersecting(poly, other.polygon(), veh.velocity * dt, other.velocity * dt)[0]:
                    return True
        return False

    def step(self, action) -> Tuple[Observation, float, bool, dict]:
        action = self._unpack_action(action)
        t, s = action

        if t > 0:
            t = min(1, t) * .5  # Only 50% of accel available, but this way stays symmetric

        obs, reward, terminated, truncated, info = super(HighwayCompatEnv, self).step((t, s))

        if not terminated and self._is_dangerously_close():
            reward -= .2

        self.compat_speed = info['speed']

        info['TimeLimit.truncated'] = truncated
        done = terminated or truncated
        return obs, reward, done, info

    def _is_truncated(self) -> bool:
        # Fix from highway env
        return self.time >= self.config["duration"]


class HighwayCompatWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(HighwayCompatWrapper, self).__init__(env)

        n_vehs, n_features = env.observation_space.shape

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-1, 1, (n_features - 1,)),
            "other_set": gym.spaces.Box(-1, 1, (n_vehs - 1, n_features)),
        })

    def observation(self, observation):
        ego, others = observation[0], observation[1:]

        # Discard present flag on ego
        ego = ego[1:]

        # Zero out vehicles outside normalized observation x range
        others[np.abs(others[:, 1]) >= 1.] = 0.

        return {"state": torch.from_numpy(ego), "other_set": torch.from_numpy(others)}


class HighwayRemapper(GenericRemapper):
    def __init__(self, env, resolution=2.):
        max_obs, n_features = env.observation_space.spaces["other_set"].shape
        super(HighwayRemapper, self).__init__((-80, 80, -20, 20), resolution, n_features)

    def encode_items(self, items):
        xs = items[:, 0] * 80
        ys = items[:, 1] * 20
        return xs, ys, torch.cat([torch.ones_like(items[:, :1]), items], dim=-1)


class HighwayCnnWrapper(gym.ObservationWrapper):
    def __init__(self, env, resolution=None):
        super(HighwayCnnWrapper, self).__init__(env)

        if resolution is None:
            resolution = 5.

        self.remapper = HighwayRemapper(env, resolution)

        self.observation_space = gym.spaces.Dict({
            "state": env.observation_space.spaces["state"],
            "other_img": self.remapper.observation_space,
        })

    def observation(self, observation):
        return {
            "state": observation["state"],
            "other_img": self.remapper(observation["other_set"]),
        }
