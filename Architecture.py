import gym
import torch
from stable_baselines3.dqn.policies import BaseFeaturesExtractor

from PreEncoder import make_preencoder
from typing import Dict, Optional


class PassthroughModel(torch.nn.Module):
    def forward(self, x):
        return x


def make_item_encoder_network(in_feature, net_arch, preencoder):
    layers = []
    preenc_layer = make_preencoder(preencoder, in_feature, net_arch[0])
    layers.append(preenc_layer)

    # A ReLU was incorrectly always applied after the preencoder before, this is now fixed

    for in_dim, out_dim in zip(net_arch[:-1], net_arch[1:]):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


def _make_static_encoder(space, static_dims):
    if len(space.shape) != 1:
        raise ValueError(f"Unsupported static space: {space}")

    if static_dims == 0:
        # Dummy
        return PassthroughModel(), space.shape[0]
    else:
        return torch.nn.Sequential(torch.nn.Linear(space.shape[0], static_dims), torch.nn.ReLU()), static_dims


class SetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, preencoder="fc", static_dims=0, item_arch: Optional[Dict[str, int]] = None, set_dims: Optional[Dict[str, int]] = None) -> None:
        super(SetFeatureExtractor, self).__init__(observation_space, 1)

        if item_arch is None:
            item_arch = dict()

        if set_dims is None:
            set_dims = dict()

        total_features = 0

        item_encoders = {}
        set_encoders = {}

        for key, space in observation_space.spaces.items():
            if "_set" in key:
                max_obs, obs_features = space.shape

                arch = item_arch.get(key, [64, 128])
                set_dim = set_dims.get(key, arch[-1])

                item_encoder_mlp = make_item_encoder_network(obs_features - 1, arch, preencoder,
                                                             freeze_preencoder=False)

                set_encoder = torch.nn.Sequential(
                    torch.nn.Linear(arch[-1], set_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(set_dim, set_dim),
                    torch.nn.ReLU(),
                )

                item_encoders[key] = ItemEncoder(item_encoder_mlp)
                set_encoders[key] = set_encoder
                total_features += set_dim
            else:
                item_encoders[key], dims = _make_static_encoder(space, static_dims)
                total_features += dims

        self._features_dim = total_features
        self.item_encoders = torch.nn.ModuleDict(item_encoders)
        self.set_encoders = torch.nn.ModuleDict(set_encoders)

    @staticmethod
    def batchify_set(values):
        batch_size, max_obs, obs_features = values.shape

        # Discard all items which do not have the "present" flag in the obs_feature
        # The present flag must always be the first feature!
        all_batch_idxs = torch.broadcast_to(torch.arange(batch_size, device=values.device)[..., None], (batch_size, max_obs))
        keep_mask = values[..., 0] > 0
        # Mask and discard the present flag
        items = values[keep_mask][..., 1:].reshape(-1, obs_features - 1)
        batch_idxs = all_batch_idxs[keep_mask].flatten()

        return items, batch_idxs, batch_size

    def forward(self, obs):
        result = []

        for key, encoder in self.item_encoders.items():
            if "_set" in key:
                items, batch_idxs, batch_size = self.batchify_set(obs[key])
                items_enc_agg = encoder(items, batch_idxs, batch_size)
                set_enc = self.set_encoders[key](items_enc_agg)
                result.append(set_enc)
            else:
                result.append(encoder(obs[key]))

        return torch.cat(result, dim=-1)


class ItemEncoder(torch.nn.Module):
    def __init__(self, mlp):
        super(ItemEncoder, self).__init__()
        self.mlp = mlp

    def forward(self, x, idxs, n_bins):
        from Util import aggregate_sum

        items_encoded = self.mlp(x)
        embeddings = aggregate_sum(items_encoded, idxs, n_bins)
        return embeddings


class FlatFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, static_dims=0, set_arch: Optional[Dict[str, int]] = None) -> None:
        super(FlatFeatureExtractor, self).__init__(observation_space, 1)

        if set_arch is None:
            set_arch = dict()

        total_features = 0

        input_encoders = {}

        for key, space in observation_space.spaces.items():
            if "_set" in key:
                max_obs, obs_features = space.shape

                arch = set_arch.get(key, [256, 256, 128])

                arch = [max_obs * obs_features] + arch

                layers = []
                for in_f, out_f in zip(arch[:-1], arch[1:]):
                    layers.append(torch.nn.Linear(in_f, out_f))
                    layers.append(torch.nn.ReLU())

                input_encoders[key] = torch.nn.Sequential(*layers)

                total_features += arch[-1]
            else:
                input_encoders[key], dims = _make_static_encoder(space, static_dims)
                total_features += dims

        self._features_dim = total_features
        self.input_encoders = torch.nn.ModuleDict(input_encoders)

    def forward(self, obs):
        result = []

        for key, encoder in self.input_encoders.items():
            if "_set" in key:
                result.append(encoder(obs[key].flatten(-2, -1)))
            else:
                result.append(encoder(obs[key]))

        return torch.cat(result, dim=-1)


def cnn_from_conf(conf, input_channels):
    from torch.nn import Conv2d, ReLU, Flatten, Sequential

    last_channels = input_channels

    model = []

    for out_channels, kernel, stride in conf:
        model.append(Conv2d(last_channels, out_channels, kernel, stride, 0))
        model.append(ReLU())
        last_channels = out_channels

    model.append(Flatten())
    return Sequential(*model)


class CnnFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dims: Optional[Dict[str, int]] = None,
                 static_dims: int = 0, conf=None):
        from torch import nn
        super(CnnFeaturesExtractor, self).__init__(observation_space, 1)

        if features_dims is None:
            features_dims = dict()

        if conf is None:
            # Nature
            conf = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
            # Huegle
            # conf = [(16, (1, 3), (1, 2)), (32, (1, 3), (1, 2))]

        total_features = 0

        input_encoders = {}

        for key, space in observation_space.spaces.items():
            if "_img" in key:
                # From SB3's NatureCNN (but we can't just use SB3's implementation which has a very strict understanding
                # of image spaces)
                features_dim = features_dims.get(key, 256)
                n_input_channels = space.shape[0]

                cnn = cnn_from_conf(conf, n_input_channels)

                # Compute shape by doing one forward pass
                with torch.no_grad():
                    n_flatten = cnn(torch.as_tensor(space.sample()[None]).float()).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
                input_encoders[key] = nn.Sequential(cnn, linear)

                total_features += features_dim
            else:
                input_encoders[key], dims = _make_static_encoder(space, static_dims)
                total_features += dims

        self._features_dim = total_features
        self.input_encoders = torch.nn.ModuleDict(input_encoders)

    def forward(self, obs):
        result = []

        for key, encoder in self.input_encoders.items():
            result.append(encoder(obs[key]))

        return torch.cat(result, dim=-1)
