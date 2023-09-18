import torch
from typing import Optional


def make_preencoder(which: str, input_dim, output_dim):
    if which == 'fc':
        return MlpPreEncoder(input_dim, output_dim)
    elif which.startswith('ffn_lin_'):
        return FfnPreencoder(input_dim, output_dim, b_scale=float(which.split('_')[-1]), act=None)
    else:
        raise ValueError(f"{which = }")


class MlpPreEncoder(torch.nn.Linear):
    def __init__(self, input_dim, f_dim):
        super(MlpPreEncoder, self).__init__(input_dim, f_dim)

    @property
    def output_dim(self) -> int:
        return self.out_features

    def forward(self, x):
        return torch.relu(super(MlpPreEncoder, self).forward(x))


class FfnPreencoder(torch.nn.Linear):
    def __init__(self, input_dim, output_dim, b_scale=0.001, act: Optional[callable] = None):
        self.b_scale = b_scale
        self.act = act
        super(FfnPreencoder, self).__init__(input_dim, output_dim)

    @property
    def output_dim(self) -> int:
        return self.out_features

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, 0., self.b_scale / self.in_features)
        torch.nn.init.uniform_(self.bias, -1., 1.)

    def forward(self, x):
        x = torch.pi * super(FfnPreencoder, self).forward(x)
        x = torch.sin(x)

        if self.act is None:
            pass
        else:
            x = self.act(x)

        return x
