import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


def uniform_spike_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    result = data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    mask = torch.rand_like(data) < cfg.spike_probability
    signs = (torch.rand_like(data) < 0.5).float() * 2.0 - 1.0
    return result + mask * signs * cfg.spike_amplitude


@configclass
class UniformSpikeNoiseCfg(NoiseCfg):
    func = uniform_spike_noise
    n_min: float = -1.0
    n_max: float = 1.0
    spike_probability: float = 0.0001
    spike_amplitude: float = 1.0
