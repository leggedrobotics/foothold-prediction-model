from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_JOINT_PAIRS = [
    (0, 1, 1.0),
    (3, 4, -1.0),
    (6, 7, -1.0),
    (9, 10, 1.0),
    (11, 12, 1.0),
    (13, 14, 1.0),
    (15, 16, -1.0),
    (17, 18, -1.0),
    (19, 20, -1.0),
    (21, 22, 1.0),
    (23, 24, -1.0),
    (25, 26, 1.0),
    (27, 28, -1.0),
]
_JOINT_SINGLES = [(2, -1.0), (5, -1.0), (8, 1.0)]

_JOINT_PERM = list(range(29))
_JOINT_SIGN = [1.0] * 29
for _a, _b, _s in _JOINT_PAIRS:
    _JOINT_PERM[_a], _JOINT_PERM[_b] = _b, _a
    _JOINT_SIGN[_a] = _JOINT_SIGN[_b] = _s
for _i, _s in _JOINT_SINGLES:
    _JOINT_SIGN[_i] = _s

_VECTOR_SIGNS = {
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
    sign = torch.tensor(_JOINT_SIGN, device=x.device, dtype=x.dtype)
    return x[..., _JOINT_PERM] * sign


def _mirror_height_scan(x: torch.Tensor, env: ManagerBasedRLEnv, term_cfg) -> torch.Tensor:
    pattern = env.scene[term_cfg.params["sensor_cfg"].name].cfg.pattern_cfg
    nx = int(round(pattern.size[0] / pattern.resolution)) + 1
    ny = int(round(pattern.size[1] / pattern.resolution)) + 1
    return x.view(-1, ny, nx).flip(dims=[1]).reshape(x.shape)


def _mirror_foothold_scan(x: torch.Tensor, term_cfg) -> torch.Tensor:
    k = term_cfg.params["patch_size"]
    stride = 2 + k * k
    feet = [x[:, :stride], x[:, stride:]]
    out = torch.empty_like(x)
    for i, src in enumerate(reversed(feet)):
        o = i * stride
        out[:, o] = src[:, 0]
        out[:, o + 1] = -src[:, 1]
        out[:, o + 2 : o + stride] = src[:, 2:].view(-1, k, k).flip(dims=[2]).reshape(-1, k * k)
    return out


def _mirror_group(env: ManagerBasedRLEnv, obs: torch.Tensor, group: str) -> torch.Tensor:
    obs_manager = env.observation_manager
    group_cfg = getattr(env.cfg.observations, group)
    out = obs.clone()
    idx = 0
    for name, shape in zip(obs_manager.active_terms[group], obs_manager.group_obs_term_dim[group]):
        dim = math.prod(shape)
        seg = obs[:, idx : idx + dim]
        if name in _VECTOR_SIGNS:
            out[:, idx : idx + dim] = seg * torch.tensor(_VECTOR_SIGNS[name], device=obs.device)
        elif name in _JOINT_TERMS:
            out[:, idx : idx + dim] = mirror_joints(seg)
        elif name == "height_scan":
            out[:, idx : idx + dim] = _mirror_height_scan(seg, env, getattr(group_cfg, name))
        elif name == "foothold_scan":
            out[:, idx : idx + dim] = _mirror_foothold_scan(seg, getattr(group_cfg, name))
        else:
            raise ValueError(f"No mirror rule for observation term '{name}' in group '{group}'.")
        idx += dim
    return out


def g1_symmetry_augmentation(
    env, obs: TensorDict | None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    env = env.unwrapped
    actions_aug = None if actions is None else torch.cat([actions, mirror_joints(actions)], dim=0)
    if obs is None:
        return None, actions_aug
    batch = obs.batch_size[0]
    aug = {}
    for key in obs.keys():
        tensor = obs[key]
        mirrored = _mirror_group(env, tensor, key) if key in ("policy", "critic") else tensor
        aug[key] = torch.cat([tensor, mirrored], dim=0)
    return TensorDict(aug, batch_size=[2 * batch], device=obs.device), actions_aug
