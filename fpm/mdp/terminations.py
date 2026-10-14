from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def nan_physics_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    bad = ~torch.isfinite(asset.data.root_pos_w).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.root_lin_vel_b).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.joint_pos).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.joint_vel).all(dim=-1)
    return bad


def displacement_from_origin(
    env: ManagerBasedRLEnv, distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1) >= distance
