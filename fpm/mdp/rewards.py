from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stand_still_nominal_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) < threshold) & (cmd[:, 2].abs() < threshold)
    error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.exp(-error / std) * standing.float()


def action_rate_l2_bounded(env: ManagerBasedRLEnv, max_delta: float = 5.0) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    delta = torch.nan_to_num(delta, nan=max_delta, posinf=max_delta, neginf=-max_delta).clamp(-max_delta, max_delta)
    return torch.sum(torch.square(delta), dim=1)


def upright_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.72,
    std: float = 0.06,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lowest_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].min(dim=1).values
    deficit = (target_height - (asset.data.root_pos_w[:, 2] - lowest_foot_z)).clamp(min=0.0)
    return torch.exp(-torch.square(deficit) / std**2)
