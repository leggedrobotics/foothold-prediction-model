from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def assist_fraction(levels: torch.Tensor, init_offset: float, cutoff_level: float) -> torch.Tensor:
    max_assist = 1.0 - float(init_offset)
    alpha = max_assist * (1.0 - levels.float() / float(cutoff_level))
    return torch.clamp(alpha, min=0.0, max=max_assist)


def _terrain_levels(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    shared = env.curriculum_manager._curriculum_state.get("terrain_levels", None)
    if shared is not None:
        level = float(shared.item()) if torch.is_tensor(shared) else float(shared)
        return torch.full((len(env_ids),), level, device=env.device)
    return env.scene.terrain.terrain_levels[env_ids].to(env.device)


def apply_gravity_compensation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    init_offset: float = 0.5,
    cutoff_level: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)

    alpha = assist_fraction(_terrain_levels(env, env_ids), init_offset, cutoff_level)
    mass = asset.data.default_mass.to(asset.device)
    if mass.dim() == 3:
        mass = mass[..., 0]
    force = alpha * mass.sum(dim=1)[env_ids] * abs(float(env.sim.cfg.gravity[2]))

    if isinstance(asset_cfg.body_ids, (list, torch.Tensor)):
        num_bodies = len(asset_cfg.body_ids)
    else:
        num_bodies = asset.num_bodies
    up_w = torch.zeros((len(env_ids), 3), device=asset.device)
    up_w[:, 2] = force / float(num_bodies)
    up_b = math_utils.quat_apply_inverse(asset.data.root_quat_w[env_ids], up_w)
    forces = up_b.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
    asset.set_external_force_and_torque(
        forces=forces, torques=torch.zeros_like(forces), env_ids=env_ids, body_ids=asset_cfg.body_ids, is_global=False
    )


def gravity_offset_level(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, init_offset: float = 0.5, cutoff_level: float = 4.0
) -> torch.Tensor:
    ids = torch.arange(env.scene.num_envs, device=env.device)
    return torch.mean(assist_fraction(_terrain_levels(env, ids), init_offset, cutoff_level)).unsqueeze(0)


def randomize_joint_default_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)
    default = asset.data.default_joint_pos
    offsets = math_utils.sample_uniform(
        position_range[0], position_range[1], (len(env_ids), default.shape[1]), default.device
    )
    default[env_ids] += offsets
    limits = asset.data.soft_joint_pos_limits
    asset.data.default_joint_pos[env_ids] = torch.clamp(
        default[env_ids], limits[env_ids, :, 0], limits[env_ids, :, 1]
    )
