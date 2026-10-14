from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from ..foot_pred import FootPredictionSupervisor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class _FootholdState:
    supervisor: FootPredictionSupervisor | None = None
    model: torch.jit.ScriptModule | None = None
    model_path: str | None = None
    model_mtime: float = 0.0
    xy_drift: torch.Tensor | None = None
    z_drift: torch.Tensor | None = None


_STATE = _FootholdState()


def _sample_uniform(shape, value_range, device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (value_range[1] - value_range[0]) + value_range[0]


def _reload_model(env: ManagerBasedEnv):
    path = _STATE.supervisor.model_path
    if not path or not os.path.exists(path):
        return
    mtime = os.path.getmtime(path)
    if _STATE.model is None or path != _STATE.model_path or mtime > _STATE.model_mtime:
        print(f"[FPM] loading {path}")
        _STATE.model = torch.jit.load(path, map_location=env.device).eval()
        _STATE.model_path = path
        _STATE.model_mtime = mtime


def _predict_footholds(env: ManagerBasedEnv, use_model: bool, bootstrap_footholds) -> torch.Tensor:
    if use_model:
        cached = getattr(getattr(env, "observation_manager", None), "_obs_buffer", None)
        if cached is not None and "system_state" in cached:
            state = cached["system_state"]
        else:
            state = torch.zeros(env.num_envs, 12 + 2 * env.scene["robot"].num_joints, device=env.device)
        with torch.inference_mode():
            return _STATE.model(state).view(env.num_envs, 2, 2)
    return torch.tensor(bootstrap_footholds, device=env.device).expand(env.num_envs, 2, 2)


def foothold_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scan_resolution: float = 0.05,
    scan_size: tuple[float, float] = (1.6, 1.0),
    patch_size: int = 7,
    xy_drift_range: tuple[float, float] = (0.0, 0.0),
    z_drift_range: tuple[float, float] = (0.0, 0.0),
    bootstrap_footholds: tuple = ((0.0, 0.1), (0.0, -0.1)),
    train_fpm: bool = True,
    force_model: bool = False,
    fpm_kwargs: dict | None = None,
) -> torch.Tensor:
    if _STATE.supervisor is None:
        _STATE.supervisor = FootPredictionSupervisor(env, train=train_fpm, **(fpm_kwargs or {}))
    _STATE.supervisor.pre_step(env)
    _reload_model(env)

    use_model = _STATE.model is not None and (force_model or not _STATE.supervisor.use_bootstrap_footholds())
    preds_b = _predict_footholds(env, use_model, bootstrap_footholds)

    num_envs, device = env.num_envs, env.device
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if _STATE.xy_drift is None or _STATE.xy_drift.shape[0] != num_envs:
        _STATE.xy_drift = _sample_uniform((num_envs, 2), xy_drift_range, device)
    if len(reset_ids) > 0 and tuple(xy_drift_range) != (0.0, 0.0):
        _STATE.xy_drift[reset_ids] = _sample_uniform((len(reset_ids), 2), xy_drift_range, device)

    asset: Articulation = env.scene["robot"]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    feet_b = torch.cat([preds_b, torch.zeros(num_envs, 2, 1, device=device)], dim=-1)
    root_quat = asset.data.root_quat_w.unsqueeze(1).repeat(1, 2, 1)
    feet_w = asset.data.root_pos_w.unsqueeze(1) + math_utils.quat_apply(root_quat, feet_b)
    sensor_quat = sensor.data.quat_w.unsqueeze(1).repeat(1, 2, 1)
    feet_s = math_utils.quat_apply(math_utils.quat_inv(sensor_quat), feet_w - sensor.data.pos_w.unsqueeze(1))
    feet_s[..., :2] += _STATE.xy_drift.unsqueeze(1)

    hit_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    height_map = sensor.data.pos_w[:, 2].unsqueeze(1) - hit_z - 0.5

    num_x = int(scan_size[0] / scan_resolution) + 1
    num_y = int(scan_size[1] / scan_resolution) + 1
    origin_x, origin_y = -scan_size[0] / 2.0, -scan_size[1] / 2.0
    half = patch_size // 2
    k_range = torch.arange(-half, half + (patch_size % 2), device=device)
    off_x, off_y = torch.meshgrid(k_range, k_range, indexing="ij")
    off_x, off_y = off_x.flatten(), off_y.flatten()

    outputs = []
    for i in range(2):
        anchor_x = torch.round((feet_s[:, i, 0] - origin_x) / scan_resolution).long()
        anchor_y = torch.round((feet_s[:, i, 1] - origin_y) / scan_resolution).long()
        sample_x = torch.clamp(anchor_x.unsqueeze(1) + off_x, 0, num_x - 1)
        sample_y = torch.clamp(anchor_y.unsqueeze(1) + off_y, 0, num_y - 1)
        heights = torch.gather(height_map, 1, sample_y * num_x + sample_x)
        outputs.append(torch.cat([preds_b[:, i, :], heights], dim=1))
    obs = torch.cat(outputs, dim=1)

    z_on = tuple(z_drift_range) != (0.0, 0.0)
    if _STATE.z_drift is None or _STATE.z_drift.shape[0] != num_envs:
        _STATE.z_drift = (
            _sample_uniform((num_envs, 1), z_drift_range, device) if z_on else torch.zeros(num_envs, 1, device=device)
        )
    if len(reset_ids) > 0 and z_on:
        _STATE.z_drift[reset_ids] = _sample_uniform((len(reset_ids), 1), z_drift_range, device)
    if z_on:
        n = patch_size**2
        obs[:, 2 : 2 + n] += _STATE.z_drift
        obs[:, n + 4 : 2 * n + 4] += _STATE.z_drift
    return obs


def current_foot_model() -> torch.jit.ScriptModule | None:
    return _STATE.model


def current_foot_model_path() -> str | None:
    return _STATE.model_path


def body_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold).float()


def base_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_pos_w, asset.data.root_quat_w], dim=-1)


def body_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose = torch.cat(
        [asset.data.body_pos_w[:, asset_cfg.body_ids], asset.data.body_quat_w[:, asset_cfg.body_ids]], dim=-1
    )
    return pose.view(pose.shape[0], -1)


def body_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
    return math_utils.quat_apply_inverse(quat, rel).view(rel.shape[0], -1)
