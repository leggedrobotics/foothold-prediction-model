import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import fpm  # noqa: F401
from fpm.foot_pred import latest_foot_model
from fpm.vecenv import FiniteRslRlVecEnvWrapper


def make_env_and_policy(task: str, env_cfg, agent_cfg, checkpoint: str, device: str | None = None):
    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    env_cfg.log_dir = run_dir
    env_cfg.load_run_dir = run_dir
    if device is not None:
        env_cfg.sim.device = device
        agent_cfg.device = device
    print(f"[INFO] policy {checkpoint}\n[INFO] FPM {latest_foot_model(run_dir)}")
    env = FiniteRslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return env, runner.get_inference_policy(device=env.unwrapped.device), run_dir


def terrain_column(env) -> torch.Tensor:
    u = env.unwrapped
    centers = u.scene.terrain.terrain_origins.mean(dim=0)[:, :2].to(u.device)
    return torch.cdist(u.scene["robot"].data.root_pos_w[:, :2], centers).argmin(dim=-1)
