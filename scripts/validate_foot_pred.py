"""Foothold prediction error per prediction step on the training and unseen terrains."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="FPM-G1-Validate-FootPred-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--num_steps", type=int, default=1100)
parser.add_argument("--save_results", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

import torch
from common import make_env_and_policy, terrain_column

from isaaclab_tasks.utils.hydra import hydra_task_config

from fpm.foot_pred import compute_strictly_future_contact_positions
from fpm.mdp import current_foot_model
from fpm.terrains import LOCOMOTION_TERRAIN_CATEGORY, LOCOMOTION_TERRAIN_NAMES


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env, policy, run_dir = make_env_and_policy(args_cli.task, env_cfg, agent_cfg, args_cli.checkpoint, args_cli.device)
    u = env.unwrapped
    foot_model = current_foot_model()
    if foot_model is None:
        raise FileNotFoundError(f"No foot_model_*.pt in {run_dir}")

    preds, contacts, poses, resets, terrains = [], [], [], [], []
    obs = env.get_observations()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            buffer = u.observation_manager._obs_buffer
            preds.append(foot_model(buffer["system_state"]).view(u.num_envs, 2, 2).cpu())
        contacts.append(buffer["system_contact"].cpu())
        poses.append(buffer["system_pose"].cpu())
        resets.append(dones.float().cpu())
        terrains.append(terrain_column(env).cpu())
        if (step + 1) % 100 == 0:
            print(f"[INFO] step {step + 1}/{args_cli.num_steps}")

    pred = torch.stack(preds, dim=1)[:, :-1]
    contact = torch.stack(contacts, dim=1)[:, 1:]
    pose = torch.stack(poses, dim=1)[:, 1:]
    reset = torch.stack(resets, dim=1)[:, :-1]
    terrain = torch.stack(terrains, dim=1)[:, :-1]
    target = compute_strictly_future_contact_positions(contact, pose[..., :7], pose[..., 7:21], reset)

    err = torch.cat(
        [torch.norm(pred[..., 0, :] - target[..., 0:2], dim=-1), torch.norm(pred[..., 1, :] - target[..., 3:5], dim=-1)]
    )
    stance = torch.cat([contact[..., 0], contact[..., 1]]) > 0.5
    valid = torch.cat([reset, reset]) < 0.5
    terrain = torch.cat([terrain, terrain])

    groups = {
        "in-distribution": [t for t, nm in LOCOMOTION_TERRAIN_NAMES.items() if LOCOMOTION_TERRAIN_CATEGORY[nm] != "ood"],
        "unseen": [t for t, nm in LOCOMOTION_TERRAIN_NAMES.items() if LOCOMOTION_TERRAIN_CATEGORY[nm] == "ood"],
        "all": list(LOCOMOTION_TERRAIN_NAMES),
    }
    phases = {"step 1 (air)": ~stance, "step 2 (stance)": stance, "both": torch.ones_like(stance)}

    def mean_cm(mask):
        return 100.0 * err[mask].mean().item() if mask.any() else float("nan")

    print(f"\n{'terrain':<24}" + "".join(f"{p:>18}" for p in phases) + "   foothold error, cm")
    for t, name in LOCOMOTION_TERRAIN_NAMES.items():
        print(f"{name:<24}" + "".join(f"{mean_cm(valid & m & (terrain == t)):>18.2f}" for m in phases.values()))
    print("-" * 78)
    results = {}
    for label, ids in groups.items():
        in_group = torch.isin(terrain, torch.tensor(ids))
        results[label] = {p: mean_cm(valid & m & in_group) for p, m in phases.items()}
        print(f"{'[' + label + ']':<24}" + "".join(f"{v:>18.2f}" for v in results[label].values()))

    if args_cli.save_results:
        path = os.path.join(run_dir, "foot_pred_validation.pt")
        torch.save({"results_cm": results, "err": err, "stance": stance, "valid": valid, "terrain": terrain}, path)
        print(f"[INFO] saved {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
