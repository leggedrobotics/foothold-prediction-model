"""Play a trained policy with its FPM, and export both for deployment."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="FPM-G1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--joystick", action="store_true", help="Command the robots with a gamepad.")
parser.add_argument("--show_footholds", action="store_true", help="Draw the predicted footholds.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils.hydra import hydra_task_config

import fpm  # noqa: F401
from fpm.foot_pred import latest_foot_model
from fpm.mdp import current_foot_model_path
from fpm.vecenv import FiniteRslRlVecEnvWrapper


def export(runner: OnPolicyRunner, export_dir: str, example_state: torch.Tensor):
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    foot_model = torch.jit.load(current_foot_model_path(), map_location="cpu").eval()
    torch.onnx.export(
        foot_model,
        example_state[:1].cpu(),
        os.path.join(export_dir, "foot_model.onnx"),
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["pred"],
        dynamic_axes={"obs": {0: "batch_size"}, "pred": {0: "batch_size"}},
    )
    print(f"[INFO] exported policy and FPM to {export_dir}")


def foothold_positions_w(env, policy_obs: torch.Tensor, patch_size: int = 7) -> torch.Tensor:
    robot, sensor = env.unwrapped.scene["robot"], env.unwrapped.scene["height_scanner"]
    stride = 2 + patch_size**2
    scan = policy_obs[:, -2 * stride :]
    points = []
    for i in range(2):
        foot = scan[:, i * stride : (i + 1) * stride]
        p_b = torch.cat([foot[:, :2], torch.zeros_like(foot[:, :1])], dim=-1)
        p_w = robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, p_b)
        p_w[:, 2] = sensor.data.pos_w[:, 2] - 0.5 - foot[:, 2 + patch_size**2 // 2]
        points.append(p_w)
    return torch.cat(points, dim=0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    run_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    env_cfg.log_dir = run_dir
    env_cfg.load_run_dir = run_dir
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if not args_cli.joystick:
        env_cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    print(f"[INFO] policy {args_cli.checkpoint}\n[INFO] FPM {latest_foot_model(run_dir)}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(run_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = FiniteRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    obs = env.get_observations()
    export(runner, os.path.join(run_dir, "exported"), obs["system_state"])

    gamepad = None
    if args_cli.joystick:
        from isaaclab.devices import Se2Gamepad, Se2GamepadCfg

        gamepad = Se2Gamepad(
            Se2GamepadCfg(v_x_sensitivity=1.0, v_y_sensitivity=0.5, omega_z_sensitivity=1.0, sim_device=u.device)
        )
    markers = None
    if args_cli.show_footholds:
        markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/PredictedFootholds",
                markers={
                    "foothold": sim_utils.SphereCfg(
                        radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
                    )
                },
            )
        )
    command = u.command_manager.get_term("base_velocity")

    step = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            if gamepad is not None:
                command.vel_command_b[:] = gamepad.advance().to(u.device).unsqueeze(0)
            obs, _, _, _ = env.step(policy(obs))
            if markers is not None:
                markers.visualize(translations=foothold_positions_w(env, obs["policy"]))
        step += 1
        if args_cli.video and step == args_cli.video_length:
            break
        sleep = u.step_dt - (time.time() - start)
        if args_cli.real_time and sleep > 0:
            time.sleep(sleep)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
