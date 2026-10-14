from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING

import torch

from .trainer import fit_foot_predictor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_MODEL_PATTERN = re.compile(r"^foot_model_(\d+)\.pt$")


def latest_foot_model(directory: str | None) -> str | None:
    if not directory or not os.path.isdir(directory):
        return None
    found = [(int(m.group(1)), f) for f in os.listdir(directory) if (m := _MODEL_PATTERN.match(f))]
    return os.path.join(directory, max(found)[1]) if found else None


class FootPredictionSupervisor:
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        train: bool = True,
        num_steps_per_iter: int = 24,
        bootstrap_iters: int = 500,
        refit_iters: int = 500,
        num_samples: int = 500_000,
        train_kwargs: dict | None = None,
    ):
        self.train = train
        self.num_steps_per_iter = num_steps_per_iter
        self.bootstrap_steps = bootstrap_iters * num_steps_per_iter
        self.refit_steps = refit_iters * num_steps_per_iter
        self.record_steps = math.ceil(num_samples / env.num_envs)
        self.train_kwargs = train_kwargs or {}
        self.device = env.device

        self.log_dir = getattr(env.cfg, "log_dir", None) or os.getcwd()
        self.load_run_dir = getattr(env.cfg, "load_run_dir", None)
        self.model_path = latest_foot_model(self.log_dir) or latest_foot_model(self.load_run_dir)
        self.model_ready = self.model_path is not None
        self.warm_started = self.model_ready

        self.step_count = 0
        self.last_sim_step = -1
        self.buffers = None
        print(
            f"[FPM] log dir {self.log_dir}, model {self.model_path}, "
            f"bootstrap {self.bootstrap_steps} steps, record {self.record_steps} steps every {self.refit_steps}"
        )

    def pre_step(self, env: ManagerBasedRLEnv):
        if not self.train or env._sim_step_counter == self.last_sim_step:
            return
        self.last_sim_step = env._sim_step_counter
        self.step_count += 1
        if self.step_count <= self.bootstrap_steps:
            return
        phase = (self.step_count - self.bootstrap_steps) % self.refit_steps
        if phase < self.record_steps:
            if self.buffers is None:
                print(f"[FPM] recording at iteration {self.step_count // self.num_steps_per_iter}")
                self.buffers = {"system_state": [], "system_contact": [], "system_pose": []}
            self._record(env)
        elif phase == self.record_steps and self.buffers is not None:
            self._refit()

    def _record(self, env: ManagerBasedRLEnv):
        obs_manager = env.observation_manager
        cached = getattr(obs_manager, "_obs_buffer", None) or {}
        for name, buf in self.buffers.items():
            value = cached[name] if name in cached else obs_manager.compute_group(name)
            buf.append(value.clone().cpu())

    def _refit(self):
        hist = {name: torch.stack(buf, dim=1) for name, buf in self.buffers.items()}
        self.buffers = None
        iteration = self.step_count // self.num_steps_per_iter
        save_path = os.path.join(self.log_dir, f"foot_model_{iteration}.pt")
        print(f"[FPM] refitting on {hist['system_state'].shape[:2]} (envs, steps)")
        fit_foot_predictor(
            hist["system_state"],
            hist["system_contact"].float(),
            hist["system_pose"],
            save_path=save_path,
            device=self.device,
            **self.train_kwargs,
        )
        self.model_path = save_path
        self.model_ready = True

    def use_bootstrap_footholds(self) -> bool:
        if not self.model_ready:
            return True
        if self.warm_started:
            return False
        return self.step_count < self.bootstrap_steps
