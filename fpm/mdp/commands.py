from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
    cfg: TurnInPlaceVelocityCommandCfg

    def __init__(self, cfg: TurnInPlaceVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_turning_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.is_turning_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turning_envs

    def _update_command(self):
        super()._update_command()
        turning_ids = self.is_turning_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[turning_ids, :2] = 0.0


@configclass
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TurnInPlaceVelocityCommand
    rel_turning_envs: float = 0.0
