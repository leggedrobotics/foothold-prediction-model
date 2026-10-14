import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _finite(x: torch.Tensor) -> torch.Tensor:
    return x if x.isfinite().all() else torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class FiniteRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def get_observations(self) -> TensorDict:
        return super().get_observations().apply(_finite)

    def reset(self):
        obs, extras = super().reset()
        return obs.apply(_finite), extras

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)
        return obs.apply(_finite), _finite(rew), dones, extras
