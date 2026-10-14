import random

import numpy as np
import torch

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass


class SeededTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        super().__init__(cfg, device)

    def _generate_curriculum_terrains(self):
        if self.cfg.seed is None:
            return super()._generate_curriculum_terrains()

        states = (np.random.get_state(), random.getstate(), torch.get_rng_state())
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_indices = np.array(
            [
                np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                for index in range(self.cfg.num_cols)
            ],
            dtype=np.int32,
        )
        sub_cfgs = list(self.cfg.sub_terrains.values())
        lower, upper = self.cfg.difficulty_range
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                tile_seed = (int(self.cfg.seed) * 6364136223846793005 + sub_col * 1000003 + sub_row * 10007) % (2**32)
                np.random.seed(tile_seed)
                random.seed(tile_seed)
                torch.manual_seed(tile_seed)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfgs[sub_indices[sub_col]])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfgs[sub_indices[sub_col]])

        np.random.set_state(states[0])
        random.setstate(states[1])
        torch.set_rng_state(states[2])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@configclass
class SeededTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = SeededTerrainGenerator
