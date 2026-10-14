import isaaclab.terrains as terrain_gen

from .generator import SeededTerrainGenerator, SeededTerrainGeneratorCfg
from .terrains_cfg import (
    ConcentricGapTerrainCfg,
    CorridorTerrainCfg,
    InvertedPyramidStairsTerrainCfg,
    PalletsTerrainCfg,
    PitGapTerrainCfg,
    PyramidStairsTerrainCfg,
    SteppingStonesTerrainCfg,
)

TRAINING_TERRAINS_CFG = SeededTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.8,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": PyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.1, 0.25), step_width_range=(0.25, 0.3), platform_width=2.0, border_width=1.0
        ),
        "pyramid_stairs_inv": InvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.1, 0.25), step_width_range=(0.25, 0.3), platform_width=2.0, border_width=1.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.05, 0.3), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.025),
        "stepping_stones": SteppingStonesTerrainCfg(
            proportion=0.3,
            stone_height_limit=(-0.05, 0.05),
            stone_width_range=(0.5, 0.3),
            stone_distance_range=(0.1, 0.2),
            platform_width=2.0,
            holes_depth_range=(-0.3, -3.0),
        ),
        "pit": PitGapTerrainCfg(
            proportion=0.05,
            pit_depth_range=(0.1, 0.4),
            platform_height_range=(0.1, 0.6),
            gap_width_range=(0.1, 0.4),
            pit_width=3.5,
            ground_width=0.6,
            gap_depth=1.0,
            border_width=0.25,
        ),
    },
)

LOCOMOTION_TERRAIN_NAMES = {
    0: "pyramid_stairs",
    1: "pyramid_stairs_inv",
    2: "boxes",
    3: "random_rough",
    4: "hf_pyramid_slope",
    5: "hf_pyramid_slope_inv",
    6: "stepping_stones",
    7: "pallets",
    8: "wave",
    9: "discrete_obstacles",
    10: "random_uniform",
    11: "concentric_gaps",
    12: "flat",
}
LOCOMOTION_TERRAIN_CATEGORY = {
    **{name: "train" for name in list(LOCOMOTION_TERRAIN_NAMES.values())[:7]},
    **{name: "ood" for name in list(LOCOMOTION_TERRAIN_NAMES.values())[7:12]},
    "flat": "flat",
}
SPARSE_TERRAINS = ("stepping_stones", "concentric_gaps", "pallets")

LOCOMOTION_VALIDATION_TERRAINS_CFG = SeededTerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=0.0,
    num_rows=1,
    num_cols=13,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.8,
    curriculum=True,
    use_cache=False,
    seed=20260910,
    difficulty_range=(0.5, 0.5),
    sub_terrains={
        "train_pyramid_stairs": PyramidStairsTerrainCfg(
            proportion=1 / 13, step_height_range=(0.1, 0.25), step_width_range=(0.25, 0.3), platform_width=2.0, border_width=12.0
        ),
        "train_pyramid_stairs_inv": InvertedPyramidStairsTerrainCfg(
            proportion=1 / 13, step_height_range=(0.1, 0.25), step_width_range=(0.25, 0.3), platform_width=2.0, border_width=12.0
        ),
        "train_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1 / 13, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "train_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1 / 13, noise_range=(0.02, 0.10), noise_step=0.02
        ),
        "train_hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1 / 13, slope_range=(0.0, 0.11), platform_width=2.0
        ),
        "train_hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1 / 13, slope_range=(0.0, 0.11), platform_width=2.0
        ),
        "train_stepping_stones": SteppingStonesTerrainCfg(
            proportion=1 / 13,
            stone_height_limit=(0.0, 0.0),
            stone_width_range=(0.5, 0.3),
            stone_distance_range=(0.1, 0.2),
            platform_width=2.0,
            holes_depth_range=(-0.75, -3.0),
        ),
        "ood_pallets": PalletsTerrainCfg(
            proportion=1 / 13,
            pallet_height_range=(-0.05, 0.05),
            pallet_width_range=(0.25, 0.5),
            pallet_length_range=(0.25, 0.5),
            pallet_thickness=0.1,
            density=0.8,
            gap_distance=0.05,
            holes_depth=-1.0,
            platform_width=1.5,
        ),
        "ood_wave": terrain_gen.HfWaveTerrainCfg(proportion=1 / 13, amplitude_range=(0.05, 0.25), num_waves=4),
        "ood_obs": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1 / 13,
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(0.1, 0.2),
            num_obstacles=20,
            platform_width=1.5,
        ),
        "ood_rand": terrain_gen.HfRandomUniformTerrainCfg(proportion=1 / 13, noise_range=(0.05, 0.15), noise_step=0.02),
        "ood_concentric_gaps": ConcentricGapTerrainCfg(
            proportion=1 / 13,
            gap_width_range=(0.2, 0.35),
            ground_width_range=(0.3, 0.6),
            ground_height_max=0.05,
            platform_width=1.5,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1 / 13),
    },
)

CORRIDOR_WALLS = ("straight", "stones", "blocks")
CORRIDOR_FLOORS = (
    "stairs",
    "stairs_inv",
    "boxes",
    "rough",
    "slope",
    "slope_inv",
    "stepping_stones",
    "pallets",
    "wave",
    "obstacles",
    "noise",
    "gaps",
    "flat",
)
CORRIDOR_FLOOR_CATEGORY = {
    "stepping_stones": "sparse",
    "gaps": "sparse",
    "pallets": "sparse",
    "stairs": "stepped",
    "stairs_inv": "stepped",
    "boxes": "stepped",
    "obstacles": "stepped",
    "rough": "continuous",
    "slope": "continuous",
    "slope_inv": "continuous",
    "wave": "continuous",
    "noise": "continuous",
    "flat": "continuous",
}
CORRIDOR_TERRAIN_NAMES = {
    i * len(CORRIDOR_FLOORS) + j: f"{w}_{f}" for i, w in enumerate(CORRIDOR_WALLS) for j, f in enumerate(CORRIDOR_FLOORS)
}

CORRIDOR_TERRAINS_CFG = SeededTerrainGeneratorCfg(
    size=(8.05, 8.05),
    border_width=5.0,
    num_rows=1,
    num_cols=len(CORRIDOR_TERRAIN_NAMES),
    horizontal_scale=0.025,
    vertical_scale=0.005,
    slope_threshold=None,
    curriculum=True,
    use_cache=False,
    seed=20260907,
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        f"{w}_{f}": CorridorTerrainCfg(
            proportion=1.0 / len(CORRIDOR_TERRAIN_NAMES),
            wall_type=w,
            floor_variant=f,
            corridor_width_range=(1.30, 1.30),
            wall_height_range=(1.5, 2.5),
            platform_width=1.0,
            edge_taper=1.0,
            stone_wall_length=0.20,
            stone_wall_gap=0.10,
            stone_wall_height=2.0,
            wall_pit_depth=3.0,
            blocks_width=1.30,
            blocks_extra_width=0.20,
            block_length=0.30,
            block_height_range=(1.0, 1.5),
            stone_width=0.20,
            stone_gap_range=(0.20, 0.20),
            stone_gap_depth=3.0,
            box_height_range=(0.02, 0.20),
            box_width_range=(0.10, 0.25),
            rough_range=(0.02, 0.06),
            stair_rise_range=(0.05, 0.115),
            noise_range=(0.05, 0.15),
            noise_step=0.02,
            noise_cell=0.1,
            slope_range=(0.0, 0.11),
            wave_amplitude_range=(0.05, 0.25),
            wave_period=7.5,
            obstacle_width_range=(0.5, 1.0),
            obstacle_height_range=(0.1, 0.2),
            obstacle_spacing_range=(1.0, 2.0),
            pallet_size_range=(0.25, 0.5),
            pallet_height_range=(-0.05, 0.05),
            pallet_gap=0.05,
            pallet_pit_depth=1.0,
            gap_ground_width_range=(0.3, 0.6),
            gap_width_range=(0.2, 0.35),
            gap_depth=1.0,
            gap_ground_height_max=0.05,
        )
        for w in CORRIDOR_WALLS
        for f in CORRIDOR_FLOORS
    },
)
