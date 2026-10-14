from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains, mesh_terrains


@configclass
class PyramidStairsTerrainCfg(SubTerrainBaseCfg):
    function = mesh_terrains.pyramid_stairs_terrain
    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    platform_width: float = 1.0


@configclass
class InvertedPyramidStairsTerrainCfg(PyramidStairsTerrainCfg):
    function = mesh_terrains.inverted_pyramid_stairs_terrain


@configclass
class PalletsTerrainCfg(SubTerrainBaseCfg):
    function = mesh_terrains.pallets_terrain
    pallet_height_range: tuple[float, float] = MISSING
    pallet_width_range: tuple[float, float] = MISSING
    pallet_length_range: tuple[float, float] = MISSING
    pallet_thickness: float = 0.05
    density: float = MISSING
    holes_depth: float = -1.0
    gap_distance: float = 0.0
    platform_width: float = 1.0


@configclass
class SteppingStonesTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.stepping_stones_terrain
    stone_height_limit: tuple[float, float] = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth_range: tuple[float, float] = MISSING
    platform_width: float = 1.0


@configclass
class PitGapTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.pit_gap_terrain
    pit_depth_range: tuple[float, float] = MISSING
    platform_height_range: tuple[float, float] | None = None
    gap_width_range: tuple[float, float] = MISSING
    pit_width: float = 3.5
    ground_width: float = 0.6
    gap_depth: float = 1.0


@configclass
class ConcentricGapTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.concentric_gap_terrain
    gap_width_range: tuple[float, float] = MISSING
    ground_width_range: tuple[float, float] = MISSING
    ground_height_max: float = MISSING
    platform_width: float = 1.0


@configclass
class CorridorTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.corridor_terrain

    wall_type: str = "straight"
    floor_variant: str = "flat"
    corridor_width_range: tuple[float, float] = (2.0, 1.0)
    platform_width: float = 1.0
    edge_taper: float = 1.0

    wall_height_range: tuple[float, float] = (1.5, 2.5)
    stone_wall_length: float = 0.20
    stone_wall_gap: float = 0.10
    stone_wall_height: float = 2.0
    wall_pit_depth: float = 3.0
    blocks_width: float = 1.5
    blocks_extra_width: float = 0.20
    block_length: float = 0.30
    block_height_range: tuple[float, float] = (1.0, 1.5)

    rough_range: tuple[float, float] = (0.02, 0.10)
    rough_step: float = 0.005
    box_width_range: tuple[float, float] = (0.10, 0.25)
    box_height_range: tuple[float, float] = (0.02, 0.25)
    box_height_step: float = 0.02
    stone_width: float = 0.20
    stone_gap_range: tuple[float, float] = (0.20, 0.20)
    stone_gap_depth: float = 3.0
    stair_rise_range: tuple[float, float] = (0.05, 0.18)
    stair_run: float = 0.30
    stair_max_rise: float = 0.60
    noise_range: tuple[float, float] = (0.05, 0.15)
    noise_step: float = 0.02
    noise_cell: float = 0.1
    slope_range: tuple[float, float] = (0.0, 0.11)
    wave_amplitude_range: tuple[float, float] = (0.05, 0.25)
    wave_period: float = 7.5
    obstacle_width_range: tuple[float, float] = (0.5, 1.0)
    obstacle_height_range: tuple[float, float] = (0.1, 0.2)
    obstacle_spacing_range: tuple[float, float] = (1.0, 2.0)
    pallet_size_range: tuple[float, float] = (0.25, 0.5)
    pallet_height_range: tuple[float, float] = (-0.05, 0.05)
    pallet_gap: float = 0.05
    pallet_pit_depth: float = 1.0
    gap_ground_width_range: tuple[float, float] = (0.3, 0.6)
    gap_width_range: tuple[float, float] = (0.2, 0.35)
    gap_depth: float = 1.0
    gap_ground_height_max: float = 0.05
