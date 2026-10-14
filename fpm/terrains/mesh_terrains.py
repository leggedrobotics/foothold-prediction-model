from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from . import terrains_cfg


def _pyramid_stairs(difficulty: float, cfg, inverted: bool) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_width = np.random.uniform(cfg.step_width_range[0], cfg.step_width_range[1])

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    meshes = []
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes += make_border(cfg.size, border_inner, step_height, border_center)

    center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        box_size = (size[0] - 2 * k * step_width, size[1] - 2 * k * step_width)
        if inverted:
            box_z = center[2] - total_height / 2 - (k + 1) * step_height / 2.0
            box_height = total_height - (k + 1) * step_height
        else:
            box_z = center[2] + k * step_height / 2.0
            box_height = (k + 2) * step_height
        offset = (k + 0.5) * step_width
        dims_tb = (box_size[0], step_width, box_height)
        dims_lr = (step_width, box_size[1] - 2 * step_width, box_height)
        for dims, pos in (
            (dims_tb, (center[0], center[1] + size[1] / 2.0 - offset, box_z)),
            (dims_tb, (center[0], center[1] - size[1] / 2.0 + offset, box_z)),
            (dims_lr, (center[0] + size[0] / 2.0 - offset, center[1], box_z)),
            (dims_lr, (center[0] - size[0] / 2.0 + offset, center[1], box_z)),
        ):
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

    middle_xy = (size[0] - 2 * num_steps * step_width, size[1] - 2 * num_steps * step_width)
    if inverted:
        middle = (*middle_xy, step_height)
        middle_pos = (center[0], center[1], center[2] - total_height - step_height / 2)
        origin_z = -(num_steps + 1) * step_height
    else:
        middle = (*middle_xy, (num_steps + 2) * step_height)
        middle_pos = (center[0], center[1], center[2] + num_steps * step_height / 2)
        origin_z = (num_steps + 1) * step_height
    meshes.append(trimesh.creation.box(middle, trimesh.transformations.translation_matrix(middle_pos)))
    return meshes, np.array([center[0], center[1], origin_z])


def pyramid_stairs_terrain(difficulty: float, cfg: terrains_cfg.PyramidStairsTerrainCfg):
    return _pyramid_stairs(difficulty, cfg, inverted=False)


def inverted_pyramid_stairs_terrain(difficulty: float, cfg: terrains_cfg.PyramidStairsTerrainCfg):
    return _pyramid_stairs(difficulty, cfg, inverted=True)


def pallets_terrain(difficulty: float, cfg: terrains_cfg.PalletsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    width_hi = cfg.pallet_width_range[1] - difficulty * (cfg.pallet_width_range[1] - cfg.pallet_width_range[0])
    length_hi = cfg.pallet_length_range[1] - difficulty * (cfg.pallet_length_range[1] - cfg.pallet_length_range[0])
    width_lo = min(cfg.pallet_width_range[0], width_hi)
    length_lo = min(cfg.pallet_length_range[0], length_hi)

    meshes = []
    floor_thick = 0.5
    floor_pose = np.eye(4)
    floor_pose[:3, 3] = [cfg.size[0] / 2, cfg.size[1] / 2, cfg.holes_depth - floor_thick / 2.0]
    meshes.append(trimesh.creation.box([cfg.size[0], cfg.size[1], floor_thick], transform=floor_pose))

    plat_h = max(abs(cfg.holes_depth), 0.1)
    plat_pose = np.eye(4)
    plat_pose[:3, 3] = [cfg.size[0] / 2, cfg.size[1] / 2, -plat_h / 2.0]
    meshes.append(trimesh.creation.box([cfg.platform_width, cfg.platform_width, plat_h], transform=plat_pose))

    gap = max(cfg.gap_distance, 0.0)
    safe_zone = cfg.platform_width / 2.0

    cells = []
    y = 0.0
    while y < cfg.size[1]:
        cell_l = float(np.random.uniform(length_lo, length_hi))
        if y + cell_l > cfg.size[1]:
            break
        x = 0.0
        while x < cfg.size[0]:
            cell_w = float(np.random.uniform(width_lo, width_hi))
            if x + cell_w > cfg.size[0]:
                break
            cells.append((x, y, cell_w, cell_l))
            x += cell_w + gap
        y += cell_l + gap

    def clear_of_platform(x, y, w, l):
        cx, cy = x + w / 2.0, y + l / 2.0
        return not (
            abs(cx - cfg.size[0] / 2) < safe_zone + w / 2.0 + gap and abs(cy - cfg.size[1] / 2) < safe_zone + l / 2.0 + gap
        )

    cells = [c for c in cells if clear_of_platform(*c)]

    target_area = max(cfg.density, 0.0) * (cfg.size[0] * cfg.size[1] - cfg.platform_width**2)
    covered = 0.0
    for idx in np.random.permutation(len(cells)):
        if covered >= target_area:
            break
        x, y, cell_w, cell_l = cells[idx]
        shrink = float(np.random.uniform(0.85, 1.0))
        w = max(cell_w * shrink, 1e-3)
        l = max(cell_l * shrink, 1e-3)

        theta = 0.0
        for cand in np.deg2rad(np.arange(20.0, 0.0, -2.0)):
            ct, st = abs(np.cos(cand)), abs(np.sin(cand))
            if w * ct + l * st <= cell_w and w * st + l * ct <= cell_l:
                theta = float(np.random.uniform(-cand, cand))
                break

        ct, st = abs(np.cos(theta)), abs(np.sin(theta))
        ext_x, ext_y = w * ct + l * st, w * st + l * ct
        cx = x + ext_x / 2.0 + float(np.random.uniform(0.0, max(cell_w - ext_x, 0.0)))
        cy = y + ext_y / 2.0 + float(np.random.uniform(0.0, max(cell_l - ext_y, 0.0)))
        h = float(np.random.uniform(cfg.pallet_height_range[0], cfg.pallet_height_range[1]))

        transform = np.eye(4)
        transform[:3, 3] = [cx, cy, h - cfg.pallet_thickness / 2.0]
        transform = transform @ trimesh.transformations.rotation_matrix(theta, [0, 0, 1])
        meshes.append(trimesh.creation.box([w, l, cfg.pallet_thickness], transform=transform))
        covered += w * l

    return meshes, np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
