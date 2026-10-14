from __future__ import annotations

import copy
from collections import defaultdict
from random import randint
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh, height_field_to_mesh

if TYPE_CHECKING:
    from . import terrains_cfg


def _center_square(hf: np.ndarray, width_px: int, value: float = 0.0):
    w, l = hf.shape
    hf[(w - width_px) // 2 : (w + width_px) // 2, (l - width_px) // 2 : (l + width_px) // 2] = value


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: terrains_cfg.SteppingStonesTerrainCfg) -> np.ndarray:
    def lerp(r):
        return r[0] + difficulty * (r[1] - r[0])

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    stone_px = int(lerp(cfg.stone_width_range) / cfg.horizontal_scale)
    gap_px = int(lerp(cfg.stone_distance_range) / cfg.horizontal_scale)
    max_height_px = int(lerp(cfg.stone_height_limit) / cfg.vertical_scale)
    holes_px = int(lerp(cfg.holes_depth_range) / cfg.vertical_scale)
    heights = np.arange(-max_height_px, max_height_px + 1) if max_height_px > 0 else np.array([0])

    hf = np.full((width_px, length_px), holes_px)
    if length_px >= width_px:
        start_y = 0
        while start_y < length_px:
            stop_y = min(length_px, start_y + stone_px)
            start_x = np.random.randint(0, stone_px)
            stop_x = max(0, start_x - gap_px)
            if stop_x > 0:
                hf[0:stop_x, start_y:stop_y] = np.random.choice(heights)
            while start_x < width_px:
                stop_x = min(width_px, start_x + stone_px)
                hf[start_x:stop_x, start_y:stop_y] = np.random.choice(heights)
                start_x += stone_px + gap_px
            start_y += stone_px + gap_px
    else:
        start_x = 0
        while start_x < width_px:
            stop_x = min(width_px, start_x + stone_px)
            start_y = np.random.randint(0, stone_px)
            stop_y = max(0, start_y - gap_px)
            if stop_y > 0:
                hf[start_x:stop_x, 0:stop_y] = np.random.choice(heights)
            while start_y < length_px:
                stop_y = min(length_px, start_y + stone_px)
                hf[start_x:stop_x, start_y:stop_y] = np.random.choice(heights)
                start_y += stone_px + gap_px
            start_x += stone_px + gap_px

    _center_square(hf, int(cfg.platform_width / cfg.horizontal_scale))
    return np.rint(hf).astype(np.int16)


@height_field_to_mesh
def pit_gap_terrain(difficulty: float, cfg: terrains_cfg.PitGapTerrainCfg) -> np.ndarray:
    sign = -1 if randint(0, 1) == 0 else 1
    platform_range = cfg.platform_height_range if cfg.platform_height_range is not None else cfg.pit_depth_range
    offset_range = cfg.pit_depth_range if sign < 0 else platform_range
    depth = offset_range[0] + difficulty * (offset_range[1] - offset_range[0])
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    depth_px = int(abs(depth) / cfg.vertical_scale)
    gap_depth_px = int(cfg.gap_depth / cfg.vertical_scale)
    gap_px = max(int(gap_width / cfg.horizontal_scale), 0)
    ground_px = max(int(cfg.ground_width / cfg.horizontal_scale), 1)
    half_pit_px = max(int(0.5 * cfg.pit_width / cfg.horizontal_scale), 1)

    dx = np.abs(np.arange(width_px) - width_px // 2)
    dy = np.abs(np.arange(length_px) - length_px // 2)
    cheb = np.maximum(dx[:, None], dy[None, :])

    hf = np.zeros((width_px, length_px), dtype=float)
    if gap_px > 0:
        d = cheb - half_pit_px
        phase = np.mod(d, ground_px + gap_px)
        hf[(d >= 0) & (phase >= ground_px)] = -gap_depth_px
    hf[cheb < half_pit_px] = sign * depth_px
    return np.rint(hf).astype(np.int16)


@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: terrains_cfg.ConcentricGapTerrainCfg) -> np.ndarray:
    gap_depth_px = int(2.0 / cfg.vertical_scale)
    gap_px = int((cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])) / cfg.horizontal_scale)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])
    ground_px = int(ground_width / cfg.horizontal_scale)
    ground_h_px = int(cfg.ground_height_max / cfg.vertical_scale)

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    platform_px = int(cfg.platform_width / cfg.horizontal_scale)

    hf = np.zeros((width_px, length_px))
    x0, y0, x1, y1 = 0, 0, width_px, length_px
    is_gap = True
    while (x1 - x0) > platform_px and (y1 - y0) > platform_px:
        if is_gap:
            hf[x0:x1, y0:y1] = -gap_depth_px
            step = gap_px
        else:
            hf[x0:x1, y0:y1] = randint(-ground_h_px, ground_h_px)
            step = ground_px
        x0, y0, x1, y1 = x0 + step, y0 + step, x1 - step, y1 - step
        is_gap = not is_gap
    _center_square(hf, platform_px)
    return np.rint(hf).astype(np.int16)



_TILE_COUNTER: dict = defaultdict(int)


def _fold_1d(n):
    return np.abs(np.arange(n) - (n - 1) / 2.0)


def _lerp(pair, difficulty):
    return pair[0] + difficulty * (pair[1] - pair[0])


def _mirror_d4(hf):
    w, l = hf.shape
    i = np.minimum(np.arange(w), w - 1 - np.arange(w))
    j = np.minimum(np.arange(l), l - 1 - np.arange(l))
    ii, jj = np.meshgrid(i, j, indexing="ij")
    if w == l:
        return hf[np.minimum(ii, jj), np.maximum(ii, jj)]
    return hf[ii, jj]


def _stone_mask_1d(n, stone_px, gap_px, limit=None, inner_limit=None):
    d = _fold_1d(n)
    period = stone_px + gap_px
    on = ((d + stone_px / 2.0) % period) < stone_px
    k = np.round(d / period)
    if limit is not None:
        on = on & ((k * period + stone_px / 2.0) <= limit)
    if inner_limit is not None:
        on = on & ((k * period - stone_px / 2.0) >= inner_limit)
    return on


def _quantised_field(shape, amp, step, cfg, rng):
    step_px = max(1, int(round(step / cfg.vertical_scale)))
    amp_px = int(round(amp / cfg.vertical_scale))
    return rng.choice(np.arange(-amp_px, amp_px + step_px, step_px), size=shape).astype(float)


def _floor_flat(hf, mask, cheb, cfg, difficulty, rng):
    return hf


def _floor_rough(hf, mask, cheb, cfg, difficulty, rng):
    hf[mask] += _mirror_d4(_quantised_field(hf.shape, _lerp(cfg.rough_range, difficulty), cfg.rough_step, cfg, rng))[mask]
    return hf


def _floor_boxes(hf, mask, cheb, cfg, difficulty, rng):
    h_max_px = int(round(_lerp(cfg.box_height_range, difficulty) / cfg.vertical_scale))
    step_px = max(1, int(round(cfg.box_height_step / cfg.vertical_scale)))
    steps = np.arange(0, h_max_px + step_px, step_px)
    w_lo = max(1, int(round(cfg.box_width_range[0] / cfg.horizontal_scale)))
    w_hi = max(w_lo, int(round(cfg.box_width_range[1] / cfg.horizontal_scale)))

    def strides(n):
        out, i = [], 0
        while i < n:
            w = int(rng.integers(w_lo, w_hi + 1))
            out.append((i, min(i + w, n)))
            i += w
        return out

    field = np.zeros_like(hf)
    for i0, i1 in strides(hf.shape[0]):
        for j0, j1 in strides(hf.shape[1]):
            field[i0:i1, j0:j1] = float(rng.choice(steps))
    hf[mask] += _mirror_d4(field)[mask]
    return hf


def _floor_stepping_stones(hf, mask, cheb, cfg, difficulty, rng):
    stone_px = max(1, int(round(cfg.stone_width / cfg.horizontal_scale)))
    gap_px = max(1, int(round(_lerp(cfg.stone_gap_range, difficulty) / cfg.horizontal_scale)))
    depth_px = int(round(cfg.stone_gap_depth / cfg.vertical_scale))
    w, l = hf.shape
    half_px = 0.5 * _lerp(cfg.corridor_width_range, difficulty) / cfg.horizontal_scale
    plat_px = 0.5 * cfg.platform_width / cfg.horizontal_scale

    free_x = _stone_mask_1d(w, stone_px, gap_px, limit=0.5 * (w - 1), inner_limit=plat_px)
    free_y = _stone_mask_1d(l, stone_px, gap_px, limit=0.5 * (l - 1), inner_limit=plat_px)
    bnd_x = _stone_mask_1d(w, stone_px, gap_px, limit=half_px)
    bnd_y = _stone_mask_1d(l, stone_px, gap_px, limit=half_px)

    gx = np.broadcast_to(_fold_1d(w)[:, None], (w, l))
    gy = np.broadcast_to(_fold_1d(l)[None, :], (w, l))
    on = ((gy < half_px) & np.outer(free_x, bnd_y)) | ((gx < half_px) & np.outer(bnd_x, free_y))
    hf[mask & ~on] -= depth_px
    return hf


def _stair_profile(cheb, cfg, difficulty, sign):
    rise = _lerp(cfg.stair_rise_range, difficulty)
    rise_px = max(1, int(round(rise / cfg.vertical_scale)))
    run_px = max(1, int(round(cfg.stair_run / cfg.horizontal_scale)))
    plat_px = 0.5 * cfg.platform_width / cfg.horizontal_scale
    n_step = np.maximum(0.0, np.floor((cheb - plat_px) / run_px) + 1.0)
    n_step = np.minimum(n_step, max(1.0, np.floor(cfg.stair_max_rise / rise)))
    return sign * rise_px * n_step


def _floor_noise(hf, mask, cheb, cfg, difficulty, rng):
    cell = max(1, int(round(cfg.noise_cell / cfg.horizontal_scale)))
    w, l = hf.shape
    coarse = _quantised_field((w // cell + 1, l // cell + 1), _lerp(cfg.noise_range, difficulty), cfg.noise_step, cfg, rng)
    field = np.repeat(np.repeat(coarse, cell, axis=0), cell, axis=1)[:w, :l]
    hf[mask] += _mirror_d4(field)[mask]
    return hf


def _ring_profile(n, first_px, fill):
    prof = np.zeros(n)
    r = float(first_px)
    while r < n:
        band_px, value_px, advance_px = fill(r)
        r0, r1 = int(round(r)), min(int(round(r + band_px)), n)
        if r1 > r0:
            prof[r0:r1] = value_px
        r = r + advance_px
        if advance_px <= 0:
            break
    return prof


def _floor_obstacles(hf, mask, cheb, cfg, difficulty, rng):
    h_lo, h_hi = cfg.obstacle_height_range
    w_lo = max(1, int(round(cfg.obstacle_width_range[0] / cfg.horizontal_scale)))
    w_hi = max(w_lo, int(round(cfg.obstacle_width_range[1] / cfg.horizontal_scale)))
    sp_lo = max(1, int(round(cfg.obstacle_spacing_range[0] / cfg.horizontal_scale)))
    sp_hi = max(sp_lo, int(round(cfg.obstacle_spacing_range[1] / cfg.horizontal_scale)))

    def fill(_r):
        band = int(rng.integers(w_lo, w_hi + 1))
        val = int(round(float(rng.uniform(h_lo, h_hi)) / cfg.vertical_scale))
        return band, val, band + int(rng.integers(sp_lo, sp_hi + 1))

    prof = _ring_profile(max(hf.shape), 0.5 * cfg.platform_width / cfg.horizontal_scale, fill)
    hf[mask] += prof[np.clip(np.rint(cheb).astype(int), 0, len(prof) - 1)][mask]
    return hf


def _floor_gaps(hf, mask, cheb, cfg, difficulty, rng):
    gw_lo, gw_hi = cfg.gap_ground_width_range
    gap_px = max(1, int(round(_lerp(cfg.gap_width_range, difficulty) / cfg.horizontal_scale)))
    depth_px = int(round(cfg.gap_depth / cfg.vertical_scale))
    hmax_px = max(0, int(round(cfg.gap_ground_height_max / cfg.vertical_scale)))
    plat_px = 0.5 * cfg.platform_width / cfg.horizontal_scale

    def fill(_r):
        band = max(1, int(round(float(rng.uniform(gw_lo, gw_hi)) / cfg.horizontal_scale)))
        val = int(rng.integers(-hmax_px, hmax_px + 1)) if hmax_px > 0 else 0
        return band, val + 1_000_000, band + gap_px

    prof = _ring_profile(max(hf.shape), plat_px, fill)
    sampled = prof[np.clip(np.rint(cheb).astype(int), 0, len(prof) - 1)]
    height = np.where(sampled >= 500_000, sampled - 1_000_000, -float(depth_px))
    height[cheb < plat_px] = 0.0
    hf[mask] += height[mask]
    return hf


def _slab_labels_1d(n, cfg, size_lo, size_hi, gap_px, rng):
    lab = np.full(n, -1, dtype=int)
    r, k = 0.0, 0
    while r < n:
        w = max(1, int(round(float(rng.uniform(size_lo, size_hi)) / cfg.horizontal_scale)))
        r0, r1 = int(round(r)), min(int(round(r + w)), n)
        if r1 > r0:
            lab[r0:r1] = k
        k += 1
        r = r + w + gap_px
    return lab


def _floor_pallets(hf, mask, cheb, cfg, difficulty, rng):
    w, l = hf.shape
    gap_px = max(1, int(round(cfg.pallet_gap / cfg.horizontal_scale)))
    pit_px = int(round(cfg.pallet_pit_depth / cfg.vertical_scale))
    plat_px = 0.5 * cfg.platform_width / cfg.horizontal_scale
    size_lo, size_hi = cfg.pallet_size_range

    lab_x = _slab_labels_1d(w, cfg, size_lo, size_hi, gap_px, rng)
    lab_y = _slab_labels_1d(l, cfg, size_lo, size_hi, gap_px, rng)
    ix = lab_x[np.clip(np.rint(_fold_1d(w)).astype(int), 0, w - 1)][:, None] * np.ones((1, l), dtype=int)
    iy = lab_y[np.clip(np.rint(_fold_1d(l)).astype(int), 0, l - 1)][None, :] * np.ones((w, 1), dtype=int)
    on = (ix >= 0) & (iy >= 0)

    n_lab = max(int(lab_x.max()), int(lab_y.max())) + 2
    table = rng.uniform(*cfg.pallet_height_range, size=(n_lab, n_lab)) / cfg.vertical_scale
    heights = table[np.clip(ix, 0, n_lab - 1), np.clip(iy, 0, n_lab - 1)]
    hf[mask & on] += np.rint(heights)[mask & on]
    hf[mask & ~on] -= pit_px
    hf[cheb < plat_px] = 0.0
    return hf


def _slope_profile(cheb, cfg, difficulty, sign):
    plat_px = 0.5 * cfg.platform_width / cfg.horizontal_scale
    r_m = np.maximum(cheb - plat_px, 0.0) * cfg.horizontal_scale
    return sign * (_lerp(cfg.slope_range, difficulty) * r_m) / cfg.vertical_scale


def _wave_profile(cheb, cfg, difficulty):
    amp = _lerp(cfg.wave_amplitude_range, difficulty)
    return (amp / cfg.vertical_scale) * np.sin(2.0 * np.pi * cheb * cfg.horizontal_scale / float(cfg.wave_period))


_FLOORS = {
    "flat": _floor_flat,
    "rough": _floor_rough,
    "boxes": _floor_boxes,
    "stepping_stones": _floor_stepping_stones,
    "obstacles": _floor_obstacles,
    "noise": _floor_noise,
    "pallets": _floor_pallets,
    "gaps": _floor_gaps,
}

_BASES = {
    "stairs": lambda cheb, cfg, d: _stair_profile(cheb, cfg, d, 1.0),
    "stairs_inv": lambda cheb, cfg, d: _stair_profile(cheb, cfg, d, -1.0),
    "slope": lambda cheb, cfg, d: _slope_profile(cheb, cfg, d, 1.0),
    "slope_inv": lambda cheb, cfg, d: _slope_profile(cheb, cfg, d, -1.0),
    "wave": lambda cheb, cfg, d: _wave_profile(cheb, cfg, d),
}


def _corridor_mask(gx, gy, w, l, cfg, difficulty, rng):
    if cfg.wall_type == "blocks":
        half_px = 0.5 * cfg.blocks_width / cfg.horizontal_scale
        extra_px = cfg.blocks_extra_width / cfg.horizontal_scale
        blk_px = max(1, int(round(cfg.block_length / cfg.horizontal_scale)))
        recess = rng.random(int(np.ceil(max(w, l) / blk_px)) + 2) * (extra_px / 2.0)
        idx_x = (_fold_1d(w) // blk_px).astype(int)
        idx_y = (_fold_1d(l) // blk_px).astype(int)
        corridor = (gy < (half_px + recess[idx_x][:, None])) | (gx < (half_px + recess[idx_y][None, :]))
        return corridor, np.full((w, l), float(rng.uniform(*cfg.block_height_range) / cfg.vertical_scale))

    half_px = 0.5 * _lerp(cfg.corridor_width_range, difficulty) / cfg.horizontal_scale
    corridor = (gy < half_px) | (gx < half_px)
    wall_h = np.full((w, l), float(rng.uniform(*cfg.wall_height_range) / cfg.vertical_scale))
    if cfg.wall_type == "stones":
        seg_px = max(1, int(round(cfg.stone_wall_length / cfg.horizontal_scale)))
        period = seg_px + max(1, int(round(cfg.stone_wall_gap / cfg.horizontal_scale)))
        on_x = ((_fold_1d(w) + seg_px / 2.0) % period) < seg_px
        on_y = ((_fold_1d(l) + seg_px / 2.0) % period) < seg_px
        pillar = np.outer(on_x, on_y) & ~corridor
        wall_h = np.where(
            pillar, float(cfg.stone_wall_height / cfg.vertical_scale), -float(cfg.wall_pit_depth / cfg.vertical_scale)
        )
    return corridor, wall_h


def _corridor_hf(difficulty: float, cfg) -> np.ndarray:
    w = int(cfg.size[0] / cfg.horizontal_scale)
    l = int(cfg.size[1] / cfg.horizontal_scale)

    seed = getattr(cfg, "seed", None)
    if seed is None:
        rng = np.random.default_rng(np.random.randint(0, 2**32))
    else:
        key = (int(seed), cfg.wall_type, cfg.floor_variant)
        n = _TILE_COUNTER[key]
        _TILE_COUNTER[key] = n + 1
        rng = np.random.default_rng(abs(hash((key, n))) % (2**32))

    gx = np.broadcast_to(_fold_1d(w)[:, None], (w, l))
    gy = np.broadcast_to(_fold_1d(l)[None, :], (w, l))
    cheb = np.maximum(gx, gy)
    corridor, wall_h = _corridor_mask(gx, gy, w, l, cfg, difficulty, rng)

    smooth = _BASES.get(cfg.floor_variant)
    base = smooth(cheb, cfg, difficulty) if smooth is not None else np.zeros((w, l))
    hf = base.copy()
    if smooth is None:
        hf = _FLOORS[cfg.floor_variant](hf, corridor, cheb, cfg, difficulty, rng)

    taper = np.clip((np.minimum(w, l) / 2.0 - cheb) / max(1.0, cfg.edge_taper / cfg.horizontal_scale), 0.0, 1.0)
    hf = hf - base + base * taper
    base = base * taper

    on_platform = cheb < 0.5 * cfg.platform_width / cfg.horizontal_scale
    hf[on_platform] = 0.0
    base[on_platform] = 0.0
    return np.rint(np.where(corridor, hf, base + wall_h)).astype(np.int16)


def corridor_terrain(difficulty: float, cfg: terrains_cfg.CorridorTerrainCfg):
    width_px = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_px = int(cfg.size[1] / cfg.horizontal_scale) + 1
    border_px = int(cfg.border_width / cfg.horizontal_scale) + 1

    heights = np.zeros((width_px, length_px), dtype=np.int16)
    full_size = copy.deepcopy(cfg.size)
    cfg.size = ((width_px - 2 * border_px) * cfg.horizontal_scale, (length_px - 2 * border_px) * cfg.horizontal_scale)
    z = _corridor_hf(difficulty, cfg)
    cfg.size = full_size
    heights[border_px:-border_px, border_px:-border_px] = z

    vertices, triangles = convert_height_field_to_mesh(heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return [mesh], np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
