import torch


def world_to_root(points_w: torch.Tensor, root_pose_w: torch.Tensor) -> torch.Tensor:
    rel = points_w - root_pose_w[..., None, 0:3]
    q_w = root_pose_w[..., None, 3:4]
    q_vec = -root_pose_w[..., None, 4:7]
    t = torch.cross(q_vec, rel, dim=-1) * 2.0
    return rel + q_w * t + torch.cross(q_vec, t, dim=-1)


def compute_strictly_future_contact_positions(
    contact_data: torch.Tensor,
    root_pose_w: torch.Tensor,
    foot_pose_w: torch.Tensor,
    reset_data: torch.Tensor,
) -> torch.Tensor:
    num_envs, num_steps, num_feet = contact_data.shape
    if reset_data.dim() == 3:
        reset_data = reset_data.squeeze(-1)
    is_reset = reset_data > 0.5

    targets = []
    for i in range(num_feet):
        pos_w = foot_pose_w[..., 7 * i : 7 * i + 3]
        contact = contact_data[..., i]

        latch_next = pos_w[:, -1].clone()
        latch_future = pos_w[:, -1].clone()
        was_in_air = torch.zeros(num_envs, dtype=torch.bool, device=contact_data.device)
        foot_targets = torch.zeros_like(pos_w)

        for t in range(num_steps - 1, -1, -1):
            reset = is_reset[:, t]
            if reset.any():
                latch_next[reset] = pos_w[reset, t]
                latch_future[reset] = pos_w[reset, t]
                was_in_air[reset] = False

            in_contact = contact[:, t] > 0.5
            lift_off = in_contact & was_in_air
            if lift_off.any():
                latch_future[lift_off] = latch_next[lift_off]
            if in_contact.any():
                latch_next[in_contact] = pos_w[in_contact, t]

            foot_targets[:, t] = torch.where(in_contact.unsqueeze(-1), latch_future, latch_next)
            was_in_air = ~in_contact

        targets.append(foot_targets)

    targets_w = torch.stack(targets, dim=2)
    return world_to_root(targets_w, root_pose_w).reshape(num_envs, num_steps, 3 * num_feet)


def detect_resets(root_pose_w: torch.Tensor, xy_jump_thresh: float = 0.5) -> torch.Tensor:
    root_xy = root_pose_w[..., :2]
    jump = (root_xy[:, 1:] - root_xy[:, :-1]).norm(dim=-1) > xy_jump_thresh
    resets = torch.zeros(root_xy.shape[:2], device=root_xy.device)
    resets[:, :-1] = jump.float()
    return resets


def foothold_labels(
    contact_hist: torch.Tensor, pose_hist: torch.Tensor, num_feet: int = 2, xy_jump_thresh: float = 0.5
) -> torch.Tensor:
    contact = contact_hist[:, 1:]
    pose = pose_hist[:, 1:]
    root_pose_w = pose[..., :7]
    foot_pose_w = pose[..., 7 : 7 + 7 * num_feet]
    resets = detect_resets(root_pose_w, xy_jump_thresh)
    targets = compute_strictly_future_contact_positions(contact, root_pose_w, foot_pose_w, resets)
    xy_idx = [3 * i + j for i in range(num_feet) for j in (0, 1)]
    return targets[..., xy_idx]
