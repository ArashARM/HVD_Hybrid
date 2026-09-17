from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def seed_separation_loss(
    seed_xyz: torch.Tensor,
    *,
    min_seed_spacing: float | torch.Tensor,
    valid_seed_mask: torch.Tensor | None = None,
    spacing_power: float = 2.0,
    safety_factor: float = 1.05,
    aggregate_temperature: float = 1.0e-4,
    repulsion_factor: float = 1.10,
    repulsion_temperature_ratio: float = 0.02,
    repulsion_weight: float = 0.10,
    eps: float = 1.0e-12,
    return_components: bool = False,
):
    """
    Combined seed-separation loss.

    1. Barrier:
       Strongly controls the worst pair and enforces minimum spacing.

    2. Collective repulsion:
       Weakly discourages several seeds from clustering near the limit.

    If valid_seed_mask is provided, only pairs where both seeds are valid
    contribute to the loss. This lets boundary/outside-domain handling own
    invalid seeds instead of letting separation push valid seeds away from them.
    """
    if seed_xyz.ndim != 2 or seed_xyz.shape[-1] != 3:
        raise ValueError(
            "seed_xyz must have shape [N, 3], "
            f"got {tuple(seed_xyz.shape)}."
        )

    if valid_seed_mask is not None:
        valid_seed_mask = valid_seed_mask.to(
            device=seed_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if valid_seed_mask.numel() != seed_xyz.shape[0]:
            raise ValueError(
                "valid_seed_mask must contain one value per seed, "
                f"got {valid_seed_mask.numel()} for {seed_xyz.shape[0]} seeds."
            )
        seed_xyz = seed_xyz[valid_seed_mask.detach()]

    n_seed = int(seed_xyz.shape[0])
    if n_seed < 2:
        zero = seed_xyz.sum() * 0.0
        if return_components:
            return zero, {
                "barrier": zero,
                "repulsion": zero,
            }
        return zero

    d_min = torch.as_tensor(
        min_seed_spacing,
        dtype=seed_xyz.dtype,
        device=seed_xyz.device,
    ).clamp_min(eps)

    distances = torch.cdist(seed_xyz, seed_xyz)

    pair_mask = torch.triu(
        torch.ones(
            (n_seed, n_seed),
            dtype=torch.bool,
            device=seed_xyz.device,
        ),
        diagonal=1,
    )

    pair_distances = distances[pair_mask]

    # ============================================================
    # 1. Smooth worst-pair spacing barrier
    # ============================================================

    d_safe = float(safety_factor) * d_min

    barrier_violation = torch.relu(
        (d_safe - pair_distances) / d_safe
    )

    pair_barrier = barrier_violation.pow(float(spacing_power))

    tau_aggregate = torch.as_tensor(
        aggregate_temperature,
        dtype=seed_xyz.dtype,
        device=seed_xyz.device,
    ).clamp_min(eps)

    barrier_loss = tau_aggregate * (
        torch.logsumexp(
            pair_barrier / tau_aggregate,
            dim=0,
        )
        - math.log(pair_barrier.numel())
    )

    # ============================================================
    # 2. Weak collective repulsion
    # ============================================================

    d_repulsion = float(repulsion_factor) * d_min

    tau_repulsion = (
        float(repulsion_temperature_ratio) * d_min
    ).clamp_min(eps)

    repulsion_violation = (
        F.softplus(
            (d_repulsion - pair_distances) / tau_repulsion
        )
        * tau_repulsion
        / d_repulsion
    )

    pair_repulsion = repulsion_violation.square()

    # Sum/N instead of mean over all N(N-1)/2 pairs.
    # This gives a more consistent per-seed scale.
    collective_repulsion = pair_repulsion.sum() / float(n_seed)

    # ============================================================
    # Combined loss
    # ============================================================

    total_loss = (
        barrier_loss
        + float(repulsion_weight) * collective_repulsion
    )

    if return_components:
        return total_loss, {
            "barrier": barrier_loss,
            "repulsion": collective_repulsion,
            "weighted_repulsion": (
                float(repulsion_weight) * collective_repulsion
            ),
            "minimum_distance": pair_distances.min(),
            "safe_distance": d_safe,
            "repulsion_distance": d_repulsion,
        }

    return total_loss


def minimum_seed_spacing_loss(
    seed_xyz: torch.Tensor,
    *,
    min_seed_spacing: float | torch.Tensor,
    spacing_power: float = 2.0,
    valid_seed_mask: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if seed_xyz.ndim != 2 or seed_xyz.shape[-1] != 3:
        raise ValueError(
            "seed_xyz must have shape [N, 3], "
            f"got {tuple(seed_xyz.shape)}."
        )
    if valid_seed_mask is not None:
        valid_seed_mask = valid_seed_mask.to(
            device=seed_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if valid_seed_mask.numel() != seed_xyz.shape[0]:
            raise ValueError(
                "valid_seed_mask must contain one value per seed, "
                f"got {valid_seed_mask.numel()} for {seed_xyz.shape[0]} seeds."
            )
        seed_xyz = seed_xyz[valid_seed_mask.detach()]

    n_seed = int(seed_xyz.shape[0])
    if n_seed < 2:
        return seed_xyz.sum() * 0.0

    distances = torch.cdist(seed_xyz, seed_xyz)
    pair_mask = torch.triu(
        torch.ones((n_seed, n_seed), dtype=torch.bool, device=seed_xyz.device),
        diagonal=1,
    )
    target = torch.as_tensor(
        min_seed_spacing,
        dtype=seed_xyz.dtype,
        device=seed_xyz.device,
    ).clamp_min(eps)
    violation = torch.relu((target - distances[pair_mask]) / target)
    return violation.pow(float(spacing_power)).mean()


def minimum_physical_seed_distance(
    seed_xyz: torch.Tensor,
    *,
    valid_seed_mask: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if seed_xyz.ndim != 2 or seed_xyz.shape[-1] != 3:
        raise ValueError(
            "seed_xyz must have shape [N, 3], "
            f"got {tuple(seed_xyz.shape)}."
        )
    if valid_seed_mask is not None:
        valid_seed_mask = valid_seed_mask.to(
            device=seed_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if valid_seed_mask.numel() != seed_xyz.shape[0]:
            raise ValueError(
                "valid_seed_mask must contain one value per seed, "
                f"got {valid_seed_mask.numel()} for {seed_xyz.shape[0]} seeds."
            )
        seed_xyz = seed_xyz[valid_seed_mask.detach()]

    n_seed = int(seed_xyz.shape[0])
    if n_seed < 2:
        return seed_xyz.new_tensor(float("inf"))
    diff = seed_xyz[:, None, :] - seed_xyz[None, :, :]
    distances = torch.sqrt((diff * diff).sum(dim=-1) + float(eps))
    distances = distances.masked_fill(
        torch.eye(n_seed, dtype=torch.bool, device=seed_xyz.device),
        float("inf"),
    )
    return distances.min()


def _segments_from_boundary_curves(
    boundary_curve_uv: torch.Tensor,
    boundary_curve_offsets: torch.Tensor,
) -> torch.Tensor:
    points = boundary_curve_uv
    offsets = boundary_curve_offsets.to(device=points.device, dtype=torch.long).reshape(-1)
    segments = []
    for loop_id in range(max(int(offsets.numel()) - 1, 0)):
        start = int(offsets[loop_id].detach().item())
        end = int(offsets[loop_id + 1].detach().item())
        loop = points[start:end]
        if loop.shape[0] < 2:
            continue
        segments.append(torch.stack((loop[:-1], loop[1:]), dim=1))
        if loop.shape[0] > 2:
            closing = torch.stack((loop[-1], loop[0]), dim=0).unsqueeze(0)
            if not bool(torch.allclose(loop[0].detach(), loop[-1].detach())):
                segments.append(closing)
    if not segments:
        return points.new_empty((0, 2, 2))
    return torch.cat(segments, dim=0)


def _point_in_loop_detached(points_uv: torch.Tensor, loop_uv: torch.Tensor, eps: float) -> torch.Tensor:
    p = points_uv.detach()
    loop = loop_uv.detach()
    if loop.shape[0] < 3:
        return torch.zeros((p.shape[0],), dtype=torch.bool, device=p.device)
    a = loop
    b = torch.roll(loop, shifts=-1, dims=0)
    px = p[:, 0:1]
    py = p[:, 1:2]
    ax = a[:, 0].unsqueeze(0)
    ay = a[:, 1].unsqueeze(0)
    bx = b[:, 0].unsqueeze(0)
    by = b[:, 1].unsqueeze(0)
    crosses_y = (ay > py) != (by > py)
    x_at_y = (bx - ax) * (py - ay) / (by - ay + float(eps)) + ax
    return (crosses_y & (px < x_at_y)).sum(dim=1).remainder(2).to(torch.bool)


def _boundary_loops_from_offsets(
    boundary_curve_uv: torch.Tensor,
    boundary_curve_offsets: torch.Tensor,
    boundary_curve_loop_id: torch.Tensor | None,
) -> list[torch.Tensor]:
    offsets = boundary_curve_offsets.reshape(-1)
    piece_count = max(int(offsets.numel()) - 1, 0)
    if piece_count <= 0:
        return []

    if boundary_curve_loop_id is None:
        return [
            boundary_curve_uv[int(offsets[piece_id].item()):int(offsets[piece_id + 1].item())]
            for piece_id in range(piece_count)
        ]

    loop_ids = boundary_curve_loop_id.to(
        device=boundary_curve_uv.device,
        dtype=torch.long,
    ).reshape(-1)

    if loop_ids.numel() == boundary_curve_uv.shape[0]:
        loops = []
        for loop_id_tensor in torch.unique(loop_ids):
            loop_id = int(loop_id_tensor.detach().item())
            loops.append(boundary_curve_uv[loop_ids == loop_id])
        return loops

    if loop_ids.numel() == piece_count:
        loops = []
        for loop_id_tensor in torch.unique(loop_ids):
            loop_id = int(loop_id_tensor.detach().item())
            pieces = []
            for piece_id in range(piece_count):
                if int(loop_ids[piece_id].detach().item()) != loop_id:
                    continue
                start = int(offsets[piece_id].detach().item())
                end = int(offsets[piece_id + 1].detach().item())
                piece = boundary_curve_uv[start:end]
                if piece.numel() > 0:
                    pieces.append(piece)
            if pieces:
                loops.append(torch.cat(pieces, dim=0))
        return loops

    raise ValueError(
        "boundary_curve_loop_id must contain either one value per boundary point "
        "or one value per boundary curve piece."
    )


def signed_trim_boundary_distance(
    seeds_uv: torch.Tensor,
    boundary_curve_uv: torch.Tensor,
    boundary_curve_offsets: torch.Tensor,
    *,
    boundary_curve_loop_id: torch.Tensor | None = None,
    inside_mask: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
        raise ValueError(
            "seeds_uv must have shape [N, 2], "
            f"got {tuple(seeds_uv.shape)}."
        )
    if boundary_curve_uv is None or boundary_curve_uv.numel() == 0:
        return seeds_uv.new_full((seeds_uv.shape[0],), float("inf"))
    boundary_curve_uv = boundary_curve_uv.to(device=seeds_uv.device, dtype=seeds_uv.dtype)
    boundary_curve_offsets = boundary_curve_offsets.to(device=seeds_uv.device, dtype=torch.long)
    segments = _segments_from_boundary_curves(boundary_curve_uv, boundary_curve_offsets)
    if segments.numel() == 0:
        return seeds_uv.new_full((seeds_uv.shape[0],), float("inf"))

    a = segments[:, 0, :]
    b = segments[:, 1, :]
    ab = b - a
    ap = seeds_uv[:, None, :] - a[None, :, :]
    denom = (ab * ab).sum(dim=-1).clamp_min(float(eps))
    t = (ap * ab[None, :, :]).sum(dim=-1) / denom[None, :]
    closest = a[None, :, :] + t.clamp(0.0, 1.0).unsqueeze(-1) * ab[None, :, :]
    diff = seeds_uv[:, None, :] - closest
    distance = torch.sqrt((diff * diff).sum(dim=-1) + float(eps)).min(dim=1).values

    if inside_mask is None:
        inside = torch.zeros((seeds_uv.shape[0],), dtype=torch.bool, device=seeds_uv.device)
        for loop in _boundary_loops_from_offsets(
            boundary_curve_uv,
            boundary_curve_offsets.reshape(-1),
            boundary_curve_loop_id,
        ):
            inside ^= _point_in_loop_detached(seeds_uv, loop, eps)
    else:
        inside = inside_mask.to(device=seeds_uv.device, dtype=torch.bool).reshape(-1).detach()
        if inside.numel() != seeds_uv.shape[0]:
            raise ValueError("inside_mask must contain one value per seed.")

    sign = torch.where(inside, torch.ones_like(distance), -torch.ones_like(distance))
    return sign * distance


def seed_trim_boundary_loss(
    seeds_uv: torch.Tensor,
    boundary_curve_uv: torch.Tensor,
    boundary_curve_offsets: torch.Tensor,
    *,
    boundary_curve_loop_id: torch.Tensor | None = None,
    inside_mask: torch.Tensor | None = None,
    boundary_margin: float | torch.Tensor = 0.05,
    boundary_power: float = 2.0,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    signed_distance = signed_trim_boundary_distance(
        seeds_uv,
        boundary_curve_uv,
        boundary_curve_offsets,
        boundary_curve_loop_id=boundary_curve_loop_id,
        inside_mask=inside_mask,
        eps=eps,
    )
    finite = torch.isfinite(signed_distance)
    if not bool(finite.any().detach().item()):
        return seeds_uv.sum() * 0.0, signed_distance
    margin = torch.as_tensor(
        boundary_margin,
        dtype=seeds_uv.dtype,
        device=seeds_uv.device,
    ).clamp_min(eps)
    violation = torch.relu(-signed_distance[finite] / margin)
    return violation.pow(float(boundary_power)).mean(), signed_distance
