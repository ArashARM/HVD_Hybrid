from __future__ import annotations

import torch


def seed_spacing_barrier_loss(
    seed_positions: torch.Tensor,
    *,
    safe_distance: float | torch.Tensor,
    seed_active_weights: torch.Tensor | None = None,
    power: float = 2.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """
    Penalize active seed pairs whose distance is below safe_distance.

    The loss is zero for distances greater than or equal to safe_distance.
    """
    if seed_positions.ndim != 2:
        raise ValueError(
            "seed_positions must have shape [N, D], "
            f"got {tuple(seed_positions.shape)}"
        )

    n_seed = int(seed_positions.shape[0])
    if n_seed < 2:
        return seed_positions.sum() * 0.0

    safe = torch.as_tensor(
        safe_distance,
        dtype=seed_positions.dtype,
        device=seed_positions.device,
    ).clamp_min(eps)

    distances = torch.cdist(seed_positions, seed_positions)
    pair_mask = torch.triu(
        torch.ones(
            (n_seed, n_seed),
            dtype=torch.bool,
            device=seed_positions.device,
        ),
        diagonal=1,
    )
    pair_distances = distances[pair_mask]
    normalized_violation = torch.relu((safe - pair_distances) / safe).pow(float(power))

    if seed_active_weights is None:
        return normalized_violation.mean()

    activity = seed_active_weights.reshape(-1)
    if activity.numel() != n_seed:
        raise ValueError("seed_active_weights must contain one value per seed")

    pair_weights_full = torch.sqrt(
        activity[:, None].clamp_min(0.0) * activity[None, :].clamp_min(0.0)
    )
    pair_weights = pair_weights_full[pair_mask]

    denominator = pair_weights.sum().clamp_min(eps)

    return (
        pair_weights * normalized_violation
    ).sum() / denominator
