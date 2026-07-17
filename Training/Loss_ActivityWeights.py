from __future__ import annotations

import torch


def _validate_activity_shape(
    weights: torch.Tensor,
    *,
    num_seeds: int,
    name: str,
) -> None:
    if weights.numel() != num_seeds:
        raise ValueError(
            f"{name} must contain one value per seed; "
            f"got {weights.numel()} values for {num_seeds} seeds."
        )


def _sanitize_seed_activity_weights(
    seed_active_weights: torch.Tensor,
    *,
    num_seeds: int,
    reference: torch.Tensor,
    name: str = "seed_active_weights",
) -> torch.Tensor:
    g = seed_active_weights.reshape(-1).to(
        dtype=reference.dtype,
        device=reference.device,
    )
    _validate_activity_shape(g, num_seeds=num_seeds, name=name)
    return torch.nan_to_num(
        g,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)


def prepare_seed_activity_weights(
    seed_active_weights: torch.Tensor | None,
    *,
    num_seeds: int,
    reference: torch.Tensor,
    floor: float = 0.02,
    power: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    # Activity weights are differentiable functions of seed locations, not
    # independent trainable parameters. Gradients through these weights update
    # seed coordinates through the decoder's activity computation.
    if not (0.0 <= float(floor) < 1.0):
        raise ValueError(f"floor must satisfy 0 <= floor < 1, got {floor}")
    if float(power) <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")

    if seed_active_weights is None:
        return torch.ones(
            num_seeds,
            dtype=reference.dtype,
            device=reference.device,
        )

    g = _sanitize_seed_activity_weights(
        seed_active_weights,
        num_seeds=num_seeds,
        reference=reference,
    )
    return (
        float(floor)
        + (1.0 - float(floor))
        * g.pow(float(power))
    )


def prepare_seed_recovery_weights(
    seed_active_weights: torch.Tensor | None,
    *,
    num_seeds: int,
    reference: torch.Tensor,
    floor: float = 0.05,
    power: float = 1.0,
) -> torch.Tensor:
    if not (0.0 <= float(floor) < 1.0):
        raise ValueError(
            f"floor must satisfy 0 <= floor < 1, got {floor}"
        )
    if float(power) <= 0.0:
        raise ValueError(
            f"power must be > 0, got {power}"
        )

    if seed_active_weights is None:
        return torch.full(
            (num_seeds,),
            float(floor),
            dtype=reference.dtype,
            device=reference.device,
        )

    g = _sanitize_seed_activity_weights(
        seed_active_weights,
        num_seeds=num_seeds,
        reference=reference,
    )
    return (
        float(floor)
        + (1.0 - float(floor))
        * (1.0 - g).pow(float(power))
    )
