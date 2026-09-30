from __future__ import annotations

import torch


class LossDensityWeightedCVT:
    def __call__(
        self,
        seeds_uv: torch.Tensor,
        sample_uv: torch.Tensor,
        seed_xyz: torch.Tensor,
        sample_xyz: torch.Tensor,
        sample_area_weights: torch.Tensor,
        importance: torch.Tensor | None = None,
        temperature: float = 0.001,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(
                f"seeds_uv must have shape [N, 2], got {tuple(seeds_uv.shape)}"
            )

        if sample_uv.ndim != 2 or sample_uv.shape[-1] != 2:
            raise ValueError(
                f"sample_uv must have shape [Q, 2], got {tuple(sample_uv.shape)}"
            )

        if seed_xyz.ndim != 2 or seed_xyz.shape[-1] != 3:
            raise ValueError(
                f"seed_xyz must have shape [N, 3], got {tuple(seed_xyz.shape)}"
            )

        if sample_xyz.ndim != 2 or sample_xyz.shape[-1] != 3:
            raise ValueError(
                f"sample_xyz must have shape [Q, 3], got {tuple(sample_xyz.shape)}"
            )

        num_seeds = seeds_uv.shape[0]
        num_samples = sample_uv.shape[0]

        if num_seeds == 0 or num_samples == 0:
            return seeds_uv.new_zeros(())

        if seed_xyz.shape[0] != num_seeds:
            raise ValueError(
                "seed_xyz and seeds_uv must contain the same number of seeds"
            )

        if sample_xyz.shape[0] != num_samples:
            raise ValueError(
                "sample_xyz and sample_uv must contain the same number of samples"
            )

        device = seeds_uv.device
        dtype = seeds_uv.dtype

        sample_uv = sample_uv.to(device=device, dtype=dtype)
        seed_xyz = seed_xyz.to(device=device, dtype=dtype)
        sample_xyz = sample_xyz.to(device=device, dtype=dtype)

        sample_area_weights = sample_area_weights.to(
            device=device,
            dtype=dtype,
        ).reshape(-1)

        if sample_area_weights.numel() != num_samples:
            raise ValueError(
                "sample_area_weights must contain one value per surface sample"
            )

        if importance is None:
            importance = torch.ones_like(sample_area_weights)
        else:
            importance = importance.to(
                device=device,
                dtype=dtype,
            ).reshape(-1)

        if importance.numel() != num_samples:
            raise ValueError(
                "importance must contain one value per surface sample"
            )

        importance = torch.nan_to_num(
            importance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        area = torch.nan_to_num(
            sample_area_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        delta_uv = sample_uv[:, None, :] - seeds_uv[None, :, :]
        distance_uv_squared = delta_uv.square().sum(dim=-1)

        tau = torch.as_tensor(
            temperature,
            device=device,
            dtype=dtype,
        ).clamp_min(eps)

        logits = -distance_uv_squared / tau

        ownership = torch.softmax(logits, dim=1)

        delta_xyz = sample_xyz[:, None, :] - seed_xyz[None, :, :]
        distance_xyz_squared = delta_xyz.square().sum(dim=-1)

        spatial_weight = area * importance
        denominator = spatial_weight.sum().clamp_min(eps)

        numerator = (
            spatial_weight[:, None]
            * ownership
            * distance_xyz_squared
        ).sum()

        return numerator / denominator
