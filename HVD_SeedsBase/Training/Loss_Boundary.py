import torch

try:
    from .Loss_ActivityWeights import prepare_seed_activity_weights
except ImportError:
    from Loss_ActivityWeights import prepare_seed_activity_weights


class Loss_Boundary:
    def __call__(
        self,
        seeds: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        seed_active_weights: torch.Tensor | None = None,
        activity_floor: float = 0.02,
        activity_power: float = 1.0,
        margin: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)

        dmin = torch.cdist(seeds, boundary_uv).amin(dim=1)
        penalty = torch.exp(-dmin / (margin + eps))

        if seed_active_weights is None:
            return penalty.mean()

        g = prepare_seed_activity_weights(
            seed_active_weights,
            num_seeds=seeds.shape[0],
            reference=seeds,
            floor=activity_floor,
            power=activity_power,
            eps=eps,
        )
        penalty = (g * penalty).sum() / (g.sum() + eps)
        return penalty
