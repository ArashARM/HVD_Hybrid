import torch
import torch.nn.functional as F


class Loss_SeedActive:
    def __call__(
        self,
        seed_active_weights: torch.Tensor,
        target_active: float,
        mode: str = "minimum",
        temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if seed_active_weights.numel() == 0:
            return seed_active_weights.new_zeros(())

        g = torch.nan_to_num(
            seed_active_weights.reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        active_mass = g.sum()

        target = g.new_tensor(float(target_active))

        if mode == "minimum":
            tau = g.new_tensor(float(temperature)).clamp_min(eps)
            shortfall = target - active_mass

            return (
                tau * F.softplus(shortfall / tau)
            ).square()

        if mode == "target":
            return (active_mass - target).square()

        raise ValueError("mode must be one of: 'minimum', 'target'")
