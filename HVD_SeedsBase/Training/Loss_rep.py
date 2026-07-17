
import torch

try:
    from .Loss_ActivityWeights import (
        prepare_seed_activity_weights,
        prepare_seed_recovery_weights,
    )
except ImportError:
    from Loss_ActivityWeights import (
        prepare_seed_activity_weights,
        prepare_seed_recovery_weights,
    )


class Loss_rep:
    def __call__(
        self,
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None = None,
        sigma: float = 0.08,
        min_dist: float | None = None,
        activity_floor: float = 0.02,
        activity_power: float = 1.0,
        recovery_floor: float = 0.05,
        recovery_power: float = 1.0,
        duplicate_recovery_strength: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        num_seeds = seeds.shape[0]

        if num_seeds < 2:
            return seeds.new_zeros(())

        distances = torch.cdist(seeds, seeds)

        pair_mask = torch.triu(
            torch.ones(
                (num_seeds, num_seeds),
                dtype=torch.bool,
                device=seeds.device,
            ),
            diagonal=1,
        )

        if min_dist is not None and min_dist > 0.0:
            target = seeds.new_tensor(min_dist)

            pair_penalty = (
                torch.relu(target - distances).square()
                / target.square().clamp_min(eps)
            )
        else:
            sigma_tensor = seeds.new_tensor(sigma)

            pair_penalty = torch.exp(
                -distances.square()
                / sigma_tensor.square().clamp_min(eps)
            )

        pair_penalty = pair_penalty[pair_mask]

        if seed_active_weights is None:
            return pair_penalty.mean()

        weights = prepare_seed_activity_weights(
            seed_active_weights,
            num_seeds=num_seeds,
            reference=seeds,
            floor=activity_floor,
            power=activity_power,
            eps=eps,
        )
        recovery = prepare_seed_recovery_weights(
            seed_active_weights,
            num_seeds=num_seeds,
            reference=seeds,
            floor=recovery_floor,
            power=recovery_power,
        )

        pair_weights_matrix = torch.sqrt(
            (weights[:, None] * weights[None, :]).clamp_min(eps)
        )
        if duplicate_recovery_strength != 0.0:
            pair_weights_matrix = (
                pair_weights_matrix
                + float(duplicate_recovery_strength)
                * torch.sqrt((recovery[:, None] * recovery[None, :]).clamp_min(eps))
            )

        pair_weights = pair_weights_matrix[pair_mask]

        weight_sum = pair_weights.sum()

        return (
            pair_weights * pair_penalty
        ).sum() / weight_sum.clamp_min(eps)
