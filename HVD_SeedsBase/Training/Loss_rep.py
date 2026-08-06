
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
        seed_positions: torch.Tensor,
        target_dist: float,
        seed_active_weights: torch.Tensor | None = None,
        transition: float | None = None,
        activity_floor: float = 0.02,
        activity_power: float = 1.0,
        recovery_floor: float = 0.05,
        recovery_power: float = 1.0,
        duplicate_recovery_strength: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        num_seeds = seed_positions.shape[0]

        if num_seeds < 2:
            return seed_positions.new_zeros(())

        distances = torch.cdist(seed_positions, seed_positions)

        pair_mask = torch.triu(
            torch.ones(
                (num_seeds, num_seeds),
                dtype=torch.bool,
                device=seed_positions.device,
            ),
            diagonal=1,
        )

        target = seed_positions.new_tensor(max(float(target_dist), 0.0)).clamp_min(eps)
        transition_t = seed_positions.new_tensor(
            0.1 * float(target_dist) if transition is None else float(transition)
        ).clamp_min(eps)
        pair_penalty = torch.nn.functional.softplus((target - distances) / transition_t).square()

        pair_penalty = pair_penalty[pair_mask]

        if seed_active_weights is None:
            return pair_penalty.mean()

        weights = prepare_seed_activity_weights(
            seed_active_weights,
            num_seeds=num_seeds,
            reference=seed_positions,
            floor=activity_floor,
            power=activity_power,
            eps=eps,
        )
        recovery = prepare_seed_recovery_weights(
            seed_active_weights,
            num_seeds=num_seeds,
            reference=seed_positions,
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
