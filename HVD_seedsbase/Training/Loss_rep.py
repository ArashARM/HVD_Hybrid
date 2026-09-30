
import torch


class Loss_rep:
    def __call__(
        self,
        seed_positions: torch.Tensor,
        target_dist: float,
        transition: float | None = None,
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

        return pair_penalty.mean()
