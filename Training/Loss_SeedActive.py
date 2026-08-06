import torch
import torch.nn.functional as F


class Loss_SeedActive:
    def __call__(
        self,
        seed_active_weights: torch.Tensor,
        minimum_active: float,
        possible_min_active: float,
        binarization_temperature: float = 0.1,
        minimum_loss_strength: float = 1.0,
        barrier_strength: float = 10.0,
        barrier_temperature: float = 0.1,
        eps: float = 1e-12,
    ) -> torch.Tensor:

        if seed_active_weights.numel() == 0:
            return seed_active_weights.new_zeros(())

        # Clean and restrict original activation values to [0, 1].
        g = torch.nan_to_num(
            seed_active_weights.reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        # Differentiable binarization around 0.5.
        tau_bin = g.new_tensor(
            float(binarization_temperature)
        ).clamp_min(eps)

        g_binary_soft = torch.sigmoid(
            (g - 0.5) / tau_bin
        )

        # Differentiable approximation of the number of active seeds.
        active_count = g_binary_soft.sum()

        minimum = g.new_tensor(float(minimum_active))
        possible_minimum = g.new_tensor(float(possible_min_active))

        if possible_minimum > minimum:
            raise ValueError(
                "possible_min_active must be less than or equal to minimum_active."
            )

        # Zero when active_count >= minimum_active.
        minimum_shortfall = torch.relu(
            minimum - active_count
        )

        minimum_loss = (
            float(minimum_loss_strength)
            * minimum_shortfall.square()
        )

        # Strong additional penalty below possible_min_active.
        tau_barrier = g.new_tensor(
            float(barrier_temperature)
        ).clamp_min(eps)

        possible_shortfall = (
            possible_minimum - active_count
        )

        barrier_loss = float(barrier_strength) * (
            tau_barrier
            * F.softplus(possible_shortfall / tau_barrier)
        ).square()

        return minimum_loss + barrier_loss