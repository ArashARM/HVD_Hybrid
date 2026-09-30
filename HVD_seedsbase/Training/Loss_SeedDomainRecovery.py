from __future__ import annotations

import torch


class LossSeedDomainRecovery:
    def __call__(
        self,
        domain_activity: torch.Tensor,
    ) -> torch.Tensor:
        g = domain_activity.reshape(-1)
        if g.numel() == 0:
            return torch.zeros((), dtype=domain_activity.dtype, device=domain_activity.device)

        g = torch.nan_to_num(
            g,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return (1.0 - g).square().mean()
