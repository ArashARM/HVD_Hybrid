from __future__ import annotations

import torch

from Training.Loss_ActivityWeights import (
    prepare_seed_activity_weights,
    prepare_seed_recovery_weights,
)
from Training.Loss_Boundary import Loss_Boundary
from Training.Loss_DensityWeightedCVT import LossDensityWeightedCVT
from Training.Loss_rep import Loss_rep
from Training.Loss_SeedActive import Loss_SeedActive
from Training.Loss_SeedDomainRecovery import LossSeedDomainRecovery


def test_prepare_seed_activity_weights_floor_shape_and_gradient() -> None:
    g = torch.tensor([0.0, 0.25, 1.0], dtype=torch.double, requires_grad=True)
    reference = torch.zeros((3, 2), dtype=torch.double)

    weights = prepare_seed_activity_weights(
        g,
        num_seeds=3,
        reference=reference,
        floor=0.02,
        power=1.0,
    )
    weights.sum().backward()

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0.0)
    assert torch.all(weights <= 1.0)
    assert torch.allclose(weights[0], torch.tensor(0.02, dtype=torch.double))
    assert g.grad is not None
    assert torch.isfinite(g.grad).all()
    assert g.grad[1].abs() > 0.0


def test_prepare_seed_recovery_weights_inactive_is_larger() -> None:
    g = torch.tensor([0.0, 0.5, 1.0], dtype=torch.double, requires_grad=True)

    weights = prepare_seed_recovery_weights(
        g,
        num_seeds=3,
        reference=g,
        floor=0.05,
        power=1.0,
    )
    expected = torch.tensor([1.0, 0.525, 0.05], dtype=torch.double)

    assert torch.allclose(weights, expected)
    weights.sum().backward()

    assert g.grad is not None
    assert torch.isfinite(g.grad).all()
    assert torch.all(g.grad < 0.0)


def test_density_weighted_cvt_low_activity_changes_ownership_without_zero_grad() -> None:
    seeds = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]],
        dtype=torch.double,
        requires_grad=True,
    )
    sample_uv = torch.tensor(
        [[0.03, 0.0], [0.97, 0.0], [0.5, 0.75]],
        dtype=torch.double,
    )
    seed_xyz = torch.cat((seeds, torch.zeros((3, 1), dtype=torch.double)), dim=1)
    sample_xyz = torch.cat((sample_uv, torch.zeros((3, 1), dtype=torch.double)), dim=1)
    activity = torch.tensor([1.0, 1e-3, 1.0], dtype=torch.double, requires_grad=True)

    loss = LossDensityWeightedCVT()(
        seeds_uv=seeds,
        sample_uv=sample_uv,
        seed_xyz=seed_xyz,
        sample_xyz=sample_xyz,
        sample_area_weights=torch.ones(3, dtype=torch.double),
        seed_active_weights=activity,
        activity_floor=0.02,
        activity_log_floor=1e-4,
        temperature=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert seeds.grad is not None
    assert activity.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.isfinite(activity.grad).all()
    assert activity.grad[1].abs() > 0.0


def test_repulsion_inactive_duplicates_still_receive_gradient() -> None:
    seeds_xy = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [1.0, 1.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    seeds = torch.cat((seeds_xy, torch.zeros((3, 1), dtype=torch.double)), dim=1)
    seeds.retain_grad()
    activity = torch.tensor([1e-4, 1e-4, 1.0], dtype=torch.double, requires_grad=True)

    loss = Loss_rep()(
        seed_positions=seeds,
        target_dist=0.1,
        seed_active_weights=activity,
        activity_floor=0.02,
        recovery_floor=0.05,
        duplicate_recovery_strength=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad[:2]) > 0.0


def test_boundary_loss_uses_smooth_weight_floor() -> None:
    seeds = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.double, requires_grad=True)
    boundary = torch.tensor([[0.0, 0.1], [1.0, 1.0]], dtype=torch.double)
    activity = torch.tensor([0.0, 1.0], dtype=torch.double, requires_grad=True)

    loss = Loss_Boundary()(
        seeds,
        boundary,
        seed_active_weights=activity,
        activity_floor=0.02,
        margin=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert seeds.grad[0].abs().sum() > 0.0


def test_seed_active_loss_modes_are_differentiable() -> None:
    seeds = torch.tensor(
        [[-0.1, 0.5], [0.5, 0.5], [1.1, 0.5]],
        dtype=torch.double,
        requires_grad=True,
    )
    distance_outside = torch.stack(
        [
            -seeds[:, 0],
            seeds[:, 0] - 1.0,
            -seeds[:, 1],
            seeds[:, 1] - 1.0,
        ],
        dim=0,
    ).amax(dim=0)
    activity = torch.sigmoid(-distance_outside / 0.25)

    minimum = Loss_SeedActive()(
        seed_active_weights=activity,
        target_active=3.0,
        mode="minimum",
        temperature=0.25,
    )
    target = Loss_SeedActive()(
        seed_active_weights=activity,
        target_active=3.0,
        mode="target",
    )
    minimum.backward(retain_graph=True)

    assert torch.isfinite(minimum)
    assert torch.isfinite(target)
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert seeds.grad.abs().sum() > 0.0


def test_seed_active_loss_baseline_and_hard_floor_barrier() -> None:
    loss_fn = Loss_SeedActive()

    above_target = loss_fn(
        torch.full((12,), 1.0, dtype=torch.double),
        target_active=10.0,
        possible_min_active=6.0,
        baseline_strength=2.0,
        hard_barrier_strength=100.0,
        temperature=0.25,
    )
    near_target = loss_fn(
        torch.full((9,), 1.0, dtype=torch.double),
        target_active=10.0,
        possible_min_active=6.0,
        baseline_strength=2.0,
        hard_barrier_strength=100.0,
        temperature=0.25,
    )
    below_floor = loss_fn(
        torch.full((5,), 1.0, dtype=torch.double),
        target_active=10.0,
        possible_min_active=6.0,
        baseline_strength=2.0,
        hard_barrier_strength=100.0,
        temperature=0.25,
    )

    assert above_target.item() < near_target.item()
    assert below_floor.item() > 10.0 * near_target.item()


def test_seed_domain_recovery_penalizes_low_domain_activity() -> None:
    domain_activity = torch.tensor([1.0, 0.25, 0.0], dtype=torch.double, requires_grad=True)

    loss = LossSeedDomainRecovery()(domain_activity)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss > 0.0
    assert domain_activity.grad is not None
    assert torch.isfinite(domain_activity.grad).all()
    assert domain_activity.grad[-1] < 0.0
