import torch

from Training.FEMControl import checkpoint_feasibility_key
from Training.Loss_SeedValidity import (
    minimum_seed_spacing_loss,
    seed_trim_boundary_loss,
    signed_trim_boundary_distance,
)


def _square_with_hole(dtype=torch.float64):
    outer = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=dtype,
    )
    hole = torch.tensor(
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        dtype=dtype,
    )
    boundary = torch.cat((outer, hole), dim=0)
    offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    loop_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return boundary, offsets, loop_id


def test_spacing_loss_zero_positive_and_monotonic():
    far = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    close = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float64)
    less_close = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)

    assert minimum_seed_spacing_loss(far, min_seed_spacing=1.0).item() == 0.0
    assert minimum_seed_spacing_loss(close, min_seed_spacing=1.0).item() > 0.0
    assert minimum_seed_spacing_loss(less_close, min_seed_spacing=1.0) < minimum_seed_spacing_loss(close, min_seed_spacing=1.0)


def test_spacing_gradient_separates_close_pair():
    seeds = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    loss = minimum_seed_spacing_loss(seeds, min_seed_spacing=1.0)
    loss.backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert seeds.grad[0, 0] > 0.0
    assert seeds.grad[1, 0] < 0.0


def test_trim_boundary_loss_outer_and_inner_hole():
    boundary, offsets, loop_id = _square_with_hole()
    safe_inside = torch.tensor([[0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    near_boundary = torch.tensor([[0.02, 0.5]], dtype=torch.float64)
    outside = torch.tensor([[-0.1, 0.5]], dtype=torch.float64)
    in_hole = torch.tensor([[0.5, 0.5]], dtype=torch.float64)

    safe_loss, _ = seed_trim_boundary_loss(
        safe_inside,
        boundary,
        offsets,
        boundary_curve_loop_id=loop_id,
        boundary_margin=0.05,
    )
    near_loss, _ = seed_trim_boundary_loss(near_boundary, boundary, offsets, boundary_curve_loop_id=loop_id, boundary_margin=0.05)
    outside_loss, _ = seed_trim_boundary_loss(outside, boundary, offsets, boundary_curve_loop_id=loop_id, boundary_margin=0.05)
    hole_signed = signed_trim_boundary_distance(in_hole, boundary, offsets, boundary_curve_loop_id=loop_id)

    assert safe_loss.item() == 0.0
    assert near_loss.item() == 0.0
    assert outside_loss.item() > 0.0
    assert hole_signed.item() < 0.0


def test_trim_boundary_loss_accepts_piece_level_loop_ids():
    outer = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    hole = torch.tensor(
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        dtype=torch.float64,
    )
    boundary = torch.cat((outer[:2], outer[2:], hole[:2], hole[2:]), dim=0)
    offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    piece_loop_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    signed_inside = signed_trim_boundary_distance(
        torch.tensor([[0.2, 0.2]], dtype=torch.float64),
        boundary,
        offsets,
        boundary_curve_loop_id=piece_loop_id,
    )
    signed_hole = signed_trim_boundary_distance(
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        boundary,
        offsets,
        boundary_curve_loop_id=piece_loop_id,
    )

    assert signed_inside.item() > 0.0
    assert signed_hole.item() < 0.0


def test_outside_boundary_gradient_moves_toward_domain():
    boundary, offsets, loop_id = _square_with_hole()
    seed = torch.tensor([[-0.1, 0.5]], dtype=torch.float64, requires_grad=True)
    loss, _ = seed_trim_boundary_loss(seed, boundary, offsets, boundary_curve_loop_id=loop_id, boundary_margin=0.05)
    loss.backward()
    assert seed.grad is not None
    assert torch.isfinite(seed.grad).all()
    assert seed.grad[0, 0] < 0.0


def test_checkpoint_feasibility_uses_spacing_only():
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.5,
        physical_stress_ratio=0.5,
        seed_spacing_feasible=True,
        design_score=2.0,
        raw_total_fiber_length=3.0,
        mechanical_violation=0.25,
        total_loss_is_finite=True,
        fem_is_valid=True,
        global_step=7,
    )
    assert feasible is True
    assert key == (2.0, 3.0, 7.0)
