import pytest
import torch

from Training.FEMControl import (
    AutomaticConstraintController,
    checkpoint_feasibility_key,
)
from Training.Loss_FEM import Loss_FEM
from Training.Loss_SeedValidity import minimum_seed_spacing_loss, seed_separation_loss
from Training.MainTrain import NN_Trainer, TrainingConfig


class _ScalarParam(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float64))


def test_asymmetric_target_length_band_is_below_target():
    out = NN_Trainer._target_total_length_band_loss(
        total_length=torch.tensor(99.95),
        target_total_length=100.0,
        lower_tolerance=10.0,
        upper_buffer=0.1,
        under_weight=1.0,
        over_weight=1.0,
        eps=1.0e-12,
    )
    inside = NN_Trainer._target_total_length_band_loss(
        total_length=torch.tensor(95.0),
        target_total_length=100.0,
        lower_tolerance=10.0,
        upper_buffer=0.1,
        under_weight=1.0,
        over_weight=1.0,
        eps=1.0e-12,
    )

    assert inside["penalty"].item() == pytest.approx(0.0)
    assert out["over_violation"].item() == pytest.approx(0.05, abs=1.0e-5)


def test_target_length_band_endpoints_are_exactly_feasible():
    cfg = TrainingConfig(
        optimization_mode="target_length_constrained_displacement",
        target_total_length=100.0,
        target_length_lower_tolerance=10.0,
        target_length_upper_buffer=0.1,
    )

    assert cfg.target_total_length - cfg.target_length_lower_tolerance == pytest.approx(90.0)
    assert cfg.target_total_length - cfg.target_length_upper_buffer == pytest.approx(99.9)

    for value in (90.0, 99.9):
        out = NN_Trainer._target_total_length_band_loss(
            total_length=torch.tensor(value, dtype=torch.float64),
            target_total_length=100.0,
            lower_tolerance=10.0,
            upper_buffer=0.1,
            under_weight=1.0,
            over_weight=1.0,
            eps=1.0e-12,
        )
        assert out["range_violation"].item() == pytest.approx(0.0)
        assert out["penalty"].item() == pytest.approx(0.0)


def test_target_length_band_rejects_invalid_interval():
    with pytest.raises(ValueError):
        TrainingConfig(
            optimization_mode="target_length_constrained_displacement",
            target_total_length=100.0,
            target_length_lower_tolerance=0.1,
            target_length_upper_buffer=0.1,
        )
    with pytest.raises(ValueError):
        TrainingConfig(
            optimization_mode="target_length_constrained_displacement",
            target_total_length=10.0,
            target_length_lower_tolerance=10.0,
            target_length_upper_buffer=0.1,
        )


def test_independent_constraint_controller_decays_satisfied_terms_only():
    controller = AutomaticConstraintController(growth=2.0, decay=0.5, update_interval=2)
    before = dict(controller.penalties)

    for _ in range(3):
        after = controller.update(
            violations={
                "stress": 0.2,
                "displacement": 0.0,
                "seed_spacing": 0.0,
                "length_under": 0.1,
                "length_over": 0.0,
            },
            objective_grad_norm=1.0,
            constraint_grad_norms={"stress": 2.0, "length_under": 2.0},
        )

    assert after["stress"] > before["stress"]
    assert after["length_under"] > before["length_under"]
    assert after["displacement"] < before["displacement"]
    assert after["length_over"] < before["length_over"]


def test_constraint_controller_uses_gradient_scales_when_available():
    controller = AutomaticConstraintController(
        growth=1.01,
        decay=0.5,
        gradient_ratio=2.0,
    )
    after = controller.update(
        violations={"stress": 0.1},
        objective_grad_norm=10.0,
        constraint_grad_norms={"stress": 2.0},
    )

    # The calibration target is 10, but one update may grow by at most 1.01.
    assert after["stress"] == pytest.approx(1.01)
    assert controller.last_diagnostics["stress"]["gradient_target"] == pytest.approx(10.0)


def test_seed_spacing_controller_decays_when_physical_spacing_feasible_with_repulsion():
    seeds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]],
        dtype=torch.float64,
    )
    combined, components = seed_separation_loss(
        seeds,
        min_seed_spacing=1.0,
        safety_factor=1.05,
        repulsion_factor=1.10,
        repulsion_weight=0.5,
        eps=1.0e-12,
        return_components=True,
    )
    physical_violation = torch.relu(
        (seeds.new_tensor(1.0) - components["minimum_distance"])
        / seeds.new_tensor(1.0)
    )

    assert physical_violation.item() == pytest.approx(0.0)
    assert components["weighted_repulsion"].item() > 0.0
    assert combined.item() > 0.0

    controller = AutomaticConstraintController(growth=10.0, decay=0.5, update_interval=2)
    controller.penalties["seed_spacing"] = 100.0
    for _ in range(2):
        after = controller.update(
            violations={"seed_spacing": physical_violation.item()},
            objective_grad_norm=1.0,
            constraint_grad_norms={"seed_spacing": 1.0},
        )

    assert after["seed_spacing"] == pytest.approx(50.0)


def test_seed_spacing_controller_recovers_then_decays_after_feasibility():
    controller = AutomaticConstraintController(growth=2.0, decay=0.5, update_interval=2)
    controller.penalties["seed_spacing"] = 1.0

    for _ in range(3):
        grown = controller.update(
            violations={"seed_spacing": 0.1},
            objective_grad_norm=1.0,
            constraint_grad_norms={"seed_spacing": 2.0},
        )
    assert grown["seed_spacing"] == pytest.approx(2.0)

    for _ in range(2):
        decayed = controller.update(violations={"seed_spacing": 0.0})
    assert decayed["seed_spacing"] == pytest.approx(1.0)


def test_seed_spacing_gradient_calibration_is_unweighted_by_current_coefficient():
    model = _ScalarParam(0.9)
    min_spacing = 1.0
    seed_xyz = torch.stack(
        (
            torch.zeros(3, dtype=torch.float64),
            torch.stack(
                (
                    model.value,
                    model.value.new_tensor(0.0),
                    model.value.new_tensor(0.0),
                )
            ),
        )
    )
    physical_violation = torch.relu(
        (seed_xyz.new_tensor(min_spacing) - model.value)
        / seed_xyz.new_tensor(min_spacing)
    )
    hard_loss = physical_violation.pow(2.0)

    unweighted_grad = NN_Trainer._autograd_grad_norm(hard_loss, [model])
    weighted_grad_low = NN_Trainer._autograd_grad_norm(0.5 * hard_loss, [model])
    weighted_grad_high = NN_Trainer._autograd_grad_norm(1000.0 * hard_loss, [model])

    assert physical_violation.item() == pytest.approx(0.1)
    assert unweighted_grad > 0.0
    assert NN_Trainer._autograd_grad_norm(hard_loss, [model]) == pytest.approx(
        unweighted_grad
    )
    assert weighted_grad_low == pytest.approx(0.5 * unweighted_grad)
    assert weighted_grad_high == pytest.approx(1000.0 * unweighted_grad)


def test_autograd_grad_norm_has_finite_guards():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)

    value = model(torch.ones(1, 1)).sum()
    assert NN_Trainer._autograd_grad_norm(value, [model]) == pytest.approx(1.0)
    assert NN_Trainer._autograd_grad_norm(torch.tensor(float("nan")), [model]) == 0.0


def test_automatic_calibration_terms_match_l_total_contributions():
    eps = 1.0e-12
    target = 100.0
    lower_tolerance = 10.0
    upper_buffer = 0.1
    lower = target - lower_tolerance
    upper = target - upper_buffer
    band_width = lower_tolerance - upper_buffer
    assert (lower, upper) == pytest.approx((90.0, 99.9))

    length_model = _ScalarParam(80.0)
    length_reference = torch.tensor(100.0, dtype=torch.float64)
    length_weight = 3.0
    length_primary = length_weight * (length_model.value / length_reference)
    assert NN_Trainer._autograd_grad_norm(
        length_primary,
        [length_model],
        eps=eps,
    ) == pytest.approx(length_weight / float(length_reference))

    under_model = _ScalarParam(lower - 0.01)
    under_band = NN_Trainer._target_total_length_band_loss(
        total_length=under_model.value,
        target_total_length=target,
        lower_tolerance=lower_tolerance,
        upper_buffer=upper_buffer,
        under_weight=1.0,
        over_weight=1.0,
        eps=eps,
    )
    under_unweighted = (under_band["under_violation"] / band_width).pow(2.0)
    under_coeff = 17.0
    assert under_band["under_violation"].item() == pytest.approx(0.01)
    assert under_band["over_violation"].item() == pytest.approx(0.0)
    assert NN_Trainer._autograd_grad_norm(
        under_coeff * under_unweighted,
        [under_model],
        eps=eps,
    ) == pytest.approx(
        under_coeff
        * NN_Trainer._autograd_grad_norm(under_unweighted, [under_model], eps=eps)
    )

    over_model = _ScalarParam(upper + 0.01)
    over_band = NN_Trainer._target_total_length_band_loss(
        total_length=over_model.value,
        target_total_length=target,
        lower_tolerance=lower_tolerance,
        upper_buffer=upper_buffer,
        under_weight=1.0,
        over_weight=1.0,
        eps=eps,
    )
    over_unweighted = (over_band["over_violation"] / band_width).pow(2.0)
    over_coeff = 19.0
    assert over_band["under_violation"].item() == pytest.approx(0.0)
    assert over_band["over_violation"].item() == pytest.approx(0.01)
    assert NN_Trainer._autograd_grad_norm(
        over_coeff * over_unweighted,
        [over_model],
        eps=eps,
    ) == pytest.approx(
        over_coeff
        * NN_Trainer._autograd_grad_norm(over_unweighted, [over_model], eps=eps)
    )

    seed_model = _ScalarParam(0.99)
    seed_xyz = torch.stack(
        (
            torch.zeros(3, dtype=torch.float64),
            torch.stack(
                (
                    seed_model.value,
                    seed_model.value.new_tensor(0.0),
                    seed_model.value.new_tensor(0.0),
                )
            ),
        )
    )
    seed_loss = minimum_seed_spacing_loss(seed_xyz, min_seed_spacing=1.0, eps=eps)
    physical_seed_violation = torch.relu((1.0 - seed_model.value) / 1.0)
    seed_coeff = 23.0
    assert seed_loss.item() == pytest.approx(physical_seed_violation.item() ** 2)
    assert NN_Trainer._autograd_grad_norm(
        seed_coeff * seed_loss,
        [seed_model],
        eps=eps,
    ) == pytest.approx(
        seed_coeff
        * NN_Trainer._autograd_grad_norm(seed_loss, [seed_model], eps=eps)
    )

    stress_model = _ScalarParam(1.001)
    stress_excess = torch.relu(stress_model.value - 1.0)
    stress_hard_loss = stress_excess + 0.5 * stress_excess.pow(2.0)
    stress_coeff = 29.0
    assert stress_hard_loss.item() > 0.0
    assert NN_Trainer._autograd_grad_norm(
        stress_coeff * stress_hard_loss,
        [stress_model],
        eps=eps,
    ) == pytest.approx(
        stress_coeff
        * NN_Trainer._autograd_grad_norm(stress_hard_loss, [stress_model], eps=eps)
    )

    displacement_model = _ScalarParam(1.001)
    displacement_excess = torch.relu(displacement_model.value - 1.0)
    displacement_hard_loss = displacement_excess + 0.5 * displacement_excess.pow(2.0)
    displacement_coeff = 31.0
    assert displacement_hard_loss.item() > 0.0
    assert NN_Trainer._autograd_grad_norm(
        displacement_coeff * displacement_hard_loss,
        [displacement_model],
        eps=eps,
    ) == pytest.approx(
        displacement_coeff
        * NN_Trainer._autograd_grad_norm(
            displacement_hard_loss,
            [displacement_model],
            eps=eps,
        )
    )


def test_upper_p_norm_responds_to_localized_displacement_peak():
    field = torch.zeros(4096, dtype=torch.float64)
    field[0] = 1.23

    mean_like = Loss_FEM._soft_p_norm(field, p=12.0, eps=1.0e-12)
    upper = Loss_FEM._upper_p_norm(field, p=12.0, eps=1.0e-12)

    assert mean_like.item() < 1.0
    assert upper.item() == pytest.approx(1.23)


def test_hard_limit_penalty_has_gradient_just_above_each_limit():
    for ratio_value in (1.0001, 1.01):
        ratio = torch.tensor(ratio_value, dtype=torch.float64, requires_grad=True)
        excess = torch.relu(ratio - 1.0)
        penalty = excess + 0.5 * excess.pow(2.0)
        penalty.backward()

        assert penalty.item() > 0.0
        assert ratio.grad is not None
        assert ratio.grad.item() > 0.9


def test_feasible_checkpoint_ranking_respects_mode_priorities():
    min_mode_ok, min_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.8,
        physical_stress_ratio=0.9,
        physical_feasible=True,
        seed_spacing_feasible=True,
        L_total=10.0,
        raw_total_fiber_length=80.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
        optimization_mode="constrained_displacement",
    )
    target_ok, target_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.6,
        physical_stress_ratio=0.95,
        physical_feasible=True,
        seed_spacing_feasible=True,
        L_total=10.0,
        raw_total_fiber_length=99.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
        optimization_mode="target_length_constrained_displacement",
        target_length_feasible=True,
    )
    invalid_short, _ = checkpoint_feasibility_key(
        physical_displacement_ratio=1.1,
        physical_stress_ratio=0.9,
        physical_feasible=False,
        seed_spacing_feasible=True,
        L_total=1.0,
        raw_total_fiber_length=40.0,
        mechanical_violation=0.1,
        total_loss_is_finite=True,
        fem_is_valid=True,
        optimization_mode="constrained_displacement",
    )

    assert min_mode_ok
    assert min_key[:2] == pytest.approx((80.0, 0.9))
    assert target_ok
    assert target_key[:3] == pytest.approx((0.6, 0.95, 99.0))
    assert not invalid_short


def test_target_checkpoint_replacement_uses_final_tuple_element_as_step():
    older_key = (0.6, 0.95, 99.0, 10.0)
    better_key = (0.5, 0.98, 99.5, 20.0)

    assert better_key < older_key
    assert NN_Trainer._best_feasible_step(better_key) == 20


def test_stage1_stage2_fem_activation_is_explicit():
    cfg = TrainingConfig(stage1_enable_fem=False, stage2_enable_fem=True)
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = cfg

    assert not trainer._stage_settings_for_stage_id(1)["enable_fem"]
    assert trainer._stage_settings_for_stage_id(2)["enable_fem"]
    assert trainer._first_physical_stage_id() == 2
