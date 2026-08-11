import torch
from types import SimpleNamespace

from Training.FEMControl import (
    checkpoint_feasibility_key,
    select_final_checkpoint,
    update_adaptive_fem_lambda,
    validate_optimizer_parameter_coverage,
)
from Training.Loss_FEM import Loss_FEM
from neuraltomo_fem.anisotropicFE_new import H8_anisotropic_K


class _DummyShellProblem:
    def build_fem_fields_from_decoder_torch(self, rho_surface, fiber_surface):
        return {
            "density": rho_surface,
            "shell_occupancy": torch.ones_like(rho_surface),
            "phi": torch.zeros_like(rho_surface),
            "theta": torch.zeros_like(rho_surface),
        }


class _DummyFEM:
    def __init__(self, stress_scale: float, displacement_scale: float, displacement_mag_scale: float | None = None):
        self.stress_scale = float(stress_scale)
        self.displacement_scale = float(displacement_scale)
        self.displacement_mag_scale = displacement_mag_scale
        self.fe = SimpleNamespace(
            stress_vm=None,
            displacement_mag_elem=None,
            displacement_load_dir_elem=None,
            displacement_mag_loaded_boundary=None,
            displacement_load_dir_loaded_boundary=None,
        )

    def __call__(self, stiffness_factor, phi, theta, penal=1.0):
        self.fe.stress_vm = stiffness_factor * self.stress_scale
        if self.displacement_mag_scale is not None:
            self.fe.displacement_mag_elem = stiffness_factor * float(self.displacement_mag_scale)
            self.fe.displacement_mag_loaded_boundary = self.fe.displacement_mag_elem
        self.fe.displacement_load_dir_elem = stiffness_factor * self.displacement_scale
        self.fe.displacement_load_dir_loaded_boundary = self.fe.displacement_load_dir_elem
        return self.fe.stress_vm, self.fe.stress_vm.sum()


def _dummy_trainer(stress_scale: float, displacement_scale: float, displacement_mag_scale: float | None = None):
    trainer = SimpleNamespace()
    trainer.cfg = SimpleNamespace(fem_training_safety_factor=0.95)
    trainer.shell_problem = _DummyShellProblem()
    trainer.fem = _DummyFEM(stress_scale, displacement_scale, displacement_mag_scale)
    trainer.fem_debug_history = []
    trainer.last_fem_debug = None
    trainer._record_invalid_fem_debug = lambda debug, reason, save: None
    return trainer


def test_fem_loss_is_nonzero_below_limits_and_keeps_gradient():
    trainer = _dummy_trainer(stress_scale=5.0, displacement_scale=2.0)
    loss_fn = Loss_FEM(trainer)
    rho = torch.ones(4, dtype=torch.float64, requires_grad=True)
    fiber = torch.ones(4, 3, dtype=torch.float64)

    out = loss_fn.evaluate(
        rho_surface=rho,
        fiber_surface=fiber,
        max_displacement=10.0,
        yield_strength=10.0,
        baseline_weight=0.05,
        violation_power=4.0,
        stress_density_threshold=None,
    )

    assert out["fem_valid"]
    assert out["training_feasible"]
    assert out["physical_feasible"]
    assert out["baseline_fem_loss"].item() > 0.0
    assert out["violation_fem_loss"].item() == 0.0
    assert out["fem_total"].item() > 0.0

    out["fem_total"].backward()
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()
    assert rho.grad.abs().sum().item() > 0.0


def test_fem_loss_rises_sharply_above_limit():
    fiber = torch.ones(4, 3, dtype=torch.float64)
    safe_rho = torch.ones(4, dtype=torch.float64, requires_grad=True)
    failed_rho = torch.ones(4, dtype=torch.float64, requires_grad=True)

    safe_out = Loss_FEM(_dummy_trainer(stress_scale=5.0, displacement_scale=2.0)).evaluate(
        rho_surface=safe_rho,
        fiber_surface=fiber,
        max_displacement=10.0,
        yield_strength=10.0,
        baseline_weight=0.05,
        violation_power=4.0,
        stress_density_threshold=None,
    )
    failed_out = Loss_FEM(_dummy_trainer(stress_scale=15.0, displacement_scale=2.0)).evaluate(
        rho_surface=failed_rho,
        fiber_surface=fiber,
        max_displacement=10.0,
        yield_strength=10.0,
        baseline_weight=0.05,
        violation_power=4.0,
        stress_density_threshold=None,
    )

    assert failed_out["physical_stress_ratio"].item() > 1.0
    assert failed_out["violation_fem_loss"].item() > 0.0
    assert failed_out["fem_total"].item() > safe_out["fem_total"].item() * 5.0


def test_fem_displacement_constraint_prefers_magnitude_over_load_direction():
    trainer = _dummy_trainer(
        stress_scale=0.0,
        displacement_scale=2.0,
        displacement_mag_scale=12.0,
    )
    loss_fn = Loss_FEM(trainer)
    rho = torch.ones(4, dtype=torch.float64, requires_grad=True)
    fiber = torch.ones(4, 3, dtype=torch.float64)

    out = loss_fn.evaluate(
        rho_surface=rho,
        fiber_surface=fiber,
        max_displacement=10.0,
        yield_strength=10.0,
        baseline_weight=0.0,
        stress_density_threshold=None,
    )

    assert torch.allclose(out["displacement_max"], out["displacement_max"].new_tensor(12.0))
    assert torch.allclose(out["physical_displacement_ratio"], out["physical_displacement_ratio"].new_tensor(1.2))
    assert not out["physical_feasible"]


def test_constraint_excess_is_zero_inside_limit_and_ratio_based():
    reference = torch.ones((), requires_grad=True)
    value = reference * 12.0

    loss, excess, ratio = Loss_FEM._constraint_excess(
        value=value,
        limit=10.0,
        power=2.0,
        reference=reference,
        eps=0.0,
    )

    assert torch.allclose(ratio, ratio.new_tensor(1.2))
    assert torch.allclose(excess, excess.new_tensor(0.2))
    assert torch.allclose(loss, loss.new_tensor(0.04))

    loss.backward()
    assert reference.grad is not None
    assert torch.isfinite(reference.grad)
    assert reference.grad.abs() > 0

    safe_reference = torch.ones((), requires_grad=True)

    safe_loss, safe_excess, safe_ratio = Loss_FEM._constraint_excess(
        value=safe_reference * 5.0,
        limit=10.0,
        power=2.0,
        reference=safe_reference,
        eps=0.0,
    )

    assert torch.allclose(safe_ratio, safe_ratio.new_tensor(0.5))
    assert torch.allclose(safe_excess, safe_excess.new_tensor(0.0))
    assert torch.allclose(safe_loss, safe_loss.new_tensor(0.0))
    safe_loss.backward()
    assert safe_reference.grad is not None
    assert torch.allclose(safe_reference.grad, safe_reference.grad.new_tensor(0.0))


def test_simp_stiffness_floor_is_applied_once():
    rho = torch.zeros(3)
    rho_min_ratio = 1.0e-6
    penal = 3.0
    stiffness_factor = rho_min_ratio + (1.0 - rho_min_ratio) * rho.pow(penal)

    assert torch.allclose(stiffness_factor, stiffness_factor.new_full((3,), 1.0e-6))


def test_physical_b_matrix_scales_by_derivative_direction():
    h8 = H8_anisotropic_K(
        device=torch.device("cpu"),
        element_size=(1.0, 2.0, 4.0),
        material_E1=100.0,
        material_E2=10.0,
        material_E3=10.0,
        material_nu12=0.25,
        material_nu23=0.25,
        material_nu13=0.25,
        material_G12=5.0,
        material_G23=4.0,
        material_G13=5.0,
    )
    B = h8.physical_B(torch.tensor([[0.0, 0.0, 0.0]]))[0]

    assert torch.allclose(B[5, 0], B.new_tensor(-0.125))  # du/dy uses 1 / hy.
    assert torch.allclose(B[5, 1], B.new_tensor(-0.25))   # dv/dx uses 1 / hx.
    assert torch.allclose(B[4, 0], B.new_tensor(-0.0625)) # du/dz uses 1 / hz.


def test_invalid_fem_attempt_leaves_parameters_and_scheduler_unchanged():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    scheduler = CountingScheduler()
    before = [p.detach().clone() for p in model.parameters()]

    fem_required = True
    fem_is_valid = False
    if not (fem_required and not fem_is_valid):
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    after = [p.detach().clone() for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(after, before))
    assert scheduler.steps == 0
    assert all(p.grad is None for p in model.parameters())


def test_invalid_fem_grows_adaptive_lambda():
    lambda_after, reason = update_adaptive_fem_lambda(
        lambda_before=2.0,
        fem_is_valid=False,
        constraint_violation=float("nan"),
        tolerance=0.0,
        growth=1.5,
        decay=0.9,
        lambda_min=1.0,
        lambda_max=10.0,
    )

    assert lambda_after > 2.0
    assert reason == "grow_invalid_fem"


def test_downstream_element_stiffness_uses_rho_min_directly():
    h8 = H8_anisotropic_K(
        device=torch.device("cpu"),
        element_size=(1.0, 1.0, 1.0),
        material_E1=100.0,
        material_E2=10.0,
        material_E3=10.0,
        material_nu12=0.25,
        material_nu23=0.25,
        material_nu13=0.25,
        material_G12=5.0,
        material_G23=4.0,
        material_G13=5.0,
    )
    phi = torch.zeros(1)
    theta = torch.zeros(1)
    rho_min = 1.0e-6
    k_min = h8.angle2Ke(phi, theta, torch.full((1,), rho_min))[0]
    k_solid = h8.angle2Ke(phi, theta, torch.ones(1))[0]
    mask = k_solid.abs() > 1.0e-12
    ratio = (k_min[mask] / k_solid[mask]).mean()

    assert torch.allclose(ratio, ratio.new_tensor(rho_min), rtol=1e-4, atol=1e-10)


def test_feasible_checkpoint_beats_shorter_failed_checkpoint():
    failed_feasible, failed_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.8,
        physical_stress_ratio=1.2,
            seed_spacing_feasible=True,
        design_score=10.0,
        raw_total_fiber_length=1.0,
        mechanical_violation=0.2,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    ok_feasible, ok_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.9,
        physical_stress_ratio=0.9,
        seed_spacing_feasible=True,
        design_score=20.0,
        raw_total_fiber_length=2.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert failed_feasible is False
    assert ok_feasible is True
    assert ok_key[0] == 20.0

    shorter_feasible, shorter_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.95,
        physical_stress_ratio=0.95,
        seed_spacing_feasible=True,
        design_score=15.0,
        raw_total_fiber_length=1.5,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    assert shorter_feasible is True
    assert shorter_key < ok_key


def test_orientation_rotates_e1_direction():
    h8 = H8_anisotropic_K(
        device=torch.device("cpu"),
        element_size=(1.0, 1.0, 1.0),
        material_E1=100.0,
        material_E2=10.0,
        material_E3=10.0,
        material_nu12=0.25,
        material_nu23=0.25,
        material_nu13=0.25,
        material_G12=5.0,
        material_G23=4.0,
        material_G13=5.0,
    )
    h8.angle2Ke(
        phi=torch.tensor([0.0]),
        theta=torch.tensor([torch.pi / 2.0]),
        stiffness_factor=torch.ones(1),
    )
    c_x = h8.temp_C[0]
    h8.angle2Ke(
        phi=torch.tensor([torch.pi / 2.0]),
        theta=torch.tensor([torch.pi / 2.0]),
        stiffness_factor=torch.ones(1),
    )
    c_y = h8.temp_C[0]

    assert c_x[0, 0] > c_x[1, 1]
    assert c_y[1, 1] > c_y[0, 0]


def test_nonadaptive_final_selection_prefers_feasible_design():
    feasible = {"stage_monitor_raw": 8.0, "row": {"stage_monitor_raw": 8.0, "design_score": 3.0, "loss_total_fiber_length": 12.0}}
    infeasible = {"row": {"loss_total_fiber_length": 5.0}}

    source, selected, score = select_final_checkpoint(
        proposed_checkpoint=None,
        proposed_source="unavailable",
        proposed_score=float("inf"),
        best_feasible_checkpoint=feasible,
        best_feasible_key=(3.0, 12.0, 10),
        best_infeasible_checkpoint=infeasible,
        best_infeasible_key=(0.2, 2.0, 11),
        last_valid_checkpoint=infeasible,
    )

    assert source == "best_feasible"
    assert selected is feasible
    assert score == 3.0


def test_returned_best_score_equals_selected_feasible_design_score():
    feasible = {"stage_id": 2, "row": {"stage": 2, "design_score": -4.5, "loss_total_fiber_length": 22.0}}

    source, selected, score = select_final_checkpoint(
        best_feasible_checkpoint=feasible,
        best_feasible_key=(-4.5, 22.0, 20),
        first_physical_stage=2,
    )

    assert source == "best_feasible"
    assert selected is feasible
    assert score == -4.5


def test_adaptive_stage_proposal_is_overridden_by_feasible_design():
    stage_failed = {"row": {"L_total": 0.1}}
    feasible = {"stage_monitor_raw": 8.0, "row": {"stage_monitor_raw": 8.0, "design_score": 3.0, "loss_total_fiber_length": 12.0}}

    source, selected, _score = select_final_checkpoint(
        proposed_checkpoint=stage_failed,
        proposed_source="stage_best_raw",
        proposed_score=0.1,
        best_feasible_checkpoint=feasible,
        best_feasible_key=(3.0, 12.0, 10),
        best_infeasible_checkpoint=stage_failed,
        best_infeasible_key=(0.4, 2.0, 11),
        last_valid_checkpoint=stage_failed,
    )

    assert source == "best_feasible"
    assert selected is feasible


def test_final_selection_ignores_stage1_candidate_for_physical_result():
    stage1_short = {"stage_id": 1, "row": {"stage": 1, "loss_total_fiber_length": 1.0}}
    stage2_feasible = {"stage_id": 2, "stage_monitor_raw": 6.0, "row": {"stage": 2, "stage_monitor_raw": 6.0, "design_score": 2.0, "loss_total_fiber_length": 4.0}}

    source, selected, score = select_final_checkpoint(
        proposed_checkpoint=stage1_short,
        proposed_source="stage_best_raw",
        proposed_score=1.0,
        best_feasible_checkpoint=stage2_feasible,
        best_feasible_key=(2.0, 4.0, 20),
        best_infeasible_checkpoint=stage1_short,
        best_infeasible_key=(0.1, 3.0, 10),
        last_valid_checkpoint=stage1_short,
        first_physical_stage=2,
    )

    assert source == "best_feasible"
    assert selected is stage2_feasible
    assert score == 2.0


def test_final_selection_returns_no_checkpoint_when_only_stage1_exists():
    stage1_checkpoint = {
        "stage_id": 1,
        "row": {"stage": 1, "stage_monitor_raw": 1.0},
        "stage_monitor_raw": 1.0,
    }

    source, selected, score = select_final_checkpoint(
        proposed_checkpoint=stage1_checkpoint,
        proposed_source="stage_best_raw",
        proposed_score=1.0,
        best_feasible_checkpoint=None,
        best_feasible_key=(float("inf"), float("inf"), float("inf")),
        best_infeasible_checkpoint=stage1_checkpoint,
        best_infeasible_key=(1.0, 1.0, 1),
        last_valid_checkpoint=stage1_checkpoint,
        first_physical_stage=2,
    )

    assert source == "unavailable"
    assert selected is None
    assert score == float("inf")


def test_feasible_checkpoint_key_prefers_design_score_before_fiber_and_step():
    old_key = (10.0, 97.0, 97)
    lower_design_score_key = (8.0, 101.0, 128)
    same_design_shorter_fiber_key = (10.0, 96.0, 128)
    same_design_same_fiber_earlier_key = (10.0, 97.0, 96)

    assert lower_design_score_key < old_key
    assert same_design_shorter_fiber_key < old_key
    assert same_design_same_fiber_earlier_key < old_key


def test_best_feasible_score_defaults_to_checkpoint_stage_monitor():
    feasible = {
        "stage_id": 2,
        "stage_monitor_raw": 3.25,
        "row": {"stage": 2, "stage_monitor_raw": 3.25, "design_score": 1.25, "loss_total_fiber_length": 9.5},
    }

    source, selected, score = select_final_checkpoint(
        best_feasible_checkpoint=feasible,
        best_feasible_key=None,
        first_physical_stage=2,
    )

    assert source == "best_feasible"
    assert selected is feasible
    assert score == 1.25


def test_optimizer_coverage_succeeds_for_ppnet_and_decoder():
    ppnet = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(
        list(ppnet.parameters()) + list(decoder.parameters()),
        lr=0.1,
    )

    validate_optimizer_parameter_coverage(
        named_trainable_modules=[("ppnet", ppnet), ("decoder", decoder)],
        optimizer=optimizer,
    )


def test_optimizer_coverage_detects_missing_decoder_params():
    ppnet = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(ppnet.parameters(), lr=0.1)

    try:
        validate_optimizer_parameter_coverage(
            named_trainable_modules=[("ppnet", ppnet), ("decoder", decoder)],
            optimizer=optimizer,
        )
    except RuntimeError as exc:
        assert "decoder.weight" in str(exc)
    else:
        raise AssertionError("Expected missing decoder parameter error.")


def test_optimizer_coverage_detects_duplicate_parameters():
    ppnet = torch.nn.Linear(2, 2)

    class DummyOptimizer:
        param_groups = [
            {"params": [ppnet.weight]},
            {"params": [ppnet.weight]},
        ]

    optimizer = DummyOptimizer()

    try:
        validate_optimizer_parameter_coverage(
            named_trainable_modules=[("ppnet", ppnet)],
            optimizer=optimizer,
        )
    except RuntimeError as exc:
        assert "more than one optimizer parameter group" in str(exc)
    else:
        raise AssertionError("Expected duplicate optimizer parameter error.")


def test_nan_physical_ratio_is_infeasible_and_not_ranked_as_nan():
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=float("nan"),
        physical_stress_ratio=0.5,
        seed_spacing_feasible=True,
        design_score=3.0,
        raw_total_fiber_length=3.0,
        mechanical_violation=float("inf"),
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert feasible is False
    assert key[0] == float("inf")
    assert not any(torch.isnan(torch.tensor(v)) for v in key)


def test_inf_physical_ratio_is_infeasible_with_infinite_violation():
    for ratio in (float("inf"), float("-inf")):
        feasible, key = checkpoint_feasibility_key(
            physical_displacement_ratio=ratio,
            physical_stress_ratio=0.5,
        seed_spacing_feasible=True,
            design_score=3.0,
            raw_total_fiber_length=3.0,
            mechanical_violation=float("inf"),
            total_loss_is_finite=True,
            fem_is_valid=True,
        )

        assert feasible is False
        assert key[0] == float("inf")


def test_nonfinite_fiber_length_is_infeasible_and_ranked_as_inf():
    for length in (float("nan"), float("inf"), -1.0):
        feasible, key = checkpoint_feasibility_key(
            physical_displacement_ratio=0.5,
            physical_stress_ratio=0.5,
        seed_spacing_feasible=True,
            design_score=3.0,
            raw_total_fiber_length=length,
            mechanical_violation=0.0,
            total_loss_is_finite=True,
            fem_is_valid=True,
        )

        assert feasible is False
        assert key[1] == float("inf")


def test_spacing_infeasibility_rejects_checkpoint():
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.5,
        physical_stress_ratio=0.5,
        seed_spacing_feasible=False,
        design_score=3.0,
        raw_total_fiber_length=3.0,
        mechanical_violation=0.25,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert feasible is False
    assert key == (0.25, 3.0, 0.0)


def test_feasible_ranking_uses_design_score_not_physical_ratio():
    a_feasible, a_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.99,
        physical_stress_ratio=0.99,
        seed_spacing_feasible=True,
        design_score=4.0,
        raw_total_fiber_length=10.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    b_feasible, b_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.7,
        physical_stress_ratio=0.7,
        seed_spacing_feasible=True,
        design_score=5.0,
        raw_total_fiber_length=10.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert a_feasible is True
    assert b_feasible is True
    assert a_key < b_key


def test_physical_feasible_false_rejects_candidate_even_when_ratios_pass():
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.5,
        physical_stress_ratio=0.5,
        physical_feasible=False,
        seed_spacing_feasible=True,
        design_score=3.0,
        raw_total_fiber_length=10.0,
        mechanical_violation=0.25,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert feasible is False
    assert key == (0.25, 3.0, 0.0)


def test_feasible_ranking_uses_design_score_not_fem_training_contribution():
    higher_train_loss = 1.90 + 2.0 * 0.20
    lower_train_loss = 2.00 + 2.0 * 0.01
    assert higher_train_loss > lower_train_loss

    a_feasible, a_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.8,
        physical_stress_ratio=0.8,
        seed_spacing_feasible=True,
        design_score=1.90,
        raw_total_fiber_length=10.0,
        mechanical_violation=0.0,
        global_step=210,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    b_feasible, b_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.8,
        physical_stress_ratio=0.8,
        seed_spacing_feasible=True,
        design_score=2.00,
        raw_total_fiber_length=10.0,
        mechanical_violation=0.0,
        global_step=211,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert a_feasible is True
    assert b_feasible is True
    assert a_key < b_key


def test_feasible_ranking_uses_global_step_as_final_tiebreak():
    earlier_feasible, earlier_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.7,
        physical_stress_ratio=0.7,
        seed_spacing_feasible=True,
        design_score=7.0,
        raw_total_fiber_length=100.0,
        mechanical_violation=0.0,
        global_step=500,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    later_feasible, later_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.7,
        physical_stress_ratio=0.7,
        seed_spacing_feasible=True,
        design_score=7.0,
        raw_total_fiber_length=100.0,
        mechanical_violation=0.0,
        global_step=600,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert earlier_feasible is True
    assert later_feasible is True
    assert earlier_key < later_key


def test_spacing_violation_enters_infeasible_ranking():
    spacing_violation = 0.25
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.7,
        physical_stress_ratio=0.7,
        seed_spacing_feasible=False,
        design_score=6.0,
        raw_total_fiber_length=50.0,
        mechanical_violation=spacing_violation,
        global_step=10,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert feasible is False
    assert spacing_violation > 0.0
    assert key == (0.25, 6.0, 10.0)
