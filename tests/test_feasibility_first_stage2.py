import math
import json
from pathlib import Path

import pytest
import torch

from Training.Loss_SeedValidity import minimum_seed_spacing_loss
from Training.MainTrain import NN_Trainer, TrainingConfig
from Training.FEMControl import checkpoint_feasibility_key


def _trainer(cfg: TrainingConfig | None = None) -> NN_Trainer:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = cfg if cfg is not None else TrainingConfig()
    return trainer


def test_stage2_fiber_reference_normalization_keeps_current_gradient() -> None:
    raw_start = torch.tensor(80.0, requires_grad=True)
    raw_current = torch.tensor(60.0, requires_grad=True)

    reference = raw_start.detach().clone()
    normalized = NN_Trainer._stage2_fiber_normalized(
        loss_total_fiber_length=raw_current,
        stage2_fiber_reference=reference,
        reference_eps=1.0e-12,
        fallback_normalizer=1.0,
    )

    assert normalized.item() == pytest.approx(0.75)
    normalized.backward()
    assert raw_current.grad.item() == pytest.approx(1.0 / 80.0)
    assert raw_start.grad is None


def test_fixed_stage2_reference_is_captured_once_and_detached() -> None:
    references: dict[str, torch.Tensor] = {}
    first = torch.tensor(80.0, requires_grad=True)
    second = torch.tensor(40.0, requires_grad=True)

    NN_Trainer._capture_fixed_stage2_reference(
        references,
        "total_fiber_length",
        first,
        enabled=True,
        reference_eps=1.0e-12,
    )
    NN_Trainer._capture_fixed_stage2_reference(
        references,
        "total_fiber_length",
        second,
        enabled=True,
        reference_eps=1.0e-12,
    )

    assert references["total_fiber_length"].item() == pytest.approx(80.0)
    assert not references["total_fiber_length"].requires_grad
    normalized = NN_Trainer._fixed_reference_normalized(
        second,
        references["total_fiber_length"],
        reference_eps=1.0e-12,
    )
    normalized.backward()
    assert second.grad.item() == pytest.approx(1.0 / 80.0)
    assert first.grad is None


def test_disabled_stage2_loss_does_not_capture_reference() -> None:
    references: dict[str, torch.Tensor] = {}

    NN_Trainer._capture_fixed_stage2_reference(
        references,
        "cvt",
        torch.tensor(2.0, requires_grad=True),
        enabled=False,
        reference_eps=1.0e-12,
    )

    assert "cvt" not in references


def test_spacing_barrier_is_zero_when_safe() -> None:
    seeds = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = minimum_seed_spacing_loss(
    seeds,
    min_seed_spacing=1.0,
)

    assert loss.item() == pytest.approx(0.0)


def test_spacing_barrier_increases_when_seeds_approach() -> None:
    losses = []
    for distance in (0.9, 0.5, 0.1):
        seeds = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        loss = minimum_seed_spacing_loss(
    seeds,
    min_seed_spacing=1.0,
)
        loss.backward()
        assert torch.isfinite(seeds.grad).all()
        assert seeds.grad.abs().sum().item() > 0.0
        losses.append(loss.item())

    assert losses[0] < losses[1] < losses[2]


def test_spacing_barrier_uses_all_seed_pairs() -> None:
    seeds = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )

    loss = minimum_seed_spacing_loss(
    seeds,
    min_seed_spacing=1.0,
)

    assert loss.item() > 0.0


def test_feasible_stage2_objective_includes_complete_fem_total() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(2.0),
        fem_violation_loss=torch.tensor(0.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(0.8),
        loss_cvt_normalized=torch.tensor(1.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=10.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.03,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "feasible_design"
    assert design_score.item() == pytest.approx(0.8 + 0.03)
    assert loss.item() == pytest.approx(design_score.item() + 10.0 * 2.0)
    assert loss.item() > design_score.item()


def test_stage2_training_loss_keeps_fem_total_separate_from_design_score() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(0.1094),
        fem_violation_loss=torch.tensor(0.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(1.94624),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=2.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "feasible_design"
    assert design_score.item() == pytest.approx(1.94624)
    assert loss.item() == pytest.approx(2.16504)


def test_stage2_margin_only_fem_loss_stays_feasible_design() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(0.5),
        fem_violation_loss=torch.tensor(0.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(1.2),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=2.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "feasible_design"
    assert design_score.item() == pytest.approx(1.2)
    assert loss.item() == pytest.approx(2.2)


def test_target_total_length_band_penalizes_overrun_harder_than_underrun() -> None:
    under = NN_Trainer._target_total_length_band_loss(
        total_length=torch.tensor(89.0),
        target_total_length=100.0,
        lower_tolerance=10.0,
        upper_buffer=0.1,
        under_weight=1.0,
        over_weight=100.0,
        eps=1.0e-12,
    )
    over = NN_Trainer._target_total_length_band_loss(
        total_length=torch.tensor(100.9),
        target_total_length=100.0,
        lower_tolerance=10.0,
        upper_buffer=0.1,
        under_weight=1.0,
        over_weight=100.0,
        eps=1.0e-12,
    )
    inside = NN_Trainer._target_total_length_band_loss(
        total_length=torch.tensor(99.9),
        target_total_length=100.0,
        lower_tolerance=10.0,
        upper_buffer=0.1,
        under_weight=1.0,
        over_weight=100.0,
        eps=1.0e-12,
    )

    assert inside["penalty"].item() == pytest.approx(0.0)
    assert under["under_violation"].item() == pytest.approx(1.0)
    assert over["over_violation"].item() == pytest.approx(1.0)
    assert over["penalty"].item() == pytest.approx(100.0 * under["penalty"].item())


def test_target_length_stage2_objective_uses_displacement_and_length_penalty() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(0.25),
        fem_violation_loss=torch.tensor(0.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(99.0),
        loss_cvt_normalized=torch.tensor(0.2),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.3),
        displacement_objective=torch.tensor(0.7),
        target_length_penalty=torch.tensor(4.0),
        optimization_mode="target_length_constrained_displacement",
        lam_fem_step=2.0,
        lam_total_fiber_length_step=10.0,
        lam_cvt_step=3.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=5.0,
        displacement_objective_weight=11.0,
    )

    expected_design = 11.0 * 0.7 + 10.0 * 4.0 + 3.0 * 0.2 + 5.0 * 0.3
    assert mode == "target_length_feasible_displacement_design"
    assert design_score.item() == pytest.approx(expected_design)
    assert loss.item() == pytest.approx(expected_design + 2.0 * 0.25)


def test_feasible_stage2_baseline_gradient_is_preserved() -> None:
    baseline_fem_loss = torch.tensor(2.0, requires_grad=True)
    violation_fem_loss = torch.tensor(0.0)
    fem_total_loss = baseline_fem_loss + violation_fem_loss

    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=fem_total_loss,
        fem_violation_loss=violation_fem_loss,
        loss_total_fiber_length_stage2_norm=torch.tensor(0.8),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=10.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "feasible_design"
    assert loss.item() == pytest.approx(design_score.item() + 10.0 * 2.0)
    loss.backward()
    assert baseline_fem_loss.grad.item() == pytest.approx(10.0)


def test_stage2_objective_matches_design_score_when_complete_fem_total_is_zero() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(0.0),
        fem_violation_loss=torch.tensor(0.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(0.8),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=10.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "feasible_design"
    assert design_score.item() == pytest.approx(0.8)
    assert loss.item() == pytest.approx(design_score.item())


def test_stage2_design_score_increases_monotonically_with_each_design_loss() -> None:
    def design_score(
        *,
        fiber: float = 1.0,
        cvt: float = 1.0,
        rep: float = 1.0,
        curve_cell: float = 1.0,
        seed: float = 1.0,
    ) -> float:
        _loss, score, _mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
            fem_total_loss=torch.tensor(0.0),
            fem_violation_loss=torch.tensor(0.0),
            loss_total_fiber_length_stage2_norm=torch.tensor(fiber),
            loss_cvt_normalized=torch.tensor(cvt),
            loss_rep_normalized=torch.tensor(rep),
            validity_loss=torch.tensor(seed),
            loss_curve_cell_normalized=torch.tensor(curve_cell),
            lam_fem_step=10.0,
            lam_total_fiber_length_step=2.0,
            lam_cvt_step=3.0,
            lam_rep_step=5.0,
            lam_l_curve_cell_step=11.0,
        )
        return float(score.item())

    baseline = design_score()
    assert design_score(cvt=2.0) > baseline
    assert design_score(rep=2.0) > baseline
    assert design_score(curve_cell=2.0) > baseline
    assert design_score(seed=2.0) > baseline
    assert design_score(fiber=2.0) > baseline


def test_infeasible_stage2_objective_is_fem_dominated() -> None:
    loss, design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=torch.tensor(6.0),
        fem_violation_loss=torch.tensor(4.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(0.8),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=10.0,
        lam_total_fiber_length_step=1.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "infeasible_recovery"
    assert design_score.item() == pytest.approx(0.8)
    assert loss.item() == pytest.approx(0.8 + 10.0 * 6.0)


def test_stage2_objective_does_not_normalize_fem() -> None:
    loss_fem = torch.tensor(4.0, requires_grad=True)

    loss, _design_score, mode = NN_Trainer._assemble_feasibility_first_stage2_loss(
        fem_total_loss=loss_fem,
        fem_violation_loss=torch.tensor(4.0),
        loss_total_fiber_length_stage2_norm=torch.tensor(0.0),
        loss_cvt_normalized=torch.tensor(0.0),
        loss_rep_normalized=torch.tensor(0.0),
        validity_loss=torch.tensor(0.0),
        loss_curve_cell_normalized=torch.tensor(0.0),
        lam_fem_step=10.0,
        lam_total_fiber_length_step=0.0,
        lam_cvt_step=0.0,
        lam_rep_step=0.0,
        lam_l_curve_cell_step=0.0,
    )

    assert mode == "infeasible_recovery"
    assert loss.item() == pytest.approx(40.0)
    loss.backward()
    assert loss_fem.grad.item() == pytest.approx(10.0)


def test_stage2_progress_log_reports_updated_feasible_best_without_generic_best() -> None:
    row = {
        "step": 210,
        "stage": 2,
        "stage_local_step": 60,
        "stage_max_steps": 500,
        "L_total": 2.16504,
        "primary_objective": 155.6,
        "stage_monitor_raw": 1.94624,
        "stage_monitor_mode": "design",
        "fem_total_loss": 0.1094,
        "fem_baseline_loss": 0.1094,
        "fem_violation_loss": 0.0,
        "lam_fem_eff": 2.0,
        "fem_constraints_active": True,
        "fem_was_evaluated": True,
        "fem_valid": True,
        "stress_max": 101.6,
        "disp_max": 0.0165,
        "min_seed_distance": 1.24,
        "loss_total_fiber_length": 155.6,
        "fem_max_displacement": 0.03,
        "fem_yield_strength": 400.0,
        "min_seed_spacing": 1.0,
        "target_length_active": False,
        "target_total_length_feasible": True,
        "physical_stress_ratio": 0.9,
        "physical_displacement_ratio": 0.8,
        "total_seed_count": 12,
        "VolFrac": 0.468,
        "curve_length_min": 0.0632,
        "grad_mean": 0.0277,
        "physical_feasible": True,
        "seed_spacing_feasible": True,
        "overall_feasible": True,
        "overall_constraint_violation": 0.0,
        "patience_active": True,
        "design_patience_active": True,
        "stage_patience_counter": 0,
        "stage_patience_limit": 100,
    }

    text = NN_Trainer._format_stage_progress_log(
        row=row,
        total_step_budget=650,
        best_feasible_key=(155.6, 0.9, 210),
        best_infeasible_key=(0.01, 2.0, 173),
    )

    assert "[best=" not in text
    assert "Optimization mode:" not in text
    assert "target_length_constrained_displacement" not in text
    assert (
        "Best feasible: step=210 | primary=1.5560e+02 | "
        "max_stress=1.016e+02 | max_disp=1.650e-02 | "
        "min_seed_dist=1.240e+00 | total_length=1.556e+02"
    ) in text
    assert "best_recovery_violation" not in text
    assert "L_total=2.1650e+00" in text
    assert "monitor=1.9462e+00 (design)" in text
    assert "max_stress=1.016e+02" in text
    assert "max_disp=1.650e-02" in text
    assert "min_seed_dist=1.240e+00" in text
    assert "total_length=1.556e+02" in text
    assert "Constraints: stress=PASS | displacement=PASS | spacing=PASS | length_band=INACTIVE | overall=PASS" in text
    assert "Hard limits: length=INACTIVE | max_disp<=0.03 | max_stress<=400 | min_seed_dist>=1" in text
    assert "Diagnostics:" in text


def test_no_obsolete_stage2_violation_only_training_objective_remains() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_paths = [
        repo_root / "Training/MainTrain.py",
        repo_root / "HVD_SeedsBase/Training/MainTrain.py",
    ]
    combined_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_paths
        if path.exists()
    )

    assert "L_train = design_score + float(lam_fem_step) * fem_violation_loss" not in combined_source
    assert "fem_total_loss=loss_fem" in combined_source


def test_target_length_mode_is_active_before_stage2_transition() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "Training/MainTrain.py").read_text(encoding="utf-8")

    assert "target_length_active = target_length_mode" in source
    assert "or bool(target_length_feasible)" in source
    assert "and bool(seed_activity_classification_feasible)" in source
    assert "visual_partial_active_seed_count" in source
    assert "target_length_stage1_prepare = target_length_mode and int(stage_id) == 1" in source
    assert "if target_length_stage1_prepare and bool(target_length_uniform_cleanup_active)" in source
    assert "target_length_uniform_cleanup_active = bool(" in source
    assert "if target_length_active else None" in source
    assert "if target_length_active" in source


def test_target_length_stage1_uses_length_band_weight_when_default_zero() -> None:
    cfg = TrainingConfig(
        optimization_mode="target_length_constrained_displacement",
        target_total_length=100.0,
        stage1_lam_total_fiber_length=0.0,
        stage2_lam_total_fiber_length=3.0,
    )
    trainer = _trainer(cfg)

    settings = trainer._stage_settings_for_stage_id(1)

    assert settings["lam_total_fiber_length"] == pytest.approx(3.0)


def test_target_length_final_log_labels_feasible_key_by_displacement() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "Training/MainTrain.py").read_text(encoding="utf-8")

    assert "best_feasible_key_label" in source
    assert "displacement_ratio, stress_ratio, raw_fiber_length, step" in source


def test_optimization_reporting_uses_objective_terms_without_total_chart_bar() -> None:
    row = {
        "stage": 2,
        "optimization_mode": "target_length_constrained_displacement",
        "L_total": 49.8,
        "displacement_objective": 0.7,
        "displacement_objective_weight": 11.0,
        "fem_displacement_p_norm": 0.7,
        "fem_max_displacement": 1.0,
        "target_total_length": 100.0,
        "target_total_length_lower": 95.0,
        "target_total_length_upper": 99.9,
        "target_total_length_penalty": 4.0,
        "loss_total_fiber_length": 110.0,
        "target_total_length_under_violation": 0.0,
        "target_total_length_over_violation": 5.0,
        "target_length_under_weight": 1.0,
        "target_length_over_weight": 100.0,
        "lam_total_fiber_length_eff": 10.0,
        "loss_cvt_norm": 0.2,
        "loss_cvt": 2.0,
        "loss_cvt_reference": 10.0,
        "lam_cvt_eff": 3.0,
        "loss_l_curve_cell_norm": 0.3,
        "loss_l_curve_cell": 6.0,
        "loss_l_curve_cell_reference": 20.0,
        "lam_l_curve_cell_eff": 5.0,
        "loss_seed_spacing": 0.0,
        "lam_seed_spacing_eff": 0.0,
        "fem_total_loss": 0.25,
        "lam_fem_eff": 2.0,
        "fem_constraints_active": True,
    }

    NN_Trainer._attach_objective_report(row)
    chart = NN_Trainer._timelapse_loss_chart_dict(row)

    assert "objective_terms_text" in row
    assert "Displacement Lu: 0.7" in row["objective_terms_text"]
    assert "Length band: 4" in row["objective_terms_text"]
    assert row["objective_displacement_contribution"] == pytest.approx(7.7)
    assert row["objective_length_band_contribution"] == pytest.approx(40.0)
    assert not any(key == "Total" for key in chart)
    assert chart["L-total"] == pytest.approx(49.8)
    assert chart["Displacement x11"] == pytest.approx(7.7)
    assert not any("Displacement Lu" in key for key in chart)
    assert chart["Length band x10"] == pytest.approx(40.0)


def test_target_mode_best_primary_reporting_uses_checkpoint_total() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "Training/MainTrain.py").read_text(encoding="utf-8")

    assert "checkpoint_L_total(best_feasible_checkpoint)" in source
    assert '"best_primary_objective": final_primary_objective' in source
    assert '"best_feasible_primary_objective": (' in source


def test_target_length_seed_domain_lock_is_configured_and_logged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "Training/MainTrain.py").read_text(encoding="utf-8")

    assert "lock_seed_domain_after_target_length_feasible: bool" in source
    assert "target_length_lock_patience: int" in source
    assert "target_length_domain_lock_applied" in source
    assert "allow_seed_outside_domain_step = False" in source


def test_target_length_displacement_objective_uses_smooth_p_norm_by_default() -> None:
    cfg = TrainingConfig(
        optimization_mode="target_length_constrained_displacement",
        target_total_length=100.0,
    )
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "Training/MainTrain.py").read_text(encoding="utf-8")

    assert cfg.displacement_objective_mode == "p_norm"
    assert "loss_displacement_ratio" in source
    assert "displacement_objective_mode" in source


def test_stage1_placeholder_fem_is_excluded_from_stage2_checkpoint_ranking() -> None:
    assert not NN_Trainer._is_stage2_physical_checkpoint_candidate(
        stage_id=1,
        first_physical_stage=2,
        fem_was_evaluated=False,
        fem_is_valid=True,
        total_loss_is_finite=True,
    )
    assert not NN_Trainer._is_stage2_physical_checkpoint_candidate(
        stage_id=2,
        first_physical_stage=2,
        fem_was_evaluated=False,
        fem_is_valid=True,
        total_loss_is_finite=True,
        fem_constraints_active=True,
    )
    assert NN_Trainer._is_stage2_physical_checkpoint_candidate(
        stage_id=2,
        first_physical_stage=2,
        fem_was_evaluated=False,
        fem_is_valid=True,
        total_loss_is_finite=True,
        fem_constraints_active=False,
    )
    assert NN_Trainer._is_stage2_physical_checkpoint_candidate(
        stage_id=2,
        first_physical_stage=2,
        fem_was_evaluated=True,
        fem_is_valid=True,
        total_loss_is_finite=True,
    )


def test_stage2_fem_violation_is_zero_within_constraints() -> None:
    stress_ratio = torch.tensor(0.95)
    displacement_ratio = torch.tensor(1.0)
    stress_excess = torch.relu(stress_ratio - 1.0)
    displacement_excess = torch.relu(displacement_ratio - 1.0)
    fem_violation_loss = stress_excess.pow(2.0) + displacement_excess.pow(2.0)

    assert stress_excess.item() == pytest.approx(0.0)
    assert displacement_excess.item() == pytest.approx(0.0)
    assert fem_violation_loss.item() == pytest.approx(0.0)


def test_no_obsolete_raw_fiber_first_checkpoint_ranking_remains() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    combined_source = "\n".join(
        (repo_root / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "Training/FEMControl.py",
            "Training/MainTrain.py",
        )
    )

    assert "return True, (fiber_length, physical_max_ratio, step)" in combined_source
    assert "best_feasible_checkpoint[\"best_feasible_fiber_length\"]" not in combined_source
    assert "feasible_key=(fiber_length, physical_max_ratio, step)" not in combined_source
    assert "- float(lam_cvt_step) * loss_cvt_normalized" not in combined_source
    assert "- float(lam_rep_step) * loss_rep_normalized" not in combined_source
    assert "- float(lam_l_curve_cell_step) * loss_curve_cell_normalized" not in combined_source
    assert "lam_l_seed_step" not in combined_source


def test_min_length_best_feasible_key_uses_raw_length_first() -> None:
    a = (97.0, 0.8, 1)
    b = (101.0, 0.1, 2)

    assert a < b


def test_min_length_best_feasible_key_uses_physical_ratio_tiebreak() -> None:
    higher_ratio = (100.0, 0.9, 1)
    lower_ratio = (100.0, 0.7, 2)

    assert lower_ratio < higher_ratio


def test_best_feasible_key_uses_global_step_as_final_tiebreak() -> None:
    earlier = (10.0, 100.0, 1)
    later = (10.0, 100.0, 2)

    assert earlier < later


def test_topology_invalid_candidate_is_rejected_when_required() -> None:
    trainer = _trainer(TrainingConfig())
    invalid = {
        "loss_spacing_barrier": 1.0,
    }
    valid = {
        "loss_spacing_barrier": 0.0,
    }

    assert not trainer._stage2_topology_valid_from_row(invalid)
    assert trainer._stage2_topology_valid_from_row(valid)


def test_infeasible_shorter_fiber_cannot_replace_feasible_checkpoint() -> None:
    feasible, feasible_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.9,
        physical_stress_ratio=0.9,
        seed_spacing_feasible=True,
        design_score=7.0,
        raw_total_fiber_length=70.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )
    infeasible, _ = checkpoint_feasibility_key(
        physical_displacement_ratio=1.2,
        physical_stress_ratio=0.9,
        seed_spacing_feasible=True,
        design_score=6.0,
        raw_total_fiber_length=60.0,
        mechanical_violation=0.2,
        total_loss_is_finite=True,
        fem_is_valid=True,
    )

    assert feasible
    assert math.isfinite(feasible_key[0])
    assert not infeasible


def test_target_length_checkpoint_key_requires_length_feasibility_and_prefers_displacement() -> None:
    feasible, key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.7,
        physical_stress_ratio=0.9,
        physical_feasible=True,
        seed_spacing_feasible=True,
        design_score=12.0,
        raw_total_fiber_length=101.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
        global_step=4,
        optimization_mode="target_length_constrained_displacement",
        target_length_feasible=True,
    )
    lower_displacement_feasible, lower_displacement_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.5,
        physical_stress_ratio=0.95,
        physical_feasible=True,
        seed_spacing_feasible=True,
        design_score=20.0,
        raw_total_fiber_length=100.0,
        mechanical_violation=0.0,
        total_loss_is_finite=True,
        fem_is_valid=True,
        global_step=5,
        optimization_mode="target_length_constrained_displacement",
        target_length_feasible=True,
    )
    outside_length_feasible, outside_length_key = checkpoint_feasibility_key(
        physical_displacement_ratio=0.2,
        physical_stress_ratio=0.8,
        physical_feasible=True,
        seed_spacing_feasible=True,
        design_score=1.0,
        raw_total_fiber_length=120.0,
        mechanical_violation=0.4,
        total_loss_is_finite=True,
        fem_is_valid=True,
        global_step=6,
        optimization_mode="target_length_constrained_displacement",
        target_length_feasible=False,
    )

    assert feasible
    assert lower_displacement_feasible
    assert lower_displacement_key < key
    assert lower_displacement_key == pytest.approx((0.5, 0.95, 100.0, 5.0))
    assert not outside_length_feasible
    assert outside_length_key[0] == pytest.approx(0.4)


def test_live_best_feasible_metadata_uses_stage_monitor(tmp_path) -> None:
    trainer = _trainer()
    checkpoint = {
        "global_step": 42,
        "stage_id": 2,
        "source": "best_feasible",
        "stage_monitor_raw": 3.5,
        "row": {
            "loss_total_fiber_length": 12.0,
            "physical_stress_ratio": 0.8,
            "physical_displacement_ratio": 0.7,
            "total_seed_count": 9,
        },
        "pred_list": [],
    }

    trainer._save_live_best_feasible_checkpoint(
        output_folder=str(tmp_path),
        checkpoint=checkpoint,
        decoder=None,
        ppnet=None,
        face_tensor=None,
    )

    metadata_path = tmp_path / "BestFeasibleCheckpoint" / "best_feasible_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    removed_tokens = [
        "fixed" + "_eval",
        "fixed" + "_evaluation",
        "best" + "_fixed",
        "best" + "_fixed" + "_evaluation",
    ]

    assert metadata["stage_monitor_raw"] == pytest.approx(3.5)
    assert metadata["raw_total_fiber_length"] == pytest.approx(12.0)
    assert metadata["physical_max_ratio"] == pytest.approx(0.8)
    assert all(token not in json.dumps(metadata) for token in removed_tokens)


def test_stage1_monitor_ignores_stage2_terms() -> None:
    trainer = _trainer()
    monitor = trainer.calculate_stage_monitor(
        1,
        {
            "loss_seed_spacing": torch.tensor(2.0),
            "loss_cvt_norm": torch.tensor(3.0),
            "loss_rep_norm": torch.tensor(5.0),
            "loss_l_curve_cell_norm": torch.tensor(7.0),
            "loss_total_fiber_length_stage2_norm": torch.tensor(100.0),
            "loss_topology_validity": torch.tensor(100.0),
            "loss_fem": torch.tensor(100.0),
        },
        {
            "lam_seed_spacing": 11.0,
            "lam_cvt": 13.0,
            "lam_rep": 17.0,
            "lam_l_curve_cell": 19.0,
            "lam_total_fiber_length": 23.0,
            "lam_fem": 29.0,
        },
    )

    expected = 11.0 * 2.0 + 13.0 * 3.0 + 17.0 * 5.0 + 19.0 * 7.0
    assert monitor.item() == pytest.approx(expected)
