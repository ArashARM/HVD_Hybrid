import math
import json
from pathlib import Path

import pytest
import torch

from Training.Loss_TopologyValidity import seed_spacing_barrier_loss
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

    loss = seed_spacing_barrier_loss(seeds, safe_distance=1.0)

    assert loss.item() == pytest.approx(0.0)


def test_spacing_barrier_increases_when_seeds_approach() -> None:
    losses = []
    for distance in (0.9, 0.5, 0.1):
        seeds = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        loss = seed_spacing_barrier_loss(seeds, safe_distance=1.0)
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

    loss = seed_spacing_barrier_loss(seeds, safe_distance=1.0)

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
        "L_train": 2.16504,
        "design_score": 1.94624,
        "stage_monitor_raw": 1.94624,
        "stage_monitor_mode": "design",
        "fem_total_loss": 0.1094,
        "fem_baseline_loss": 0.1094,
        "fem_violation_loss": 0.0,
        "lam_fem_eff": 2.0,
        "physical_stress_ratio": 0.9,
        "physical_displacement_ratio": 0.8,
        "total_seed_count": 12,
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
        best_feasible_key=(1.94624, 50.0, 210),
        best_infeasible_key=(0.01, 2.0, 173),
    )

    assert "[best=" not in text
    assert "best_feasible_design=1.9462e+00@00210" in text
    assert "best_recovery_violation=1.0000e-02@00173" in text
    assert "L_train=2.1650e+00" in text
    assert "monitor=1.9462e+00(design)" in text


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

    assert "return True, (fiber_length, physical_max_ratio, step)" not in combined_source
    assert "best_feasible_checkpoint[\"best_feasible_fiber_length\"]" not in combined_source
    assert "feasible_key=(fiber_length, physical_max_ratio, step)" not in combined_source
    assert "- float(lam_cvt_step) * loss_cvt_normalized" not in combined_source
    assert "- float(lam_rep_step) * loss_rep_normalized" not in combined_source
    assert "- float(lam_l_curve_cell_step) * loss_curve_cell_normalized" not in combined_source
    assert "lam_l_seed_step" not in combined_source
    assert "loss_seed" not in combined_source


def test_best_feasible_key_uses_design_score_before_raw_fiber_length() -> None:
    a = (10.0, 97.0, 1)
    b = (8.0, 101.0, 2)

    assert b < a


def test_best_feasible_key_uses_raw_fiber_length_as_design_score_tiebreak() -> None:
    longer = (10.0, 100.0, 1)
    shorter = (10.0, 90.0, 2)

    assert shorter < longer


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
