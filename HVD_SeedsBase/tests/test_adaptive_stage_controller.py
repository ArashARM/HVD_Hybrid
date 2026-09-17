from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

from Training.MainTrain import NN_Trainer, StageRuntime, StageSpec, TrainingConfig


class TinyPPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seed_refine = nn.Linear(2, 2)
        self.seed_id_embed = nn.Embedding(2, 2)
        self.delta_head = nn.Linear(2, 2)
        self.global_latent = nn.Parameter(torch.zeros(1))
        self.independent_seed_offsets = nn.Parameter(torch.zeros(2, 2))


def make_trainer(**cfg_kwargs):
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(**cfg_kwargs)
    return trainer


def test_meaningful_improvement_respects_abs_and_relative_thresholds():
    assert NN_Trainer.is_meaningful_improvement(0.89, 1.0, 1e-4, 1e-3)
    assert not NN_Trainer.is_meaningful_improvement(0.9995, 1.0, 1e-4, 1e-3)
    assert NN_Trainer.is_meaningful_improvement(1.0, float("inf"), 1e-4, 1e-3)


def test_main_stage_monitor_uses_stage2_design_terms_without_fem():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        2,
        {
            "loss_total_fiber_length_norm": torch.tensor(3.0),
            "loss_fem_norm": torch.tensor(5.0),
            "_zero": torch.tensor(0.0),
        },
        {"lam_total_fiber_length": 2.0, "lam_fem": 4.0},
    )

    assert torch.allclose(monitor, torch.tensor(6.0), atol=1e-12)


def test_feasible_stage2_monitor_reuses_design_score():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        2,
        {
            "design_score": torch.tensor(7.5),
            "mechanical_violation": torch.tensor(0.0),
            "physical_feasible": True,
            "loss_fem_norm": torch.tensor(100.0),
            "_zero": torch.tensor(0.0),
        },
        {"lam_fem": 4.0},
    )

    assert torch.allclose(monitor, torch.tensor(7.5), atol=1e-12)


def test_infeasible_stage2_monitor_uses_mechanical_violation():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        2,
        {
            "design_score": torch.tensor(7.5),
            "mechanical_violation": torch.tensor(0.25),
            "physical_feasible": False,
            "loss_fem_norm": torch.tensor(100.0),
            "_zero": torch.tensor(0.0),
        },
        {"lam_fem": 4.0},
    )

    assert torch.allclose(monitor, torch.tensor(0.25), atol=1e-12)


def test_overall_infeasible_stage2_monitor_uses_geometric_violation():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        2,
        {
            "design_score": torch.tensor(1.5),
            "mechanical_violation": torch.tensor(0.0),
            "overall_constraint_violation": torch.tensor(0.25),
            "overall_feasible": False,
            "_zero": torch.tensor(0.0),
        },
        {},
    )

    assert torch.allclose(monitor, torch.tensor(0.25), atol=1e-12)


def test_stage1_monitor_uses_effective_weighted_normalized_terms():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        1,
        {
            "loss_cvt_norm": torch.tensor(13.0),
            "loss_fem_norm": torch.tensor(2.0),
            "loss_l_curve_cell_norm": torch.tensor(5.0),
            "loss_rep_norm": torch.tensor(7.0),
            "loss_seed_spacing": torch.tensor(11.0),
            "L_total": torch.tensor(999.0),
        },
        {
            "lam_cvt": 3.0,
            "lam_l_curve_cell": 0.0,
            "lam_rep": 1.0,
            "lam_seed_spacing": 2.0,
        },
    )

    assert torch.allclose(monitor, torch.tensor(68.0))


def test_removed_eval_config_fields_are_gone():
    removed_names = {
        "fixed" + "_eval_lam_fem",
        "fixed" + "_eval_lam_l_curve_cell",
        "fixed" + "_eval_lam_rep",
        "fixed" + "_eval_lam_total_fiber_length",
    }

    config_names = {field.name for field in fields(TrainingConfig)}

    assert removed_names.isdisjoint(config_names)
    for name in removed_names:
        try:
            TrainingConfig(**{name: 0.0})
        except TypeError:
            pass
        else:
            raise AssertionError(f"{name} should not be accepted")


def test_removed_eval_score_function_is_gone():
    trainer = make_trainer()

    assert not hasattr(trainer, "_calculate_" + "fixed" + "_evaluation_score")

def test_legacy_joint_optimization_config_fields_are_gone():
    removed_names = {
        "num_steps",
        "use_adaptive_stage_stopping",
        "early_stop_start",
        "patience",
        "min_delta",
        "scheduler_milestones",
        "lam_fem",
        "lam_cvt",
        "lam_rep",
        "lam_seed_spacing",
        "lam_total_fiber_length",
        "lam_l_curve_cell",
    }

    config_names = {field.name for field in fields(TrainingConfig)}

    assert removed_names.isdisjoint(config_names)
    for name in removed_names:
        try:
            TrainingConfig(**{name: 1})
        except TypeError:
            pass
        else:
            raise AssertionError(f"{name} should not be accepted")


def test_stage_configs_include_all_stage_objective_lambdas():
    config_names = {field.name for field in fields(TrainingConfig)}
    loss_names = {
        "lam_fem",
        "lam_cvt",
        "lam_rep",
        "lam_seed_spacing",
        "lam_total_fiber_length",
        "lam_l_curve_cell",
    }

    for stage in ("stage1", "stage2"):
        assert {f"{stage}_{name}" for name in loss_names}.issubset(config_names)


def test_allow_seed_outside_domain_is_stage_configurable():
    trainer = make_trainer(
        allow_seed_outside_domain=True,
        allow_seed_outside_domain_warmup_frac=0.5,
        stage1_allow_seed_outside_domain=False,
        stage2_allow_seed_outside_domain=True,
    )

    assert trainer._stage_settings_for_stage_id(1)["allow_seed_outside_domain"] is False
    assert trainer._stage_settings_for_stage_id(2)["allow_seed_outside_domain"] is True
    assert not trainer.allow_seed_outside_domain_for_step(
        100,
        100,
        stage_allow_seed_outside_domain=False,
    )
    assert not trainer.allow_seed_outside_domain_for_step(
        49,
        100,
        stage_allow_seed_outside_domain=True,
    )
    assert trainer.allow_seed_outside_domain_for_step(
        50,
        100,
        stage_allow_seed_outside_domain=True,
    )


def test_stage_local_seed_offset_scale_uses_stage_max_steps():
    trainer = make_trainer(
        Offset_scale=1.0,
        seed_offset_scale_start=1.0,
        seed_offset_scale_final=0.1,
        seed_offset_scale_ramp_frac=0.5,
    )

    assert math.isclose(trainer.seed_offset_scale_for_step(0, 100), 1.0)
    assert math.isclose(trainer.seed_offset_scale_for_step(50, 100), 0.1)
    assert trainer.seed_offset_scale_for_step(25, 100) < 1.0


def test_stage_scheduler_milestones_are_stage_local():
    trainer = make_trainer(
        stage1_max_steps=100,
        stage2_max_steps=200,
        stage1_scheduler_milestones=(0.5, 75),
        stage2_scheduler_milestones=(0.25, 150),
    )

    assert trainer._stage_scheduler_milestones(1) == [50, 75]
    assert trainer._stage_scheduler_milestones(2) == [50, 150]


def test_removed_eval_terms_are_absent_from_runtime_output_surfaces():
    removed_tokens = [
        "fixed" + "_eval",
        "fixed" + "_evaluation",
        "best" + "_fixed",
        "best" + "_fixed" + "_evaluation",
    ]
    paths = [
        Path("Training/MainTrain.py"),
        Path("Training/FEMControl.py"),
        Path("Main.ipynb"),
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in removed_tokens), path


def test_spacing_metric_names_are_used_in_runtime_output_surfaces():
    forbidden_tokens = [
        "min_active_seed_dist",
        "d_active=",
        "anchor_guard_min_active_seed_dist_factor",
    ]
    main_text = Path("Training/MainTrain.py").read_text(encoding="utf-8")

    assert all(token not in main_text for token in forbidden_tokens)
    assert "min_seed_distance" in main_text


def test_clone_pred_list_preserves_fixed_thickness_metadata():
    pred = {
        "face_id": 0,
        "seeds_raw": torch.zeros(2, 2),
        "centerline_radius": torch.tensor(0.125),
        "strut_thickness": 0.25,
    }

    cloned = NN_Trainer._clone_pred_list([pred])

    assert torch.equal(cloned[0]["centerline_radius"], pred["centerline_radius"])
    assert cloned[0]["centerline_radius"] is not pred["centerline_radius"]


def test_transition_selection_stage_monitor_prefers_best_raw_candidate():
    trainer = make_trainer(stage_transition_selection="stage_monitor")
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_best_raw_checkpoint = {"valid": True, "stage_monitor_raw": 1.0}
    runtime.stage_last_valid_checkpoint = {"valid": True, "stage_monitor_raw": 2.0}

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "best_raw"
    assert checkpoint is runtime.stage_best_raw_checkpoint
    assert score == 1.0


def test_transition_selection_stage_monitor_falls_back_to_raw_then_last():
    trainer = make_trainer(stage_transition_selection="stage_monitor")
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_last_valid_checkpoint = {"valid": True, "stage_monitor_raw": 2.0}

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "last"
    assert checkpoint is runtime.stage_last_valid_checkpoint
    assert score == 2.0


def test_final_stage_selection_uses_raw_checkpoint():
    trainer = make_trainer()
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[-1])
    runtime.stage_best_raw_checkpoint = {"valid": True, "stage_monitor_raw": 1.0}
    runtime.stage_last_valid_checkpoint = {"valid": True, "stage_monitor_raw": 2.0}

    name, checkpoint, score = trainer._select_final_stage_checkpoint(runtime)

    assert name == "stage_best_raw"
    assert checkpoint is runtime.stage_best_raw_checkpoint
    assert score == 1.0


def test_next_stage_objective_selects_best_incoming_stage2_monitor_from_stored_metrics():
    trainer = make_trainer(
        stage_transition_selection="next_stage_objective",
        stage2_lam_fem=10.0,
        stage2_lam_total_fiber_length=2.0,
        stage2_lam_l_curve_cell=0.5,
        stage2_lam_rep=0.0,
        stage2_lam_cvt=0.0,
        lam_seed_spacing=0.0,
    )
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_best_raw_checkpoint = {
        "valid": True,
        "row": {"loss_total_fiber_length_norm": 4.0, "loss_fem_norm": 2.0, "loss_l_curve_cell_norm": 6.0, "loss_rep_norm": 100.0, "VolFrac": 0.5},
    }
    runtime.stage_last_valid_checkpoint = {
        "valid": True,
        "row": {"loss_total_fiber_length_norm": 1.0, "loss_fem_norm": 1.0, "loss_l_curve_cell_norm": 2.0, "loss_rep_norm": 100.0, "VolFrac": 0.5},
    }

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "last"
    assert checkpoint is runtime.stage_last_valid_checkpoint
    assert score == 3.0


def test_adaptive_stage_specs_use_min_max_steps_and_patience():
    trainer = make_trainer(
        stage1_min_steps=10,
        stage1_max_steps=100,
        stage1_patience=11,
        stage2_min_steps=20,
        stage2_max_steps=200,
        stage2_patience=22,
    )

    specs = trainer._adaptive_stage_specs()

    assert [spec.min_steps for spec in specs] == [10, 20]
    assert [spec.max_steps for spec in specs] == [100, 200]
    assert [spec.patience for spec in specs] == [11, 22]

def test_freeze_settings_exclude_frozen_parameters_from_optimizer():
    trainer = make_trainer()
    ppnet = TinyPPNet()
    trainer._apply_stage_trainability(ppnet, {"freeze_seeds": True})

    opt = trainer._build_optimizer(ppnet, decoder=None)
    opt_params = {id(p) for group in opt.param_groups for p in group["params"]}

    assert all(not p.requires_grad for p in ppnet.seed_refine.parameters())
    assert all(not p.requires_grad for p in ppnet.seed_id_embed.parameters())
    assert not ppnet.independent_seed_offsets.requires_grad
    assert id(ppnet.global_latent) in opt_params
    assert all(id(p) not in opt_params for p in ppnet.seed_refine.parameters())


def test_training_config_validates_adaptive_stage_fields():
    try:
        TrainingConfig(stage2_min_steps=10, stage2_max_steps=9)
    except ValueError as exc:
        assert "stage2_max_steps" in str(exc)
    else:
        raise AssertionError("stage max below min should fail")

    try:
        TrainingConfig(stage_transition_selection="bad")
    except ValueError as exc:
        assert "stage_transition_selection" in str(exc)
    else:
        raise AssertionError("invalid transition policy should fail")



def make_stage_runtime(min_steps=1, max_steps=20, patience=3):
    return StageRuntime(
        spec=StageSpec(
            stage_id=3,
            name="Stage 3",
            min_steps=min_steps,
            max_steps=max_steps,
            patience=patience,
            min_delta_abs=1e-4,
            min_delta_rel=1e-3,
        )
    )


def controller_row(identifier, *, total_edges=3, seed_count=4):
    return {
        "topology_identifier": identifier,
        "number_of_total_edges": total_edges,
        "total_seed_count": seed_count,
    }


def test_topology_identifier_canonicalizes_pairs_order_and_duplicates():
    graph_a = {
        "edge_seed_pair": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        "edge_type": torch.tensor([0, 0, 1]),
    }
    graph_b = {
        "edge_seed_pair": torch.tensor([[3, 2], [1, 0], [2, 1]]),
        "edge_type": torch.tensor([1, 0, 0]),
    }
    graph_dup = {
        "edge_seed_pair": torch.tensor([[0, 1], [1, 2], [2, 3], [1, 0]]),
        "edge_type": torch.tensor([0, 0, 1, 0]),
    }
    graph_c = {
        "edge_seed_pair": torch.tensor([[0, 1], [1, 3], [2, 3]]),
        "edge_type": torch.tensor([0, 0, 1]),
    }

    identifier_a = NN_Trainer.topology_identifier_from_graph(graph_a)

    assert identifier_a == NN_Trainer.topology_identifier_from_graph(graph_b)
    assert identifier_a == NN_Trainer.topology_identifier_from_graph(graph_dup)
    assert identifier_a != NN_Trainer.topology_identifier_from_graph(graph_c)


def test_topology_identifier_prefers_original_pairs_and_ignores_node_ids():
    graph_a = {
        "edge_seed_pair_original": torch.tensor([[0, 1], [1, 2]]),
        "edge_seed_pair": torch.tensor([[10, 11], [11, 12]]),
        "edge_type": torch.tensor([0, 1]),
        "edge_index": torch.tensor([[0, 1], [1, 2]]),
    }
    graph_b = {
        "edge_seed_pair_original": torch.tensor([[2, 1], [1, 0]]),
        "edge_seed_pair": torch.tensor([[99, 98], [98, 97]]),
        "edge_type": torch.tensor([1, 0]),
        "edge_index": torch.tensor([[50, 40], [40, 30]]),
    }

    assert NN_Trainer.topology_identifier_from_graph(graph_a) == NN_Trainer.topology_identifier_from_graph(graph_b)


def test_topology_edge_count_uses_seed_pair_records_before_edge_index():
    graph = {
        "edge_seed_pair": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        "edge_index": torch.tensor([[0, 1]]),
    }

    assert NN_Trainer.topology_edge_count_from_graph(graph) == 3


def test_false_topology_changes_do_not_reset_grace_or_patience():
    identifier = NN_Trainer.topology_identifier_from_graph({
        "edge_seed_pair": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        "edge_type": torch.tensor([0, 0, 1]),
    })
    runtime = make_stage_runtime(patience=3)

    row = controller_row(identifier)
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        meaningful_improvement=True,
        stage_topology_grace_steps=2,
    )
    assert diagnostics["topology_changed"] is False
    assert runtime.patience_counter == 0

    reordered_same = NN_Trainer.topology_identifier_from_graph({
        "edge_seed_pair": torch.tensor([[3, 2], [1, 0], [2, 1]]),
        "edge_type": torch.tensor([1, 0, 0]),
        "edge_index": torch.tensor([[9, 8], [8, 7], [7, 6]]),
    })
    row = controller_row(reordered_same)
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
    )

    assert diagnostics["topology_changed"] is False
    assert runtime.patience_counter == 1
    assert runtime.topology_grace_remaining == 0

    changed = NN_Trainer.topology_identifier_from_graph({
        "edge_seed_pair": torch.tensor([[0, 1], [1, 3], [2, 3]]),
        "edge_type": torch.tensor([0, 0, 1]),
    })
    row = controller_row(changed)
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
    )

    assert diagnostics["topology_changed"] is True
    assert diagnostics["identifier_changed"] is True
    assert runtime.patience_counter == 1
    assert runtime.topology_grace_remaining == 2


def test_topology_change_before_min_steps_does_not_arm_grace():
    runtime = make_stage_runtime(min_steps=5, patience=3)
    first_id = "topology-a"
    changed_id = "topology-b"

    NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row(first_id),
        meaningful_improvement=True,
        stage_topology_grace_steps=20,
    )
    runtime.local_step += 1

    row = controller_row(changed_id)
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        meaningful_improvement=False,
        stage_topology_grace_steps=20,
    )

    assert diagnostics["topology_changed"] is True
    assert diagnostics["patience_active"] is False
    assert runtime.patience_counter == 0
    assert runtime.topology_grace_remaining == 0


def test_topology_grace_counts_down_then_patience_increments():
    identifier = "stable"
    runtime = make_stage_runtime(patience=3)
    runtime.previous_active_count = 4
    runtime.previous_topology_identifier = identifier
    runtime.previous_edge_count = 3
    runtime.topology_grace_remaining = 3

    observed = []
    for _ in range(4):
        row = controller_row(identifier)
        NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            meaningful_improvement=False,
            stage_topology_grace_steps=3,
        )
        observed.append(runtime.topology_grace_remaining)

    assert observed == [2, 1, 0, 0]
    assert runtime.patience_counter == 1


def test_topology_grace_reset_budget_prevents_patience_starvation():
    runtime = make_stage_runtime(patience=3)
    runtime.previous_active_count = 4
    runtime.previous_topology_identifier = "topology-0"
    runtime.previous_edge_count = 3

    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row("topology-1"),
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
        stage_topology_grace_max_resets=1,
    )
    assert diagnostics["topology_changed"] is True
    assert diagnostics["topology_grace_reset"] is True
    assert runtime.topology_grace_remaining == 2
    assert runtime.topology_grace_resets_used == 1
    assert runtime.patience_counter == 0

    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row("topology-2"),
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
        stage_topology_grace_max_resets=1,
    )
    assert diagnostics["topology_changed"] is True
    assert diagnostics["topology_grace_reset"] is False
    assert runtime.topology_grace_remaining == 1
    assert runtime.topology_grace_resets_used == 1
    assert runtime.patience_counter == 0

    NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row("topology-3"),
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
        stage_topology_grace_max_resets=1,
    )
    assert runtime.topology_grace_remaining == 0

    NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row("topology-4"),
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
        stage_topology_grace_max_resets=1,
    )
    assert runtime.patience_counter == 1


def test_recovery_step_does_not_overwrite_feasible_stage_best_checkpoint():
    runtime = make_stage_runtime()
    feasible_checkpoint = {"source": "best_raw", "row": {"overall_feasible": True, "design_score": 1.5}}
    recovery_checkpoint = {"source": "last_valid", "row": {"overall_feasible": False, "overall_constraint_violation": 0.01}}
    runtime.best_raw_monitor = 1.5
    runtime.stage_best_raw_checkpoint = feasible_checkpoint

    result = NN_Trainer._update_stage_runtime_checkpoint(
        runtime,
        row={"overall_feasible": False, "overall_constraint_violation": 0.01},
        checkpoint=recovery_checkpoint,
        stage_monitor_raw=0.01,
        meaningful_improvement=False,
        stage_id=2,
    )

    assert result == "recovery_best"
    assert runtime.stage_best_raw_checkpoint is feasible_checkpoint
    assert runtime.stage_recovery_best_checkpoint is not None
    assert runtime.recovery_best_monitor == 0.01


def test_recovery_steps_do_not_consume_feasible_design_patience():
    runtime = make_stage_runtime(min_steps=1, patience=3)
    runtime.patience_counter = 2

    row = controller_row("recovery")
    row.update(
        {
            "overall_feasible": False,
            "stage_monitor_mode": "recovery",
            "overall_constraint_violation": 0.01,
        }
    )
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        meaningful_improvement=False,
        stage_topology_grace_steps=0,
    )

    assert diagnostics["stage_monitor_mode"] == "recovery"
    assert diagnostics["design_patience_active"] is False
    assert runtime.patience_counter == 2
    assert runtime.recovery_best_monitor == 0.01


def test_feasible_design_patience_resumes_after_recovery():
    runtime = make_stage_runtime(min_steps=1, patience=3)
    runtime.patience_counter = 2

    recovery_row = controller_row("same")
    recovery_row.update(
        {
            "overall_feasible": False,
            "stage_monitor_mode": "recovery",
            "overall_constraint_violation": 0.01,
        }
    )
    NN_Trainer.update_adaptive_stage_controller(
        runtime,
        recovery_row,
        meaningful_improvement=False,
        stage_topology_grace_steps=0,
    )

    feasible_row = controller_row("same")
    feasible_row.update(
        {
            "overall_feasible": True,
            "stage_monitor_mode": "design",
            "overall_constraint_violation": 0.0,
        }
    )
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        feasible_row,
        meaningful_improvement=True,
        stage_topology_grace_steps=0,
    )

    assert diagnostics["stage_monitor_mode"] == "design"
    assert diagnostics["design_patience_active"] is True
    assert runtime.patience_counter == 0


def test_patience_starts_after_min_steps_and_can_stop_with_patience_reason():
    runtime = make_stage_runtime(min_steps=5, max_steps=20, patience=3)
    identifier = "stable"
    monitors = [10.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    stop_reason = None

    for monitor in monitors:
        meaningful = NN_Trainer.is_meaningful_improvement(
            monitor,
            runtime.best_raw_monitor,
            runtime.spec.min_delta_abs,
            runtime.spec.min_delta_rel,
        )
        if meaningful:
            runtime.best_raw_monitor = monitor
        row = controller_row(identifier)
        NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            meaningful_improvement=meaningful,
            stage_topology_grace_steps=2,
        )
        runtime.local_step += 1
        if runtime.local_step < runtime.spec.min_steps:
            assert runtime.patience_counter == 0
        if (
            runtime.local_step >= runtime.spec.min_steps
            and runtime.patience_counter >= runtime.spec.patience
        ):
            stop_reason = "patience"
            break

    assert runtime.local_step == 7
    assert stop_reason == "patience"


def test_meaningful_improvement_resets_patience_after_min_steps():
    runtime = make_stage_runtime(min_steps=5, max_steps=20, patience=3)
    identifier = "stable"
    monitors = [10.0, 9.0, 8.0, 8.0, 7.0, 7.0, 7.0, 7.0]
    patience_history = []

    for monitor in monitors:
        meaningful = NN_Trainer.is_meaningful_improvement(
            monitor,
            runtime.best_raw_monitor,
            runtime.spec.min_delta_abs,
            runtime.spec.min_delta_rel,
        )
        if meaningful:
            runtime.best_raw_monitor = monitor
        row = controller_row(identifier)
        NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            meaningful_improvement=meaningful,
            stage_topology_grace_steps=2,
        )
        patience_history.append(runtime.patience_counter)
        runtime.local_step += 1

    assert patience_history[:4] == [0, 0, 0, 0]
    assert patience_history[4] == 0
    assert patience_history[-3:] == [1, 2, 3]


def test_patience_uses_stable_anchor_for_slow_cumulative_improvement():
    runtime = make_stage_runtime(min_steps=1, max_steps=30, patience=20)
    identifier = "stable"
    monitors = [10.0] + [10.0 - 0.001 * index for index in range(1, 12)]
    patience_history = []

    for monitor in monitors:
        meaningful = NN_Trainer.is_meaningful_improvement(
            monitor,
            runtime.best_raw_monitor,
            runtime.spec.min_delta_abs,
            runtime.spec.min_delta_rel,
        )
        if monitor < runtime.best_raw_monitor:
            runtime.best_raw_monitor = monitor
        row = controller_row(identifier)
        diagnostics = NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            meaningful_improvement=meaningful,
            stage_monitor_raw=monitor,
            stage_topology_grace_steps=0,
        )
        patience_history.append(runtime.patience_counter)
        runtime.local_step += 1

    assert any(count > 0 for count in patience_history)
    assert patience_history[-1] == 0
    assert diagnostics["patience_meaningful_improvement"] is True
    assert math.isclose(runtime.patience_anchor_monitor, monitors[-1])
