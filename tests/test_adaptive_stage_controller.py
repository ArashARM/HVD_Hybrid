from __future__ import annotations

import math

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
        self.w_head = nn.Linear(2, 1)
        self.freeze_w = False


def make_trainer(**cfg_kwargs):
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(**cfg_kwargs)
    return trainer


def test_meaningful_improvement_respects_abs_and_relative_thresholds():
    assert NN_Trainer.is_meaningful_improvement(0.89, 1.0, 1e-4, 1e-3)
    assert not NN_Trainer.is_meaningful_improvement(0.9995, 1.0, 1e-4, 1e-3)
    assert NN_Trainer.is_meaningful_improvement(1.0, float("inf"), 1e-4, 1e-3)


def test_stage3_monitor_uses_volume_error_when_fem_disabled():
    trainer = make_trainer(target_volfrac=0.4, stage3_lam_fem_scale=0.0)

    monitor = trainer.calculate_stage_monitor(
        3,
        {"loss_fem_norm": torch.tensor(1000.0), "_zero": torch.tensor(0.0)},
        {"vol_frac": 0.47},
        {"lam_fem": 0.0},
    )

    assert torch.allclose(monitor, torch.tensor(0.07), atol=1e-12)


def test_stage1_monitor_uses_effective_weighted_normalized_terms():
    trainer = make_trainer()

    monitor = trainer.calculate_stage_monitor(
        1,
        {
            "loss_fem_norm": torch.tensor(2.0),
            "loss_density_weighted_cvt_norm": torch.tensor(3.0),
            "loss_cell_edge_uniform_norm": torch.tensor(5.0),
            "loss_rep_norm": torch.tensor(7.0),
            "L_total": torch.tensor(999.0),
        },
        {},
        {"lam_fem": 0.5, "lambda_cvt": 2.0, "lam_cell_edge_uniform": 0.0, "lam_rep": 1.0},
    )

    assert torch.allclose(monitor, torch.tensor(14.0))


def test_fixed_evaluation_score_does_not_use_stage_scaled_volume_loss():
    trainer = make_trainer(target_volfrac=0.5)

    score = trainer._calculate_fixed_evaluation_score(
        {
            "loss_fem_norm": torch.tensor(2.0),
            "loss_density_weighted_cvt_norm": torch.tensor(3.0),
            "loss_cell_edge_uniform_norm": torch.tensor(4.0),
            "loss_rep_norm": torch.tensor(5.0),
        },
        {"vol_frac": 0.6},
    )

    assert math.isclose(score, 2.0 + 0.1 + 0.2 * 3.0 + 0.05 * 4.0)


def test_transition_selection_stage_monitor_prefers_best_ema_candidate():
    trainer = make_trainer(stage_transition_selection="stage_monitor")
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_best_checkpoint = {"valid": True, "stage_monitor_ema": 3.0}
    runtime.stage_best_raw_checkpoint = {"valid": True, "stage_monitor_raw": 1.0, "stage_monitor_ema": 10.0}
    runtime.stage_last_valid_checkpoint = {"valid": True, "stage_monitor_ema": 2.0}

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "best_ema"
    assert checkpoint is runtime.stage_best_checkpoint
    assert score == 3.0


def test_transition_selection_stage_monitor_falls_back_to_raw_then_last():
    trainer = make_trainer(stage_transition_selection="stage_monitor")
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_best_raw_checkpoint = {"valid": True, "stage_monitor_raw": 1.0}
    runtime.stage_last_valid_checkpoint = {"valid": True, "stage_monitor_ema": 2.0}

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "best_raw"
    assert checkpoint is runtime.stage_best_raw_checkpoint
    assert score == 1.0


def test_next_stage_objective_selects_best_incoming_monitor_from_stored_metrics():
    trainer = make_trainer(stage_transition_selection="next_stage_objective")
    runtime = StageRuntime(spec=trainer._adaptive_stage_specs()[0])
    runtime.stage_best_checkpoint = {
        "valid": True,
        "row": {"loss_fem_norm": 10.0, "loss_density_weighted_cvt_norm": 10.0, "loss_cell_edge_uniform_norm": 10.0, "loss_rep_norm": 10.0, "VolFrac": 0.5},
    }
    runtime.stage_best_raw_checkpoint = {
        "valid": True,
        "row": {"loss_fem_norm": 1.0, "loss_density_weighted_cvt_norm": 1.0, "loss_cell_edge_uniform_norm": 1.0, "loss_rep_norm": 1.0, "VolFrac": 0.5},
    }
    runtime.stage_last_valid_checkpoint = {
        "valid": True,
        "row": {"loss_fem_norm": 5.0, "loss_density_weighted_cvt_norm": 5.0, "loss_cell_edge_uniform_norm": 5.0, "loss_rep_norm": 5.0, "VolFrac": 0.5},
    }

    name, checkpoint, score = trainer._select_transition_checkpoint(runtime, next_stage_id=2)

    assert name == "best_raw"
    assert checkpoint is runtime.stage_best_raw_checkpoint
    assert math.isfinite(score)


def test_adaptive_stage_specs_use_min_max_steps_and_patience():
    trainer = make_trainer(
        stage1_min_steps=10,
        stage1_max_steps=100,
        stage1_patience=11,
        stage2_min_steps=20,
        stage2_max_steps=200,
        stage2_patience=22,
        stage3_min_steps=30,
        stage3_max_steps=300,
        stage3_patience=33,
    )

    specs = trainer._adaptive_stage_specs()

    assert [spec.min_steps for spec in specs] == [10, 20, 30]
    assert [spec.max_steps for spec in specs] == [100, 200, 300]
    assert [spec.patience for spec in specs] == [11, 22, 33]


def test_freeze_settings_exclude_frozen_parameters_from_optimizer():
    trainer = make_trainer()
    ppnet = TinyPPNet()
    trainer._apply_stage_trainability(ppnet, {"freeze_seeds": True, "freeze_w": True})

    opt = trainer._build_optimizer(ppnet, decoder=None)
    opt_params = {id(p) for group in opt.param_groups for p in group["params"]}

    assert all(not p.requires_grad for p in ppnet.seed_refine.parameters())
    assert all(not p.requires_grad for p in ppnet.seed_id_embed.parameters())
    assert not ppnet.independent_seed_offsets.requires_grad
    assert all(not p.requires_grad for p in ppnet.w_head.parameters())
    assert id(ppnet.global_latent) in opt_params
    assert all(id(p) not in opt_params for p in ppnet.seed_refine.parameters())
    assert all(id(p) not in opt_params for p in ppnet.w_head.parameters())


def test_training_config_validates_adaptive_stage_fields():
    try:
        TrainingConfig(stage_monitor_ema_beta=1.0)
    except ValueError as exc:
        assert "stage_monitor_ema_beta" in str(exc)
    else:
        raise AssertionError("stage_monitor_ema_beta=1.0 should fail")

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


def controller_row(identifier, *, total_edges=3, active=4):
    return {
        "topology_identifier": identifier,
        "number_of_total_edges": total_edges,
        "active_units_total": active,
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
        stage_monitor_ema=1.0,
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
        stage_monitor_ema=1.0,
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
        stage_monitor_ema=1.0,
        meaningful_improvement=False,
        stage_topology_grace_steps=2,
    )

    assert diagnostics["topology_changed"] is True
    assert diagnostics["identifier_changed"] is True
    assert runtime.patience_counter == 0
    assert runtime.topology_grace_remaining == 2


def test_topology_change_before_min_steps_does_not_arm_grace():
    runtime = make_stage_runtime(min_steps=5, patience=3)
    first_id = "topology-a"
    changed_id = "topology-b"

    NN_Trainer.update_adaptive_stage_controller(
        runtime,
        controller_row(first_id),
        stage_monitor_ema=1.0,
        meaningful_improvement=True,
        stage_topology_grace_steps=20,
    )
    runtime.local_step += 1

    row = controller_row(changed_id)
    diagnostics = NN_Trainer.update_adaptive_stage_controller(
        runtime,
        row,
        stage_monitor_ema=1.0,
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
            stage_monitor_ema=1.0,
            meaningful_improvement=False,
            stage_topology_grace_steps=3,
        )
        observed.append(runtime.topology_grace_remaining)

    assert observed == [2, 1, 0, 0]
    assert runtime.patience_counter == 1


def test_patience_starts_after_min_steps_and_can_stop_with_patience_reason():
    runtime = make_stage_runtime(min_steps=5, max_steps=20, patience=3)
    identifier = "stable"
    monitors = [10.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    stop_reason = None

    for monitor in monitors:
        meaningful = NN_Trainer.is_meaningful_improvement(
            monitor,
            runtime.best_monitor_ema,
            runtime.spec.min_delta_abs,
            runtime.spec.min_delta_rel,
        )
        if meaningful:
            runtime.best_monitor_ema = monitor
        row = controller_row(identifier)
        NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            stage_monitor_ema=monitor,
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
            runtime.best_monitor_ema,
            runtime.spec.min_delta_abs,
            runtime.spec.min_delta_rel,
        )
        if meaningful:
            runtime.best_monitor_ema = monitor
        row = controller_row(identifier)
        NN_Trainer.update_adaptive_stage_controller(
            runtime,
            row,
            stage_monitor_ema=monitor,
            meaningful_improvement=meaningful,
            stage_topology_grace_steps=2,
        )
        patience_history.append(runtime.patience_counter)
        runtime.local_step += 1

    assert patience_history[:4] == [0, 0, 0, 0]
    assert patience_history[4] == 0
    assert patience_history[-3:] == [1, 2, 3]
