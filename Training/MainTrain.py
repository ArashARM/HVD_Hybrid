from dataclasses import asdict, dataclass
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from Utils.TimelapseRecorder import TimelapseRecorder
from tqdm.auto import tqdm
import numpy as np
from Utils.DifferentiableFilters import (
    smooth_heaviside_projection,
    surface_density_filter_metric_aware,
)

import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    from .Loss_FEM import Loss_FEM
    from .FEMControl import (
        checkpoint_design_score,
        checkpoint_feasibility_key,
        checkpoint_raw_fiber_length,
        checkpoint_stage_id,
        select_final_checkpoint,
        update_adaptive_fem_lambda,
        validate_optimizer_parameter_coverage,
    )
    from .Loss_rep import Loss_rep
    from .Loss_DensityWeightedCVT import LossDensityWeightedCVT
    from .Loss_ActivityWeights import prepare_seed_activity_weights
    from .Loss_SeedActive import Loss_SeedActive
except ImportError:
    from Loss_FEM import Loss_FEM
    from FEMControl import (
        checkpoint_design_score,
        checkpoint_feasibility_key,
        checkpoint_raw_fiber_length,
        checkpoint_stage_id,
        select_final_checkpoint,
        update_adaptive_fem_lambda,
        validate_optimizer_parameter_coverage,
    )
    from Loss_rep import Loss_rep
    from Loss_DensityWeightedCVT import LossDensityWeightedCVT
    from Loss_ActivityWeights import prepare_seed_activity_weights
    from Loss_SeedActive import Loss_SeedActive


EDGE_INTERIOR_VORONOI = 0
EDGE_CLIPPED_ONE_SIDE = 1
EDGE_RESERVED = 2
EDGE_CLIPPED_TWO_SIDE = 3
EDGE_DOMAIN_SHELL = 4
EDGE_TYPES_BY_LOSS_MODE = {
    "Interior": (EDGE_INTERIOR_VORONOI,),
    "VDonly": (EDGE_INTERIOR_VORONOI, EDGE_CLIPPED_ONE_SIDE, EDGE_CLIPPED_TWO_SIDE),
    "all": (EDGE_INTERIOR_VORONOI, EDGE_CLIPPED_ONE_SIDE, EDGE_CLIPPED_TWO_SIDE, EDGE_DOMAIN_SHELL),
}
CELL_GEOMETRY_EDGE_TYPES = (
    EDGE_INTERIOR_VORONOI,
    EDGE_CLIPPED_ONE_SIDE,
    EDGE_CLIPPED_TWO_SIDE,
    EDGE_DOMAIN_SHELL,
)
CELL_LENGTH_EDGE_TYPES = (
    EDGE_INTERIOR_VORONOI,
    EDGE_CLIPPED_ONE_SIDE,
    EDGE_CLIPPED_TWO_SIDE,
)


def canonical_edge_in_losses_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError(
            f"Edge_in_losses must be a string, got {type(mode).__name__}."
        )

    normalized = mode.strip().lower()
    canonical_by_normalized = {
        "interior": "Interior",
        "vdonly": "VDonly",
        "all": "all",
    }

    if normalized not in canonical_by_normalized:
        raise ValueError(
            "Invalid Edge_in_losses value "
            f"{mode!r}. Expected one of: 'Interior', 'VDonly', 'all'."
        )

    return canonical_by_normalized[normalized]
def resolve_edge_types_in_losses(mode: str) -> tuple[int, ...]:
    return EDGE_TYPES_BY_LOSS_MODE[canonical_edge_in_losses_mode(mode)]
def build_edge_type_mask(
    edge_type: torch.Tensor,
    allowed_types: tuple[int, ...],
) -> torch.Tensor:
    if edge_type.ndim != 1:
        raise ValueError(
            f"edge_type must be one-dimensional, got {edge_type.shape}."
        )

    mask = torch.zeros_like(edge_type, dtype=torch.bool)
    for edge_type_value in allowed_types:
        mask |= edge_type == int(edge_type_value)
    return mask
def compute_all_edge_curve_lengths(
    edge_curves_xyz: torch.Tensor,
) -> torch.Tensor:
    if edge_curves_xyz.ndim != 3:
        raise ValueError(
            "edge_curves_xyz must have shape [E, K, 3], "
            f"got {tuple(edge_curves_xyz.shape)}."
        )

    if edge_curves_xyz.shape[-1] != 3:
        raise ValueError(
            "The final edge_curves_xyz dimension must be 3."
        )

    edge_count = edge_curves_xyz.shape[0]
    sample_count = edge_curves_xyz.shape[1]

    if edge_count == 0:
        return edge_curves_xyz.new_empty((0,))

    if sample_count < 2:
        return edge_curves_xyz.new_zeros((edge_count,))

    segments = edge_curves_xyz[:, 1:, :] - edge_curves_xyz[:, :-1, :]

    return torch.linalg.vector_norm(
        segments,
        ord=2,
        dim=-1,
    ).sum(dim=-1)

@dataclass
class SharedCurveGeometry:
    curves_xyz: torch.Tensor
    edge_lengths: torch.Tensor
    finite_edge_mask: torch.Tensor
    edge_type: torch.Tensor
    edge_seed_pair: torch.Tensor | None
    cell_boundary_edge_indices: list[torch.Tensor] | None = None
    cell_boundary_edge_directions: list[torch.Tensor] | None = None
    cell_boundary_seed_ids: torch.Tensor | None = None

def build_shared_curve_geometry(
    decoder_out: dict,
    *,
    require_edge_type: bool = True,
    require_edge_seed_pair: bool = False,
) -> SharedCurveGeometry | None:
    curves = decoder_out.get("edge_curves_xyz", None)
    if curves is None:
        return None

    edge_lengths = compute_all_edge_curve_lengths(curves)
    graph = decoder_out.get("graph", None)
    edge_count = curves.shape[0]

    edge_type = None
    edge_seed_pair = None
    if isinstance(graph, dict):
        edge_type = graph.get("edge_type", None)
        edge_seed_pair = graph.get(
            "edge_seed_pair_original",
            graph.get("edge_seed_pair", None),
        )
        cell_boundary_edge_indices = graph.get("cell_boundary_edge_indices", None)
        cell_boundary_edge_directions = graph.get("cell_boundary_edge_directions", None)
        cell_boundary_seed_ids = graph.get(
            "cell_boundary_seed_ids_original",
            graph.get("cell_boundary_seed_ids", None),
        )
    else:
        cell_boundary_edge_indices = None
        cell_boundary_edge_directions = None
        cell_boundary_seed_ids = None

    if edge_type is None:
        if require_edge_type:
            raise ValueError(
                "decoder_out['graph']['edge_type'] is required when "
                "edge_curves_xyz is present."
            )
        edge_type = torch.full((edge_count,), EDGE_INTERIOR_VORONOI, dtype=torch.long, device=curves.device)
    else:
        edge_type = torch.as_tensor(
            edge_type,
            dtype=torch.long,
            device=curves.device,
        ).reshape(-1)
        if edge_type.shape[0] != edge_count:
            raise ValueError(
                "decoder_out['graph']['edge_type'] must have one value per "
                f"edge curve, got {edge_type.shape[0]} for {edge_count} curves."
            )

    if edge_seed_pair is None:
        if require_edge_seed_pair:
            raise ValueError(
                "decoder_out['graph']['edge_seed_pair'] is required "
                "when the cell edge-uniformity loss is enabled."
            )
    else:
        edge_seed_pair = torch.as_tensor(
            edge_seed_pair,
            dtype=torch.long,
            device=curves.device,
        )
        if edge_seed_pair.ndim != 2 or edge_seed_pair.shape != (edge_count, 2):
            raise ValueError(
                "decoder_out['graph']['edge_seed_pair'] must have shape "
                f"[{edge_count}, 2], got {tuple(edge_seed_pair.shape)}."
            )

    finite_edge_mask = (
        torch.isfinite(edge_lengths)
        & torch.isfinite(curves).all(dim=-1).all(dim=-1)
    )

    return SharedCurveGeometry(
        curves_xyz=curves,
        edge_lengths=edge_lengths,
        finite_edge_mask=finite_edge_mask,
        edge_type=edge_type,
        edge_seed_pair=edge_seed_pair,
        cell_boundary_edge_indices=cell_boundary_edge_indices,
        cell_boundary_edge_directions=cell_boundary_edge_directions,
        cell_boundary_seed_ids=cell_boundary_seed_ids,
    )
def needs_shared_curve_geometry(
    *,
    compute_total_fiber_length_loss: bool,
    compute_l_curve_cell_loss: bool,
    collect_curve_metrics: bool,
    collect_topology_metrics: bool,
) -> bool:
    return (
        compute_total_fiber_length_loss
        or compute_l_curve_cell_loss
        or collect_curve_metrics
        or collect_topology_metrics
    )
@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    min_steps: int
    max_steps: int
    patience: int
    min_delta_abs: float
    min_delta_rel: float
@dataclass
class StageRuntime:
    spec: StageSpec
    local_step: int = 0
    best_raw_monitor: float = float("inf")
    recovery_best_monitor: float = float("inf")
    patience_counter: int = 0
    topology_grace_remaining: int = 0
    topology_grace_resets_used: int = 0
    previous_active_count: int | None = None
    previous_topology_identifier: str | None = None
    previous_edge_count: int | None = None
    stage_best_raw_checkpoint: dict[str, Any] | None = None
    stage_recovery_best_checkpoint: dict[str, Any] | None = None
    stage_last_valid_checkpoint: dict[str, Any] | None = None
    end_reason: str | None = None
@dataclass
class TrainingConfig:
    seed_init_fps_seed: int | None = None
    use_balanced_seed_init: bool = True
    seed_number: int = 15
    training_face_index: int = 0
    LoadingCase: str = "Unspecified loading case"

    strut_thickness: float = 0.25
    boundary_margin: float = 0.05

    curve_length_worst_weight: float = 1.0
    curve_length_outlier_weight: float = 5.0

    cell_edge_uniform_eps: float = 1e-8
    cell_angle_eps: float = 1e-8
    cell_vertex_merge_tolerance: float = 1e-5

    beta: float = 0.05
    centerline_beta: float = 0.02
    centerline_softmin_tau: float = 0.01
    tube_curve_samples: int = 64
    tube_lift_tau: float = 0.02
    tube_lift_max_values: int = 4_000_000
    rho_min: float = 0.0
    decoder_eps: float = 1e-8
    decoder_solve_reg: float = 1e-6
    decoder_tau_voronoi: float = 0.01
    decoder_tau_box: float = 0.01
    decoder_tau_trim: float = 0.01
    decoder_use_trim_activity: bool = True
    decoder_return_xyz: bool = True
    decoder_vertex_boundary_margin: float = 0.02
    decoder_edge_trim_samples: int = 32
    decoder_edge_trim_reduction: str = "softmin"
    decoder_edge_trim_reduce_tau: float = 0.05
    decoder_use_edge_trim_gate: bool = True
    decoder_nearest_segment_k: int = 4
    decoder_use_segment_distance: bool = True
    decoder_use_spatial_pruning: bool = True
    decoder_min_tube_spacing: float = 1e-3
    decoder_tube_target_spacing_ratio: float = 0.75
    decoder_use_seed_activation: bool = True
    decoder_duplicate_effect_temp_ratio: float = 0.25

    use_3d_density_filter: bool = False
    filter_radius_3d: float = 0.03
    filter_self_weight: float = 1.0
    filter_projection_strength: float = 1.0
    filter_projection_beta: float = 8.0
    filter_projection_eta: float = 0.5
    visualize_filtered_density: bool = True
    visualize_raw_density: bool = False
    generate_decoder_density_fiber: bool = True

    cvt_temperature: float = 0.02
    cvt_activity_log_floor: float = 1e-4
    cvt_activity_temperature: float = 1.0
    use_smooth_seed_activity_in_losses: bool = True
    activity_weight_floor: float = 0.02
    activity_weight_power: float = 1.0
    activity_pair_weight_mode: str = "geometric_mean"
    activity_recovery_floor: float = 0.05
    activity_recovery_power: float = 1.0
    repulsion_inactive_recovery_strength: float = 1.0
    seed_active_loss_mode: str = "minimum"
    seed_active_loss_temperature: float = 0.25
    possible_min_active: int | None = None
    seed_active_baseline_strength: float = 1.0
    seed_active_baseline_power: float = 2.0
    seed_active_hard_barrier_strength: float = 100.0
    seed_active_hard_barrier_temperature: float | None = None
    curve_length_eps: float = 1e-8
    curve_length_tolerance: float = 0.15
    Edge_in_losses: str = "Interior"
    lam_cell_angle_uniform: float = 1.0
    lam_cell_radial_uniform: float = 0.5
    include_shell_in_length_loss: bool = False

    fem_max_displacement: float | None = None
    fem_yield_strength: float | None = None
    fem_training_safety_factor: float = 0.95
    fem_constraint_weight: float = 2.0
    fem_baseline_weight: float = 0.05
    fem_violation_power: float = 4.0
    fem_stress_density_threshold: float = 1.0e-3
    normalize_losses: bool = True
    adaptive_fem_penalty: bool = True
    fem_lambda_initial: float = 1.0
    fem_lambda_min: float = 1.0
    fem_lambda_max: float = 1.0e8
    fem_lambda_growth: float = 1.05
    fem_lambda_decay: float = 0.995
    fem_constraint_tolerance: float = 0.0
    fem_rho_min_ratio: float = 1.0e-5
    fem_penal: float = 3.0
    fem_rho_min_start: float = 1.0e-3
    fem_rho_min_end: float = 1.0e-6
    fem_density_floor: float = 0.0
    skip_bad_fem_steps: bool = True
    invalid_fem_patience: int = 3
    invalid_lr_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-7
    invalid_fem_consumes_budget: bool = True
    debug_fem_integrity: bool = False

    tau: float = 0.02
    tau_anneal_final: float | None = None
    tau__anneal_final: float | None = None
    tau_anneal_start_frac: float = 0.0
    tau_anneal_ramp_frac: float = 0.5

    seed_anchor_momentum: float = 0.20
    seed_anchor_warmup_frac: float = 0.05
    use_rolling_seed_anchors: bool = True
    disable_rolling_seed_anchors_for_curve_only: bool = True
    anchor_guard_updates: bool = True
    anchor_guard_rep_max: float = 0.30
    anchor_guard_vol_eff_min: float = 0.10
    anchor_guard_min_active_seed_dist_factor: float = 2.0

    allow_seed_outside_domain: bool = True
    allow_seed_outside_domain_warmup_frac: float = 0.50
    seed_domain_margin: float = 0.25
    use_seed_domain_mask: bool = True
    seed_domain_mask_threshold: float = 0.5
    seed_domain_temp: float = 0.05
    seed_domain_mask_support_scale: float = 2.5
    seed_domain_mask_max_points: int = 2048
    use_independent_seed_offsets: bool = True
    independent_seed_offset_max: float = 0.05

    lr_seed_refine: float = 1e-1
    lr_independent_seed_offsets: float = 1e-3
    lr_delta_head: float = 2e-4
    lr_mlp: float = 2e-4
    lr_decoder: float = 2e-4

    stage1_lam_fem: float = 0.0
    stage1_lam_cvt: float = 1.0
    stage1_lam_rep: float = 2.0
    stage1_lam_l_seed: float = 1.0
    stage1_lam_total_fiber_length: float = 0.0
    stage1_lam_l_curve_cell: float = 0.05
    stage1_freeze_seeds: bool = True
    stage1_allow_seed_outside_domain: bool = True
    stage2_lam_fem: float = 1.0
    stage2_lam_cvt: float = 0.0
    stage2_lam_rep: float = 0.10
    stage2_lam_l_seed: float = 0.10
    stage2_lam_total_fiber_length: float = 1.0
    stage2_lam_l_curve_cell: float = 0.05
    stage2_freeze_seeds: bool = True
    stage2_allow_seed_outside_domain: bool = True
    # Adaptive stage scheduling
    stage1_min_steps: int = 150
    stage1_max_steps: int = 300
    stage1_patience: int = 60
    stage2_min_steps: int = 250
    stage2_max_steps: int = 500
    stage2_patience: int = 100
    stage1_min_delta_abs: float = 1e-4
    stage1_min_delta_rel: float = 1e-3
    stage2_min_delta_abs: float = 1e-4
    stage2_min_delta_rel: float = 1e-3
    stage_topology_grace_steps: int = 20
    stage_topology_grace_max_resets: int = 3
    debug_stage_controller: bool = False
    restore_transition_checkpoint: bool = True
    reset_optimizer_between_stages: bool = True
    stage_transition_selection: str = "next_stage_objective"

    log_every: int = 50

    min_active_seeds: int | None = None

    eps: float = 1e-12

    Offset_scale: float = 1.00
    seed_offset_scale_start: float | None = None
    seed_offset_scale_final: float | None = None
    seed_offset_scale_ramp_frac: float = 0.60
    stage1_scheduler_milestones: tuple[float, ...] = ()
    stage2_scheduler_milestones: tuple[float, ...] = ()
    scheduler_gamma: float = 0.5

    save_fem_debug_history: bool = True
    grad_clip_norm: float | None = 1.0
    debug_anomaly_detection: bool = False

    tensorboard_enabled: bool = True
    tensorboard_log_root: str = "runs"
    experiment_name: str | None = None
    tb_flush_secs: int = 10
    tb_log_histograms_every: int = 200

    MakeTimelaps: bool = True
    timelapse_output_folder: str | None = None

    timelapse_frame_step: int = 10
    TM_laps_Thr: float = 0.45
    timelapse_show_3d_tubes: bool = True
    timelapse_tube_radius_scale: float = 1.0
    timelapse_tube_n_sides: int = 12

    def __post_init__(self):
        for name in (
            "curve_length_worst_weight",
            "curve_length_outlier_weight",
            "cell_edge_uniform_eps",
            "cell_angle_eps",
            "cell_vertex_merge_tolerance",
        ):
            value = getattr(self, name, None)
            if isinstance(value, tuple) and len(value) == 1:
                setattr(self, name, value[0])

        self.Edge_in_losses = canonical_edge_in_losses_mode(self.Edge_in_losses)
        self.strut_thickness = float(self.strut_thickness)
        if self.strut_thickness <= 0.0:
            raise ValueError(f"strut_thickness must be > 0, got {self.strut_thickness}")

        self.training_face_index = int(self.training_face_index)
        if self.training_face_index < 0:
            raise ValueError(
                f"training_face_index must be >= 0, got {self.training_face_index}"
            )

        if self.tau__anneal_final is not None:
            self.tau_anneal_final = self.tau__anneal_final

        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.tau_anneal_final is not None and self.tau_anneal_final <= 0.0:
            raise ValueError(f"tau_anneal_final must be > 0, got {self.tau_anneal_final}")
        if not (0.0 <= self.filter_projection_strength <= 1.0):
            raise ValueError(
                "filter_projection_strength must be in [0,1], "
                f"got {self.filter_projection_strength}"
            )
        if self.filter_projection_beta <= 0.0:
            raise ValueError(
                f"filter_projection_beta must be > 0, got {self.filter_projection_beta}"
            )
        if self.centerline_beta <= 0.0:
            raise ValueError(f"centerline_beta must be > 0, got {self.centerline_beta}")
        if self.centerline_softmin_tau <= 0.0:
            raise ValueError(
                f"centerline_softmin_tau must be > 0, got {self.centerline_softmin_tau}"
            )
        if self.tube_curve_samples < 2:
            raise ValueError(f"tube_curve_samples must be >= 2, got {self.tube_curve_samples}")
        if self.tube_lift_tau <= 0.0:
            raise ValueError(f"tube_lift_tau must be > 0, got {self.tube_lift_tau}")
        if self.tube_lift_max_values < 1:
            raise ValueError(
                f"tube_lift_max_values must be >= 1, got {self.tube_lift_max_values}"
            )
        if not (0.0 <= self.rho_min < 1.0):
            raise ValueError(f"rho_min must satisfy 0 <= rho_min < 1, got {self.rho_min}")
        if not (0.0 < self.filter_projection_eta < 1.0):
            raise ValueError(
                "filter_projection_eta must be in (0,1), "
                f"got {self.filter_projection_eta}"
            )
        if not (0.0 <= self.seed_anchor_momentum <= 1.0):
            raise ValueError(
                f"seed_anchor_momentum must be in [0,1], got {self.seed_anchor_momentum}"
            )
        if not (0.0 <= self.seed_anchor_warmup_frac <= 1.0):
            raise ValueError(
                f"seed_anchor_warmup_frac must be in [0,1], got {self.seed_anchor_warmup_frac}"
            )
        if self.anchor_guard_min_active_seed_dist_factor < 0.0:
            raise ValueError(
                "anchor_guard_min_active_seed_dist_factor must be >= 0, "
                f"got {self.anchor_guard_min_active_seed_dist_factor}"
            )
        if not (0.0 <= self.allow_seed_outside_domain_warmup_frac <= 1.0):
            raise ValueError(
                "allow_seed_outside_domain_warmup_frac must be in [0,1], "
                f"got {self.allow_seed_outside_domain_warmup_frac}"
            )
        if self.seed_domain_margin < 0.0:
            raise ValueError(f"seed_domain_margin must be >= 0, got {self.seed_domain_margin}")
        if not (0.0 <= self.seed_domain_mask_threshold <= 1.0):
            raise ValueError(
                "seed_domain_mask_threshold must be in [0,1], "
                f"got {self.seed_domain_mask_threshold}"
            )
        if self.seed_domain_temp <= 0.0:
            raise ValueError(f"seed_domain_temp must be > 0, got {self.seed_domain_temp}")
        if self.seed_domain_mask_support_scale <= 0.0:
            raise ValueError(
                "seed_domain_mask_support_scale must be > 0, "
                f"got {self.seed_domain_mask_support_scale}"
            )
        if self.seed_domain_mask_max_points < 1:
            raise ValueError(
                "seed_domain_mask_max_points must be >= 1, "
                f"got {self.seed_domain_mask_max_points}"
            )
        if self.independent_seed_offset_max < 0.0:
            raise ValueError(
                "independent_seed_offset_max must be >= 0, "
                f"got {self.independent_seed_offset_max}"
            )
        if self.lr_independent_seed_offsets < 0.0:
            raise ValueError(
                "lr_independent_seed_offsets must be >= 0, "
                f"got {self.lr_independent_seed_offsets}"
            )
        if self.lr_decoder < 0.0:
            raise ValueError(f"lr_decoder must be >= 0, got {self.lr_decoder}")
        if self.fem_constraint_weight <= 0.0:
            raise ValueError(
                f"fem_constraint_weight must be > 0, got {self.fem_constraint_weight}"
            )
        if self.fem_baseline_weight < 0.0:
            raise ValueError(
                f"fem_baseline_weight must be >= 0, got {self.fem_baseline_weight}"
            )
        if self.fem_violation_power <= 0.0:
            raise ValueError(
                f"fem_violation_power must be > 0, got {self.fem_violation_power}"
            )
        if not (0.0 < float(self.fem_training_safety_factor) <= 1.0):
            raise ValueError(
                "fem_training_safety_factor must satisfy 0 < factor <= 1, "
                f"got {self.fem_training_safety_factor}"
            )
        for name in (
            "fem_lambda_initial",
            "fem_lambda_min",
            "fem_lambda_max",
            "fem_lambda_growth",
            "fem_lambda_decay",
            "fem_rho_min_ratio",
            "fem_penal",
            "fem_rho_min_start",
            "fem_rho_min_end",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite, got {value}")
        if self.fem_lambda_max < self.fem_lambda_min:
            raise ValueError("fem_lambda_max must be >= fem_lambda_min")
        if self.fem_lambda_initial < self.fem_lambda_min:
            self.fem_lambda_initial = self.fem_lambda_min
        if self.fem_lambda_initial > self.fem_lambda_max:
            self.fem_lambda_initial = self.fem_lambda_max
        if self.fem_constraint_tolerance < 0.0:
            raise ValueError("fem_constraint_tolerance must be >= 0")
        for name in ("fem_max_displacement", "fem_yield_strength"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"{name} must be a positive finite value or None, got {value}")
        if self.fem_stress_density_threshold is not None and self.fem_stress_density_threshold < 0.0:
            raise ValueError(
                "fem_stress_density_threshold must be >= 0 or None, "
                f"got {self.fem_stress_density_threshold}"
            )
        if int(self.invalid_fem_patience) < 1:
            raise ValueError("invalid_fem_patience must be >= 1")
        if not (0.0 < float(self.invalid_lr_factor) <= 1.0):
            raise ValueError("invalid_lr_factor must satisfy 0 < factor <= 1")
        if float(self.minimum_learning_rate) < 0.0:
            raise ValueError("minimum_learning_rate must be >= 0")
        if self.cvt_temperature <= 0.0:
            raise ValueError(f"cvt_temperature must be > 0, got {self.cvt_temperature}")
        if self.cvt_activity_log_floor <= 0.0:
            raise ValueError(
                f"cvt_activity_log_floor must be > 0, got {self.cvt_activity_log_floor}"
            )
        if self.cvt_activity_temperature <= 0.0:
            raise ValueError(
                f"cvt_activity_temperature must be > 0, got {self.cvt_activity_temperature}"
            )
        for name in (
            "stage1_lam_fem",
            "stage1_lam_cvt",
            "stage1_lam_rep",
            "stage1_lam_l_seed",
            "stage1_lam_total_fiber_length",
            "stage1_lam_l_curve_cell",
            "stage2_lam_fem",
            "stage2_lam_cvt",
            "stage2_lam_rep",
            "stage2_lam_l_seed",
            "stage2_lam_total_fiber_length",
            "stage2_lam_l_curve_cell",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.seed_offset_scale_start is not None and self.seed_offset_scale_start <= 0.0:
            raise ValueError(
                f"seed_offset_scale_start must be > 0, got {self.seed_offset_scale_start}"
            )
        if self.seed_offset_scale_final is not None and self.seed_offset_scale_final <= 0.0:
            raise ValueError(
                f"seed_offset_scale_final must be > 0, got {self.seed_offset_scale_final}"
            )
        if not (0.0 < self.seed_offset_scale_ramp_frac <= 1.0):
            raise ValueError(
                "seed_offset_scale_ramp_frac must be in (0,1], "
                f"got {self.seed_offset_scale_ramp_frac}"
            )
        if self.min_active_seeds is not None and self.min_active_seeds < 1:
            raise ValueError(f"min_active_seeds must be >= 1, got {self.min_active_seeds}")

        if int(self.stage_topology_grace_max_resets) < 0:
            raise ValueError(
                "stage_topology_grace_max_resets must be >= 0, "
                f"got {self.stage_topology_grace_max_resets}"
            )
        for sid in (1, 2):
            min_steps = int(getattr(self, f"stage{sid}_min_steps"))
            max_steps = int(getattr(self, f"stage{sid}_max_steps"))
            patience = int(getattr(self, f"stage{sid}_patience"))
            if min_steps < 0:
                raise ValueError(f"stage{sid}_min_steps must be >= 0, got {min_steps}")
            if max_steps < min_steps:
                raise ValueError(
                    f"stage{sid}_max_steps must be >= stage{sid}_min_steps, "
                    f"got {max_steps} < {min_steps}"
                )
            if patience < 1:
                raise ValueError(f"stage{sid}_patience must be >= 1, got {patience}")
        if self.stage_transition_selection not in {
            "next_stage_objective",
            "stage_monitor",
            "last",
        }:
            raise ValueError(
                "stage_transition_selection must be one of "
                "{'next_stage_objective', 'stage_monitor', 'last'}, "
                f"got {self.stage_transition_selection!r}"
            )
        if not (0.0 <= self.activity_weight_floor < 1.0):
            raise ValueError(
                "activity_weight_floor must satisfy 0 <= floor < 1, "
                f"got {self.activity_weight_floor}"
            )
        if self.activity_weight_power <= 0.0:
            raise ValueError(
                f"activity_weight_power must be > 0, got {self.activity_weight_power}"
            )
        if self.activity_pair_weight_mode != "geometric_mean":
            raise ValueError(
                "activity_pair_weight_mode must be 'geometric_mean', "
                f"got {self.activity_pair_weight_mode!r}"
            )
        if not (0.0 <= self.activity_recovery_floor < 1.0):
            raise ValueError(
                "activity_recovery_floor must satisfy 0 <= floor < 1, "
                f"got {self.activity_recovery_floor}"
            )
        if self.activity_recovery_power <= 0.0:
            raise ValueError(
                f"activity_recovery_power must be > 0, got {self.activity_recovery_power}"
            )
        if self.repulsion_inactive_recovery_strength < 0.0:
            raise ValueError(
                "repulsion_inactive_recovery_strength must be >= 0, "
                f"got {self.repulsion_inactive_recovery_strength}"
            )
        if self.seed_active_loss_mode not in {"minimum", "target"}:
            raise ValueError(
                "seed_active_loss_mode must be one of: 'minimum', 'target', "
                f"got {self.seed_active_loss_mode!r}"
            )
        if self.seed_active_loss_temperature <= 0.0:
            raise ValueError(
                "seed_active_loss_temperature must be > 0, "
                f"got {self.seed_active_loss_temperature}"
            )
        if self.seed_active_baseline_strength < 0.0:
            raise ValueError(
                "seed_active_baseline_strength must be >= 0, "
                f"got {self.seed_active_baseline_strength}"
            )
        if self.seed_active_baseline_power <= 0.0:
            raise ValueError(
                "seed_active_baseline_power must be > 0, "
                f"got {self.seed_active_baseline_power}"
            )
        if self.seed_active_hard_barrier_strength < 0.0:
            raise ValueError(
                "seed_active_hard_barrier_strength must be >= 0, "
                f"got {self.seed_active_hard_barrier_strength}"
            )
        if (
            self.seed_active_hard_barrier_temperature is not None
            and self.seed_active_hard_barrier_temperature <= 0.0
        ):
            raise ValueError(
                "seed_active_hard_barrier_temperature must be > 0 or None, "
                f"got {self.seed_active_hard_barrier_temperature}"
            )
        # if self.min_active_seeds is None:
        #     self.min_active_seeds = 10
        # elif self.min_active_seeds < 10:
        #     raise ValueError(
        #         "At least 10 seeds must remain active; "
        #         f"got min_active_seeds={self.min_active_seeds}"
        #     )
        if self.possible_min_active is not None:
            self.possible_min_active = int(self.possible_min_active)
            if self.possible_min_active < 1:
                raise ValueError(
                    "possible_min_active must be >= 1 or None, "
                    f"got {self.possible_min_active}"
                )
            if self.possible_min_active >= int(self.min_active_seeds):
                raise ValueError(
                    "possible_min_active must be smaller than min_active_seeds, "
                    f"got possible_min_active={self.possible_min_active} and "
                    f"min_active_seeds={self.min_active_seeds}"
                )
        self.use_balanced_seed_init = bool(self.use_balanced_seed_init)


def _cfg_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)

def _density_postprocess_debug(
    rho_raw: torch.Tensor,
    rho_filtered: torch.Tensor,
    rho_projected: torch.Tensor,
    rho_final: torch.Tensor,
) -> dict[str, float]:
    filter_delta = (rho_filtered - rho_raw).abs()
    projection_delta = (rho_projected - rho_filtered).abs()
    return {
        "filter_delta_mean": float(filter_delta.detach().mean().item()),
        "filter_delta_max": float(filter_delta.detach().max().item()),
        "projection_delta_mean": float(projection_delta.detach().mean().item()),
        "projection_delta_max": float(projection_delta.detach().max().item()),
        "raw_mean": float(rho_raw.detach().mean().item()),
        "filtered_mean": float(rho_filtered.detach().mean().item()),
        "projected_mean": float(rho_projected.detach().mean().item()),
        "final_mean": float(rho_final.detach().mean().item()),
    }


def _fiber_angles_from_3d(fiber3d: torch.Tensor, eps: float = 1e-6):
    fiber3d = fiber3d / torch.linalg.norm(fiber3d, dim=-1, keepdim=True).clamp_min(eps)
    ax, ay, az = fiber3d.unbind(dim=-1)
    phi = torch.atan2(ay, ax)
    theta = torch.acos(az.clamp(-1.0 + eps, 1.0 - eps))
    return fiber3d, phi, theta


def normalize_decoder_density_fiber_output(out: dict, eps: float = 1e-6) -> dict:
    """
    Accept the decoder's current density/fiber aliases and publish the stable
    training contract: rho, density, fiber3d, phi, theta.
    """
    if "rho" not in out:
        if "density" in out:
            out["rho"] = out["density"]
        elif "rho_surface" in out:
            out["rho"] = out["rho_surface"]
        else:
            raise KeyError("Decoder output must contain one of: rho, density, rho_surface")

    out["density"] = out["rho"]

    if "fiber3d" not in out:
        if "fiber" in out:
            out["fiber3d"] = out["fiber"]
        elif "fiber_direction" in out:
            out["fiber3d"] = out["fiber_direction"]
        elif "3d_fiberDir" in out:
            out["fiber3d"] = out["3d_fiberDir"]
        else:
            raise KeyError("Decoder output must contain one of: fiber3d, fiber, fiber_direction, 3d_fiberDir")

    out["fiber3d"], out["phi"], out["theta"] = _fiber_angles_from_3d(out["fiber3d"], eps=eps)
    out["fiber"] = out["fiber3d"]
    return out


def apply_density_postprocess(
    rho,
    face_tensor,
    config,
    return_debug: bool = False,
):
    """
    Canonical decoder-density postprocess.

    The 3D filter is graph-based, so callers must pass a face_tensor whose
    points/faces correspond to the density samples in `rho`.
    """
    rho_raw = rho
    eps = float(_cfg_value(config, "eps", 1e-8))

    use_density_filter = bool(_cfg_value(config, "use_3d_density_filter", False))
    if use_density_filter:
        rho_filtered = surface_density_filter_metric_aware(
            rho=rho_raw,
            points_xyz=face_tensor["points_xyz"],
            faces=face_tensor["faces_ijk"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            base_radius=float(_cfg_value(config, "filter_radius_3d", 0.03)),
            self_weight=float(_cfg_value(config, "filter_self_weight", 1.0)),
            eps=eps,
        )
    else:
        rho_filtered = rho_raw

    projection_strength = float(_cfg_value(config, "filter_projection_strength", 0.0))
    if projection_strength > 0.0:
        rho_projected = smooth_heaviside_projection(
            rho_filtered,
            beta=float(_cfg_value(config, "filter_projection_beta", 8.0)),
            eta=float(_cfg_value(config, "filter_projection_eta", 0.5)),
            strength=projection_strength,
            eps=eps,
            debug=False,
        )
    else:
        rho_projected = rho_filtered

    rho_final = rho_projected
    if not return_debug:
        return rho_final
    return rho_final, _density_postprocess_debug(
        rho_raw=rho_raw,
        rho_filtered=rho_filtered,
        rho_projected=rho_projected,
        rho_final=rho_final,
    )


def apply_density_postprocess_to_output(
    out: dict,
    face_tensor,
    config,
    return_debug: bool = False,
):
    out = normalize_decoder_density_fiber_output(
        out,
        eps=float(_cfg_value(config, "eps", 1e-6)),
    )
    rho_raw = out["rho"]
    rho_final, stats = apply_density_postprocess(
        rho_raw,
        face_tensor,
        config,
        return_debug=True,
    )

    out["rho_raw_decoder"] = rho_raw
    out["rho"] = rho_final
    out["density"] = rho_final
    out["rho_postprocessed"] = rho_final
    out["fiber3d"], out["phi"], out["theta"] = _fiber_angles_from_3d(
        out["fiber3d"],
        eps=float(_cfg_value(config, "eps", 1e-6)),
    )
    out["fiber"] = out["fiber3d"]
    if return_debug:
        out["density_postprocess_stats"] = stats
        return out, stats
    return out


class RunningNorm:
    def __init__(
        self,
        momentum: float = 0.99,
        eps: float = 1e-12,
        min_scale: float = 1.0,
    ):
        self.val = None
        self.momentum = momentum
        self.eps = eps
        self.min_scale = min_scale

    def update(self, x: float) -> float:
        x = abs(float(x))
        if not math.isfinite(x):
            return max(self.val if self.val is not None else self.min_scale, self.min_scale)

        if x <= self.min_scale:
            return max(self.val if self.val is not None else self.min_scale, self.min_scale)

        x = x + self.eps
        if self.val is None:
            self.val = x
        else:
            self.val = self.momentum * self.val + (1.0 - self.momentum) * x
        return max(self.val, self.min_scale)

    def update_if_active(self, x: float, active: bool) -> float:
        if not active:
            return max(self.val if self.val is not None else self.min_scale, self.min_scale)
        return self.update(x)


def _cpu_detached_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_detached_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_detached_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_detached_tree(v) for v in value)
    return value


def _tree_to_device(value, device=None, dtype=None):
    if torch.is_tensor(value):
        out = value.to(device=device) if device is not None else value
        if dtype is not None and out.is_floating_point():
            out = out.to(dtype=dtype)
        return out
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(v, device=device, dtype=dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(v, device=device, dtype=dtype) for v in value)
    return value


def _safe_int_or_none(value, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().reshape(-1)[0].cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _portable_face_tensor(face_tensor):
    if face_tensor is None:
        return None
    keep_keys = {
        "uv",
        "Xu",
        "Xv",
        "points_xyz",
        "faces_ijk",
        "face_areas",
        "global_vertex_idx",
        "boundary_idx_ring1",
        "u_periodic",
        "v_periodic",
        "face_id",
        "seed_domain_uv_support",
        "seed_domain_sigma",
        "seed_domain_mask_grid",
        "seed_domain_sdf_grid",
        "seed_domain_sdf_uv_min",
        "seed_domain_sdf_uv_max",
        "boundary_curve_uv",
        "boundary_curve_offsets",
        "boundary_curve_loop_id",
    }
    return {
        key: _cpu_detached_tree(value)
        for key, value in dict(face_tensor).items()
        if key in keep_keys
    }


def _portable_cad_domain(face_tensor=None, best_pred=None):
    data = {}
    sources = []
    if isinstance(face_tensor, dict):
        sources.append(face_tensor)
    if isinstance(best_pred, dict) and isinstance(best_pred.get("graph"), dict):
        sources.append(best_pred["graph"])

    for source in sources:
        for key in (
            "boundary_curve_uv",
            "boundary_curve_offsets",
            "boundary_curve_loop_id",
            "seed_domain_sdf_grid",
            "seed_domain_sdf_uv_min",
            "seed_domain_sdf_uv_max",
        ):
            value = source.get(key, None)
            if value is not None and key not in data:
                data[key] = _cpu_detached_tree(value)

    return data or None


def _import_symbol(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class OptimizedShellFunction:
    """
    Reloadable single-face implicit shell field.

    The object evaluates the optimized decoder field on UV points:
        (u, v), Xu, Xv -> density rho and 3D fiber direction.
    """

    package_version = 2

    def __init__(self, package: dict[str, Any], decoder_cls=None, device=None):
        self.package = package
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if decoder_cls is None:
            decoder_info = package.get("decoder_class", {})
            decoder_cls = _import_symbol(
                decoder_info.get("module", "Decoder_CLasses.VoronoiDecorder"),
                decoder_info.get("name", "VoronoiDecoder"),
            )
        self.decoder_cls = decoder_cls

        self.config = package.get("config", {})
        self.face_tensor = _tree_to_device(
            package.get("face_tensor", None),
            device=self.device,
        )
        decoder_init_kwargs_raw = dict(package["decoder_init_kwargs"])
        if self.face_tensor is not None:
            decoder_init_kwargs_raw["face_mesh"] = self.face_tensor
        if decoder_init_kwargs_raw.get("Cad_domain", None) is None:
            decoder_init_kwargs_raw["Cad_domain"] = package.get("cad_domain", None)
        self.decoder_init_kwargs = _tree_to_device(
            decoder_init_kwargs_raw,
            device=self.device,
        )
        self.decoder = self.decoder_cls(**self.decoder_init_kwargs).to(self.device)
        state = package.get("decoder_state_dict", None)
        if state:
            self.decoder.load_state_dict(_tree_to_device(state, device=self.device))
        self.decoder.eval()

        self.best_pred = _tree_to_device(package["best_pred"], device=self.device)
        self.face_metadata = package.get("face_metadata", {})
        self.final_shape_density = _tree_to_device(
            package.get("final_shape_density", None),
            device=self.device,
        )
        self.final_shape_fiber_direction = _tree_to_device(
            package.get("final_shape_fiber_direction", None),
            device=self.device,
        )

    @classmethod
    def load(cls, path, decoder_cls=None, device=None):
        try:
            package = torch.load(path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            package = torch.load(path, map_location=device or "cpu")
        return cls(package=package, decoder_cls=decoder_cls, device=device)

    @staticmethod
    def _true_open_boundary_idx(ft, tol=None):
        if ("boundary_idx_ring1" not in ft) or ft["boundary_idx_ring1"] is None:
            return torch.empty(0, dtype=torch.long, device=ft["uv"].device)

        bidx = torch.unique(ft["boundary_idx_ring1"].to(dtype=torch.long))
        if bidx.numel() == 0:
            return bidx

        uv = ft["uv"]
        u = uv[:, 0]
        v = uv[:, 1]
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        if tol is None:
            u_span = (u.max() - u.min()).abs()
            v_span = (v.max() - v.min()).abs()
            base_span = torch.maximum(
                u_span,
                v_span,
            ).clamp_min(torch.as_tensor(1.0, device=uv.device, dtype=uv.dtype))
            tol = 1e-4 * float(base_span.detach().item())

        ub = u[bidx]
        vb = v[bidx]
        keep = torch.ones_like(bidx, dtype=torch.bool)

        if u_periodic:
            umin = u.min()
            umax = u.max()
            is_u_seam = (ub - umin).abs() <= tol
            is_u_seam = is_u_seam | ((ub - umax).abs() <= tol)
            keep = keep & (~is_u_seam)

        if v_periodic:
            vmin = v.min()
            vmax = v.max()
            is_v_seam = (vb - vmin).abs() <= tol
            is_v_seam = is_v_seam | ((vb - vmax).abs() <= tol)
            keep = keep & (~is_v_seam)

        return bidx[keep]

    def _seed_domain_mask_for_face(self, ft):
        if not bool(self.config.get("use_seed_domain_mask", False)):
            return None

        mask_grid = ft.get("seed_domain_mask_grid", None)
        if mask_grid is not None:
            return mask_grid

        uv_face = ft.get("seed_domain_uv_support", ft["uv"])
        if uv_face.numel() == 0:
            return None

        cfg = self.config
        uv_support = uv_face.detach()
        max_points = int(cfg.get("seed_domain_mask_max_points", 2048))
        if uv_support.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                uv_support.shape[0] - 1,
                max_points,
                device=uv_support.device,
            ).round().to(torch.long)
            uv_support = uv_support[sample_idx]

        sigma_value = ft.get("seed_domain_sigma", None)
        if sigma_value is None:
            sigma = NN_Trainer._estimate_uv_mask_tol(
                uv_support,
                u_periodic=bool(ft.get("u_periodic", False)),
                v_periodic=bool(ft.get("v_periodic", False)),
                fallback=float(cfg.get("boundary_margin", 0.05)),
                scale=float(cfg.get("seed_domain_mask_support_scale", 2.5)),
            )
        elif torch.is_tensor(sigma_value):
            sigma = float(sigma_value.detach().cpu().item())
        else:
            sigma = float(sigma_value)
        sigma = max(float(sigma), float(cfg.get("eps", 1e-12)))
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        def mask_fn(seeds):
            support = uv_support.to(device=seeds.device, dtype=seeds.dtype)
            diff = seeds.unsqueeze(1) - support.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            dmin = torch.norm(diff, dim=-1).amin(dim=1)
            sigma_t = torch.as_tensor(sigma, device=seeds.device, dtype=seeds.dtype)
            return torch.exp(-0.5 * (dmin / sigma_t.clamp_min(float(cfg.get("eps", 1e-12)))).pow(2))

        return mask_fn

    def evaluate_at_uv(
        self,
        points_uv,
        Xu,
        Xv,
        points_xyz=None,
        face_tensor=None,
        boundary_uv=None,
        hard_seed_mask=True,
    ):
        points_uv = torch.as_tensor(points_uv, device=self.device)
        dtype = points_uv.dtype if points_uv.is_floating_point() else torch.float32
        points_uv = points_uv.to(dtype=dtype)
        Xu = torch.as_tensor(Xu, device=self.device, dtype=dtype)
        Xv = torch.as_tensor(Xv, device=self.device, dtype=dtype)
        points_xyz = (
            None
            if points_xyz is None
            else torch.as_tensor(points_xyz, device=self.device, dtype=dtype)
        )

        ft = None
        if face_tensor is not None:
            ft = _tree_to_device(dict(face_tensor), device=self.device, dtype=dtype)

        points_face_id = torch.zeros(points_uv.shape[0], dtype=torch.long, device=self.device)
        boundary_face_id = None
        if boundary_uv is None and ft is not None:
            bidx = self._true_open_boundary_idx(ft)
            if bidx.numel() > 0:
                boundary_uv = ft["uv"][bidx]
                boundary_face_id = torch.zeros(
                    boundary_uv.shape[0],
                    dtype=torch.long,
                    device=self.device,
                )
        elif boundary_uv is not None:
            boundary_uv = torch.as_tensor(boundary_uv, device=self.device, dtype=dtype)
            boundary_face_id = torch.zeros(
                boundary_uv.shape[0],
                dtype=torch.long,
                device=self.device,
            )

        seed_domain_mask = self._seed_domain_mask_for_face(ft) if ft is not None else None
        pred = _tree_to_device(self.best_pred, device=self.device, dtype=dtype)
        tau = pred.get("tau", None)
        if tau is None:
            tau = float(self.config.get("tau", 0.02))

        with torch.no_grad():
            # Raw arbitrary-point evaluation: graph density postprocess needs a
            # full mesh/face tensor and is applied by evaluate_face().
            use_u_periodic = self.decoder._bool_value(getattr(self.decoder, "face_u_periodic", False))
            use_v_periodic = self.decoder._bool_value(getattr(self.decoder, "face_v_periodic", False))
            if hasattr(self.decoder, "build_swept_tube_fields"):
                return self.decoder.build_swept_tube_fields(
                    points_uv=points_uv,
                    points_3d=points_xyz,
                    seeds_uv=pred["seeds_raw"],
                    Xu=Xu,
                    Xv=Xv,
                    cad_domain=getattr(self.decoder, "Cad_domain", None),
                    u_periodic=use_u_periodic,
                    v_periodic=use_v_periodic,
                    return_xyz=True,
                    generate_density_fiber=bool(self.config.get("generate_decoder_density_fiber", True)),
                )

            if points_xyz is None:
                raise ValueError(
                    "points_xyz is required for decoders that evaluate from their face mesh."
                )

            old_points_uv = getattr(self.decoder, "points_uv", None)
            old_points_3d = getattr(self.decoder, "points_3d", None)
            old_Xu = getattr(self.decoder, "Xu", None)
            old_Xv = getattr(self.decoder, "Xv", None)
            old_cad_domain = getattr(self.decoder, "Cad_domain", None)
            try:
                self.decoder.points_uv = points_uv
                self.decoder.points_3d = points_xyz
                self.decoder.Xu = Xu
                self.decoder.Xv = Xv
                if face_tensor is not None and face_tensor.get("Cad_domain", None) is not None:
                    self.decoder.Cad_domain = face_tensor["Cad_domain"]
                return self.decoder(
                    seeds_uv=pred["seeds_raw"],
                    generate_density_fiber=bool(self.config.get("generate_decoder_density_fiber", True)),
                )
            finally:
                self.decoder.points_uv = old_points_uv
                self.decoder.points_3d = old_points_3d
                self.decoder.Xu = old_Xu
                self.decoder.Xv = old_Xv
                self.decoder.Cad_domain = old_cad_domain

    def evaluate_face(self, face_tensor=None, hard_seed_mask=True):
        if face_tensor is None:
            face_tensor = self.face_tensor
        if face_tensor is None:
            raise ValueError(
                "face_tensor is required because this optimized shell package "
                "does not contain a saved face_tensor snapshot."
            )
        ft = _tree_to_device(
            dict(face_tensor),
            device=self.device,
            dtype=face_tensor["uv"].dtype if torch.is_tensor(face_tensor["uv"]) else None,
        )
        out = self.evaluate_at_uv(
            points_uv=face_tensor["uv"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            points_xyz=face_tensor["points_xyz"],
            face_tensor=ft,
            hard_seed_mask=hard_seed_mask,
        )
        return apply_density_postprocess_to_output(
            out,
            ft,
            self.config,
            return_debug=False,
        )

    def build_fem_fields(self, shell_problem, face_tensor, rho_void=1e-3, hard_seed_mask=True):
        out = self.evaluate_face(face_tensor, hard_seed_mask=hard_seed_mask)
        return shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=out["rho"],
            fiber_surface=out["fiber3d"],
            rho_void=rho_void,
        )


def evaluate_optimized_shell_function(
    optimized_function,
    face_tensors=None,
    face_index: int = 0,
    hard_seed_mask: bool = True,
):
    """
    Evaluate a loaded optimized single-face shell function on a face tensor.

    Returns surface density and 3D fiber direction, ready for later
    visualization or export to a custom FEM workflow.
    """
    if face_tensors is None:
        face_tensor = getattr(optimized_function, "face_tensor", None)
    elif isinstance(face_tensors, dict) and "face_tensors" in face_tensors:
        face_tensors = face_tensors["face_tensors"]
        face_tensor = face_tensors[int(face_index)]
    elif isinstance(face_tensors, (list, tuple)):
        face_tensor = face_tensors[int(face_index)]
    else:
        face_tensor = face_tensors

    if face_tensors is None:
        final_density = getattr(optimized_function, "final_shape_density", None)
        final_fiber = getattr(optimized_function, "final_shape_fiber_direction", None)
        if final_density is not None and final_fiber is not None:
            density = final_density
            fiber_3d = final_fiber
            density_binary = (density >= 0.5).to(dtype=density.dtype)
            return {
                "2d_density": density,
                "2d_fiberDir": None,
                "3d_density": density,
                "3d_fiberDir": fiber_3d,
                "density": density,
                "density_binary": density_binary,
                "fiber_direction": fiber_3d,
                "rho": density,
                "rho_raw_decoder": density,
                "rho_postprocessed": density,
                "fiber3d": fiber_3d,
                "t_uv": None,
                "decoder_output": None,
                "face_tensor": face_tensor,
            }

    out = optimized_function.evaluate_face(
        face_tensor,
        hard_seed_mask=hard_seed_mask,
    )
    density = out["rho"]
    fiber_2d = out.get("t_uv", out.get("t_uv_raw", None))
    fiber_3d = out["fiber3d"]
    rho_raw_decoder = out.get("rho_raw_decoder", density)
    density_binary = (density >= 0.5).to(dtype=density.dtype)
    return {
        "2d_density": density,
        "2d_fiberDir": fiber_2d,
        "3d_density": density,
        "3d_fiberDir": fiber_3d,
        "density": density,
        "density_binary": density_binary,
        "fiber_direction": fiber_3d,
        "rho": density,
        "rho_raw_decoder": rho_raw_decoder,
        "rho_postprocessed": out.get("rho_postprocessed", density),
        "fiber3d": fiber_3d,
        "t_uv": fiber_2d,
        "decoder_output": out,
        "face_tensor": face_tensor,
    }


def sanity_check_density_postprocess_pipeline(
    optimized_function,
    face_tensor,
    expected_training_rho=None,
    tol: float = 1e-6,
    small_tolerance: float = 1e-8,
):
    out = optimized_function.evaluate_face(face_tensor)
    rho = out["rho"]
    rho_raw = out.get("rho_raw_decoder", rho)
    cfg = optimized_function.config
    postprocess_enabled = bool(cfg.get("use_3d_density_filter", False))
    if postprocess_enabled:
        delta = (rho - rho_raw).abs().mean()
        assert float(delta.detach().item()) > float(small_tolerance), (
            "Postprocess is enabled but evaluate_face returned density too close "
            "to rho_raw_decoder."
        )
    fields = evaluate_optimized_shell_function(optimized_function, face_tensor)
    assert fields["rho"] is out["rho"] or torch.allclose(fields["rho"], out["rho"], atol=tol, rtol=0.0)
    assert fields["density"] is fields["rho"] or torch.allclose(fields["density"], fields["rho"], atol=tol, rtol=0.0)
    assert "rho_raw_decoder" in fields
    if expected_training_rho is not None:
        expected = torch.as_tensor(expected_training_rho, device=rho.device, dtype=rho.dtype)
        assert torch.allclose(rho, expected, atol=tol, rtol=0.0), (
            "Loaded optimized_shell_function.evaluate_face(...) does not match "
            "the expected training-time postprocessed density."
        )
    return {
        "mean_abs_postprocess_delta": float((rho - rho_raw).abs().mean().detach().item()),
        "rho_mean": float(rho.detach().mean().item()),
        "rho_raw_mean": float(rho_raw.detach().mean().item()),
    }


def load_optimized_shell_function(path, decoder_cls=None, device=None):
    return OptimizedShellFunction.load(path, decoder_cls=decoder_cls, device=device)


def _field_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _surface_density_volume_fraction(face_tensor, density_values):
    density = np.asarray(density_values, dtype=np.float64).reshape(-1)
    faces = _field_to_numpy(face_tensor.get("faces_ijk", np.empty((0, 3)))).astype(np.int64)

    if faces.size == 0:
        valid = np.isfinite(density)
        value = float(np.mean(density[valid])) if np.any(valid) else float("nan")
        return value, "point-mean"

    face_areas_raw = face_tensor.get("face_areas", None)
    if face_areas_raw is not None:
        face_areas = _field_to_numpy(face_areas_raw).reshape(-1).astype(np.float64)
    else:
        xyz = _field_to_numpy(face_tensor["points_xyz"]).astype(np.float64)
        tri = xyz[faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )

    if face_areas.shape[0] != faces.shape[0]:
        valid = np.isfinite(density)
        value = float(np.mean(density[valid])) if np.any(valid) else float("nan")
        return value, "point-mean"

    weights = np.zeros((density.shape[0],), dtype=np.float64)
    local_weight = face_areas / 3.0
    np.add.at(weights, faces[:, 0], local_weight)
    np.add.at(weights, faces[:, 1], local_weight)
    np.add.at(weights, faces[:, 2], local_weight)

    valid = np.isfinite(density) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), "area-weighted"
    return float(np.sum(density[valid] * weights[valid]) / np.sum(weights[valid])), "area-weighted"


def visualize_optimized_shell_fields(
    fields,
    show_2d: bool = True,
    show_3d: bool = True,
    density_cmap: str = "viridis",
    fiber_stride: int = 20,
    fiber_min_density: float = 0.05,
    fiber_scale_2d: float = 0.06,
    fiber_scale_3d: float | None = None,
    fiber_vector_style: str = "arrow",
    fiber_color: str = "#1f4fa3",
    show_fiber_surface: bool = True,
    fiber_surface_opacity: float = 0.25,
    show_fiber_background: bool = False,
    show_edges: bool = False,
    window_size: tuple[int, int] = (1500, 700),
):
    """
    Visualize loaded optimized shell fields in UV and on the 3D surface.

    Returns a dictionary with optional:
        uv_fig: matplotlib figure for 2D UV density/fiber
        plotter: pyvista plotter for 3D density/fiber
    """
    face_tensor = fields["face_tensor"]
    uv = _field_to_numpy(face_tensor["uv"]).astype(np.float64)
    xyz = _field_to_numpy(face_tensor["points_xyz"]).astype(np.float64)
    faces = _field_to_numpy(face_tensor["faces_ijk"]).astype(np.int64)

    density_2d = _field_to_numpy(fields["2d_density"]).reshape(-1).astype(np.float64)
    fiber_2d_raw = fields.get("2d_fiberDir", None)
    fiber_2d = (
        None
        if fiber_2d_raw is None
        else _field_to_numpy(fiber_2d_raw).reshape(-1, 2).astype(np.float64)
    )
    density_3d = _field_to_numpy(fields["3d_density"]).reshape(-1).astype(np.float64)
    fiber_3d = _field_to_numpy(fields["3d_fiberDir"]).reshape(-1, 3).astype(np.float64)

    result = {}

    if show_2d:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax_density, ax_fiber = axes

        if faces.size > 0:
            density_artist = ax_density.tripcolor(
                uv[:, 0],
                uv[:, 1],
                faces,
                density_2d,
                shading="gouraud",
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            density_artist = ax_density.scatter(
                uv[:, 0],
                uv[:, 1],
                c=density_2d,
                s=10,
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
            )
        ax_density.set_title("2D UV Density")
        ax_density.set_xlabel("u")
        ax_density.set_ylabel("v")
        ax_density.set_aspect("equal", adjustable="box")
        fig.colorbar(density_artist, ax=ax_density, label="density")

        if show_fiber_background and faces.size > 0:
            ax_fiber.tripcolor(
                uv[:, 0],
                uv[:, 1],
                faces,
                density_2d,
                shading="gouraud",
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.30,
            )
        elif show_fiber_background:
            ax_fiber.scatter(
                uv[:, 0],
                uv[:, 1],
                c=density_2d,
                s=10,
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.30,
                linewidths=0,
            )

        if fiber_2d is not None:
            fiber_norm_2d = np.linalg.norm(fiber_2d, axis=1)
            mask_2d = np.isfinite(density_2d) & np.isfinite(fiber_2d).all(axis=1)
            mask_2d &= density_2d >= float(fiber_min_density)
            mask_2d &= fiber_norm_2d > 1e-12
            if fiber_stride > 1:
                stride_mask = np.zeros(mask_2d.shape[0], dtype=bool)
                stride_mask[::int(fiber_stride)] = True
                mask_2d &= stride_mask
        else:
            mask_2d = np.zeros(density_2d.shape[0], dtype=bool)

        if fiber_2d is not None and np.any(mask_2d):
            ax_fiber.quiver(
                uv[mask_2d, 0],
                uv[mask_2d, 1],
                fiber_2d[mask_2d, 0],
                fiber_2d[mask_2d, 1],
                density_2d[mask_2d],
                cmap=density_cmap,
                angles="xy",
                scale_units="xy",
                scale=max(float(fiber_scale_2d), 1e-8) ** -1,
                width=0.003,
                pivot="mid",
            )
        ax_fiber.set_title("2D UV Fiber Direction")
        ax_fiber.set_xlabel("u")
        ax_fiber.set_ylabel("v")
        ax_fiber.set_aspect("equal", adjustable="box")
        result["uv_fig"] = fig

    if show_3d:
        volume_fraction, volume_fraction_method = _surface_density_volume_fraction(
            face_tensor,
            density_3d,
        )
        result["density_volume_fraction"] = volume_fraction
        result["density_volume_fraction_method"] = volume_fraction_method
        print(
            "3D density volume fraction "
            f"({volume_fraction_method}): {volume_fraction:.6f}"
        )

        if faces.size > 0:
            pv_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces
            mesh = pv.PolyData(xyz, pv_faces.reshape(-1))
        else:
            mesh = pv.PolyData(xyz)
        mesh["density"] = density_3d.astype(np.float32)

        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)

        plotter.subplot(0, 0)
        plotter.add_text("3D Surface Density", font_size=10)
        plotter.add_mesh(
            mesh,
            scalars="density",
            cmap=density_cmap,
            clim=[0.0, 1.0],
            show_edges=show_edges,
        )
        plotter.show_axes()

        plotter.subplot(0, 1)
        plotter.add_text("3D Surface Fiber Direction", font_size=10)
        if show_fiber_background:
            plotter.add_mesh(
                mesh.copy(),
                scalars="density",
                cmap=density_cmap,
                clim=[0.0, 1.0],
                opacity=0.30,
                show_edges=show_edges,
            )
        elif show_fiber_surface:
            plotter.add_mesh(
                mesh.copy(),
                color="white",
                opacity=float(fiber_surface_opacity),
                show_edges=False,
                smooth_shading=True,
            )

        fiber_norm_3d = np.linalg.norm(fiber_3d, axis=1)
        mask_3d = np.isfinite(density_3d) & np.isfinite(fiber_3d).all(axis=1)
        mask_3d &= density_3d >= float(fiber_min_density)
        mask_3d &= fiber_norm_3d > 1e-12
        if fiber_stride > 1:
            stride_mask = np.zeros(mask_3d.shape[0], dtype=bool)
            stride_mask[::int(fiber_stride)] = True
            mask_3d &= stride_mask

        if np.any(mask_3d):
            diag = float(np.linalg.norm(np.ptp(xyz, axis=0)))
            glyph_scale = 0.04 * max(diag, 1e-6) if fiber_scale_3d is None else float(fiber_scale_3d)
            cloud = pv.PolyData(xyz[mask_3d])
            cloud["vectors"] = fiber_3d[mask_3d].astype(np.float32)
            cloud["density"] = density_3d[mask_3d].astype(np.float32)
            style = str(fiber_vector_style).lower()
            if style == "arrow":
                glyph_geom = pv.Arrow(
                    start=(0.0, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0),
                    tip_length=0.30,
                    tip_radius=0.045,
                    shaft_radius=0.014,
                    shaft_resolution=8,
                    tip_resolution=12,
                )
            elif style == "line":
                glyph_geom = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
            else:
                raise ValueError("fiber_vector_style must be 'arrow' or 'line'")
            glyphs = cloud.glyph(
                orient="vectors",
                scale=False,
                factor=glyph_scale,
                geom=glyph_geom,
            )
            plotter.add_mesh(glyphs, color=fiber_color, line_width=2)

        plotter.show_axes()
        plotter.link_views()
        result["plotter"] = plotter

    return result


def visualize_optimized_shell_fields_2d(fields, **kwargs):
    return visualize_optimized_shell_fields(
        fields,
        show_2d=True,
        show_3d=False,
        **kwargs,
    )["uv_fig"]


def visualize_optimized_shell_fields_3d(fields, **kwargs):
    result = visualize_optimized_shell_fields(
        fields,
        show_2d=False,
        show_3d=True,
        **kwargs,
    )
    return result["plotter"], result["density_volume_fraction"]


def binarize_optimized_shell_fields(
    fields,
    density_threshold: float = 0.5,
    solid_density: float = 1.0,
    void_density: float = 1e-3,
    mask_void_fibers: bool = True,
):
    """
    Convert optimized continuous surface density to solid/void density.

    Fiber directions are directions, so they are not thresholded into binary
    values. They are normalized and optionally zeroed in void regions.
    """
    out = dict(fields)
    density = fields["3d_density"]
    fiber_2d = fields.get("2d_fiberDir", None)
    fiber_3d = fields["3d_fiberDir"]

    solid_mask = density >= float(density_threshold)
    binary_density = torch.where(
        solid_mask,
        torch.as_tensor(solid_density, dtype=density.dtype, device=density.device),
        torch.as_tensor(void_density, dtype=density.dtype, device=density.device),
    )

    def normalize_and_mask(fiber):
        if fiber is None:
            return None
        norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(1e-12)
        fiber_out = fiber / norm
        if mask_void_fibers:
            fiber_out = torch.where(solid_mask[:, None], fiber_out, torch.zeros_like(fiber_out))
        return fiber_out

    binary_fiber_2d = normalize_and_mask(fiber_2d)
    binary_fiber_3d = normalize_and_mask(fiber_3d)

    out["2d_density_continuous"] = fields["2d_density"]
    out["3d_density_continuous"] = fields["3d_density"]
    out["2d_fiberDir_continuous"] = fields.get("2d_fiberDir", None)
    out["3d_fiberDir_continuous"] = fields["3d_fiberDir"]

    out["solid_mask"] = solid_mask
    out["2d_density"] = binary_density
    out["3d_density"] = binary_density
    out["density"] = binary_density
    out["rho"] = binary_density

    if binary_fiber_2d is not None:
        out["2d_fiberDir"] = binary_fiber_2d
        out["t_uv"] = binary_fiber_2d
    out["3d_fiberDir"] = binary_fiber_3d
    out["fiber_direction"] = binary_fiber_3d
    out["fiber3d"] = binary_fiber_3d

    return out


class Load_Model:
    @staticmethod
    def load(path, decoder_cls=None, device=None):
        return load_optimized_shell_function(
            path=path,
            decoder_cls=decoder_cls,
            device=device,
        )

    @staticmethod
    def evaluate(
        optimized_function,
        face_tensors,
        face_index: int = 0,
        hard_seed_mask: bool = True,
    ):
        return evaluate_optimized_shell_function(
            optimized_function=optimized_function,
            face_tensors=face_tensors,
            face_index=face_index,
            hard_seed_mask=hard_seed_mask,
        )

    @staticmethod
    def visualize(
        fields,
        show_2d: bool = True,
        show_3d: bool = True,
        **kwargs,
    ):
        return visualize_optimized_shell_fields(
            fields,
            show_2d=show_2d,
            show_3d=show_3d,
            **kwargs,
        )

    @staticmethod
    def visualize_2d(fields, **kwargs):
        return visualize_optimized_shell_fields_2d(fields, **kwargs)

    @staticmethod
    def visualize_3d(fields, **kwargs):
        return visualize_optimized_shell_fields_3d(fields, **kwargs)

    @staticmethod
    def binarize(
        fields,
        density_threshold: float = 0.5,
        solid_density: float = 1.0,
        void_density: float = 1e-3,
        mask_void_fibers: bool = True,
    ):
        return binarize_optimized_shell_fields(
            fields,
            density_threshold=density_threshold,
            solid_density=solid_density,
            void_density=void_density,
            mask_void_fibers=mask_void_fibers,
        )

class NN_Trainer:
    def __init__(
        self,
        generator,
        viz,
        decoder_cls,
        ppnet_cls,
        fem,
        shell_problem,
        config: TrainingConfig,
        loading_img=None,
        Cad_domain=None,
        cad_domain=None,
        face_mesh=None,
    ):
        self.generator = generator
        self.viz = viz
        self.decoder_cls = decoder_cls
        self.ppnet_cls = ppnet_cls
        self.fem = fem
        self.shell_problem = shell_problem
        self.cfg = config
        self.Cad_domain = Cad_domain if Cad_domain is not None else cad_domain
        if self.Cad_domain is None:
            self.Cad_domain = generator
        self.face_mesh = face_mesh

        self.last_fem_debug = {}
        self.fem_debug_history = []
        self.loss_fem = Loss_FEM(self)
        self.loss_rep = Loss_rep()
        self.loss_cvt = LossDensityWeightedCVT()
        self.loss_l_seed = Loss_SeedActive()
        self.timelapse_loading_img = (
            None if loading_img is None else self._composite_to_white(np.asarray(loading_img))
        )

        self.writer = None
        self.tensorboard_log_dir = None
        self._init_tensorboard()

    def curve_3d_edge_lengths(
        self,
        curve_geometry: SharedCurveGeometry,
        edge_types: tuple[int, ...] | list[int] | None = None,
    ):
        allowed_types = (
            tuple(edge_types)
            if edge_types is not None
            else resolve_edge_types_in_losses(self.cfg.Edge_in_losses)
        )
        type_mask = build_edge_type_mask(curve_geometry.edge_type, allowed_types)
        keep = curve_geometry.finite_edge_mask & type_mask
        return curve_geometry.edge_lengths[keep]

    def curve_edge_activity_weights(
        self,
        curve_geometry: SharedCurveGeometry,
        seed_active_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        curves = curve_geometry.curves_xyz
        edge_count = curves.shape[0]
        pairs = curve_geometry.edge_seed_pair
        if seed_active_weights is None or pairs is None:
            return torch.ones(edge_count, dtype=curves.dtype, device=curves.device)

        num_seeds = int(seed_active_weights.reshape(-1).numel())
        g_eff = prepare_seed_activity_weights(
            seed_active_weights,
            num_seeds=num_seeds,
            reference=curves,
            floor=float(getattr(self.cfg, "activity_weight_floor", 0.02)),
            power=float(getattr(self.cfg, "activity_weight_power", 1.0)),
        )

        pairs = pairs.to(device=curves.device, dtype=torch.long)
        valid = (pairs >= 0) & (pairs < num_seeds)
        safe_pairs = pairs.clamp(0, max(num_seeds - 1, 0))
        endpoint_weights = g_eff[safe_pairs] * valid.to(dtype=curves.dtype)
        valid_count = valid.to(dtype=curves.dtype).sum(dim=1)

        two_endpoint = valid_count >= 2.0
        one_endpoint = valid_count == 1.0
        edge_activity = torch.ones(edge_count, dtype=curves.dtype, device=curves.device)
        edge_activity = torch.where(
            two_endpoint,
            torch.sqrt(endpoint_weights.prod(dim=1).clamp_min(float(getattr(self.cfg, "eps", 1e-12)))),
            edge_activity,
        )
        edge_activity = torch.where(
            one_endpoint,
            endpoint_weights.sum(dim=1),
            edge_activity,
        )
        return edge_activity

    def curve_network_length_loss(
        self,
        curve_geometry: SharedCurveGeometry,
        seed_active_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Minimize the total selected Voronoi curve network length.

        This is the final geometry objective: fixed width controls volume, while
        this term shrinks the interior VD network. Activity weights remain
        graph-connected, so inactive or invalid seeds are softly discounted
        without detaching gradients from the decoder activity computation.
        """
        cfg = self.cfg
        allowed_types = resolve_edge_types_in_losses("VDonly")
        type_mask = build_edge_type_mask(curve_geometry.edge_type, allowed_types)
        keep = curve_geometry.finite_edge_mask & type_mask
        edge_lengths = curve_geometry.edge_lengths[keep]
        if edge_lengths.numel() == 0:
            return curve_geometry.curves_xyz.reshape(-1)[0] * 0.0
        return edge_lengths.sum()

    @staticmethod
    def topology_identifier_from_graph(graph: dict | None) -> str:
        """
        Build a stable identifier for meaningful Voronoi topology.

        The signature ignores:
        - node numbering;
        - edge ordering;
        - differentiable node coordinates;
        - edge_index endpoint IDs.

        It uses canonical seed-pair ownership and edge type.
        """
        if not isinstance(graph, dict):
            return ""

        edge_seed_pair = graph.get(
            "edge_seed_pair_original",
            graph.get("edge_seed_pair"),
        )
        edge_type = graph.get("edge_type")

        if edge_seed_pair is None:
            return ""

        if torch.is_tensor(edge_seed_pair):
            pairs = edge_seed_pair.detach().cpu().to(torch.long).numpy()
        else:
            pairs = np.asarray(edge_seed_pair, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            return ""

        if edge_type is None:
            types = np.zeros((pairs.shape[0],), dtype=np.int64)
        elif torch.is_tensor(edge_type):
            types = edge_type.detach().cpu().to(torch.long).numpy().reshape(-1)
        else:
            types = np.asarray(edge_type, dtype=np.int64).reshape(-1)

        if types.shape[0] != pairs.shape[0]:
            return ""

        # Seed ownership is undirected.
        pairs = np.sort(pairs, axis=1)

        # Each row represents one meaningful edge identity.
        records = np.column_stack((pairs, types)).astype(np.int64, copy=False)

        # Remove exact duplicates if graph construction produces any.
        records = np.unique(records, axis=0)

        # Canonical ordering independent of decoder/SciPy output order.
        if records.shape[0] > 1:
            order = np.lexsort(
                (
                    records[:, 2],  # edge type
                    records[:, 1],  # larger seed ID
                    records[:, 0],  # smaller seed ID
                )
            )
            records = records[order]

        digest = hashlib.sha1()
        digest.update(str(records.shape).encode("utf-8"))
        digest.update(records.tobytes())

        return digest.hexdigest()[:16]

    @staticmethod
    def topology_edge_count_from_graph(graph: dict | None) -> int | None:
        if not isinstance(graph, dict):
            return None

        edge_seed_pair = graph.get(
            "edge_seed_pair_original",
            graph.get("edge_seed_pair"),
        )
        if edge_seed_pair is not None:
            if torch.is_tensor(edge_seed_pair):
                return int(edge_seed_pair.shape[0])
            return int(np.asarray(edge_seed_pair).shape[0])

        edge_index = graph.get("edge_index", None)
        if edge_index is not None:
            if torch.is_tensor(edge_index):
                return int(edge_index.shape[0])
            return int(np.asarray(edge_index).shape[0])

        return None

    @staticmethod
    def update_adaptive_stage_controller(
        runtime: StageRuntime,
        row: dict[str, Any],
        *,
        meaningful_improvement: bool,
        stage_topology_grace_steps: int,
        stage_topology_grace_max_resets: int | None = None,
        debug_stage_controller: bool = False,
        global_step: int | None = None,
    ) -> dict[str, bool | int | str | None]:
        spec = runtime.spec
        topology_identifier = str(row.get("topology_identifier", ""))

        edge_count_raw = row.get("number_of_total_edges", None)
        edge_count = (
            int(edge_count_raw)
            if isinstance(edge_count_raw, (int, float))
            and math.isfinite(float(edge_count_raw))
            else None
        )

        active_count = int(round(float(row.get("active_units_total", 0.0))))

        previous_active_count = runtime.previous_active_count
        previous_identifier = runtime.previous_topology_identifier
        previous_edge_count = runtime.previous_edge_count

        active_count_changed = (
            previous_active_count is not None
            and active_count != previous_active_count
        )
        identifier_changed = (
            previous_identifier is not None
            and bool(previous_identifier)
            and bool(topology_identifier)
            and topology_identifier != previous_identifier
        )
        edge_count_changed = (
            previous_edge_count is not None
            and edge_count is not None
            and edge_count != previous_edge_count
        )
        topology_changed = (
            active_count_changed
            or identifier_changed
            or edge_count_changed
        )

        runtime.previous_active_count = active_count
        runtime.previous_topology_identifier = topology_identifier
        runtime.previous_edge_count = edge_count

        patience_active = runtime.local_step + 1 >= spec.min_steps
        grace_reset_budget = (
            int(stage_topology_grace_max_resets)
            if stage_topology_grace_max_resets is not None
            else None
        )
        grace_reset_allowed = (
            topology_changed
            and int(stage_topology_grace_steps) > 0
            and (
                grace_reset_budget is None
                or int(runtime.topology_grace_resets_used) < grace_reset_budget
            )
        )
        if patience_active:
            overall_feasible = bool(row.get("overall_feasible", True))
            design_patience_active = bool(overall_feasible)
            if not design_patience_active:
                runtime.recovery_best_monitor = min(
                    float(runtime.recovery_best_monitor),
                    float(row.get("overall_constraint_violation", row.get("mechanical_violation", float("inf")))),
                )
            elif grace_reset_allowed:
                runtime.topology_grace_remaining = int(stage_topology_grace_steps)
                runtime.topology_grace_resets_used += 1
            elif runtime.topology_grace_remaining > 0:
                runtime.topology_grace_remaining = max(
                    int(runtime.topology_grace_remaining) - 1,
                    0,
                )
            elif meaningful_improvement:
                runtime.patience_counter = 0
            else:
                runtime.patience_counter += 1
        else:
            overall_feasible = bool(row.get("overall_feasible", True))
            design_patience_active = False
            runtime.patience_counter = 0
            runtime.topology_grace_remaining = 0
        diagnostics = {
            "topology_changed": bool(topology_changed),
            "topology_identifier_short": topology_identifier[:8] if topology_identifier else "",
            "patience_active": bool(patience_active),
            "design_patience_active": bool(patience_active and design_patience_active),
            "stage_monitor_mode": str(row.get("stage_monitor_mode", "design" if overall_feasible else "recovery")),
            "meaningful_improvement": bool(meaningful_improvement),
            "active_count_changed": bool(active_count_changed),
            "identifier_changed": bool(identifier_changed),
            "edge_count_changed": bool(edge_count_changed),
            "topology_grace_reset": bool(patience_active and grace_reset_allowed),
            "topology_grace_resets_used": int(runtime.topology_grace_resets_used),
            "previous_topology_identifier": previous_identifier,
            "current_topology_identifier": topology_identifier,
            "previous_edge_count": previous_edge_count,
            "current_edge_count": edge_count,
        }

        if topology_changed and debug_stage_controller:
            print(
                "[Topology change] "
                f"global_step={global_step if global_step is not None else 'None'} "
                f"stage={runtime.spec.stage_id} "
                f"local_step={runtime.local_step} "
                f"active_changed={active_count_changed} "
                f"identifier_changed={identifier_changed} "
                f"edge_count_changed={edge_count_changed} "
                f"previous_id={previous_identifier[:8] if previous_identifier else 'None'} "
                f"current_id={topology_identifier[:8] if topology_identifier else 'None'} "
                f"previous_edges={previous_edge_count} "
                f"current_edges={edge_count}"
            )

        row["best_stage_monitor"] = runtime.best_raw_monitor
        row["recovery_best_monitor"] = runtime.recovery_best_monitor
        row["stage_patience_counter"] = runtime.patience_counter
        row["topology_grace_remaining"] = runtime.topology_grace_remaining
        row["topology_grace_resets_used"] = runtime.topology_grace_resets_used
        row.update(diagnostics)

        return diagnostics

    def solution_topology_metrics(
        self,
        decoder_out: dict,
        *,
        curve_lengths: torch.Tensor | None = None,
    ) -> dict[str, float | int | str]:
        if curve_lengths is None:
            raise ValueError("solution_topology_metrics requires preselected curve_lengths.")

        graph = decoder_out.get("graph", None)
        topology_edge_count = self.topology_edge_count_from_graph(graph)
        if topology_edge_count is not None:
            number_of_total_edges = topology_edge_count
        else:
            curves = decoder_out.get("edge_curves_xyz", decoder_out.get("edge_curves_uv", None))
            number_of_total_edges = int(curves.shape[0]) if isinstance(curves, torch.Tensor) and curves.ndim >= 1 else 0
        number_of_selected_edges = int(curve_lengths.numel()) if curve_lengths is not None else 0

        if curve_lengths is not None and curve_lengths.numel() > 0:
            lengths = curve_lengths.detach()
            minimum_length = float(lengths.min().item())
            maximum_length = float(lengths.max().item())
            mean_length = float(lengths.mean().item())
            standard_deviation = float(lengths.std(unbiased=False).item())
            coefficient_of_variation = (
                standard_deviation / mean_length
                if math.isfinite(mean_length) and abs(mean_length) > float(self.cfg.eps)
                else float("nan")
            )
            maximum_minimum_ratio = maximum_length / max(minimum_length, float(self.cfg.eps))
        else:
            minimum_length = float("nan")
            maximum_length = float("nan")
            mean_length = float("nan")
            standard_deviation = float("nan")
            coefficient_of_variation = float("nan")
            maximum_minimum_ratio = float("nan")

        return {
            "minimum_length": minimum_length,
            "maximum_length": maximum_length,
            "mean_length": mean_length,
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": coefficient_of_variation,
            "maximum_minimum_ratio": maximum_minimum_ratio,
            "number_of_selected_edges": number_of_selected_edges,
            "number_of_total_edges": number_of_total_edges,
            "number_of_edges": number_of_selected_edges,
            "topology_identifier": self.topology_identifier_from_graph(graph),
        }


    def curve_length_similarity_loss(
        self,
        curve_geometry: SharedCurveGeometry,
        seed_active_weights: torch.Tensor | None = None,
    ):
        """
        Encourage selected 3D Voronoi edges to have similar lengths.

        Uses a tolerance region, an average penalty, and a smooth worst-edge
        penalty. The objective is scale-invariant.
        """
        cfg = self.cfg

        allowed_types = resolve_edge_types_in_losses(cfg.Edge_in_losses)
        type_mask = build_edge_type_mask(curve_geometry.edge_type, allowed_types)
        keep = curve_geometry.finite_edge_mask & type_mask
        edge_lengths = curve_geometry.edge_lengths[keep]
        edge_weights = self.curve_edge_activity_weights(
            curve_geometry,
            seed_active_weights,
        )[keep]

        if edge_lengths.numel() <= 1:
            return edge_lengths.new_zeros(())

        eps = float(getattr(cfg, "curve_length_eps", 1e-8))
        tolerance = max(
            float(getattr(cfg, "curve_length_tolerance", 0.10)),
            0.0,
        )
        worst_weight = max(
            float(getattr(cfg, "curve_length_worst_weight", 1.0)),
            0.0,
        )
        outlier_weight = max(
            float(getattr(cfg, "curve_length_outlier_weight", 1.0)),
            0.0,
        )

        weight_sum = edge_weights.sum().clamp_min(eps)
        mean_length = (edge_weights * edge_lengths).sum().div(weight_sum).clamp_min(eps)
        log_ratio = torch.log(
            edge_lengths.clamp_min(eps) / mean_length
        )

        # No penalty for edges inside the requested relative tolerance.
        excess = torch.relu(log_ratio.abs() - tolerance)

        mean_loss = (edge_weights * excess.square()).sum() / weight_sum

        # Smooth worst-edge approximation, instead of hard max.
        temperature = 12.0
        squared_excess = excess.square()
        smooth_worst = (
            torch.logsumexp(
                temperature * squared_excess,
                dim=0,
            )
            - math.log(max(int(squared_excess.numel()), 1))
        ) / temperature

        # Additional emphasis on outliers, weighted smoothly by severity.
        # Detaching only the adaptive weights preserves gradients through the
        # selected shared edge lengths while avoiding gradients through the
        # weighting decision itself.
        severity_weights = torch.softmax(
            torch.log(edge_weights.clamp_min(eps)) + outlier_weight * excess.detach(),
            dim=0,
        )
        outlier_loss = (
            severity_weights * squared_excess
        ).sum()

        return (
            mean_loss
            + worst_weight * smooth_worst
            + outlier_loss
        )

    def cell_edge_uniformity_loss(
        self,
        curve_geometry: SharedCurveGeometry,
        seed_active_weights: torch.Tensor | None = None,
    ):
        cfg = self.cfg

        curves = curve_geometry.curves_xyz

        pairs = curve_geometry.edge_seed_pair
        edge_type = curve_geometry.edge_type

        if pairs is None or curves.ndim != 3 or curves.shape[0] == 0:
            return curves.new_zeros(())

        pairs = pairs.to(device=curves.device)

        eps = float(getattr(cfg, "cell_edge_uniform_eps", 1e-8))
        angle_eps = float(getattr(cfg, "cell_angle_eps", 1e-8))
        merge_tol = float(
            getattr(cfg, "cell_vertex_merge_tolerance", 1e-5)
        )

        lam_angle = float(
            getattr(cfg, "lam_cell_angle_uniform", 0.0)
        )
        lam_radial = float(
            getattr(cfg, "lam_cell_radial_uniform", 0.0)
        )

        edge_len = curve_geometry.edge_lengths
        edge_activity = self.curve_edge_activity_weights(
            curve_geometry,
            seed_active_weights,
        )
        finite_curves = torch.isfinite(curves).all(dim=(1, 2))
        finite_lengths = curve_geometry.finite_edge_mask

        if bool(getattr(cfg, "include_shell_in_length_loss", False)):
            cell_length_types = CELL_GEOMETRY_EDGE_TYPES
        else:
            cell_length_types = CELL_LENGTH_EDGE_TYPES
        loss_edge_type_mask = build_edge_type_mask(
            edge_type,
            cell_length_types,
        )
        cell_geometry_type_mask = build_edge_type_mask(
            edge_type,
            CELL_GEOMETRY_EDGE_TYPES,
        )

        # Edges associated with at least one valid seed cell. Cell polygon
        # reconstruction uses all valid geometry edges (0, 1, 3, 4), while
        # the scalar length component uses CELL_LENGTH_EDGE_TYPES by default.
        base_cell_mask = (
            finite_curves
            & finite_lengths
            & (pairs >= 0).any(dim=1)
        )
        geometry_mask = base_cell_mask & cell_geometry_type_mask
        uniform_mask = base_cell_mask & loss_edge_type_mask

        if not bool(base_cell_mask.any().detach().item()):
            return curves.new_zeros(())

        def unique_vertices(points):
            """
            Deduplicate points for polygon reconstruction.

            Index selection is discrete, but selected coordinates retain their
            gradient connection to `curves`.
            """
            if points.ndim != 2 or points.shape[0] == 0:
                return None

            finite = torch.isfinite(points).all(dim=1)
            points = points[finite]

            if points.shape[0] < 3:
                return None

            keys = torch.round(
                points.detach() / merge_tol
            ).to(torch.int64)

            # Preserve the first occurrence of each rounded coordinate.
            seen = set()
            keep_indices = []

            for index, key in enumerate(keys.cpu().tolist()):
                key_tuple = tuple(key)
                if key_tuple not in seen:
                    seen.add(key_tuple)
                    keep_indices.append(index)

            if len(keep_indices) < 3:
                return None

            keep = torch.as_tensor(
                keep_indices,
                dtype=torch.long,
                device=points.device,
            )

            return points.index_select(0, keep)

        def order_vertices(vertices):
            """
            Order 3D polygon vertices around their best-fit local plane.
            """
            if vertices is None or vertices.shape[0] < 3:
                return None

            center = vertices.mean(dim=0)
            centered = vertices - center

            # Detached basis selection avoids differentiating through discrete/
            # unstable plane orientation while preserving gradients through the
            # projected coordinates.
            try:
                _, _, vh = torch.linalg.svd(
                    centered.detach(),
                    full_matrices=False,
                )
            except RuntimeError:
                return None

            if vh.shape[0] < 2:
                return None

            basis_u = vh[0].to(
                device=vertices.device,
                dtype=vertices.dtype,
            )
            basis_v = vh[1].to(
                device=vertices.device,
                dtype=vertices.dtype,
            )

            coord_u = centered @ basis_u
            coord_v = centered @ basis_v

            angles = torch.atan2(coord_v, coord_u)

            # Ordering is discrete; the reordered vertices remain differentiable.
            order = torch.argsort(angles.detach())

            return vertices.index_select(0, order)

        def safe_normalize(x):
            return x / torch.sqrt(
                (x * x).sum(dim=-1, keepdim=True) + angle_eps
            )

        def polygon_angle_loss(vertices, true_corner_mask=None):
            if vertices is None or vertices.shape[0] < 3:
                return None

            previous = torch.roll(vertices, shifts=1, dims=0)
            following = torch.roll(vertices, shifts=-1, dims=0)

            vector_a = previous - vertices
            vector_b = following - vertices

            norm_a = torch.sqrt((vector_a * vector_a).sum(dim=1) + angle_eps)
            norm_b = torch.sqrt((vector_b * vector_b).sum(dim=1) + angle_eps)

            valid = (
                torch.isfinite(norm_a)
                & torch.isfinite(norm_b)
                & (norm_a > angle_eps)
                & (norm_b > angle_eps)
            )
            if true_corner_mask is not None:
                true_corner_mask = true_corner_mask.to(
                    device=vertices.device,
                    dtype=torch.bool,
                )
                if true_corner_mask.shape[0] == valid.shape[0]:
                    valid = valid & true_corner_mask

            if int(valid.detach().sum().item()) < 3:
                return None

            cosine = (
                safe_normalize(vector_a[valid])
                * safe_normalize(vector_b[valid])
            ).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

            mean_cosine = cosine.mean()
            scale = mean_cosine.abs().clamp_min(angle_eps)

            return (
                cosine.var(unbiased=False)
                / scale.square()
            )

        def polygon_radial_loss(vertices, vertex_weights=None):
            if vertices is None or vertices.shape[0] < 3:
                return None

            if vertex_weights is not None:
                vertex_weights = vertex_weights.to(
                    device=vertices.device,
                    dtype=vertices.dtype,
                ).reshape(-1)
                if vertex_weights.shape[0] != vertices.shape[0]:
                    vertex_weights = None

            if vertex_weights is None:
                center = vertices.mean(dim=0)
            else:
                weight_sum = vertex_weights.sum().clamp_min(eps)
                center = (
                    vertex_weights.unsqueeze(-1) * vertices
                ).sum(dim=0) / weight_sum

            radial = torch.linalg.vector_norm(
                vertices - center,
                dim=1,
            )

            valid = (
                torch.isfinite(radial)
                & (radial > eps)
            )

            if int(valid.detach().sum().item()) < 3:
                return None

            radial = radial[valid]
            if vertex_weights is None:
                weights = torch.ones_like(radial)
            else:
                weights = vertex_weights[valid].clamp_min(0.0)
            weight_sum = weights.sum().clamp_min(eps)
            mean_radial = (
                weights * radial
            ).sum().div(weight_sum).clamp_min(eps)

            return (
                (
                    weights
                    * (radial - mean_radial).square()
                ).sum()
                / weight_sum
                / mean_radial.square()
            )

        def ordered_cell_boundary(cell_id: int):
            indices_by_cell = curve_geometry.cell_boundary_edge_indices
            directions_by_cell = curve_geometry.cell_boundary_edge_directions
            seed_ids = curve_geometry.cell_boundary_seed_ids
            num_shell_samples = 8
            if (
                indices_by_cell is None
                or directions_by_cell is None
                or seed_ids is None
            ):
                return None, None, None

            seed_ids = torch.as_tensor(
                seed_ids,
                dtype=torch.long,
                device=curves.device,
            ).reshape(-1)
            matches = torch.nonzero(seed_ids == cell_id, as_tuple=False).flatten()
            if matches.numel() == 0:
                return None, None, None

            boundary_id = int(matches[0].detach().item())
            edge_indices = indices_by_cell[boundary_id].to(
                device=curves.device,
                dtype=torch.long,
            )
            edge_directions = directions_by_cell[boundary_id].to(
                device=curves.device,
                dtype=torch.long,
            )
            if edge_indices.numel() == 0:
                return None, None, None

            parts = []
            corner_masks = []

            def fixed_shell_samples(edge_points):
                if edge_points.shape[0] == num_shell_samples:
                    return edge_points
                if edge_points.shape[0] < 2:
                    return edge_points
                t = torch.linspace(
                    0.0,
                    1.0,
                    num_shell_samples,
                    dtype=edge_points.dtype,
                    device=edge_points.device,
                )
                scaled = t * float(edge_points.shape[0] - 1)
                left = torch.floor(scaled).to(dtype=torch.long)
                right = torch.clamp(left + 1, max=edge_points.shape[0] - 1)
                alpha = (scaled - left.to(dtype=edge_points.dtype)).unsqueeze(-1)
                return (
                    (1.0 - alpha) * edge_points.index_select(0, left)
                    + alpha * edge_points.index_select(0, right)
                )

            for local_index in range(int(edge_indices.numel())):
                edge_index_value = int(edge_indices[local_index].detach().item())
                if edge_index_value < 0 or edge_index_value >= curves.shape[0]:
                    continue
                edge_points = curves[edge_index_value]
                if edge_points.shape[0] < 2:
                    continue
                if int(edge_directions[local_index].detach().item()) < 0:
                    edge_points = edge_points.flip(0)
                if int(edge_type[edge_index_value].detach().item()) == EDGE_DOMAIN_SHELL:
                    edge_points = fixed_shell_samples(edge_points)
                edge_points = edge_points[:-1]
                if edge_points.shape[0] == 0:
                    continue
                is_corner = torch.zeros(
                    edge_points.shape[0],
                    dtype=torch.bool,
                    device=curves.device,
                )
                is_corner[0] = True
                parts.append(edge_points)
                corner_masks.append(is_corner)

            if not parts:
                return None, None, None

            boundary_points = torch.cat(parts, dim=0)
            true_corners = torch.cat(corner_masks, dim=0)
            next_points = torch.roll(boundary_points, shifts=-1, dims=0)
            segment_lengths = torch.linalg.vector_norm(
                next_points - boundary_points,
                dim=-1,
            )
            vertex_weights = 0.5 * (
                segment_lengths
                + torch.roll(segment_lengths, shifts=1, dims=0)
            )

            return boundary_points, true_corners, vertex_weights

        cell_ids = torch.unique(pairs[base_cell_mask])
        cell_ids = cell_ids[cell_ids >= 0]

        cell_losses = []
        cell_loss_weights = []

        for cell_id_tensor in cell_ids:
            cell_id = int(cell_id_tensor.detach().item())

            belongs_geometry = (
                geometry_mask
                & (pairs == cell_id).any(dim=1)
            )

            belongs_uniform = (
                uniform_mask
                & (pairs == cell_id).any(dim=1)
            )

            cell_loss = curves.new_zeros(())
            has_component = False

            # Within-cell equality of selected edge lengths.
            lengths = edge_len[belongs_uniform]
            length_weights = edge_activity[belongs_uniform]

            if lengths.numel() > 1:
                length_weight_sum = length_weights.sum().clamp_min(eps)
                mean_length = (
                    length_weights * lengths
                ).sum().div(length_weight_sum).clamp_min(eps)

                edge_loss = (
                    (length_weights * (lengths - mean_length).square()).sum()
                    / length_weight_sum
                    / mean_length.square()
                )

                cell_loss = cell_loss + edge_loss
                has_component = True

            # Reconstruct the complete cell polygon using all valid cell edges,
            # not only the selected equal-length edge types. Prefer the
            # topology-provided order so shell samples stay on the cell
            # boundary without differentiable hard sorting.
            ordered_boundary_result = ordered_cell_boundary(cell_id)
            ordered_boundary, true_corner_mask, radial_weights = (
                ordered_boundary_result
                if ordered_boundary_result[0] is not None
                else (None, None, None)
            )

            if ordered_boundary is not None and ordered_boundary.shape[0] >= 3:
                if lam_angle != 0.0:
                    angle_loss = polygon_angle_loss(
                        ordered_boundary,
                        true_corner_mask,
                    )

                    if angle_loss is not None:
                        cell_loss = (
                            cell_loss
                            + lam_angle * angle_loss
                        )
                        has_component = True

                if lam_radial != 0.0:
                    radial_loss = polygon_radial_loss(
                        ordered_boundary,
                        radial_weights,
                    )

                    if radial_loss is not None:
                        cell_loss = (
                            cell_loss
                            + lam_radial * radial_loss
                        )
                        has_component = True

            else:
                cell_curves = curves[belongs_geometry]

                if cell_curves.shape[0] == 0:
                    continue

                endpoints = torch.cat(
                    (
                        cell_curves[:, 0, :],
                        cell_curves[:, -1, :],
                    ),
                    dim=0,
                )

                vertices = unique_vertices(endpoints)
                ordered_vertices = order_vertices(vertices)

                if lam_angle != 0.0:
                    angle_loss = polygon_angle_loss(
                        ordered_vertices
                    )

                    if angle_loss is not None:
                        cell_loss = (
                            cell_loss
                            + lam_angle * angle_loss
                        )
                        has_component = True

                if lam_radial != 0.0:
                    radial_loss = polygon_radial_loss(vertices)

                    if radial_loss is not None:
                        cell_loss = (
                            cell_loss
                            + lam_radial * radial_loss
                        )
                        has_component = True

            if has_component and torch.isfinite(cell_loss):
                cell_weight = edge_activity[
                    geometry_mask & (pairs == cell_id).any(dim=1)
                ].mean().clamp_min(eps)
                cell_losses.append(cell_weight * cell_loss)
                cell_loss_weights.append(cell_weight)

        if not cell_losses:
            return curves.new_zeros(())

        return torch.stack(cell_losses).sum() / torch.stack(cell_loss_weights).sum().clamp_min(eps)

    def cell_boundary_areas(
        self,
        curve_geometry: SharedCurveGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        curves = curve_geometry.curves_xyz
        edge_type = curve_geometry.edge_type
        indices_by_cell = curve_geometry.cell_boundary_edge_indices
        directions_by_cell = curve_geometry.cell_boundary_edge_directions
        seed_ids = curve_geometry.cell_boundary_seed_ids

        if (
            curves.ndim != 3
            or curves.shape[0] == 0
            or indices_by_cell is None
            or directions_by_cell is None
            or seed_ids is None
        ):
            return (
                curves.new_empty((0,)),
                torch.empty((0,), dtype=torch.long, device=curves.device),
            )

        seed_ids = torch.as_tensor(
            seed_ids,
            dtype=torch.long,
            device=curves.device,
        ).reshape(-1)
        num_shell_samples = 8

        def fixed_shell_samples(edge_points):
            if edge_points.shape[0] == num_shell_samples:
                return edge_points
            if edge_points.shape[0] < 2:
                return edge_points
            t = torch.linspace(
                0.0,
                1.0,
                num_shell_samples,
                dtype=edge_points.dtype,
                device=edge_points.device,
            )
            scaled = t * float(edge_points.shape[0] - 1)
            left = torch.floor(scaled).to(dtype=torch.long)
            right = torch.clamp(left + 1, max=edge_points.shape[0] - 1)
            alpha = (scaled - left.to(dtype=edge_points.dtype)).unsqueeze(-1)
            return (
                (1.0 - alpha) * edge_points.index_select(0, left)
                + alpha * edge_points.index_select(0, right)
            )

        def boundary_points_for_index(boundary_id: int):
            edge_indices = indices_by_cell[boundary_id].to(
                device=curves.device,
                dtype=torch.long,
            )
            edge_directions = directions_by_cell[boundary_id].to(
                device=curves.device,
                dtype=torch.long,
            )
            parts = []
            for local_index in range(int(edge_indices.numel())):
                edge_index_value = int(edge_indices[local_index].detach().item())
                if edge_index_value < 0 or edge_index_value >= curves.shape[0]:
                    continue
                edge_points = curves[edge_index_value]
                if edge_points.shape[0] < 2 or not bool(torch.isfinite(edge_points).all().detach().item()):
                    continue
                if int(edge_directions[local_index].detach().item()) < 0:
                    edge_points = edge_points.flip(0)
                if int(edge_type[edge_index_value].detach().item()) == EDGE_DOMAIN_SHELL:
                    edge_points = fixed_shell_samples(edge_points)
                edge_points = edge_points[:-1]
                if edge_points.shape[0] > 0:
                    parts.append(edge_points)
            if not parts:
                return None
            boundary_points = torch.cat(parts, dim=0)
            if boundary_points.shape[0] < 3:
                return None
            return boundary_points

        def polygon_area_3d(points: torch.Tensor) -> torch.Tensor:
            next_points = torch.roll(points, shifts=-1, dims=0)
            area_vector = torch.cross(points, next_points, dim=1).sum(dim=0) * 0.5
            return torch.sqrt((area_vector * area_vector).sum() + float(self.cfg.eps))

        areas = []
        area_seed_ids = []
        for boundary_id in range(int(seed_ids.numel())):
            points = boundary_points_for_index(boundary_id)
            if points is None:
                continue
            area = polygon_area_3d(points)
            if torch.isfinite(area):
                areas.append(area)
                area_seed_ids.append(seed_ids[boundary_id])

        if not areas:
            return (
                curves.new_empty((0,)),
                torch.empty((0,), dtype=torch.long, device=curves.device),
            )

        return torch.stack(areas), torch.stack(area_seed_ids).to(dtype=torch.long)

    def neutral_density_fiber_fields(self, uv: torch.Tensor, Xu: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        rho = torch.zeros((uv.shape[0],), dtype=uv.dtype, device=uv.device)
        if Xu is not None:
            fiber = Xu.to(device=uv.device, dtype=uv.dtype)
            if fiber.ndim != 2 or fiber.shape != (uv.shape[0], 3):
                fiber = None
        else:
            fiber = None
        if fiber is None:
            fiber = uv.new_tensor([1.0, 0.0, 0.0]).expand(uv.shape[0], 3)
        fiber = fiber / torch.linalg.norm(fiber, dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        stats = {
            "filter_delta_mean": 0.0,
            "filter_delta_max": 0.0,
            "projection_delta_mean": 0.0,
            "projection_delta_max": 0.0,
            "raw_mean": 0.0,
            "filtered_mean": 0.0,
            "projected_mean": 0.0,
            "final_mean": 0.0,
        }
        return rho, fiber, stats

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------

    def _init_tensorboard(self):
        if not self.cfg.tensorboard_enabled:
            return

        exp_name = self.cfg.experiment_name
        if exp_name is None or str(exp_name).strip() == "":
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self.cfg.tensorboard_log_root, exp_name)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.cfg.tb_flush_secs,
        )
        self.tensorboard_log_dir = log_dir

        cfg_lines = [f"{k}: {v}" for k, v in vars(self.cfg).items()]
        self.writer.add_text("config", "\n".join(cfg_lines), global_step=0)

        print(f"TensorBoard log dir: {self.tensorboard_log_dir}")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _true_open_boundary_idx(self, ft, tol=None):
        if ("boundary_idx_ring1" not in ft) or ft["boundary_idx_ring1"] is None:
            return torch.empty(0, dtype=torch.long, device=ft["uv"].device)

        bidx = torch.unique(ft["boundary_idx_ring1"].to(dtype=torch.long))
        if bidx.numel() == 0:
            return bidx

        uv = ft["uv"]
        u = uv[:, 0]
        v = uv[:, 1]

        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        if tol is None:
            u_span = (u.max() - u.min()).abs()
            v_span = (v.max() - v.min()).abs()
            base_span = torch.maximum(
                u_span,
                v_span,
            ).clamp_min(torch.as_tensor(1.0, device=uv.device, dtype=uv.dtype))
            tol = 1e-4 * float(base_span.detach().item())

        ub = u[bidx]
        vb = v[bidx]
        keep = torch.ones_like(bidx, dtype=torch.bool)

        if u_periodic:
            umin = u.min()
            umax = u.max()
            is_u_seam = (ub - umin).abs() <= tol
            is_u_seam = is_u_seam | ((ub - umax).abs() <= tol)
            keep = keep & (~is_u_seam)

        if v_periodic:
            vmin = v.min()
            vmax = v.max()
            is_v_seam = (vb - vmin).abs() <= tol
            is_v_seam = is_v_seam | ((vb - vmax).abs() <= tol)
            keep = keep & (~is_v_seam)

        return bidx[keep]

    def _ordered_true_open_boundary(self, ft):
        bidx = self._true_open_boundary_idx(ft)
        if bidx.numel() == 0 or ft.get("faces_ijk", None) is None:
            return bidx, None

        device = bidx.device
        boundary_set = set(int(i) for i in bidx.detach().cpu().tolist())
        if len(boundary_set) < 2:
            return bidx, None

        faces = ft["faces_ijk"].detach().cpu().to(torch.long)
        edge_count = {}
        for a, b, c in faces.tolist():
            for i, j in ((a, b), (b, c), (c, a)):
                key = (i, j) if i < j else (j, i)
                edge_count[key] = edge_count.get(key, 0) + 1

        adj = {i: [] for i in boundary_set}
        for (i, j), count in edge_count.items():
            if count == 1 and i in boundary_set and j in boundary_set:
                adj[i].append(j)
                adj[j].append(i)

        if not any(adj.values()):
            return bidx, None

        ordered = []
        loop_ids = []
        visited_edges = set()

        def edge_key(i, j):
            return (i, j) if i < j else (j, i)

        starts = [i for i, nbrs in adj.items() if len(nbrs) == 1]
        starts.extend(i for i in adj.keys() if i not in starts)

        loop_id = 0
        for start in starts:
            has_unused = any(edge_key(start, nb) not in visited_edges for nb in adj[start])
            if not has_unused:
                continue

            chain = [start]
            prev = None
            cur = start
            while True:
                next_nodes = [
                    nb for nb in adj[cur]
                    if nb != prev and edge_key(cur, nb) not in visited_edges
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                visited_edges.add(edge_key(cur, nxt))
                if nxt == start:
                    break
                chain.append(nxt)
                prev, cur = cur, nxt

            if len(chain) >= 2:
                ordered.extend(chain)
                loop_ids.extend([loop_id] * len(chain))
                loop_id += 1

        if not ordered:
            return bidx, None

        ordered_idx = torch.tensor(ordered, dtype=torch.long, device=device)
        loop_id_t = torch.tensor(loop_ids, dtype=torch.long, device=device)
        return ordered_idx, loop_id_t

    @staticmethod
    def _to_float_if_finite(x):
        if isinstance(x, torch.Tensor):
            x = x.reshape(())
            if torch.isfinite(x).item():
                return float(x.detach().item())
            return None
        try:
            x = float(x)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def _tb_add_scalar(self, tag: str, value, step: int):
        if self.writer is None:
            return
        v = self._to_float_if_finite(value)
        if v is not None:
            self.writer.add_scalar(tag, v, step)

    def _tb_add_histogram(self, tag: str, value: torch.Tensor, step: int):
        if self.writer is None or value is None:
            return
        try:
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                finite_mask = torch.isfinite(value)
                if finite_mask.any():
                    self.writer.add_histogram(tag, value[finite_mask].detach().cpu(), step)
        except Exception:
            pass
    def _tb_log_step(
        self,
        step: int,
        row: dict,
        rho: torch.Tensor,
        fiber_surface: torch.Tensor,
        seeds_list: list[torch.Tensor],
        pred_list: list[dict],
    ):
        if self.writer is None:
            return

        self._tb_add_scalar("Stage/Index", row.get("stage", 0.0), step)
        self._tb_add_scalar("Stage/FreezeSeeds", row.get("stage_freeze_seeds", 0.0), step)
        self._tb_add_scalar("Stage/AllowSeedOutsideDomainConfigured", row.get("stage_allow_seed_outside_domain", 0.0), step)
        self._tb_add_scalar("Stage/AllowSeedOutsideDomainEffective", row.get("allow_seed_outside_domain", 0.0), step)
        self._tb_add_scalar("StageLambda/FEM", row.get("lam_fem_eff", 0.0), step)
        self._tb_add_scalar("StageLambda/FEMBase", row.get("lam_fem_base", 0.0), step)
        self._tb_add_scalar("StageLambda/FEMAdaptiveMultiplier", row.get("adaptive_lambda_fem", 0.0), step)
        self._tb_add_scalar("StageLambda/CVT", row.get("lam_cvt_eff", 0.0), step)
        self._tb_add_scalar("StageLambda/Repulsion", row.get("lam_rep_eff", 0.0), step)
        self._tb_add_scalar("StageLambda/L_seed", row.get("lam_l_seed_eff", 0.0), step)
        self._tb_add_scalar("StageLambda/Total_Fiber_Length", row.get("lam_total_fiber_length_eff", 0.0), step)
        self._tb_add_scalar("StageLambda/L_curve_cell", row.get("lam_l_curve_cell_eff", 0.0), step)

        self._tb_add_scalar("Loss/Total", row["L_total"], step)
        self._tb_add_scalar("Loss/TrainObjective", row.get("L_train", row["L_total"]), step)
        self._tb_add_scalar("Loss/DesignScore", row.get("design_score", 0.0), step)
        self._tb_add_scalar("Loss/Repulsion", row["loss_rep"], step)
        self._tb_add_scalar("Loss/CVT", row["loss_cvt"], step)
        self._tb_add_scalar("Loss/L_seed", row["loss_l_seed"], step)
        self._tb_add_scalar("Loss/Total_Fiber_Length", row["loss_total_fiber_length"], step)
        self._tb_add_scalar("Loss/L_curve_cell", row["loss_l_curve_cell"], step)
        self._tb_add_scalar("Geometry/CellAreaMin", row.get("cell_area_min", 0.0), step)
        self._tb_add_scalar("Geometry/CellAreaMean", row.get("cell_area_mean", 0.0), step)
        self._tb_add_scalar("Loss/FEM", row["loss_fem"], step)
        self._tb_add_scalar("Loss/FEMTotalTrainingLoss", row.get("fem_total_loss", row["loss_fem"]), step)
        self._tb_add_scalar("Loss/FEMStressConstraint", row.get("loss_fem_stress_constraint", 0.0), step)
        self._tb_add_scalar("Loss/FEMDisplacementConstraint", row.get("loss_fem_displacement_constraint", 0.0), step)
        self._tb_add_scalar("Loss/FEMBaseline", row.get("baseline_fem_loss", 0.0), step)
        self._tb_add_scalar("Loss/FEMViolation", row.get("violation_fem_loss", 0.0), step)
        self._tb_add_scalar("LossNormalized/CVT", row.get("loss_cvt_norm", 0.0), step)
        self._tb_add_scalar("LossNormalized/Repulsion", row.get("loss_rep_norm", 0.0), step)
        self._tb_add_scalar("LossNormalized/L_seed", row.get("loss_l_seed_norm", 0.0), step)
        self._tb_add_scalar("LossNormalized/Total_Fiber_Length", row.get("loss_total_fiber_length_norm", 0.0), step)
        self._tb_add_scalar("LossNormalized/L_curve_cell", row.get("loss_l_curve_cell_norm", 0.0), step)
        self._tb_add_scalar("LossReference/CVT", row.get("loss_cvt_reference", 1.0), step)
        self._tb_add_scalar("LossReference/Repulsion", row.get("loss_rep_reference", 1.0), step)
        self._tb_add_scalar("LossReference/L_seed", row.get("loss_l_seed_reference", 1.0), step)
        self._tb_add_scalar("LossReference/Total_Fiber_Length", row.get("loss_total_fiber_length_reference", 1.0), step)
        self._tb_add_scalar("LossReference/L_curve_cell", row.get("loss_l_curve_cell_reference", 1.0), step)
        self._tb_add_scalar("LossReference/FEM", row.get("loss_fem_reference", 1.0), step)
        self._tb_add_scalar("Physics/FEMStressMax", row.get("fem_stress_max", 0.0), step)
        self._tb_add_scalar("Physics/FEMDisplacementMax", row.get("fem_displacement_max", 0.0), step)
        self._tb_add_scalar("Physics/FEMStressConstraintExcess", row.get("fem_stress_constraint_excess", 0.0), step)
        self._tb_add_scalar("Physics/FEMDisplacementConstraintExcess", row.get("fem_displacement_constraint_excess", 0.0), step)
        self._tb_add_scalar("Physics/FEMStressRatio", row.get("fem_stress_ratio", 0.0), step)
        self._tb_add_scalar("Physics/FEMDisplacementRatio", row.get("fem_displacement_ratio", 0.0), step)
        self._tb_add_scalar("Physics/FEMPhysicalStressRatio", row.get("physical_stress_ratio", 0.0), step)
        self._tb_add_scalar("Physics/FEMPhysicalDisplacementRatio", row.get("physical_displacement_ratio", 0.0), step)
        self._tb_add_scalar("Physics/FEMConstraintViolation", row.get("fem_constraint_violation", 0.0), step)
        self._tb_add_scalar("Physics/FEMRhoMinRatio", row.get("fem_rho_min_ratio", 0.0), step)
        self._tb_add_scalar("Physics/VolFrac", row["VolFrac"], step)
        self._tb_add_scalar("Geometry/StrutThickness", row["strut_thickness"], step)

        self._tb_add_scalar("Density/Min", row["rho_min"], step)
        self._tb_add_scalar("Density/Mean", row["rho_mean"], step)
        self._tb_add_scalar("Density/Max", row["rho_max"], step)

        self._tb_add_scalar("Train/DeltaRho", row["drho"], step)
        self._tb_add_scalar("Train/DeltaSeed", row["dseed"], step)
        self._tb_add_scalar("Train/GradMean", row["grad_mean"], step)
        self._tb_add_scalar("Train/BestScore", row["best_score"], step)
        self._tb_add_scalar("Train/BestStep", row["best_step"], step)
        self._tb_add_scalar("Checkpoint/BestFeasibleDesignScore", row.get("best_feasible_design_score", float("nan")), step)
        self._tb_add_scalar("Checkpoint/BestFeasibleStep", row.get("best_feasible_step", -1), step)
        self._tb_add_scalar("Checkpoint/BestInfeasibleViolation", row.get("best_infeasible_violation", float("nan")), step)
        self._tb_add_scalar("Checkpoint/BestInfeasibleStep", row.get("best_infeasible_step", -1), step)
        self._tb_add_scalar("Stage/MonitorRaw", row.get("stage_monitor_raw", float("nan")), step)
        self._tb_add_scalar("Stage/DesignPatienceActive", 1.0 if row.get("design_patience_active", False) else 0.0, step)
        self._tb_add_scalar("Train/FEMValid", 1.0 if row["fem_valid"] else 0.0, step)
        self._tb_add_scalar(
            "Train/OptimizerStepSkipped",
            1.0 if row["optimizer_step_skipped"] else 0.0,
            step,
        )
        self._tb_add_scalar("Activity/SoftActiveMass", row["soft_active_mass"], step)
        self._tb_add_scalar("Activity/HardActiveCount", row["hard_active_count"], step)
        self._tb_add_scalar("Activity/MinActiveSeedDistance", row["min_active_seed_dist"], step)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)
        if fiber_norm.numel() > 0:
            self._tb_add_scalar("Fiber/NormMean", fiber_norm.mean(), step)
            self._tb_add_scalar("Fiber/NormMin", fiber_norm.min(), step)
            self._tb_add_scalar("Fiber/NormMax", fiber_norm.max(), step)

        stage_local_step = int(row.get("stage_local_step", 0))
        stage_max_steps = int(row.get("stage_max_steps", 0))
        if (
            step % self.cfg.tb_log_histograms_every == 0
            or (stage_max_steps > 0 and stage_local_step >= stage_max_steps - 1)
        ):
            self._tb_add_histogram("Density/Rho", rho, step)
            self._tb_add_histogram("Fiber/Norm", fiber_norm, step)

            if len(seeds_list) > 0:
                all_seeds = torch.cat(seeds_list, dim=0)
                self._tb_add_histogram("Seeds/All", all_seeds, step)
                if all_seeds.shape[1] >= 1:
                    self._tb_add_histogram("Seeds/U", all_seeds[:, 0], step)
                if all_seeds.shape[1] >= 2:
                    self._tb_add_histogram("Seeds/V", all_seeds[:, 1], step)

            centerline_radius_vals = []

            for p in pred_list:
                if "centerline_radius" in p and p["centerline_radius"] is not None:
                    centerline_radius_vals.append(p["centerline_radius"].reshape(-1))

            if centerline_radius_vals: self._tb_add_histogram("Geometry/CenterlineRadiusHist", torch.cat(centerline_radius_vals, dim=0), step)

        if self.last_fem_debug:
            dbg = self.last_fem_debug
            for key in [
                "density_raw_min",
                "density_raw_mean",
                "density_raw_max",
                "density_min",
                "density_mean",
                "density_max",
                "fiber_norm_min",
                "fiber_norm_mean",
                "fiber_norm_max",
                "void_fraction_lt_1e_2_raw",
                "void_fraction_lt_5e_2_raw",
                "void_fraction_lt_floor_raw",
            ]:
                if key in dbg:
                    self._tb_add_scalar(f"FEMDebug/{key}", dbg[key], step)

            if "fem_valid" in dbg:
                self._tb_add_scalar("FEMDebug/Valid", 1.0 if dbg["fem_valid"] else 0.0, step)

            if dbg.get("failure_reason"):
                self.writer.add_text("FEMDebug/FailureReason", str(dbg["failure_reason"]), step)

    # ------------------------------------------------------------------
    # Losses / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ramp_weight(step: int, total_steps: int, start_frac: float, ramp_frac: float) -> float:
        if total_steps <= 0:
            return 0.0
        start_step = max(int(start_frac * total_steps), 0)
        ramp_steps = max(int(ramp_frac * total_steps), 1)
        if step <= start_step:
            return 0.0
        if step >= start_step + ramp_steps:
            return 1.0
        return float(step - start_step) / float(ramp_steps)

    def seed_offset_scale_for_step(self, step: int, stage_max_steps: int) -> float:
        cfg = self.cfg
        start = cfg.Offset_scale if cfg.seed_offset_scale_start is None else cfg.seed_offset_scale_start
        final = start if cfg.seed_offset_scale_final is None else cfg.seed_offset_scale_final
        if int(stage_max_steps) <= 0:
            return float(final)

        t = min(
            max(
                float(step)
                / max(float(cfg.seed_offset_scale_ramp_frac) * float(stage_max_steps), 1.0),
                0.0,
            ),
            1.0,
        )
        # Smooth decay: exploration changes gently instead of snapping at a milestone.
        t = t * t * (3.0 - 2.0 * t)
        return float((1.0 - t) * float(start) + t * float(final))

    def allow_seed_outside_domain_for_step(
        self,
        step: int,
        stage_max_steps: int,
        *,
        stage_allow_seed_outside_domain: bool | None = None,
    ) -> bool:
        cfg = self.cfg
        stage_enabled = (
            bool(cfg.allow_seed_outside_domain)
            if stage_allow_seed_outside_domain is None
            else bool(stage_allow_seed_outside_domain)
        )
        if not stage_enabled:
            return False
        warmup_step = int(
            round(float(cfg.allow_seed_outside_domain_warmup_frac) * float(stage_max_steps))
        )
        return int(step) >= warmup_step

    @staticmethod
    def min_pairwise_active_seed_distance(
        seed_xyz_list: list[torch.Tensor],
        seed_active_mask_list: list[torch.Tensor],
    ) -> float:
        """
        Return the minimum 3D distance between distinct hard-active seeds.

        Inactive seeds are excluded completely. The metric is diagnostic and
        non-differentiable; activity masks and the returned value are detached
        from autograd.
        """
        if len(seed_xyz_list) != len(seed_active_mask_list):
            raise ValueError(
                "seed_xyz_list and seed_active_mask_list must have the same length."
            )

        min_active_seed_dist = float("inf")
        valid_face_found = False

        for seeds_xyz, active_mask in zip(seed_xyz_list, seed_active_mask_list):
            if not isinstance(seeds_xyz, torch.Tensor):
                raise TypeError("Each seeds_xyz entry must be a torch.Tensor.")
            if not isinstance(active_mask, torch.Tensor):
                raise TypeError("Each active_mask entry must be a torch.Tensor.")

            seeds_xyz = seeds_xyz.reshape(-1, 3)
            active_mask = active_mask.detach().to(
                device=seeds_xyz.device,
                dtype=torch.bool,
            ).reshape(-1)

            if active_mask.numel() != seeds_xyz.shape[0]:
                raise ValueError(
                    "The hard active-seed mask length must match the "
                    "number of seed XYZ positions."
                )

            active_seeds_xyz = seeds_xyz[active_mask]
            if active_seeds_xyz.shape[0] < 2:
                continue

            distances = torch.cdist(active_seeds_xyz, active_seeds_xyz)
            diagonal = torch.eye(
                active_seeds_xyz.shape[0],
                dtype=torch.bool,
                device=active_seeds_xyz.device,
            )
            distances = distances.masked_fill(
                diagonal,
                float("inf"),
            )
            face_min_distance = distances.min()

            if torch.isfinite(face_min_distance):
                min_active_seed_dist = min(
                    min_active_seed_dist,
                    float(face_min_distance.detach().item()),
                )
                valid_face_found = True

        if not valid_face_found:
            return 0.0

        return min_active_seed_dist

    @staticmethod
    def project_seed_spacing(
        seeds_list: list[torch.Tensor],
        min_dist: float,
        iters: int = 8,
        eps_uv: float = 1e-4,
        detach: bool = True,
        clamp_to_domain: bool = True,
    ) -> list[torch.Tensor]:
        repaired = [(s.detach() if detach else s).clone() for s in seeds_list]
        if min_dist <= 0.0 or iters <= 0:
            return repaired

        for _ in range(int(iters)):
            for seeds in repaired:
                s = int(seeds.shape[0])
                if s < 2:
                    continue
                for i in range(s - 1):
                    for j in range(i + 1, s):
                        diff = seeds[i] - seeds[j]
                        dist = torch.linalg.norm(diff)
                        shortfall = float(min_dist) - float(dist.detach().item())
                        if shortfall <= 0.0:
                            continue
                        if float(dist.detach().item()) > 1e-8:
                            direction = diff / dist.clamp_min(1e-8)
                        else:
                            # Deterministic fallback direction for exact overlaps.
                            angle = torch.as_tensor(
                                2.399963229728653 * float(i + 1) + 1.61803398875 * float(j + 1),
                                dtype=seeds.dtype,
                                device=seeds.device,
                            )
                            direction = torch.stack((torch.cos(angle), torch.sin(angle)))
                        step = 0.5 * shortfall * direction
                        seeds[i] = seeds[i] + step
                        seeds[j] = seeds[j] - step
                if clamp_to_domain:
                    seeds.clamp_(float(eps_uv), 1.0 - float(eps_uv))
        return repaired

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total = int(round(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d} h {minutes:02d} min {secs:02d} sec"
        return f"{minutes:d} min {secs:02d} sec"

    @staticmethod
    def _volume_metric_definitions() -> dict[str, str]:
        return {
            "VolFrac": "Reported material volume fraction. The final objective controls material by fixed width plus network-length minimization, not by a target-volume penalty.",
        }

    def _curve_length_summary(
        self,
        curve_length_values: list[torch.Tensor],
        reference: torch.Tensor,
    ) -> dict[str, float]:
        nonempty_curve_lengths = [
            v.reshape(-1) for v in curve_length_values if v.numel() > 0
        ]
        if nonempty_curve_lengths:
            curve_lengths = torch.cat(nonempty_curve_lengths, dim=0)
        else:
            curve_lengths = reference.new_empty((0,))

        if curve_lengths.numel() == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "cv": float("nan"),
                "ratio": float("nan"),
            }

        curve_length_min = float(curve_lengths.min().item())
        curve_length_max = float(curve_lengths.max().item())
        curve_length_mean = float(curve_lengths.mean().item())
        curve_length_std = float(curve_lengths.std(unbiased=False).item())
        curve_length_cv = (
            curve_length_std / curve_length_mean
            if math.isfinite(curve_length_mean) and abs(curve_length_mean) > float(self.cfg.eps)
            else float("nan")
        )
        curve_length_ratio = curve_length_max / max(curve_length_min, float(self.cfg.eps))

        return {
            "min": curve_length_min,
            "max": curve_length_max,
            "mean": curve_length_mean,
            "std": curve_length_std,
            "cv": curve_length_cv,
            "ratio": curve_length_ratio,
        }

    @staticmethod
    def _sanitize_history_rows(history: list[dict]) -> list[dict]:
        obsolete_keys = {
            "VF_total",
            "VF_eff_total",
            "VF_int",
            "VF_eff_int",
            "vol_frac",
            "vol_frac_internal",
            "vol_frac_eff_total",
            "vol_frac_eff",
            "tau",
            "h_mean",
        }
        return [
            {key: value for key, value in row.items() if key not in obsolete_keys}
            for row in list(history or [])
        ]

    def _save_optimization_logs(
        self,
        output_folder: str | None,
        history: list[dict],
        best_row: dict | None,
        best_score: float,
        best_step: int,
        computation_time_sec: float,
        returned_best_source: str,
    ) -> str | None:
        if not output_folder:
            return None

        log_dir = os.path.join(os.path.normpath(str(output_folder)), "OptimizationLogs")
        os.makedirs(log_dir, exist_ok=True)

        config_path = os.path.join(log_dir, "training_parameters.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("Training Parameters\n")
            f.write("===================\n")
            for key, value in sorted(asdict(self.cfg).items()):
                if key == "min_active_seeds":
                    f.write(f"min_active_units: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        definitions_path = os.path.join(log_dir, "volume_metric_definitions.txt")
        with open(definitions_path, "w", encoding="utf-8") as f:
            f.write("Volume Metric Definitions\n")
            f.write("=========================\n")
            for key, description in self._volume_metric_definitions().items():
                f.write(f"{key}: {description}\n")

        clean_history = self._sanitize_history_rows(history)
        clean_best_row = self._sanitize_history_rows([best_row or {}])[0]
        solution_metric_keys = [
            "minimum_length",
            "maximum_length",
            "mean_length",
            "standard_deviation",
            "coefficient_of_variation",
            "maximum_minimum_ratio",
            "minimum_active_seed_distance",
            "number_of_edges",
            "number_of_selected_edges",
            "number_of_total_edges",
            "topology_identifier",
        ]
        best_solution_metrics = {
            key: clean_best_row.get(key)
            for key in solution_metric_keys
            if key in clean_best_row
        }
        summary = {
            "best_score": best_score,
            "best_design_score": float(clean_best_row.get("design_score", float("nan"))),
            "best_raw_fiber_length": float(clean_best_row.get("loss_total_fiber_length", float("nan"))),
            "best_physical_stress_ratio": float(clean_best_row.get("physical_stress_ratio", float("nan"))),
            "best_physical_displacement_ratio": float(clean_best_row.get("physical_displacement_ratio", float("nan"))),
            "best_hard_active_count": float(clean_best_row.get("hard_active_count", float("nan"))),
            "best_step": best_step,
            "returned_best_source": returned_best_source,
            "computation_time": self._format_elapsed_time(computation_time_sec),
            "computation_time_seconds": computation_time_sec,
            "volume_metrics": self._volume_metric_definitions(),
            "best_solution_metrics": best_solution_metrics,
            "best_row": clean_best_row,
        }
        summary_path = os.path.join(log_dir, "optimization_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        history_path = os.path.join(log_dir, "optimization_history.csv")
        if clean_history:
            fieldnames = []
            for row in clean_history:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with open(history_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(clean_history)
        else:
            with open(history_path, "w", encoding="utf-8", newline="") as f:
                f.write("")

        return log_dir

    def _timelapse_geometry_summary(self, face_tensors) -> str:


        surface_pts = int(sum(int(ft["points_xyz"].shape[0]) for ft in face_tensors))

        if self.shell_problem is not None and getattr(self.shell_problem, "brep_bbox", None) is not None:
            bbox = self.shell_problem.brep_bbox
            bbox_dims = (
                float(bbox["xmax"] - bbox["xmin"]),
                float(bbox["ymax"] - bbox["ymin"]),
                float(bbox["zmax"] - bbox["zmin"]),
            )
        else:
            xyz_all = torch.cat([ft["points_xyz"].detach() for ft in face_tensors], dim=0)
            bbox_t = xyz_all.amax(dim=0) - xyz_all.amin(dim=0)
            bbox_dims = tuple(float(v) for v in bbox_t.detach().cpu().tolist())

        load_value = (
            float(getattr(self.shell_problem, "Load_magnitude", 0.0))
            if self.shell_problem is not None
            else 0.0
        )

        bbox_text = " x ".join(f"{dim:.4g}" for dim in bbox_dims)
        return (
            f"BBox: {bbox_text}, "
            f"SurfacePts={surface_pts} "
        )

    def _timelapse_optimized_parameter_summary(self) -> str:
        cfg = self.cfg
        params = [
            f"seed positions ({int(cfg.seed_number)})",
            f"strut thickness={float(cfg.strut_thickness):.6g}",
        ]
        return "Optimized: " + ", ".join(params)

    @staticmethod
    def _clone_detached_tree(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: NN_Trainer._clone_detached_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [NN_Trainer._clone_detached_tree(v) for v in value]
        if isinstance(value, tuple):
            return tuple(NN_Trainer._clone_detached_tree(v) for v in value)
        return value

    @staticmethod
    def _clone_pred_list(pred_list: list[dict]) -> list[dict]:
        def _clone_value(value):
            return NN_Trainer._clone_detached_tree(value)

        return [
            {
                "face_id": p["face_id"],
                "seeds_raw": p["seeds_raw"].detach().clone(),
                "h_raw": _clone_value(p.get("h_raw")),
                "tau": _clone_value(p.get("tau")),
                "h": _clone_value(p.get("h")),
                "strut_thickness": p.get("strut_thickness"),
                "centerline_radius": _clone_value(p.get("centerline_radius")),
                "seeds_uv": _clone_value(p.get("seeds_uv")),
                "seed_active_mask": _clone_value(p.get("seed_active_mask")),
                "active_seed_ids": _clone_value(p.get("active_seed_ids")),
                "seed_activity_weight": _clone_value(p.get("seed_activity_weight")),
                "seed_box_activity_weight": _clone_value(p.get("seed_box_activity_weight")),
                "seed_domain_activity_weight": _clone_value(p.get("seed_domain_activity_weight")),
                "seed_duplicate_activity_weight": _clone_value(p.get("seed_duplicate_activity_weight")),
                "topology_seeds_uv": _clone_value(p.get("topology_seeds_uv")),
                "seeds_xyz": _clone_value(p.get("seeds_xyz")),
                "edge_curves_uv": _clone_value(p.get("edge_curves_uv")),
                "edge_curves_xyz": _clone_value(p.get("edge_curves_xyz")),
                "edge_index": _clone_value(p.get("edge_index")),
                "edge_seed_pair": _clone_value(p.get("edge_seed_pair")),
                "edge_type": _clone_value(p.get("edge_type")),
                "graph": _clone_value(p.get("graph")),
                "number_of_edges": p.get("number_of_edges"),
                "topology_identifier": p.get("topology_identifier"),
            }
            for p in pred_list
        ]

    @staticmethod
    def _copy_activation_metadata_to_pred(pred: dict, decoder_out: dict) -> None:
        for key in (
            "seeds_uv",
            "seed_active_mask",
            "active_seed_ids",
            "seed_activity_weight",
            "seed_box_activity_weight",
            "seed_domain_activity_weight",
            "seed_duplicate_activity_weight",
            "topology_seeds_uv",
        ):
            value = decoder_out.get(key, None)
            if isinstance(value, torch.Tensor):
                pred[key] = value.detach().clone()

    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor | float | int) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).reshape(()).detach().item())
        return math.isfinite(float(x))

    def _scheduled_fem_rho_min_ratio(self, step: int, stage_max_steps: int) -> float:
        cfg = self.cfg
        default = float(getattr(cfg, "fem_rho_min_ratio", 1.0e-5))
        start = float(getattr(cfg, "fem_rho_min_start", default))
        end = float(getattr(cfg, "fem_rho_min_end", default))
        if start <= 0.0 or end <= 0.0:
            return default
        denom = max(int(stage_max_steps) - 1, 1)
        progress = min(max(float(step) / float(denom), 0.0), 1.0)
        log_rho_min = (1.0 - progress) * math.log(start) + progress * math.log(end)
        return float(math.exp(log_rho_min))

    @staticmethod
    def _require_decoder_keys(decoder_out: dict, required_keys: list[str]):
        missing = [k for k in required_keys if k not in decoder_out]
        if missing:
            raise ValueError(
                f"Decoder output missing required keys: {missing}. "
                f"Available keys: {list(decoder_out.keys())}"
            )

    def _record_invalid_fem_debug(
        self,
        debug: dict,
        reason: str,
        save_debug_history: bool,
    ):
        debug = dict(debug)
        debug["fem_valid"] = False
        debug["failure_reason"] = reason
        self.last_fem_debug = debug
        if save_debug_history:
            self.fem_debug_history.append(debug.copy())

    # ------------------------------------------------------------------
    # Model / optimizer builders
        # ------------------------------------------------------------------
    def _build_single_face_models(
        self,
        device,
        seed_number,
        u_periodic,
        v_periodic,
        boundary_solid_idx=None,
        face_tensor=None,
    ):
        decoder = self.decoder_cls(
            **self._decoder_init_kwargs(
                device=device,
                seed_number=seed_number,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
                boundary_solid_idx=boundary_solid_idx,
                face_tensor=face_tensor,
            )
        ).to(device)

        ppnet = self.ppnet_cls(
            n_seeds=seed_number,
            allow_seed_outside_domain=(
                bool(getattr(self.cfg, "stage1_allow_seed_outside_domain", self.cfg.allow_seed_outside_domain))
                and float(self.cfg.allow_seed_outside_domain_warmup_frac) <= 0.0
            ),
            seed_domain_margin=self.cfg.seed_domain_margin,
            use_independent_seed_offsets=self.cfg.use_independent_seed_offsets,
            independent_seed_offset_max=self.cfg.independent_seed_offset_max,
        ).to(device)

        return decoder, ppnet

    def _decoder_face_mesh_for_face(self, face_tensor):
        if self.face_mesh is None:
            return face_tensor
        if isinstance(self.face_mesh, dict) and "face_tensors" in self.face_mesh:
            tensors = self.face_mesh["face_tensors"]
            if isinstance(tensors, (list, tuple)):
                return tensors[int(getattr(self.cfg, "training_face_index", 0))]
            return tensors
        if isinstance(self.face_mesh, (list, tuple)):
            return self.face_mesh[int(getattr(self.cfg, "training_face_index", 0))]
        return self.face_mesh

    def _decoder_init_kwargs(self, device, seed_number, u_periodic, v_periodic, boundary_solid_idx=None, face_tensor=None):
        return {
            "Cad_domain": self.Cad_domain,
            "face_mesh": self._decoder_face_mesh_for_face(face_tensor),
            "return_xyz": bool(self.cfg.decoder_return_xyz),
            "tube_curve_samples": int(self.cfg.tube_curve_samples),
            "edge_trim_samples": int(self.cfg.decoder_edge_trim_samples),
            "tube_lift_tau": float(self.cfg.tube_lift_tau),
            "tube_lift_max_values": int(self.cfg.tube_lift_max_values),
            "tube_density_tau": 0.002,
            "tube_fiber_tau": 0.002,
            "face_u_periodic": bool(u_periodic),
            "face_v_periodic": bool(v_periodic),
            "eps": float(self.cfg.decoder_eps),
            "solve_reg": float(self.cfg.decoder_solve_reg),
            "tau_voronoi": float(self.cfg.decoder_tau_voronoi),
            "tau_box": float(self.cfg.decoder_tau_box),
            "tau_trim": float(self.cfg.decoder_tau_trim),
            "use_trim_activity": bool(self.cfg.decoder_use_trim_activity),
            "vertex_boundary_margin": float(self.cfg.decoder_vertex_boundary_margin),
            "edge_trim_reduction": self.cfg.decoder_edge_trim_reduction,
            "edge_trim_reduce_tau": float(self.cfg.decoder_edge_trim_reduce_tau),
            "use_edge_trim_gate": bool(self.cfg.decoder_use_edge_trim_gate),
            "nearest_segment_k": int(self.cfg.decoder_nearest_segment_k),
            "use_segment_distance": bool(self.cfg.decoder_use_segment_distance),
            "use_spatial_pruning": bool(self.cfg.decoder_use_spatial_pruning),
            "min_tube_spacing": float(self.cfg.decoder_min_tube_spacing),
            "tube_target_spacing_ratio": float(self.cfg.decoder_tube_target_spacing_ratio),
            "use_seed_activation": bool(self.cfg.decoder_use_seed_activation),
            "n_seeds": None if seed_number is None else int(seed_number),
            "strut_thickness": float(self.cfg.strut_thickness),
            "duplicate_effect_temp_ratio": self.cfg.decoder_duplicate_effect_temp_ratio,
            "rho_min": float(self.cfg.rho_min),
        }

    def _build_face_model(self, face_tensor, device):
        return self._build_single_face_models(
            device=device,
            seed_number=self.cfg.seed_number,
            u_periodic=face_tensor.get("u_periodic", False),
            v_periodic=face_tensor.get("v_periodic", False),
            boundary_solid_idx=self._true_open_boundary_idx(face_tensor),
            face_tensor=face_tensor,
        )

    def _save_optimized_shell_function(
        self,
        save_dir,
        decoder,
        ppnet,
        face_tensor,
        best_pred,
        best_score,
        best_step,
        returned_best_source,
        final_shape_density=None,
        final_shape_fiber_direction=None,
    ):
        if save_dir is None:
            return None
        if best_pred is None:
            return None

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "optimized_shell_function.pt")
        device = face_tensor["uv"].device
        portable_face_tensor = _portable_face_tensor(face_tensor)
        portable_cad_domain = _portable_cad_domain(
            face_tensor=face_tensor,
            best_pred=best_pred,
        )
        seed_number = _safe_int_or_none(getattr(decoder, "n_seeds", None))
        if seed_number is None:
            seeds_raw = best_pred.get("seeds_raw", None)
            if isinstance(seeds_raw, torch.Tensor):
                seed_number = int(seeds_raw.shape[0])
        if seed_number is None:
            seed_number = int(self.cfg.seed_number)
        decoder_init_kwargs = self._decoder_init_kwargs(
            device=device,
            seed_number=seed_number,
            u_periodic=face_tensor.get("u_periodic", False),
            v_periodic=face_tensor.get("v_periodic", False),
            face_tensor=face_tensor,
        )
        decoder_init_kwargs = dict(decoder_init_kwargs)
        decoder_init_kwargs["Cad_domain"] = portable_cad_domain
        decoder_init_kwargs["face_mesh"] = portable_face_tensor

        package = {
            "package_type": "OptimizedShellFunction",
            "package_version": OptimizedShellFunction.package_version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.cfg),
            "decoder_class": {
                "module": decoder.__class__.__module__,
                "name": decoder.__class__.__name__,
            },
            "ppnet_class": {
                "module": ppnet.__class__.__module__,
                "name": ppnet.__class__.__name__,
            },
            "decoder_init_kwargs": _cpu_detached_tree(decoder_init_kwargs),
            "cad_domain": _cpu_detached_tree(portable_cad_domain),
            "face_tensor": portable_face_tensor,
            "decoder_state_dict": _cpu_detached_tree(decoder.state_dict()),
            "ppnet_state_dict": _cpu_detached_tree(ppnet.state_dict()),
            "best_pred": _cpu_detached_tree(best_pred),
            "best_score": float(best_score),
            "best_step": _safe_int_or_none(best_step, default=-1),
            "returned_best_source": returned_best_source,
            "face_metadata": {
                "face_id": self._face_id_key(face_tensor.get("face_id", 0)),
                "u_periodic": bool(face_tensor.get("u_periodic", False)),
                "v_periodic": bool(face_tensor.get("v_periodic", False)),
                "num_surface_points": int(face_tensor["uv"].shape[0]),
            },
            "final_shape_density": _cpu_detached_tree(final_shape_density),
            "final_shape_fiber_direction": _cpu_detached_tree(final_shape_fiber_direction),
        }
        torch.save(package, path)
        metadata_path = os.path.join(save_dir, "optimized_shell_function.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "package_type": package["package_type"],
                    "package_version": package["package_version"],
                    "created_at": package["created_at"],
                    "best_score": package["best_score"],
                    "best_step": package["best_step"],
                    "returned_best_source": package["returned_best_source"],
                    "face_metadata": package["face_metadata"],
                    "decoder_class": package["decoder_class"],
                    "ppnet_class": package["ppnet_class"],
                    "portable": True,
                    "contains_face_tensor": portable_face_tensor is not None,
                    "contains_final_shape_fields": (
                        final_shape_density is not None
                        and final_shape_fiber_direction is not None
                    ),
                },
                f,
                indent=2,
            )
        return path

    @staticmethod
    def load_optimized_shell_function(path, decoder_cls=None, device=None):
        return OptimizedShellFunction.load(path, decoder_cls=decoder_cls, device=device)

    def _build_optimizer(self, ppnet, decoder):
        cfg = self.cfg
        param_groups = []

        def trainable(params):
            return [p for p in params if p.requires_grad]

        seed_refine_params = list(ppnet.seed_refine.parameters())
        if getattr(ppnet, "seed_id_embed", None) is not None:
            seed_refine_params.extend(ppnet.seed_id_embed.parameters())
        seed_refine_params = trainable(seed_refine_params)
        if seed_refine_params:
            param_groups.append({"params": seed_refine_params, "lr": cfg.lr_seed_refine})

        delta_head_params = trainable(ppnet.delta_head.parameters())
        if delta_head_params:
            param_groups.append({"params": delta_head_params, "lr": cfg.lr_delta_head})

        if getattr(ppnet, "global_latent", None) is not None and ppnet.global_latent.requires_grad:
            param_groups.append({"params": [ppnet.global_latent], "lr": cfg.lr_mlp})

        independent_seed_offsets = getattr(ppnet, "independent_seed_offsets", None)
        if independent_seed_offsets is not None and independent_seed_offsets.requires_grad:
            param_groups.append(
                {
                    "params": [independent_seed_offsets],
                    "lr": cfg.lr_independent_seed_offsets,
                }
            )

        optimizer_param_ids = {
            id(p)
            for group in param_groups
            for p in group.get("params", [])
        }
        decoder_params = []
        if decoder is not None:
            decoder_params = [
                p
                for p in decoder.parameters()
                if p.requires_grad and id(p) not in optimizer_param_ids
            ]
        if decoder_params:
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": float(cfg.lr_decoder),
                    "name": "decoder",
                }
            )

        if not param_groups:
            raise ValueError("No trainable parameters remain for the optimizer.")
        opt = torch.optim.Adam(param_groups)
        validate_optimizer_parameter_coverage(
            named_trainable_modules=[
                ("ppnet", ppnet),
                ("decoder", decoder),
            ],
            optimizer=opt,
        )
        return opt

    def _build_scheduler(self, opt, milestones):
        cfg = self.cfg
        if not milestones:
            return None
        return torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=list(milestones),
            gamma=cfg.scheduler_gamma,
        )

    def _stage_scheduler_milestones(self, stage_id: int) -> list[int]:
        cfg = self.cfg
        spec_max_steps = int(getattr(cfg, f"stage{int(stage_id)}_max_steps"))
        raw_milestones = getattr(cfg, f"stage{int(stage_id)}_scheduler_milestones", ())
        if raw_milestones is None:
            return []
        raw_seq = [raw_milestones] if isinstance(raw_milestones, (int, float)) else list(raw_milestones)
        milestones = []
        for milestone in raw_seq:
            m = float(milestone)
            step_m = int(round(m * spec_max_steps)) if 0.0 < m <= 1.0 else int(round(m))
            if 0 < step_m < spec_max_steps:
                milestones.append(step_m)
        return sorted(set(milestones))

    @staticmethod
    def _copy_optimizer_lrs(src_opt, dst_opt):
        for src_group, dst_group in zip(src_opt.param_groups, dst_opt.param_groups):
            dst_group["lr"] = src_group.get("lr", dst_group["lr"])

    @staticmethod
    def _clone_module_state_dict(module):
        return {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for key, value in module.state_dict().items()
        }

    @staticmethod
    def clone_state_dict(state_dict):
        return {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for key, value in state_dict.items()
        }

    @staticmethod
    def _clone_optimizer_state_dict(opt):
        return copy.deepcopy(opt.state_dict()) if opt is not None else None

    @classmethod
    def _clone_modules_state_dict(cls, modules):
        return [cls._clone_module_state_dict(module) for module in modules]

    @staticmethod
    def _restore_modules_state_dict(modules, states):
        for module, state in zip(modules, states):
            module.load_state_dict(state)

    @staticmethod
    def _reduce_optimizer_lr(opt, factor: float, minimum_lr: float) -> list[tuple[float, float]]:
        changes = []
        if opt is None:
            return changes
        for group in opt.param_groups:
            old_lr = float(group.get("lr", 0.0))
            new_lr = max(old_lr * float(factor), float(minimum_lr))
            group["lr"] = new_lr
            changes.append((old_lr, new_lr))
        return changes


    @staticmethod
    def _decoder_seed_state_for_pred(decoder, pred_i: dict, device) -> tuple[int | None, torch.Tensor | None]:
        old_n_seeds_raw = getattr(decoder, "n_seeds", None)
        old_n_seeds = None if old_n_seeds_raw is None else int(old_n_seeds_raw)
        old_seed_face_id = (
            decoder.seed_face_id.detach().clone()
            if hasattr(decoder, "seed_face_id")
            else None
        )
        pred_seed_count = int(pred_i["seeds_raw"].shape[0])
        if old_n_seeds is None or pred_seed_count != old_n_seeds:
            decoder.n_seeds = pred_seed_count
            if old_seed_face_id is not None:
                decoder.seed_face_id = torch.zeros(
                    pred_seed_count,
                    dtype=torch.long,
                    device=device,
                )
        return old_n_seeds, old_seed_face_id

    @staticmethod
    def _restore_decoder_seed_state(decoder, state: tuple[int | None, torch.Tensor | None]):
        old_n_seeds, old_seed_face_id = state
        decoder.n_seeds = None if old_n_seeds is None else int(old_n_seeds)
        if old_seed_face_id is not None:
            decoder.seed_face_id = old_seed_face_id

    @staticmethod
    def _decoder_seed_activation_counts(decoder_out: dict) -> dict[str, int | float]:
        raw_seeds_i = decoder_out.get("seeds_uv", None)
        if not isinstance(raw_seeds_i, torch.Tensor):
            raise RuntimeError(
                "Decoder output is missing raw seed positions under 'seeds_uv'; "
                "cannot calculate active units reliably."
            )

        active_mask_i = decoder_out.get("seed_active_mask", None)
        active_ids_i = decoder_out.get("active_seed_ids", None)
        topology_seeds_i = decoder_out.get("topology_seeds_uv", None)
        seed_activity_weight_i = decoder_out.get("seed_activity_weight", None)

        total_seed_i = int(raw_seeds_i.shape[0])

        active_count_i: int | None = None
        if active_mask_i is not None:
            active_count_i = int(active_mask_i.to(dtype=torch.bool).sum().item())
        elif active_ids_i is not None:
            active_count_i = int(active_ids_i.numel())
        elif topology_seeds_i is not None:
            active_count_i = int(topology_seeds_i.shape[0])
        else:
            raise RuntimeError(
                "Decoder output is missing seed activation information; "
                "cannot calculate active units reliably."
            )

        if active_mask_i is not None and active_ids_i is not None:
            assert int(active_mask_i.to(dtype=torch.bool).sum().item()) == int(active_ids_i.numel())

        if active_mask_i is not None and topology_seeds_i is not None:
            assert int(active_mask_i.to(dtype=torch.bool).sum().item()) == int(topology_seeds_i.shape[0])

        inactive_count_i = total_seed_i - active_count_i
        if inactive_count_i < 0:
            raise RuntimeError(
                f"Decoder reported more active seeds ({active_count_i}) than raw seeds ({total_seed_i})."
            )

        topology_count_i = (
            int(topology_seeds_i.shape[0])
            if isinstance(topology_seeds_i, torch.Tensor)
            else active_count_i
        )
        soft_active_i = (
            float(seed_activity_weight_i.detach().sum().item())
            if isinstance(seed_activity_weight_i, torch.Tensor)
            else float("nan")
        )

        return {
            "raw": total_seed_i,
            "active": active_count_i,
            "inactive": inactive_count_i,
            "topology": topology_count_i,
            "soft_active": soft_active_i,
        }

    @staticmethod
    def _pair_upper_values(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("Expected tensor for pair reduction")
        if t.ndim < 2:
            return t.reshape(-1)

        mask = torch.triu(
            torch.ones(t.shape[-2], t.shape[-1], device=t.device, dtype=torch.bool),
            diagonal=1,
        )
        vals = t[..., mask]
        if vals.numel() == 0:
            return t.reshape(-1)
        return vals.reshape(-1)

    @staticmethod
    def _face_id_key(face_id) -> int:
        return _safe_int_or_none(face_id, default=0)
    
    def _init_face_seed(self, face_tensor):
        cfg = self.cfg
        boundary = self._true_open_boundary_idx(face_tensor)
        if not cfg.use_balanced_seed_init:
            seed_idx = self._random_seed_indices(
                n_points=int(face_tensor["uv"].shape[0]),
                n_samples=int(cfg.seed_number),
                exclude_idx=boundary,
                seed=cfg.seed_init_fps_seed,
                device=face_tensor["uv"].device,
            )
            return face_tensor["uv"][seed_idx].clone()

        seed_idx = self.generator.fps_3d(
            face_tensor["points_xyz"],
            cfg.seed_number,
            exclude_idx=boundary,
            seed = cfg.seed_init_fps_seed,
        )
        return face_tensor["uv"][seed_idx].clone()

    @staticmethod
    def _random_seed_indices(
        n_points: int,
        n_samples: int,
        exclude_idx=None,
        seed: int | None = None,
        device=None,
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else torch.device(device)
        n_points = int(n_points)
        n_samples = min(int(n_samples), n_points)
        if n_samples <= 0 or n_points <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        candidate_mask = torch.ones((n_points,), dtype=torch.bool, device=device)
        if exclude_idx is not None:
            exclude_idx = torch.as_tensor(exclude_idx, dtype=torch.long, device=device)
            exclude_idx = exclude_idx[(exclude_idx >= 0) & (exclude_idx < n_points)]
            if exclude_idx.numel() > 0:
                candidate_mask[exclude_idx] = False

        candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(n_points, dtype=torch.long, device=device)
        n_samples = min(n_samples, int(candidates.numel()))

        if seed is None:
            order = torch.randperm(candidates.numel(), device=device)
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            order_cpu = torch.randperm(candidates.numel(), generator=gen)
            order = order_cpu.to(device=device)
        return candidates[order[:n_samples]].to(dtype=torch.long)

    def _seed_points_xyz(self, seeds, face_tensor):
        return self.generator.seeds_uv_to_xyz_nearest(
            seeds,
            face_tensor["uv"],
            face_tensor["points_xyz"],
        )

    def _stage_settings_for_stage_id(self, stage_id: int) -> dict[str, float | bool | int]:
        cfg = self.cfg
        prefix = "stage1" if int(stage_id) == 1 else "stage2"
        return {
            "stage": 1 if int(stage_id) == 1 else 2,
            "freeze_seeds": bool(getattr(cfg, f"{prefix}_freeze_seeds")),
            "allow_seed_outside_domain": bool(getattr(cfg, f"{prefix}_allow_seed_outside_domain")),
            "lam_fem": float(getattr(cfg, f"{prefix}_lam_fem")),
            "lam_cvt": float(getattr(cfg, f"{prefix}_lam_cvt")),
            "lam_rep": float(getattr(cfg, f"{prefix}_lam_rep")),
            "lam_l_seed": float(getattr(cfg, f"{prefix}_lam_l_seed")),
            "lam_total_fiber_length": float(getattr(cfg, f"{prefix}_lam_total_fiber_length")),
            "lam_l_curve_cell": float(getattr(cfg, f"{prefix}_lam_l_curve_cell")),
        }

    def _first_physical_stage_id(self, stage_specs: list[StageSpec] | None = None) -> int:
        stage_ids = (
            [int(spec.stage_id) for spec in stage_specs]
            if stage_specs
            else [1, 2]
        )
        for stage_id in sorted(set(stage_ids)):
            settings = self._stage_settings_for_stage_id(stage_id)
            if float(settings.get("lam_fem", 0.0)) > 0.0:
                return int(stage_id)
        return min(stage_ids) if stage_ids else 1

    def _adaptive_stage_specs(self) -> list[StageSpec]:
        cfg = self.cfg
        return [
            StageSpec(
                stage_id=1,
                name="Stage 1",
                min_steps=int(cfg.stage1_min_steps),
                max_steps=int(cfg.stage1_max_steps),
                patience=int(cfg.stage1_patience),
                min_delta_abs=float(cfg.stage1_min_delta_abs),
                min_delta_rel=float(cfg.stage1_min_delta_rel),
            ),
            StageSpec(
                stage_id=2,
                name="Stage 2",
                min_steps=int(cfg.stage2_min_steps),
                max_steps=int(cfg.stage2_max_steps),
                patience=int(cfg.stage2_patience),
                min_delta_abs=float(cfg.stage2_min_delta_abs),
                min_delta_rel=float(cfg.stage2_min_delta_rel),
            ),
        ]

    @staticmethod
    def is_meaningful_improvement(
        current: float,
        best: float,
        min_delta_abs: float,
        min_delta_rel: float,
    ) -> bool:
        if not math.isfinite(float(current)):
            return False
        if not math.isfinite(float(best)):
            return True
        required_delta = max(
            float(min_delta_abs),
            float(min_delta_rel) * max(abs(float(best)), 1e-12),
        )
        return float(current) < float(best) - required_delta

    def calculate_stage_monitor(
        self,
        stage_id: int,
        loss_values: dict[str, torch.Tensor | float],
        effective_lambdas: dict[str, float],
    ) -> torch.Tensor:
        def tensor_value(name: str, fallback: float = 0.0) -> torch.Tensor:
            value = loss_values.get(name, fallback)
            if isinstance(value, torch.Tensor):
                return value.detach()
            ref = next((v for v in loss_values.values() if isinstance(v, torch.Tensor)), None)
            device = ref.device if isinstance(ref, torch.Tensor) else None
            dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float64
            return torch.tensor(float(value), dtype=dtype, device=device)

        terms: list[torch.Tensor] = []
        if int(stage_id) == 1:
            for lam_name, loss_name in (
                ("lam_l_seed", "loss_l_seed_norm"),
                ("lam_cvt", "loss_cvt_norm"),
                ("lam_rep", "loss_rep_norm"),
                ("lam_l_curve_cell", "loss_l_curve_cell_norm"),
            ):
                lam = float(effective_lambdas.get(lam_name, 0.0))
                if lam != 0.0:
                    terms.append(tensor_value(loss_name) * lam)
        else:
            if bool(loss_values.get("overall_feasible", False)):
                return tensor_value("design_score")
            if "overall_constraint_violation" in loss_values:
                return tensor_value("overall_constraint_violation")
            if "mechanical_violation" in loss_values:
                return tensor_value("mechanical_violation")
            if "design_score" in loss_values:
                return tensor_value("design_score")
            for lam_name, loss_name in (
                ("lam_total_fiber_length", "loss_total_fiber_length_norm"),
                ("lam_cvt", "loss_cvt_norm"),
                ("lam_rep", "loss_rep_norm"),
                ("lam_l_curve_cell", "loss_l_curve_cell_norm"),
                ("lam_l_seed", "loss_l_seed_norm"),
            ):
                lam = float(effective_lambdas.get(lam_name, 0.0))
                if lam != 0.0:
                    terms.append(tensor_value(loss_name) * lam)
        if not terms:
            return tensor_value("L_total")
        monitor = sum(terms[1:], terms[0])
        return monitor.detach()

    @staticmethod
    def _fixed_reference_normalized(
        current_loss: torch.Tensor,
        reference: torch.Tensor | float | int | None,
        reference_eps: float,
        fallback_normalizer: float = 1.0,
    ) -> torch.Tensor:
        if not isinstance(current_loss, torch.Tensor):
            raise TypeError("current_loss must be a torch.Tensor")
        if reference is None:
            denominator = current_loss.new_tensor(max(float(fallback_normalizer), float(reference_eps)))
        elif isinstance(reference, torch.Tensor):
            denominator = reference.detach().to(device=current_loss.device, dtype=current_loss.dtype)
            denominator = denominator.reshape(()).abs().clamp_min(float(reference_eps))
        else:
            denominator = current_loss.new_tensor(abs(float(reference))).clamp_min(float(reference_eps))
        return current_loss / denominator

    @staticmethod
    def _capture_fixed_stage2_reference(
        references: dict[str, torch.Tensor],
        name: str,
        raw_loss: torch.Tensor,
        enabled: bool,
        reference_eps: float,
    ) -> None:
        if not enabled or name in references:
            return
        if not isinstance(raw_loss, torch.Tensor):
            return
        raw_scalar = raw_loss.detach().reshape(())
        if bool(torch.isfinite(raw_scalar).item()):
            references[name] = raw_scalar.abs().clamp_min(float(reference_eps)).clone()

    @staticmethod
    def _stage2_fiber_normalized(
        *,
        loss_total_fiber_length: torch.Tensor,
        stage2_fiber_reference: torch.Tensor | float | int | None,
        reference_eps: float,
        fallback_normalizer: float = 1.0,
    ) -> torch.Tensor:
        return NN_Trainer._fixed_reference_normalized(
            loss_total_fiber_length,
            stage2_fiber_reference,
            reference_eps,
            fallback_normalizer,
        )

    @staticmethod
    def _assemble_feasibility_first_stage2_loss(
        *,
        fem_total_loss: torch.Tensor,
        fem_violation_loss: torch.Tensor,
        loss_total_fiber_length_stage2_norm: torch.Tensor,
        loss_cvt_normalized: torch.Tensor,
        loss_rep_normalized: torch.Tensor,
        loss_seed: torch.Tensor,
        loss_curve_cell_normalized: torch.Tensor,
        lam_fem_step: float,
        lam_total_fiber_length_step: float,
        lam_cvt_step: float,
        lam_rep_step: float,
        lam_l_seed_step: float,
        lam_l_curve_cell_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        design_score = (
            float(lam_total_fiber_length_step) * loss_total_fiber_length_stage2_norm
            + float(lam_cvt_step) * loss_cvt_normalized
            + float(lam_rep_step) * loss_rep_normalized
            + float(lam_l_curve_cell_step) * loss_curve_cell_normalized
            + float(lam_l_seed_step) * loss_seed
        )
        L_train = design_score + float(lam_fem_step) * fem_total_loss
        mode = (
            "feasible_design"
            if bool(torch.as_tensor(fem_violation_loss.detach()).reshape(()).item() == 0.0)
            else "infeasible_recovery"
        )
        return L_train, design_score, mode

    @staticmethod
    def _stage2_topology_valid_from_row(row: dict[str, Any]) -> bool:
        spacing = float(row.get("loss_spacing_barrier", 0.0))
        return math.isfinite(spacing) and spacing <= 0.0

    @staticmethod
    def _stage_monitor_uses_design_mode(stage_id: int, row: dict[str, Any]) -> bool:
        return int(stage_id) != 2 or bool(row.get("overall_feasible", True))

    @staticmethod
    def _is_stage2_physical_checkpoint_candidate(
        *,
        stage_id: int,
        first_physical_stage: int,
        fem_was_evaluated: bool,
        fem_is_valid: bool,
        total_loss_is_finite: bool,
    ) -> bool:
        return (
            int(stage_id) >= int(first_physical_stage)
            and bool(fem_was_evaluated)
            and bool(fem_is_valid)
            and bool(total_loss_is_finite)
        )

    @staticmethod
    def _best_feasible_design_score(best_feasible_key) -> float:
        if best_feasible_key is None or len(best_feasible_key) < 1:
            return float("nan")
        value = float(best_feasible_key[0])
        return value if math.isfinite(value) else float("nan")

    @staticmethod
    def _best_feasible_step(best_feasible_key, best_feasible_checkpoint=None) -> int:
        if best_feasible_key is not None and len(best_feasible_key) >= 3:
            step = float(best_feasible_key[2])
            if math.isfinite(step):
                return int(step)
        if best_feasible_checkpoint is not None and hasattr(best_feasible_checkpoint, "get"):
            return int(best_feasible_checkpoint.get("global_step", -1))
        return -1

    @staticmethod
    def _best_infeasible_violation(best_infeasible_key) -> float:
        if best_infeasible_key is None or len(best_infeasible_key) < 1:
            return float("nan")
        value = float(best_infeasible_key[0])
        return value if math.isfinite(value) else float("nan")

    @staticmethod
    def _best_infeasible_step(best_infeasible_key, best_infeasible_checkpoint=None) -> int:
        if best_infeasible_key is not None and len(best_infeasible_key) >= 3:
            step = float(best_infeasible_key[2])
            if math.isfinite(step):
                return int(step)
        if best_infeasible_checkpoint is not None and hasattr(best_infeasible_checkpoint, "get"):
            return int(best_infeasible_checkpoint.get("global_step", -1))
        return -1

    @staticmethod
    def _format_optional_score(value: float, step: int, none_label: str) -> str:
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        if not math.isfinite(numeric) or int(step) < 0:
            return none_label
        return f"{numeric:.4e}@{int(step):05d}"

    @classmethod
    def _format_stage_progress_log(
        cls,
        *,
        row: dict[str, Any],
        total_step_budget: int,
        best_feasible_key,
        best_feasible_checkpoint=None,
        best_infeasible_key=None,
        best_infeasible_checkpoint=None,
    ) -> str:
        best_feasible_design_score = cls._best_feasible_design_score(best_feasible_key)
        best_feasible_step = cls._best_feasible_step(best_feasible_key, best_feasible_checkpoint)
        best_infeasible_violation = cls._best_infeasible_violation(best_infeasible_key)
        best_infeasible_step = cls._best_infeasible_step(best_infeasible_key, best_infeasible_checkpoint)
        best_feasible_text = cls._format_optional_score(
            best_feasible_design_score,
            best_feasible_step,
            "best_feasible_design=None",
        )
        if best_feasible_text != "best_feasible_design=None":
            best_feasible_text = f"best_feasible_design={best_feasible_text}"
        best_recovery_text = cls._format_optional_score(
            best_infeasible_violation,
            best_infeasible_step,
            "best_recovery_violation=None",
        )
        if best_recovery_text != "best_recovery_violation=None":
            best_recovery_text = f"best_recovery_violation={best_recovery_text}"
        stage_label = f"Stage {int(row.get('stage', 0))}"
        return (
            f"[{int(row.get('step', -1)):05d}/{int(total_step_budget):05d}] "
            f"({stage_label} {int(row.get('stage_local_step', 0)):05d}/"
            f"{int(row.get('stage_max_steps', 0)):05d}) "
            f"L_train={float(row.get('L_train', row.get('L_total', float('nan')))):.4e} | "
            f"design_score={float(row.get('design_score', float('nan'))):.4e} | "
            f"monitor={float(row.get('stage_monitor_raw', float('nan'))):.4e}"
            f"({str(row.get('stage_monitor_mode', ''))}) | "
            f"{best_feasible_text} | "
            f"{best_recovery_text} | "
            f"FEM_total={float(row.get('fem_total_loss', row.get('loss_fem', float('nan')))):.3e} "
            f"FEM_baseline={float(row.get('fem_baseline_loss', row.get('baseline_fem_loss', float('nan')))):.3e} "
            f"FEM_violation_loss={float(row.get('fem_violation_loss', row.get('violation_fem_loss', float('nan')))):.3e} "
            f"lam_FEM={float(row.get('lam_fem_eff', 0.0)):.3g} | "
            f"ratios(phys s/d)="
            f"{float(row.get('physical_stress_ratio', float('nan'))):.3e}/"
            f"{float(row.get('physical_displacement_ratio', float('nan'))):.3e} "
            f"hard_active={float(row.get('hard_active_seed_count', row.get('hard_active_count', float('nan')))):.0f} "
            f"physical_feasible={bool(row.get('physical_feasible', False))} "
            f"active_seed_feasible={bool(row.get('active_seed_feasible', False))} "
            f"overall_feasible={bool(row.get('overall_feasible', False))} "
            f"overall_violation={float(row.get('overall_constraint_violation', float('nan'))):.3e} "
            f"patience_active={bool(row.get('patience_active', False))} "
            f"design_patience_active={bool(row.get('design_patience_active', False))} "
            f"design_patience={int(row.get('stage_patience_counter', 0))}/"
            f"{int(row.get('stage_patience_limit', 0))}"
        )

    @staticmethod
    def _update_stage_runtime_checkpoint(
        runtime: StageRuntime,
        *,
        row: dict[str, Any],
        checkpoint: dict[str, Any],
        stage_monitor_raw: float,
        meaningful_improvement: bool,
        stage_id: int,
    ) -> str:
        if NN_Trainer._stage_monitor_uses_design_mode(stage_id, row):
            if meaningful_improvement or float(stage_monitor_raw) < float(runtime.best_raw_monitor):
                runtime.best_raw_monitor = float(stage_monitor_raw)
                runtime.stage_best_raw_checkpoint = dict(checkpoint)
                runtime.stage_best_raw_checkpoint["source"] = "best_raw"
                return "design_best"
            return "design_unchanged"

        if float(stage_monitor_raw) < float(runtime.recovery_best_monitor):
            runtime.recovery_best_monitor = float(stage_monitor_raw)
            runtime.stage_recovery_best_checkpoint = dict(checkpoint)
            runtime.stage_recovery_best_checkpoint["source"] = "best_recovery"
            return "recovery_best"
        return "recovery_unchanged"

    def _apply_stage_trainability(self, ppnet, stage_settings: dict[str, Any]) -> None:
        freeze_seeds = bool(stage_settings.get("freeze_seeds", False))
        seed_modules = [getattr(ppnet, "seed_refine", None), getattr(ppnet, "seed_id_embed", None)]
        for module in seed_modules:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(not freeze_seeds)
        independent_seed_offsets = getattr(ppnet, "independent_seed_offsets", None)
        if independent_seed_offsets is not None:
            independent_seed_offsets.requires_grad_(not freeze_seeds)
        for module in (getattr(ppnet, "delta_head", None),):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(True)
        if getattr(ppnet, "global_latent", None) is not None:
            ppnet.global_latent.requires_grad_(True)

    def _stage_checkpoint_from_step(
        self,
        *,
        source: str,
        ppnet,
        decoder,
        opt,
        scheduler,
        uv_anchor: torch.Tensor,
        row: dict[str, Any],
        pred_list: list[dict[str, Any]],
        seeds_list: list[torch.Tensor],
        rho: torch.Tensor,
        fiber_surface: torch.Tensor,
        fem_density_field,
        fem_stress_field,
        fem_displacement_field,
        stage_monitor_raw: float,
        effective_lambdas: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "source": source,
            "ppnet_state_dict": self._clone_module_state_dict(ppnet),
            "decoder_state_dict": self._clone_module_state_dict(decoder),
            "optimizer_state_dict": _cpu_detached_tree(opt.state_dict()) if opt is not None else None,
            "scheduler_state_dict": _cpu_detached_tree(scheduler.state_dict()) if scheduler is not None else None,
            "uv_anchor": uv_anchor.detach().clone(),
            "global_step": int(row.get("step", -1)),
            "stage_id": int(row.get("stage", 0)),
            "stage_local_step": int(row.get("stage_local_step", 0)),
            "row": _cpu_detached_tree(dict(row)),
            "raw_metrics": _cpu_detached_tree(dict(row)),
            "pred_list": self._clone_pred_list(pred_list),
            "seeds": [s.detach().clone() for s in seeds_list],
            "rho": rho.detach().clone(),
            "fiber_surface": fiber_surface.detach().clone(),
            "fem_density_field": fem_density_field.detach().clone() if isinstance(fem_density_field, torch.Tensor) else None,
            "fem_stress_field": fem_stress_field.detach().clone() if isinstance(fem_stress_field, torch.Tensor) else None,
            "fem_displacement_field": fem_displacement_field.detach().clone() if isinstance(fem_displacement_field, torch.Tensor) else None,
            "stage_monitor_raw": float(stage_monitor_raw),
            "effective_lambdas": dict(effective_lambdas),
            "active_seed_count": float(row.get("active_units_total", float("nan"))),
            "volume_fraction": float(row.get("VolFrac", float("nan"))),
            "selected_checkpoint_stage_loss": float(row.get("L_total", float("nan"))),
            "valid": True,
        }

    def _save_live_best_feasible_checkpoint(
        self,
        *,
        output_folder: str | None,
        checkpoint: dict[str, Any],
        decoder,
        ppnet,
        face_tensor,
    ) -> str | None:
        if not output_folder or checkpoint is None:
            return None

        save_dir = os.path.join(os.path.normpath(str(output_folder)), "BestFeasibleCheckpoint")
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, "best_feasible_checkpoint.pt")
        tmp_checkpoint_path = checkpoint_path + ".tmp"
        package = {
            "package_type": "BestFeasibleTrainingCheckpoint",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.cfg),
            "checkpoint": _cpu_detached_tree(checkpoint),
        }
        torch.save(package, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)

        row = checkpoint.get("row", {})
        metadata = {
            "package_type": package["package_type"],
            "created_at": package["created_at"],
            "global_step": int(checkpoint.get("global_step", -1)),
            "stage_id": int(checkpoint.get("stage_id", 0)),
            "source": str(checkpoint.get("source", "")),
            "stage_monitor_raw": float(checkpoint.get("stage_monitor_raw", float("nan"))),
            "design_score": float(row.get("design_score", float("nan"))),
            "raw_total_fiber_length": float(row.get("loss_total_fiber_length", float("nan"))),
            "physical_stress_ratio": float(row.get("physical_stress_ratio", float("nan"))),
            "physical_displacement_ratio": float(row.get("physical_displacement_ratio", float("nan"))),
            "physical_max_ratio": max(
                float(row.get("physical_stress_ratio", float("nan"))),
                float(row.get("physical_displacement_ratio", float("nan"))),
            ),
            "hard_active_count": float(row.get("hard_active_count", float("nan"))),
        }
        metadata_path = os.path.join(save_dir, "best_feasible_checkpoint.json")
        tmp_metadata_path = metadata_path + ".tmp"
        with open(tmp_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        os.replace(tmp_metadata_path, metadata_path)

        pred_list = checkpoint.get("pred_list", [])
        best_pred = pred_list[0] if pred_list else None
        if best_pred is not None:
            self._save_optimized_shell_function(
                save_dir=save_dir,
                decoder=decoder,
                ppnet=ppnet,
                face_tensor=face_tensor,
                best_pred=best_pred,
                best_score=float(row.get("design_score", float("nan"))),
                best_step=int(checkpoint.get("global_step", -1)),
                returned_best_source="live_best_feasible",
                final_shape_density=checkpoint.get("rho"),
                final_shape_fiber_direction=checkpoint.get("fiber_surface"),
            )

        return checkpoint_path

    def _restore_stage_checkpoint(self, checkpoint: dict[str, Any], ppnet, decoder, opt=None, scheduler=None):
        ppnet.load_state_dict(checkpoint["ppnet_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        if opt is not None and checkpoint.get("optimizer_state_dict") is not None:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["uv_anchor"].detach().clone()

    def _checkpoint_stage_monitor_for_stage(self, checkpoint: dict[str, Any], stage_id: int) -> float:
        row = checkpoint.get("row", {})
        settings = self._stage_settings_for_stage_id(stage_id)
        loss_values = {
            "L_total": float(row.get("L_total", float("inf"))),
            "loss_fem_norm": float(row.get("loss_fem_norm", row.get("loss_fem", float("inf")))),
            "loss_cvt_norm": float(row.get("loss_cvt_norm", row.get("loss_cvt", float("inf")))),
            "loss_l_curve_cell_norm": float(row.get("loss_l_curve_cell_norm", row.get("loss_l_curve_cell", float("inf")))),
            "loss_total_fiber_length_norm": float(row.get("loss_total_fiber_length_norm", row.get("loss_total_fiber_length", float("inf")))),
            "loss_rep_norm": float(row.get("loss_rep_norm", row.get("loss_rep", float("inf")))),
            "loss_l_seed_norm": float(row.get("loss_l_seed_norm", row.get("loss_l_seed", float("inf")))),
            "_zero": 0.0,
        }
        monitor = self.calculate_stage_monitor(stage_id, loss_values, settings)
        value = float(monitor.detach().cpu().item()) if isinstance(monitor, torch.Tensor) else float(monitor)
        return value if math.isfinite(value) else float("inf")

    def _select_transition_checkpoint(
        self,
        runtime: StageRuntime,
        next_stage_id: int,
    ) -> tuple[str | None, dict[str, Any] | None, float]:
        candidates = [
            ("best_raw", runtime.stage_best_raw_checkpoint),
            ("last", runtime.stage_last_valid_checkpoint),
        ]
        candidates = [(name, ckpt) for name, ckpt in candidates if ckpt is not None and ckpt.get("valid", False)]
        if not candidates:
            return None, None, float("inf")
        policy = str(getattr(self.cfg, "stage_transition_selection", "next_stage_objective"))
        if policy == "last":
            for name, ckpt in candidates:
                if name == "last":
                    return name, ckpt, float(ckpt.get("stage_monitor_raw", float("inf")))
            return None, None, float("inf")
        if policy == "stage_monitor":
            fallback_order = ("best_raw", "last")
            for fallback_name in fallback_order:
                for name, ckpt in candidates:
                    if name == fallback_name:
                        return name, ckpt, float(ckpt.get("stage_monitor_raw", float("inf")))
            return None, None, float("inf")
        scored = []
        for name, ckpt in candidates:
            score = self._checkpoint_stage_monitor_for_stage(ckpt, next_stage_id)
            if math.isfinite(float(score)):
                scored.append((score, name, ckpt))
        if not scored:
            return None, None, float("inf")
        score, name, ckpt = min(scored, key=lambda item: item[0])
        return name, ckpt, float(score)

    def _select_final_stage_checkpoint(
        self,
        runtime: StageRuntime,
    ) -> tuple[str | None, dict[str, Any] | None, float]:
        candidates = [
            ("stage_best_raw", runtime.stage_best_raw_checkpoint, "stage_monitor_raw"),
            ("stage_last_valid", runtime.stage_last_valid_checkpoint, "stage_monitor_raw"),
        ]
        valid = [
            (name, ckpt, score_key)
            for name, ckpt, score_key in candidates
            if ckpt is not None and ckpt.get("valid", False)
        ]
        if not valid:
            return None, None, float("inf")

        for preferred_name in ("stage_best_raw", "stage_last_valid"):
            for name, ckpt, score_key in valid:
                if name == preferred_name:
                    return name, ckpt, float(ckpt.get(score_key, float("inf")))

        return None, None, float("inf")

    @staticmethod
    def _stage_lambda_summary(row: dict[str, Any]) -> str:
        return (
            f"S{int(row.get('stage', 0))} "
            f"(fem={float(row.get('lam_fem_eff', 0.0)):.2g}, "
            f"cvt={float(row.get('lam_cvt_eff', 0.0)):.2g}, "
            f"fiber={float(row.get('lam_total_fiber_length_eff', 0.0)):.2g}, "
            f"rep={float(row.get('lam_rep_eff', 0.0)):.2g}, "
            f"cell={float(row.get('lam_l_curve_cell_eff', 0.0)):.2g}, "
            f"seed={float(row.get('lam_l_seed_eff', 0.0)):.2g})"
        )

    @staticmethod
    def _timelapse_loss_chart_dict(row: dict[str, Any], score: float | None = None) -> dict[str, float]:
        def row_float(name: str, default: float = float("nan")) -> float:
            try:
                return float(row.get(name, default))
            except Exception:
                return default

        total_value = (
            float(score)
            if score is not None
            else row_float("stage_monitor_raw", row_float("L_total"))
        )
        def label(name: str, lam_name: str) -> str:
            return f"{name}({row_float(lam_name, 0.0):.2g})"

        return {
            "Total": total_value,
            label("FEM", "lam_fem_eff"): row_float("loss_fem_norm"),
            label("CVT", "lam_cvt_eff"): row_float("loss_cvt_norm"),
            label("Rep", "lam_rep_eff"): row_float("loss_rep_norm"),
            label("Seed", "lam_l_seed_eff"): row_float("loss_l_seed_norm"),
            label("TotLen", "lam_total_fiber_length_eff"): row_float("loss_total_fiber_length_norm"),
            label("EdgeLen", "lam_l_curve_cell_eff"): row_float("loss_l_curve_cell_norm"),
        }

    @staticmethod
    def _timelapse_geometry_summary_text(row: dict[str, Any] | None) -> str:
        row = row or {}

        def row_float(name: str) -> float:
            try:
                return float(row.get(name, float("nan")))
            except Exception:
                return float("nan")

        fields = (
            ("Tot_Len", "loss_total_fiber_length", ".2e"),
            ("Min_cell_Ar", "cell_area_min", ".2e"),
            ("Min_Eg_Len", "curve_length_min", ".2e"),
            ("Max Disp", "disp_max", ".2e"),
            ("Max Stress", "stress_max", ".2e"),
            ("VolFrac", "VolFrac", ".2f"),
            ("Active Units", "active_units_total", ".0f"),
        )
        parts = []
        for label, key, fmt in fields:
            value = row_float(key)
            if not math.isfinite(value):
                continue
            parts.append(f"{label}={value:{fmt}}")
        return " | ".join(parts)

    def _eval_uv_to_xyz_differentiable(self, uv: torch.Tensor) -> torch.Tensor:
        evaluator_owner = self.Cad_domain if self.Cad_domain is not None else self.generator
        evaluator = getattr(evaluator_owner, "eval_uv_norm_batch_torch", None)
        if evaluator is None or not callable(evaluator):
            raise TypeError(
                "Warm-up CVT requires a differentiable "
                "eval_uv_norm_batch_torch(uv) CAD evaluator."
            )
        evaluated = evaluator(uv)
        xyz = evaluated["xyz"] if isinstance(evaluated, dict) else evaluated
        if not isinstance(xyz, torch.Tensor):
            raise TypeError("eval_uv_norm_batch_torch must return a tensor or {'xyz': tensor}.")
        if xyz.shape != (*uv.shape[:-1], 3):
            raise ValueError(
                "Differentiable CAD evaluator returned XYZ with shape "
                f"{tuple(xyz.shape)} for UV shape {tuple(uv.shape)}."
            )
        return xyz

    def _finite_or_default(self, x: torch.Tensor | float | int, default: float = float("nan")) -> float:
        if self._scalar_tensor_is_finite(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().item())
            return float(x)
        return default

    @staticmethod
    def _named_trainable_params(modules):
        for mi, module in enumerate(modules):
            for pn, p in module.named_parameters():
                if p.requires_grad:
                    yield mi, pn, p

    @classmethod
    def _trainable_zero(cls, modules, dtype, device):
        zero = torch.zeros((), dtype=dtype, device=device)
        for _mi, _pn, p in cls._named_trainable_params(modules):
            return p.reshape(-1)[0] * 0.0
        return zero

    @classmethod
    def _nonfinite_grad_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            g = p.grad
            if g is not None and not torch.isfinite(g).all():
                bad.append((mi, pn))
        return bad

    @classmethod
    def _nonfinite_grad_cause_summary(
        cls,
        modules,
        bad_grad_info,
        loss_terms=None,
        fem_is_valid=True,
        fem_failure_reason=None,
    ) -> str:
        reasons = []

        if loss_terms:
            bad_losses = []
            finite_losses = []
            for name, value in loss_terms:
                if value is None:
                    continue
                if cls._scalar_tensor_is_finite(value):
                    raw = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
                    finite_losses.append((name, raw))
                else:
                    bad_losses.append(name)

            if bad_losses:
                reasons.append("non-finite loss term(s): " + ", ".join(bad_losses[:5]))
            elif finite_losses:
                largest_name, largest_value = max(finite_losses, key=lambda item: abs(item[1]))
                reasons.append(f"all tracked losses finite; largest={largest_name}={largest_value:.3e}")

        if not fem_is_valid:
            if fem_failure_reason:
                reasons.append(f"FEM invalid: {fem_failure_reason}")
            else:
                reasons.append("FEM invalid")

        bad_set = set(bad_grad_info)
        for mi, pn, p in cls._named_trainable_params(modules):
            if (mi, pn) not in bad_set or p.grad is None:
                continue
            g = p.grad.detach()
            nan_count = int(torch.isnan(g).sum().item())
            posinf_count = int(torch.isposinf(g).sum().item())
            neginf_count = int(torch.isneginf(g).sum().item())
            reasons.append(f"bad grad at face={mi}:{pn} (nan={nan_count}, +inf={posinf_count}, -inf={neginf_count})")
            break

        if not reasons:
            reasons.append("likely backward overflow or unstable derivative")
        elif loss_terms and not any(reason.startswith("non-finite loss") for reason in reasons):
            reasons.append("likely backward overflow or unstable derivative")

        return "Cause: " + "; ".join(reasons)

    @classmethod
    def _nonfinite_param_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            if not torch.isfinite(p).all():
                bad.append((mi, pn))
        return bad

    @staticmethod
    def _restore_param_snapshot(snapshot):
        for p, saved in snapshot.items():
            p.data.copy_(saved)

    @staticmethod
    def _clear_optimizer_state_for_params(opt, params):
        for p in params:
            if p in opt.state:
                opt.state.pop(p, None)

    def _print_fem_failure(self, step: int):
        print(f"\n=== FEM FAILURE AT STEP {step} ===")
        for k, v in self.last_fem_debug.items():
            print(f"{k}: {v}")
        print("Skipping FEM term for this step.\n")



    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _validate_face_tensors(self, face_tensors):
        required_keys = [
            "face_id",
            "uv",
            "Xu",
            "Xv",
            "points_xyz",
            "faces_ijk",
            "face_areas",
            "global_vertex_idx",
        ]

        if not isinstance(face_tensors, (list, tuple)) or len(face_tensors) == 0:
            raise ValueError("face_tensors must be a non-empty list.")

        ref_uv = face_tensors[0]["uv"]
        ref_device = ref_uv.device
        ref_dtype = ref_uv.dtype

        for i, ft in enumerate(face_tensors):
            missing = [k for k in required_keys if k not in ft]
            if missing:
                raise ValueError(f"face_tensors[{i}] is missing required keys: {missing}")

            uv = ft["uv"]
            Xu = ft["Xu"]
            Xv = ft["Xv"]
            points_xyz = ft["points_xyz"]
            faces_ijk = ft["faces_ijk"]
            face_areas = ft["face_areas"]
            gidx = ft["global_vertex_idx"]

            if uv.device != ref_device:
                raise ValueError(f"face_tensors[{i}]['uv'] device mismatch: {uv.device} != {ref_device}")
            if uv.dtype != ref_dtype:
                raise ValueError(f"face_tensors[{i}]['uv'] dtype mismatch: {uv.dtype} != {ref_dtype}")

            n_local = uv.shape[0]
            if Xu.shape[0] != n_local or Xv.shape[0] != n_local or points_xyz.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}] local tensor lengths do not match uv.shape[0]={n_local}")

            if gidx.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] length mismatch with local vertex count")

            if gidx.dtype != torch.long:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] must be torch.long")

            if gidx.numel() > 0 and int(gidx.min().item()) < 0:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] contains negative indices")

            if faces_ijk.numel() > 0:
                if faces_ijk.dtype != torch.long:
                    raise ValueError(f"face_tensors[{i}]['faces_ijk'] must be torch.long")
                fmin = int(faces_ijk.min().item())
                fmax = int(faces_ijk.max().item())
                if fmin < 0 or fmax >= n_local:
                    raise ValueError(
                        f"face_tensors[{i}]['faces_ijk'] contains invalid local indices "
                        f"(min={fmin}, max={fmax}, n_local={n_local})"
                    )

            if face_areas.ndim != 1:
                raise ValueError(f"face_tensors[{i}]['face_areas'] must be 1D")

            if face_areas.shape[0] != faces_ijk.shape[0]:
                raise ValueError(
                    f"face_tensors[{i}]['face_areas'] length must match number of faces "
                    f"({face_areas.shape[0]} != {faces_ijk.shape[0]})"
                )

    def _select_single_training_face(self, face_tensors):
        if not isinstance(face_tensors, (list, tuple)) or len(face_tensors) == 0:
            raise ValueError("face_tensors must be a non-empty list.")

        face_index = int(getattr(self.cfg, "training_face_index", 0))
        if face_index < 0 or face_index >= len(face_tensors):
            raise IndexError(
                f"training_face_index={face_index} is out of range for "
                f"{len(face_tensors)} face tensor(s)"
            )

        selected = dict(face_tensors[face_index])
        selected["global_vertex_idx"] = torch.arange(
            selected["uv"].shape[0],
            dtype=torch.long,
            device=selected["uv"].device,
        )

        if len(face_tensors) > 1:
            tqdm.write(
                f"Using only face_tensors[{face_index}] for single-face training "
                f"(received {len(face_tensors)} faces)."
            )
        return [selected]

    @staticmethod
    def _build_face_uv_grid(ft, grid_res_u, grid_res_v):
        uv_face = ft["uv"]
        device = uv_face.device
        dtype = uv_face.dtype
        u = uv_face[:, 0]
        v = uv_face[:, 1]

        if bool(ft.get("u_periodic", False)):
            u_lin = torch.linspace(0.0, 1.0, grid_res_u + 1, device=device, dtype=dtype)[:-1]
        else:
            u_lin = torch.linspace(u.min(), u.max(), grid_res_u, device=device, dtype=dtype)

        if bool(ft.get("v_periodic", False)):
            v_lin = torch.linspace(0.0, 1.0, grid_res_v + 1, device=device, dtype=dtype)[:-1]
        else:
            v_lin = torch.linspace(v.min(), v.max(), grid_res_v, device=device, dtype=dtype)

        UU, VV = torch.meshgrid(u_lin, v_lin, indexing="ij")
        uv_grid = torch.stack([UU.reshape(-1), VV.reshape(-1)], dim=1)
        return uv_grid, u_lin, v_lin

    @staticmethod
    def _periodic_uv_min_dist(uv_query, uv_face, u_periodic=False, v_periodic=False, chunk_size=4096):
        if uv_query.numel() == 0 or uv_face.numel() == 0:
            return torch.empty((uv_query.shape[0],), device=uv_query.device, dtype=uv_query.dtype)

        mins = []
        for start in range(0, uv_query.shape[0], chunk_size):
            q = uv_query[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_face.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            mins.append(torch.norm(diff, dim=-1).min(dim=1).values)
        return torch.cat(mins, dim=0)

    @staticmethod
    def _estimate_uv_mask_tol(
        uv_face: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
        fallback: float = 0.05,
        scale: float = 2.5,
        max_points: int = 2048,
        chunk_size: int = 512,
    ) -> float:
        if uv_face.shape[0] < 2:
            return float(fallback)

        uv_cpu = uv_face.detach().to(device="cpu")
        n = uv_cpu.shape[0]
        if n > max_points:
            sample_idx = torch.linspace(0, n - 1, max_points).round().long()
            uv_cpu = uv_cpu[sample_idx]
            n = uv_cpu.shape[0]

        min_vals = []
        for start in range(0, n, chunk_size):
            q = uv_cpu[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_cpu.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)

            dist = torch.norm(diff, dim=-1)
            rows = q.shape[0]
            dist[torch.arange(rows), start:start + rows] = float("inf")
            min_vals.append(dist.min(dim=1).values)

        spacing = torch.cat(min_vals, dim=0).median()
        if not torch.isfinite(spacing):
            return float(fallback)
        return float(max(scale * float(spacing.item()), 1e-6))

    def _seed_domain_mask_for_face(self, ft):
        cfg = self.cfg
        if not bool(cfg.use_seed_domain_mask):
            return None
        cached = ft.get("_seed_domain_mask_callable", None)
        if cached is not None:
            return cached
        mask_grid = ft.get("seed_domain_mask_grid", None)
        if mask_grid is not None:
            return mask_grid

        uv_face = ft.get("seed_domain_uv_support", ft["uv"])
        if uv_face.numel() == 0:
            return None

        uv_support = uv_face.detach()
        max_points = int(cfg.seed_domain_mask_max_points)
        if uv_support.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                uv_support.shape[0] - 1,
                max_points,
                device=uv_support.device,
            ).round().to(torch.long)
            uv_support = uv_support[sample_idx]

        sigma_value = ft.get("seed_domain_sigma", None)
        if sigma_value is None:
            sigma = self._estimate_uv_mask_tol(
                uv_support,
                u_periodic=bool(ft.get("u_periodic", False)),
                v_periodic=bool(ft.get("v_periodic", False)),
                fallback=float(cfg.boundary_margin),
                scale=float(cfg.seed_domain_mask_support_scale),
            )
        elif torch.is_tensor(sigma_value):
            sigma = float(sigma_value.detach().cpu().item())
        else:
            sigma = float(sigma_value)
        sigma = max(float(sigma), float(cfg.eps))
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        def mask_fn(seeds):
            support = uv_support.to(device=seeds.device, dtype=seeds.dtype)
            diff = seeds.unsqueeze(1) - support.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            dmin = torch.norm(diff, dim=-1).amin(dim=1)
            sigma_t = torch.as_tensor(sigma, device=seeds.device, dtype=seeds.dtype)
            return torch.exp(-0.5 * (dmin / sigma_t.clamp_min(cfg.eps)).pow(2))

        ft["_seed_domain_mask_callable"] = mask_fn
        return mask_fn

    def build_timelapse_render_cache(
        self,
        face_tensors,
    ):
        cache = []

        for ft in face_tensors:
            device = ft["uv"].device
            uv_dense = ft["uv"]
            xyz_dense = ft["points_xyz"]
            Xu_dense = ft["Xu"]
            Xv_dense = ft["Xv"]


            cache.append({
                "face_id": self._face_id_key(ft.get("face_id", 0)),
                "uv_dense": uv_dense,
                "xyz_dense": xyz_dense,
                "points_xyz": xyz_dense,
                "Xu_dense": Xu_dense,
                "Xv_dense": Xv_dense,
                "Xu": Xu_dense,
                "Xv": Xv_dense,
                "seed_domain_mask": self._seed_domain_mask_for_face(ft),
                "faces_ijk": ft["faces_ijk"],
            })

        return cache

    def evaluate_cached_face_fields(self, render_cache, decoder, pred):
        decoder_out = decoder(
            seeds_uv=pred["seeds_raw"],
            generate_density_fiber=getattr(self.cfg, "generate_decoder_density_fiber", True),
        )
        self._copy_activation_metadata_to_pred(pred, decoder_out)

        if getattr(self.cfg, "generate_decoder_density_fiber", True):
            decoder_out = apply_density_postprocess_to_output(
                decoder_out,
                render_cache,
                self.cfg,
                return_debug=False,
            )
            rho_dense = decoder_out["rho"]
            rho_raw_decoder_dense = decoder_out["rho_raw_decoder"]
            rho_postprocessed_dense = decoder_out["rho_postprocessed"]
            fiber3d_dense = decoder_out["fiber3d"]
        else:
            rho_dense, fiber3d_dense, _ = self.neutral_density_fiber_fields(
                render_cache["uv_dense"],
                render_cache.get("Xu_dense", None),
            )
            rho_raw_decoder_dense = rho_dense
            rho_postprocessed_dense = rho_dense

        return {
            "xyz_dense": render_cache["xyz_dense"],
            "rho_dense": rho_dense,
            "rho_raw_decoder_dense": rho_raw_decoder_dense,
            "rho_postprocessed_dense": rho_postprocessed_dense,
            "fiber3d_dense": fiber3d_dense,
            "seeds_uv": decoder_out.get("seeds_uv", decoder_out.get("seeds", None)),
            "topology_seeds_uv": decoder_out.get("topology_seeds_uv", decoder_out.get("seeds_uv", decoder_out.get("seeds", None))),
            "seed_active_mask": decoder_out.get("seed_active_mask", None),
            "active_seed_ids": decoder_out.get("active_seed_ids", None),
            "seed_activity_weight": decoder_out.get("seed_activity_weight", None),
            "seed_box_activity_weight": decoder_out.get("seed_box_activity_weight", None),
            "seed_domain_activity_weight": decoder_out.get("seed_domain_activity_weight", None),
            "seed_duplicate_activity_weight": decoder_out.get("seed_duplicate_activity_weight", None),
            "seeds_xyz": decoder_out.get("seeds_xyz", None),
            "edge_curves_uv": decoder_out.get("edge_curves_uv", None),
            "edge_curves_xyz": decoder_out.get("edge_curves_xyz", None),
            "graph": decoder_out.get("graph", None),
            "faces_ijk": render_cache["faces_ijk"],
        }

    @staticmethod
    def _concat_polydata(meshes, scalar_name=None):
        if len(meshes) == 0:
            return None

        pts_parts = []
        face_parts = []
        scalar_parts = []
        offset = 0

        for mesh in meshes:
            pts = np.asarray(mesh.points, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4).copy()
            faces[:, 1:] += offset
            pts_parts.append(pts)
            face_parts.append(faces.reshape(-1))
            if scalar_name is not None:
                scalar_parts.append(np.asarray(mesh[scalar_name], dtype=np.float32))
            offset += pts.shape[0]

        out = pv.PolyData(
            np.concatenate(pts_parts, axis=0),
            np.concatenate(face_parts, axis=0),
        )
        if scalar_name is not None and len(scalar_parts) > 0:
            out[scalar_name] = np.concatenate(scalar_parts, axis=0)
        return out

    @staticmethod
    def _curves_xyz_to_polydata(curves_xyz):
        if curves_xyz is None:
            return None
        if torch.is_tensor(curves_xyz):
            curves_xyz = curves_xyz.detach().cpu().numpy()
        curves_xyz = np.asarray(curves_xyz, dtype=np.float32)
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3 or curves_xyz.shape[0] == 0 or curves_xyz.shape[1] < 2:
            return None
        points = curves_xyz.reshape(-1, 3)
        lines = []
        samples = curves_xyz.shape[1]
        for edge_id in range(curves_xyz.shape[0]):
            base = edge_id * samples
            for j in range(samples - 1):
                lines.extend([2, base + j, base + j + 1])
        return pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))

    @staticmethod
    def _concat_line_polydata(meshes):
        if len(meshes) == 0:
            return None
        point_parts = []
        line_parts = []
        offset = 0
        for mesh in meshes:
            points = np.asarray(mesh.points, dtype=np.float32)
            lines = np.asarray(mesh.lines, dtype=np.int64).copy()
            cursor = 0
            while cursor < lines.size:
                count = int(lines[cursor])
                lines[cursor + 1:cursor + 1 + count] += offset
                cursor += count + 1
            point_parts.append(points)
            line_parts.append(lines)
            offset += points.shape[0]
        return pv.PolyData(
            np.concatenate(point_parts, axis=0),
            lines=np.concatenate(line_parts, axis=0),
        )

    @staticmethod
    def _composite_to_white(img):
        if img.ndim != 3:
            return img
        if img.shape[2] == 3:
            return img
        if img.shape[2] != 4:
            return img[..., :3]

        rgb = img[..., :3].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32) / 255.0)
        white = np.full_like(rgb, 255.0)
        out = rgb * alpha + white * (1.0 - alpha)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _render_offscreen_plotter(plotter, view_name):
        tight_view = None
        if view_name == "xy":
            plotter.enable_parallel_projection()
            plotter.view_xy()
            tight_view = "xy"
        elif view_name == "xz":
            plotter.enable_parallel_projection()
            plotter.view_xz()
            tight_view = "xz"
        elif view_name == "yz":
            plotter.enable_parallel_projection()
            plotter.view_yz()
            tight_view = "yz"
        else:
            plotter.disable_parallel_projection()
            plotter.view_isometric()
            tight_view = None

        plotter.reset_camera()
        if tight_view is not None:
            try:
                plotter.camera.tight(view=tight_view, adjust_render_window=False)
            except Exception:
                pass
            try:
                plotter.camera.zoom(0.90)
            except Exception:
                pass
        else:
            try:
                plotter.camera.zoom(0.94)
            except Exception:
                pass
        img = plotter.screenshot(return_img=True, transparent_background=False)
        return NN_Trainer._composite_to_white(img)

    @staticmethod
    def _add_image_title(img, title, pad=10, band_height=42, font_scale=0.72, thickness=2):
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        title_band = np.full((band_height, img.shape[1], 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(title, font, font_scale, thickness)
        x = max(pad, (img.shape[1] - text_size[0]) // 2)
        y = max(pad + text_size[1], (band_height + text_size[1]) // 2 - baseline)
        cv2.putText(
            title_band,
            title,
            (x, y),
            font,
            font_scale,
            (32, 32, 32),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return np.vstack([title_band, img])

    @staticmethod
    def _add_panel_border(img, pad=10, border=2, bg_color=(255, 255, 255), border_color=(180, 186, 195)):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        inner = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )
        return cv2.copyMakeBorder(
            inner,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

    @staticmethod
    def _pad_to_size(img, target_h=None, target_w=None, bg_color=(255, 255, 255)):
        h, w = img.shape[:2]
        if target_h is None:
            target_h = h
        if target_w is None:
            target_w = w
        if h == target_h and w == target_w:
            return img

        top = 0
        bottom = max(0, target_h - h)
        left = max(0, (target_w - w) // 2)
        right = max(0, target_w - w - left)
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )

    @staticmethod
    def _resize_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        target_h = max(1, int(round(h * (target_w / w))))
        return cv2.resize(img, (target_w, target_h))

    @staticmethod
    def _equalize_row_heights(images):
        target_h = min(img.shape[0] for img in images)
        out = []
        for img in images:
            h, w = img.shape[:2]
            if h == target_h:
                out.append(img)
            else:
                target_w = max(1, int(round(w * (target_h / h))))
                out.append(cv2.resize(img, (target_w, target_h)))
        return out

    @staticmethod
    def _stack_row_with_gaps(images, gap=18, bg_color=(255, 255, 255)):
        images = NN_Trainer._equalize_row_heights(images)
        if len(images) == 1:
            return images[0]
        gap_tile = np.full((images[0].shape[0], gap, 3), bg_color, dtype=np.uint8)
        parts = []
        for i, img in enumerate(images):
            parts.append(img)
            if i != len(images) - 1:
                parts.append(gap_tile)
        return np.hstack(parts)

    @staticmethod
    def _center_row_to_width(images, target_w, gap=18, bg_color=(255, 255, 255)):
        row = NN_Trainer._stack_row_with_gaps(images, gap=gap, bg_color=bg_color)
        if row.shape[1] > target_w:
            row = NN_Trainer._resize_to_width(row, target_w)
        return NN_Trainer._pad_to_size(row, target_w=target_w, bg_color=bg_color)

    @staticmethod
    def _clip_segment_to_uv_box_np(p0, p1, tol=1e-12):
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        if p0.shape != (2,) or p1.shape != (2,) or not np.isfinite([*p0, *p1]).all():
            return None
        delta = p1 - p0
        t_enter, t_exit = 0.0, 1.0
        for p, q in (
            (-delta[0], p0[0]),
            (delta[0], 1.0 - p0[0]),
            (-delta[1], p0[1]),
            (delta[1], 1.0 - p0[1]),
        ):
            if abs(float(p)) <= tol:
                if float(q) < -tol:
                    return None
                continue
            ratio = float(q / p)
            if p < 0.0:
                t_enter = max(t_enter, ratio)
            else:
                t_exit = min(t_exit, ratio)
            if t_enter > t_exit + tol:
                return None
        return (
            np.clip(p0 + t_enter * delta, 0.0, 1.0),
            np.clip(p0 + t_exit * delta, 0.0, 1.0),
        )

    def _render_current_cad_frame_cached(
        self,
        seeds_list,
        decoders,
        pred_list,
        render_cache,
        thr=0.5,
        loading_img=None,
    ):


        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in pred_list}
        dec_by_face_id = {self._face_id_key(ft.get("face_id", 0)): dec for ft, dec in zip(self.current_face_tensors, decoders)}
        density_meshes = []
        solid_meshes = []
        curve_meshes = []
        fiber_xyz_parts = []
        fiber_vec_parts = []
        fiber_rho_parts = []

        for cache_i in render_cache:
            face_id = cache_i["face_id"]
            pred = pred_by_face_id[face_id]
            decoder = dec_by_face_id[face_id]

            out = self.evaluate_cached_face_fields(cache_i, decoder, pred)

            xyz = out["xyz_dense"].detach().cpu().numpy()
            rho_dense = out["rho_dense"].detach().cpu().numpy()
            fiber_dense = out["fiber3d_dense"].detach().cpu().numpy()
            faces_local = out["faces_ijk"].detach().cpu().numpy().astype(np.int64)
            curve_mesh = self._curves_xyz_to_polydata(out.get("edge_curves_xyz", None))
            if curve_mesh is not None:
                curve_meshes.append(curve_mesh)
            if faces_local.size > 0:
                pv_faces_all = np.empty((faces_local.shape[0], 4), dtype=np.int64)
                pv_faces_all[:, 0] = 3
                pv_faces_all[:, 1:] = faces_local
                mesh_all = pv.PolyData(xyz, pv_faces_all.reshape(-1))
                mesh_all["rho"] = rho_dense.astype(np.float32)
                density_meshes.append(mesh_all)

            if faces_local.size > 0:
                solid_keep = np.mean(rho_dense[faces_local], axis=1) >= float(thr)
                faces_solid_local = faces_local[solid_keep]
                if faces_solid_local.size > 0:
                    pv_faces_solid = np.empty((faces_solid_local.shape[0], 4), dtype=np.int64)
                    pv_faces_solid[:, 0] = 3
                    pv_faces_solid[:, 1:] = faces_solid_local
                    solid_meshes.append(pv.PolyData(xyz, pv_faces_solid.reshape(-1)))

            valid_fiber = np.isfinite(rho_dense)
            valid_fiber &= np.isfinite(fiber_dense).all(axis=1)
            valid_fiber &= (rho_dense >= float(thr))
            valid_fiber &= (np.linalg.norm(fiber_dense, axis=1) > 1e-10)
            if np.any(valid_fiber):
                fiber_xyz_parts.append(xyz[valid_fiber])
                fiber_vec_parts.append(fiber_dense[valid_fiber])
                fiber_rho_parts.append(rho_dense[valid_fiber])

        if density_meshes:
            rho_all = np.concatenate([m["rho"] for m in density_meshes], axis=0)
            rho_clim = [0.0, max(1.0, float(np.quantile(rho_all, 0.995)))]
        else:
            rho_clim = [0.0, 1.0]

        density_mesh_merged = self._concat_polydata(density_meshes, scalar_name="rho")
        solid_mesh_merged = self._concat_polydata(solid_meshes, scalar_name=None)
        curve_mesh_merged = self._concat_line_polydata(curve_meshes) if curve_meshes else None

        all_points = []
        for mesh in density_meshes:
            all_points.append(np.asarray(mesh.points))
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            diag = float(np.linalg.norm(np.ptp(all_points, axis=0)))
        else:
            diag = 1.0
        arrow_scale = 0.04 * max(diag, 1e-6)

        fiber_points = None
        fiber_vectors = None
        fiber_rho = None
        if fiber_xyz_parts:
            fiber_points = np.concatenate(fiber_xyz_parts, axis=0).astype(np.float32)
            fiber_vectors = np.concatenate(fiber_vec_parts, axis=0).astype(np.float32)
            fiber_rho = np.concatenate(fiber_rho_parts, axis=0).astype(np.float32)
            max_arrows = 600
            if fiber_points.shape[0] > max_arrows:
                stride = int(np.ceil(fiber_points.shape[0] / max_arrows))
                fiber_points = fiber_points[::stride]
                fiber_vectors = fiber_vectors[::stride]
                fiber_rho = fiber_rho[::stride]

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
            seeds_list=seeds_list,
            pred_list=pred_list,
            face_tensors=self.current_face_tensors,
        )
        active_seed_points = seed_vis["xyz_active"]
        inactive_seed_points = seed_vis["xyz_inactive"]
        seed_point_size = max(6.0, 0.006 * max(diag, 1.0) * 100.0)
        show_seed_points = True
        show_axes_widget = True

        first_face_voronoi_img = None
        first_face_graph_img = None
        first_face_core_curves_img = None
        if render_cache:
            first_cache = render_cache[0]
            first_face_id = first_cache["face_id"]
            first_out = self.evaluate_cached_face_fields(
                first_cache,
                dec_by_face_id[first_face_id],
                pred_by_face_id[first_face_id],
            )
            first_seed_idx = 0
            for idx, ft in enumerate(self.current_face_tensors):
                if self._face_id_key(ft.get("face_id", 0)) == first_face_id:
                    first_seed_idx = idx
                    break
            first_face_voronoi_img = self._render_first_face_density_2d(
                cache_i=first_cache,
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                pred_i=pred_by_face_id[first_face_id],
                window_size=(1050, 1050),
                show_scipy_voronoi=True,
                show_core_curves=False,
            )
            first_face_graph_img = self._render_first_face_generated_graph_2d(
                decoder=dec_by_face_id[first_face_id],
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                window_size=(1050, 1050),
                show_node_ids=False,
                show_edge_ids=False,
            )
            first_face_core_curves_img = self._render_first_face_density_2d(
                cache_i=first_cache,
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                pred_i=pred_by_face_id[first_face_id],
                window_size=(1050, 1050),
                show_scipy_voronoi=False,
                show_core_curves=True,
            )

        def make_plotter(title, mode, window_size):
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            pl.set_background("white")
            try:
                pl.disable_anti_aliasing()
            except Exception:
                pass
            try:
                pl.ren_win.SetMultiSamples(0)
            except Exception:
                pass
            pl.remove_all_lights()

            if mode == "density":
                if density_mesh_merged is not None:
                    pl.add_mesh(
                        density_mesh_merged,
                        scalars="rho",
                        cmap="viridis",
                        clim=rho_clim,
                        show_edges=False,
                        lighting=False,
                        smooth_shading=False,
                        nan_color="white",
                        interpolate_before_map=False,
                        scalar_bar_args={
                            "title": "rho",
                            "position_x": 0.28,
                            "position_y": 0.02,
                            "width": 0.64,
                            "height": 0.05,
                            "title_font_size": 12,
                            "label_font_size": 10,
                            "color": "#4b5563",
                            "fmt": "%.2f",
                            "n_labels": 5,
                        },
                    )
                if curve_mesh_merged is not None:
                    pl.add_mesh(curve_mesh_merged, color="black", line_width=3, render_lines_as_tubes=True)
            elif mode == "solid":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#8ecae6",
                        smooth_shading=False,
                        specular=0.0,
                        show_edges=False,
                        lighting=False,
                    )
            elif mode == "fiber":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#dbeafe",
                        opacity=1.0,
                        smooth_shading=False,
                        show_edges=False,
                        lighting=False,
                    )
                if fiber_points is not None and fiber_points.shape[0] > 0:
                    cloud = pv.PolyData(fiber_points)
                    cloud["vectors"] = fiber_vectors
                    cloud["rho"] = fiber_rho
                    glyphs = cloud.glyph(
                        orient="vectors",
                        scale=False,
                        factor=arrow_scale,
                        geom=pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0)),
                    )
                    pl.add_mesh(glyphs, color="#1d4ed8", line_width=2)
                if curve_mesh_merged is not None:
                    pl.add_mesh(curve_mesh_merged, color="black", line_width=3, render_lines_as_tubes=True)

            if show_seed_points and active_seed_points is not None and len(active_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(active_seed_points.astype(np.float32)),
                    color="red",
                    render_points_as_spheres=True,
                    point_size=seed_point_size,
                )
            if show_seed_points and inactive_seed_points is not None and len(inactive_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(inactive_seed_points.astype(np.float32)),
                    color="gray",
                    opacity=0.35,
                    render_points_as_spheres=True,
                    point_size=max(5.0, 0.8 * seed_point_size),
                )
            if show_axes_widget:
                pl.show_axes()
            return pl

        top_specs = [
            ("3D Heaviside Material | Front View", "solid", "xz"),
            ("3D Heaviside Material | Side View", "solid", "yz"),
            ("3D Heaviside Material | Top View", "solid", "xy"),
        ]
        perspective_spec = ("3D Heaviside Material | Perspective View", "solid", "iso")
        top_window_size = (560, 430)
        bottom_window_size = (1050, 1050)

        top_imgs = []
        if loading_img is not None:
            loading_panel_img = cv2.resize(
                loading_img,
                top_window_size,
                interpolation=cv2.INTER_AREA if loading_img.shape[1] > top_window_size[0] else cv2.INTER_CUBIC,
            )
            top_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        loading_panel_img,
                        "Voxel Loading And Boundary Conditions",
                    )
                )
            )
        for title, mode, view in top_specs:
            pl = make_plotter(title, mode, window_size=top_window_size)
            img = self._render_offscreen_plotter(pl, view)
            top_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
            pl.close()

        bottom_imgs = []
        if first_face_voronoi_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_voronoi_img,
                        "Exact SciPy Voronoi"
                        )
                )
            )
        if first_face_graph_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_graph_img,
                        "Connectivity Graph"
                    )
                )
            )
        if first_face_core_curves_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_core_curves_img,
                        "Core Curves UV"
                        )
                )
            )
        title, mode, view = perspective_spec
        pl = make_plotter(title, mode, window_size=bottom_window_size)
        img = self._render_offscreen_plotter(pl, view)
        bottom_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
        pl.close()

        col_gap = 22
        row_gap = 28
        top_row = self._stack_row_with_gaps(top_imgs, gap=col_gap)
        bottom_row = self._center_row_to_width(bottom_imgs, target_w=top_row.shape[1], gap=col_gap)
        gap_tile = np.full((row_gap, top_row.shape[1], 3), 255, dtype=np.uint8)
        cad_panel = np.vstack([top_row, gap_tile, bottom_row])
        cad_panel = cv2.copyMakeBorder(
            cad_panel,
            16,
            16,
            16,
            16,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        return cad_panel

    def _render_current_3d_tube_frame_cached(
        self,
        seeds_list,
        decoders,
        pred_list,
        render_cache,
        loading_img=None,
        fem_density_field=None,
        fem_stress_field=None,
        fem_displacement_field=None,
        history_rows=None,
    ):
        import numpy as np
        import pyvista as pv

        tube_meshes = []
        seed_meshes = []
        first_face_voronoi_img = None
        first_face_graph_img = None
        first_face_core_curves_img = None

        for face_i, (decoder, pred, cache) in enumerate(zip(decoders, pred_list, render_cache)):
            fields = self.evaluate_cached_face_fields(cache, decoder, pred)

            if first_face_voronoi_img is None:
                first_face_voronoi_img = self._render_first_face_density_2d(
                    cache_i=cache,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    pred_i=pred,
                    window_size=(650, 650),
                    show_scipy_voronoi=True,
                    show_core_curves=False,
                )
                first_face_graph_img = self._render_first_face_generated_graph_2d(
                    decoder=decoder,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    window_size=(650, 650),
                    show_node_ids=False,
                    show_edge_ids=False,
                )
                first_face_core_curves_img = self._render_first_face_density_2d(
                    cache_i=cache,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    pred_i=pred,
                    window_size=(650, 650),
                    show_scipy_voronoi=False,
                    show_core_curves=True,
                )

            curves = fields.get("edge_curves_xyz", None)
            if curves is None:
                continue

            if torch.is_tensor(curves):
                curves_np = curves.detach().cpu().numpy()
            else:
                curves_np = np.asarray(curves)

            edge_colors = {
                0: "black",
                1: "orange",
                2: "gray",
                3: "orange",
                4: "cyan",
            }
            edge_types_np = None
            graph = fields.get("graph", None)
            if isinstance(graph, dict):
                edge_types = graph.get("edge_type", None)
                if torch.is_tensor(edge_types):
                    edge_types_np = edge_types.detach().cpu().numpy()
                elif edge_types is not None:
                    edge_types_np = np.asarray(edge_types)

            radius = pred.get("centerline_radius", None)
            if radius is None:
                radius = 0.01
            elif torch.is_tensor(radius):
                radius = float(radius.detach().mean().cpu().item())
            else:
                radius = float(radius)

            radius = max(
                radius * float(getattr(self.cfg, "timelapse_tube_radius_scale", 1.0)),
                1e-4,
            )

            for edge_id, points in enumerate(curves_np):
                if points.shape[0] < 2 or not np.isfinite(points).all():
                    continue

                polyline = pv.PolyData(points.astype(np.float32))
                polyline.lines = np.concatenate(
                    ([len(points)], np.arange(len(points)))
                ).astype(np.int64)

                tube = polyline.tube(
                    radius=radius,
                    n_sides=int(getattr(self.cfg, "timelapse_tube_n_sides", 12)),
                )
                if tube.n_points > 0:
                    if edge_types_np is not None and edge_id < len(edge_types_np):
                        color = edge_colors.get(int(edge_types_np[edge_id]), "gray")
                    else:
                        color = "orange"
                    tube_meshes.append((tube, color))

            seeds_xyz = fields.get("seeds_xyz", None)
            if seeds_xyz is not None:
                if torch.is_tensor(seeds_xyz):
                    seeds_xyz = seeds_xyz.detach().cpu().numpy()
                seeds_xyz = np.asarray(seeds_xyz, dtype=np.float32)
                if seeds_xyz.ndim == 2 and seeds_xyz.shape[0] > 0 and seeds_xyz.shape[1] == 3:
                    finite_seed = np.isfinite(seeds_xyz).all(axis=1)
                    if np.any(finite_seed):
                        seed_mesh = pv.PolyData(seeds_xyz[finite_seed])
                        if seed_mesh.n_points > 0:
                            seed_meshes.append(seed_mesh)

        plotter = pv.Plotter(off_screen=True, window_size=(900, 650))
        plotter.set_background("white")

        for mesh, color in tube_meshes:
            if mesh.n_points > 0:
                plotter.add_mesh(mesh, color=color, smooth_shading=True)

        for sm in seed_meshes:
            if sm.n_points > 0:
                plotter.add_mesh(
                    sm,
                    color="red",
                    point_size=8,
                    render_points_as_spheres=True,
                )

        if not tube_meshes and not seed_meshes:
            plotter.add_text("No 3D tube curves", color="black", font_size=14)

        plotter.view_isometric()
        plotter.reset_camera()
        img = plotter.screenshot(return_img=True)
        plotter.close()

        tube_img = self._add_panel_border(
            self._add_image_title(
                self._composite_to_white(img),
                "3D Voronoi Tube Curves",
            )
        )

        def _field_to_vtk_cell_order(field):
            if field is None or self.shell_problem is None:
                return None
            if torch.is_tensor(field):
                field = field.detach().cpu().numpy()
            field = np.asarray(field, dtype=np.float32).reshape(-1)
            mesh_cfg = getattr(self.shell_problem, "mesh", None)
            if mesh_cfg is None or getattr(self.shell_problem, "grid_geom", None) is None:
                return None
            nelx, nely, nelz = int(mesh_cfg["nelx"]), int(mesh_cfg["nely"]), int(mesh_cfg["nelz"])
            if field.size != nelx * nely * nelz:
                return None
            return field.reshape((nelz, nelx, nely)).transpose(1, 2, 0).ravel(order="F")

        def _occupied_cell_mask_vtk():
            if self.shell_problem is None or getattr(self.shell_problem, "elem_occupancy", None) is None:
                return None
            occ = np.asarray(self.shell_problem.elem_occupancy, dtype=np.uint8)
            return occ.transpose(1, 2, 0).ravel(order="F")

        def _render_fem_cell_field(
            field,
            title,
            scalar_name,
            cmap,
            clim=None,
            window_size=(900, 650),
            density_mask_field=None,
            density_threshold=None,
            colorbar_label=None,
            title_font_scale=0.86,
        ):
            values = _field_to_vtk_cell_order(field)
            if values is None:
                return None
            density_values = _field_to_vtk_cell_order(density_mask_field)
            mesh_cfg = self.shell_problem.mesh
            grid_geom = self.shell_problem.grid_geom
            nelx, nely, nelz = int(mesh_cfg["nelx"]), int(mesh_cfg["nely"]), int(mesh_cfg["nelz"])
            grid_cls = getattr(pv, "ImageData", None)
            if grid_cls is None:
                grid_cls = pv.UniformGrid
            grid = grid_cls(
                dimensions=(nelx + 1, nely + 1, nelz + 1),
                spacing=(float(grid_geom["hx"]), float(grid_geom["hy"]), float(grid_geom["hz"])),
                origin=(float(grid_geom["xmin"]), float(grid_geom["ymin"]), float(grid_geom["zmin"])),
            )
            grid.cell_data[scalar_name] = values
            occ = _occupied_cell_mask_vtk()
            visible = np.ones(values.shape, dtype=np.uint8)
            if occ is not None and occ.size == values.size:
                visible &= occ.astype(np.uint8)
            if density_values is not None and density_values.size == values.size and density_threshold is not None:
                visible &= (density_values >= float(density_threshold)).astype(np.uint8)
            if visible.size == values.size:
                grid.cell_data["visible"] = visible
                mesh = grid.threshold(value=0.5, scalars="visible")
            else:
                mesh = grid
            if mesh.n_cells <= 0:
                return None

            range_values = values
            if visible.size == values.size:
                range_values = values[visible.astype(bool)]
            finite = range_values[np.isfinite(range_values)]
            if clim is None:
                if finite.size > 0:
                    vmax = float(np.quantile(finite, 0.98))
                    if vmax <= 0.0:
                        vmax = float(np.max(finite)) if finite.size else 1.0
                    clim = [0.0, max(vmax, 1e-8)]
                else:
                    clim = [0.0, 1.0]

            pl = pv.Plotter(off_screen=True, window_size=window_size)
            pl.set_background("white")
            try:
                pl.disable_anti_aliasing()
                pl.ren_win.SetMultiSamples(0)
            except Exception:
                pass
            pl.add_mesh(
                mesh,
                scalars=scalar_name,
                cmap=cmap,
                clim=clim,
                show_edges=False,
                lighting=False,
                smooth_shading=False,
                nan_color="white",
                scalar_bar_args={
                    "title": colorbar_label or scalar_name,
                    "position_x": 0.08,
                    "position_y": 0.02,
                    "width": 0.86,
                    "height": 0.14,
                    "title_font_size": 34,
                    "label_font_size": 30,
                    "fmt": "%.2g",
                    "n_labels": 5,
                },
            )
            pl.view_isometric()
            pl.reset_camera()
            try:
                pl.camera.zoom(0.94)
            except Exception:
                pass
            field_img = self._composite_to_white(pl.screenshot(return_img=True, transparent_background=False))
            pl.close()
            return self._add_panel_border(
                self._add_image_title(
                    field_img,
                    title,
                    band_height=54,
                    font_scale=title_font_scale,
                    thickness=2,
                )
            )

        def _history_cap(key, default=1.0):
            vals = []
            for row in list(history_rows or []):
                try:
                    value = float(row.get(key, float("nan")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    vals.append(value)
            if not vals:
                return float(default)
            return max(max(vals), 1e-8)

        stress_threshold = getattr(self.cfg, "fem_yield_strength", None)
        disp_threshold = getattr(self.cfg, "fem_max_displacement", None)
        stress_vmax = (
            float(stress_threshold)
            if stress_threshold is not None and math.isfinite(float(stress_threshold)) and float(stress_threshold) > 0.0
            else _history_cap("stress_max", default=_history_cap("fem_stress_max", default=1.0))
        )
        disp_vmax = (
            float(disp_threshold)
            if disp_threshold is not None and math.isfinite(float(disp_threshold)) and float(disp_threshold) > 0.0
            else _history_cap("disp_field_max", default=_history_cap("disp_max", default=1.0))
        )
        stress_normalize = Normalize(vmin=0.0, vmax=stress_vmax, clip=True)
        disp_normalize = Normalize(vmin=0.0, vmax=disp_vmax, clip=True)

        density_img = _render_fem_cell_field(
            fem_density_field,
            "3D FEM Density Distribution",
            "density",
            "viridis",
            clim=[0.0, 1.0],
            window_size=(1150, 820),
            colorbar_label="density",
        )
        stress_img = _render_fem_cell_field(
            fem_stress_field,
            "FEM Stress Diagnostic (Material Only)",
            "stress",
            "turbo",
            clim=[float(stress_normalize.vmin), float(stress_normalize.vmax)],
            window_size=(1150, 820),
            density_mask_field=fem_density_field,
            density_threshold=max(float(getattr(self.cfg, "vis_thr", 0.5)), float(getattr(self.cfg, "fem_rho_min_ratio", 1.0e-5)) * 1.05),
            colorbar_label="stress",
        )
        displacement_img = _render_fem_cell_field(
            fem_displacement_field,
            "FEM Displacement in Load Direction",
            "u_load",
            "plasma",
            clim=[float(disp_normalize.vmin), float(disp_normalize.vmax)],
            window_size=(1150, 820),
            density_mask_field=fem_density_field,
            density_threshold=max(float(getattr(self.cfg, "vis_thr", 0.5)), float(getattr(self.cfg, "fem_rho_min_ratio", 1.0e-5)) * 1.05),
            colorbar_label="u_load",
        )

        def _history_ylim(keys, log_y=False):
            vals = []
            for row in list(history_rows or []):
                for key in keys:
                    try:
                        value = float(row.get(key, float("nan")))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and (not log_y or value > 0.0):
                        vals.append(value)
            if not vals:
                return None
            vmin = min(vals)
            vmax = max(vals)
            if log_y:
                return [max(vmin * 0.9, 1e-12), max(vmax * 1.1, 1e-12)]
            if abs(vmax - vmin) < 1e-12:
                pad = max(abs(vmax) * 0.05, 0.05)
                return [vmin - pad, vmax + pad]
            pad = 0.06 * (vmax - vmin)
            return [vmin - pad, vmax + pad]

        def _render_history_plot(history_rows, keys, title, ylabel, colors=None, log_y=False, ylim=None, window_size=(900, 360)):
            width, height = int(window_size[0]), int(window_size[1])
            rows = list(history_rows or [])
            fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
            ax = fig.add_subplot(111)
            ax.set_facecolor("white")
            colors = colors or ["#2563eb", "#dc2626", "#16a34a"]
            plotted = False
            for idx, key in enumerate(keys):
                xs = []
                ys = []
                for row in rows:
                    if key not in row:
                        continue
                    value = row.get(key)
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value):
                        continue
                    if log_y and value <= 0.0:
                        continue
                    xs.append(float(row.get("step", len(xs))))
                    ys.append(value)
                if xs:
                    label = key.replace("loss_fem", "L_FEM").replace("_", " ")
                    ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=2.4, label=label)
                    plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No history yet", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            if log_y and plotted:
                ax.set_yscale("log")
            if plotted and ylim is not None:
                ax.set_ylim(ylim)
            ax.set_title(title, fontsize=18, weight="bold", pad=8)
            ax.set_xlabel("Iteration", fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.75)
            ax.tick_params(axis="both", labelsize=11)
            if plotted and len(keys) > 1:
                ax.legend(loc="best", fontsize=10, frameon=True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout(pad=1.2)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3].copy()
            plt.close(fig)
            return self._add_panel_border(img)

        normalized_history_rows = []
        total_loss_baseline = None
        for row in list(history_rows or []):
            try:
                value = float(row.get("L_total", float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value) and abs(value) > 1e-12:
                total_loss_baseline = abs(value)
                break
        for row in list(history_rows or []):
            plot_row = dict(row)
            if total_loss_baseline is not None:
                try:
                    value = float(row.get("L_total", float("nan")))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    plot_row["L_total_norm"] = value / total_loss_baseline
            normalized_history_rows.append(plot_row)

        total_loss_hist_img = _render_history_plot(
            normalized_history_rows,
            keys=["L_total_norm"],
            title="Normalized Total Loss History",
            ylabel="Normalized Total Loss",
            colors=["#dc2626"],
            ylim=[0.0, 2.0],
        )
        fem_hist_img = _render_history_plot(
            normalized_history_rows,
            keys=["L_FEM_norm"],
            title="Normalized FEM Loss History",
            ylabel="Normalized FEM Loss",
            colors=["#2563eb"],
            ylim=[0.0, 2.0],
        )
        fiber_length_hist_img = _render_history_plot(
            normalized_history_rows,
            keys=["loss_total_fiber_length_norm"],
            title="Normalized Fiber Length Diagnostic",
            ylabel="Normalized Fiber Length Diagnostic",
            colors=["#0891b2"],
            ylim=[0.0, 2.0],
        )

        if first_face_voronoi_img is None or first_face_graph_img is None or first_face_core_curves_img is None:
            panels = [p for p in [loading_img, tube_img, density_img, stress_img, displacement_img] if p is not None]
            return self._stack_row_with_gaps(panels, gap=22) if panels else tube_img

        voronoi_img = self._add_panel_border(
            self._add_image_title(
                first_face_voronoi_img,
                "Exact SciPy Voronoi",
            )
        )
        graph_img = self._add_panel_border(
            self._add_image_title(
                first_face_graph_img,
                "Connectivity Graph",
            )
        )
        core_curves_img = self._add_panel_border(
            self._add_image_title(
                first_face_core_curves_img,
                "Core Curves UV",
            )
        )
        def _fit_panel_to_box(img, width, height):
            h, w = img.shape[:2]
            scale = min(float(width) / max(float(w), 1.0), float(height) / max(float(h), 1.0))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
            out = np.full((int(height), int(width), 3), 255, dtype=np.uint8)
            y0 = max((int(height) - new_h) // 2, 0)
            x0 = max((int(width) - new_w) // 2, 0)
            out[y0:y0 + new_h, x0:x0 + new_w] = resized[..., :3]
            return out

        generate_density_fiber = bool(getattr(self.cfg, "generate_decoder_density_fiber", True))

        top_panels = []
        if loading_img is not None:
            loading_panel = cv2.resize(
                loading_img,
                (520, 300),
                interpolation=cv2.INTER_AREA if loading_img.shape[1] > 520 else cv2.INTER_CUBIC,
            )
            top_panels.append(
                self._add_panel_border(
                    self._add_image_title(loading_panel, "Loading And Boundary Conditions")
                )
            )
        if generate_density_fiber:
            if loading_img is not None:
                top_panels.extend([graph_img, tube_img])
            else:
                top_panels.extend([voronoi_img, graph_img, tube_img])
        else:
            top_panels.extend([voronoi_img, graph_img])
        top_panels = top_panels[:3]

        if generate_density_fiber:
            bottom_panels = [p for p in [density_img, stress_img, displacement_img] if p is not None]
            if not bottom_panels:
                bottom_panels = [core_curves_img]
            bottom_panels = bottom_panels[:3]
        else:
            bottom_panels = [core_curves_img, tube_img]

        panel_w = 460
        top_h = 280
        fem_h = 520
        hist_h = 240
        col_gap = 22
        row_gap = 22
        top_row_imgs = [_fit_panel_to_box(p, panel_w, top_h) for p in top_panels]
        fem_row_imgs = [_fit_panel_to_box(p, panel_w, fem_h) for p in bottom_panels]
        if not generate_density_fiber and len(fem_row_imgs) == 2:
            fem_row_imgs.insert(1, np.full((fem_h, panel_w, 3), 255, dtype=np.uint8))
        hist_panels = [total_loss_hist_img, fem_hist_img, fiber_length_hist_img]
        hist_row_imgs = [_fit_panel_to_box(p, panel_w, hist_h) for p in hist_panels]
        top_row = self._stack_row_with_gaps(top_row_imgs, gap=col_gap)
        fem_row = self._stack_row_with_gaps(fem_row_imgs, gap=col_gap)
        hist_row = self._stack_row_with_gaps(hist_row_imgs, gap=col_gap)
        target_w = max(top_row.shape[1], fem_row.shape[1], hist_row.shape[1])
        top_row = self._pad_to_size(top_row, target_w=target_w)
        fem_row = self._pad_to_size(fem_row, target_w=target_w)
        hist_row = self._pad_to_size(hist_row, target_w=target_w)
        gap_tile = np.full((row_gap, target_w, 3), 255, dtype=np.uint8)
        frame = np.vstack([top_row, gap_tile, fem_row, gap_tile, hist_row])
        return cv2.copyMakeBorder(
            frame,
            16,
            16,
            16,
            16,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _render_first_face_density_2d(
        self,
        cache_i,
        out_i,
        seeds_i,
        pred_i,
        window_size=(820, 820),
        show_scipy_voronoi=True,
        show_core_curves=True,
    ):
        width, height = int(window_size[0]), int(window_size[1])
        uv = cache_i["uv_dense"].detach().cpu().numpy().astype(np.float64)
        faces = cache_i["faces_ijk"].detach().cpu().numpy().astype(np.int64)
        raw_seeds_t = pred_i.get("seeds_uv", seeds_i)
        seeds = raw_seeds_t.detach().cpu().numpy().astype(np.float64)
        topology_seeds_t = out_i.get("topology_seeds_uv", None)
        topology_seeds = None
        if isinstance(topology_seeds_t, torch.Tensor):
            topology_seeds = topology_seeds_t.detach().cpu().numpy().astype(np.float64)
        elif topology_seeds_t is not None:
            topology_seeds = np.asarray(topology_seeds_t, dtype=np.float64)
        curves_uv_t = out_i.get("edge_curves_uv", None)
        curves_uv = None
        if isinstance(curves_uv_t, torch.Tensor):
            curves_uv = curves_uv_t.detach().cpu().numpy().astype(np.float64)

        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.88, 0.84])

        if faces.size > 0:
            ax.triplot(
                uv[:, 0],
                uv[:, 1],
                faces,
                color="#d1d5db",
                linewidth=0.35,
                alpha=0.5,
            )
        else:
            ax.scatter(
                uv[:, 0],
                uv[:, 1],
                c="#d1d5db",
                s=5,
                alpha=0.45,
                linewidths=0,
            )

        def plot_clipped_segment(p0, p1, color, linewidth, alpha, zorder):
            clipped = self._clip_segment_to_uv_box_np(p0, p1)
            if clipped is None:
                return
            q0, q1 = clipped
            ax.plot(
                [q0[0], q1[0]],
                [q0[1], q1[1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

        if show_scipy_voronoi and topology_seeds is not None and topology_seeds.ndim == 2 and topology_seeds.shape[0] >= 3:
            finite_topology_seeds = topology_seeds[np.isfinite(topology_seeds).all(axis=1)]
            if finite_topology_seeds.shape[0] >= 3:
                try:
                    from scipy.spatial import Voronoi, voronoi_plot_2d
                    raw_voronoi = Voronoi(finite_topology_seeds)
                    voronoi_plot_2d(
                        raw_voronoi,
                        ax=ax,
                        show_vertices=False,
                        show_points=False,
                        line_colors="#2563eb",
                        line_width=1.05,
                        line_alpha=0.58,
                        point_size=0,
                    )
                    if raw_voronoi.vertices.size > 0:
                        ax.scatter(
                            raw_voronoi.vertices[:, 0],
                            raw_voronoi.vertices[:, 1],
                            marker="x",
                            c="#374151",
                            s=38,
                            linewidths=1.0,
                            alpha=0.82,
                            zorder=4,
                        )
                except Exception:
                    pass

        active_values = pred_i.get("seed_active_mask", None)
        if active_values is None:
            if not getattr(self, "_warned_missing_seed_active_mask_for_timelapse", False):
                tqdm.write(
                    "Timelapse seed overlay is using all seeds as active because "
                    "this cached prediction has no seed_active_mask."
                )
                self._warned_missing_seed_active_mask_for_timelapse = True
            active = np.ones((seeds.shape[0],), dtype=bool)
        else:
            active = active_values.detach().cpu().numpy().reshape(-1).astype(bool)
            if active.shape[0] != seeds.shape[0]:
                if not getattr(self, "_warned_missing_seed_active_mask_for_timelapse", False):
                    tqdm.write(
                        "Timelapse seed overlay is using all seeds as active because "
                        "seed_active_mask length does not match raw seed positions."
                    )
                    self._warned_missing_seed_active_mask_for_timelapse = True
                active = np.ones((seeds.shape[0],), dtype=bool)
        weight_values = pred_i.get("seed_active_weights", None)
        if weight_values is not None:
            weights = weight_values.detach().cpu().numpy().reshape(-1)
            active = active & (weights >= 0.5)

        if seeds.shape[0] > 0:
            if np.any(~active):
                ax.scatter(
                    seeds[~active, 0],
                    seeds[~active, 1],
                    s=72,
                    facecolors="none",
                    edgecolors="#6b7280",
                    linewidths=1.4,
                    alpha=0.55,
                    zorder=5,
                )
            if np.any(active):
                ax.scatter(
                    seeds[active, 0],
                    seeds[active, 1],
                    s=92,
                    c="#ef4444",
                    edgecolors="white",
                    linewidths=1.6,
                    zorder=6,
                )

        if show_core_curves and curves_uv is not None and curves_uv.ndim == 3:
            for curve in curves_uv:
                if curve.shape[0] >= 2:
                    ax.plot(
                        curve[:, 0],
                        curve[:, 1],
                        color="black",
                        linewidth=1.8,
                        alpha=0.95,
                        zorder=4,
                    )

        ax.set_xlim(float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 0])))
        ax.set_ylim(float(np.nanmin(uv[:, 1])), float(np.nanmax(uv[:, 1])))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("u", fontsize=11)
        ax.set_ylabel("v", fontsize=11)
        ax.tick_params(labelsize=9, colors="#374151")
        ax.grid(color="#e5e7eb", linewidth=0.7, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()
        plt.close(fig)
        return img

    def _render_first_face_generated_graph_2d(
        self,
        decoder,
        out_i,
        seeds_i,
        window_size=(820, 820),
        show_node_ids=False,
        show_edge_ids=False,
    ):
        width, height = int(window_size[0]), int(window_size[1])
        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.88, 0.84])
        topology_seeds = out_i.get("topology_seeds_uv", seeds_i)
        try:
            decoder._draw_generated_graph(
                ax,
                topology_seeds,
                out_i,
                show_node_ids=show_node_ids,
                show_edge_ids=show_edge_ids,
                node_id_fontsize=7,
                color_by_edge_type=True,
            )
            ax.set_title("Generated Connectivity Graph", fontsize=12)
        except Exception as error:
            ax.text(
                0.5,
                0.5,
                f"Generated graph unavailable\n{error}",
                ha="center",
                va="center",
                fontsize=10,
                color="#111827",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=6,
                loc="upper right",
                framealpha=0.78,
            )
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()
        plt.close(fig)
        return img

    def _seed_points_xyz_and_activity_all_faces(self, seeds_list, pred_list, face_tensors):
        xyz_active = []
        xyz_inactive = []
        active_weight = []
        inactive_weight = []

        for seeds, pred, ft in zip(seeds_list, pred_list, face_tensors):
            if isinstance(pred.get("seeds_xyz"), torch.Tensor):
                xyz_i = pred["seeds_xyz"].detach().cpu().numpy()
            else:
                xyz_i = self.generator.seeds_uv_to_xyz_nearest(
                    seeds,
                    ft["uv"],
                    ft["points_xyz"],
                )

            active_mask_i = pred.get("seed_active_mask", None)
            active_weights_i = pred.get("seed_active_weights", None)

            if active_mask_i is None:
                xyz_active.append(xyz_i)
                continue

            active_mask = active_mask_i.detach().cpu().numpy().astype(bool)
            weights = (
                active_weights_i.detach().cpu().numpy()
                if active_weights_i is not None
                else active_mask.astype(float)
            )
            participating_mask = active_mask & (weights >= 0.5)
            inactive_mask = ~participating_mask

            xyz_i_active = xyz_i[participating_mask]
            xyz_i_inactive = xyz_i[inactive_mask]

            if len(xyz_i_active) > 0:
                xyz_active.append(xyz_i_active)
                active_weight.append(weights[participating_mask])

            if len(xyz_i_inactive) > 0:
                xyz_inactive.append(xyz_i_inactive)
                inactive_weight.append(weights[inactive_mask])

        import numpy as np

        xyz_active = np.concatenate(xyz_active, axis=0) if len(xyz_active) > 0 else None
        xyz_inactive = np.concatenate(xyz_inactive, axis=0) if len(xyz_inactive) > 0 else None
        active_weight = np.concatenate(active_weight, axis=0) if len(active_weight) > 0 else None
        inactive_weight = np.concatenate(inactive_weight, axis=0) if len(inactive_weight) > 0 else None

        return {
            "xyz_active": xyz_active,
            "xyz_inactive": xyz_inactive,
            "active_weight": active_weight,
            "inactive_weight": inactive_weight,
        }
    
    def visualize_best_seed_activity(self, result, points_xyz=None, faces_ijk=None):
        best_seeds = result["best_seeds"]
        best_pred = result["best_pred"]
        face_tensors = result["face_tensors"]

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
            seeds_list=best_seeds,
            pred_list=best_pred,
            face_tensors=face_tensors,
        )

        plotter = pv.Plotter()

        if points_xyz is not None and faces_ijk is not None:
            pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)
            mesh = pv.PolyData(points_xyz.detach().cpu().numpy(), pv_faces_fixed)
            plotter.add_mesh(mesh, color="white", opacity=0.25, show_edges=False)

        if seed_vis["xyz_active"] is not None and len(seed_vis["xyz_active"]) > 0:
            active_cloud = pv.PolyData(seed_vis["xyz_active"])
            plotter.add_mesh(
                active_cloud,
                color="red",
                render_points_as_spheres=True,
                point_size=14,
                label="Active seeds",
            )

        if seed_vis["xyz_inactive"] is not None and len(seed_vis["xyz_inactive"]) > 0:
            inactive_cloud = pv.PolyData(seed_vis["xyz_inactive"])
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                render_points_as_spheres=True,
                point_size=10,
                opacity=0.4,
                label="Inactive seeds",
            )

        plotter.add_legend()
        plotter.show()
    
    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _get_result_final_density(self, result):
        density = result.get("Final_shape_density", None)
        if density is None:
            density = result.get("best_rho", None)
        if density is None:
            available = ", ".join(sorted(str(k) for k in result.keys()))
            raise KeyError(
                "Could not find final density in result. Expected "
                "'Final_shape_density' or legacy fallback 'best_rho'. "
                f"Available keys: {available}"
            )
        return density



    def visualize_result_final(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr=0.5,
        show_solid=True,
        show_rho_overlay=False,
    ):
        points_np = (
            points_xyz.detach().cpu().numpy()
            if isinstance(points_xyz, torch.Tensor)
            else np.asarray(points_xyz)
        )
        faces_np = (
            faces_ijk.detach().cpu().numpy()
            if isinstance(faces_ijk, torch.Tensor)
            else np.asarray(faces_ijk)
        ).astype(np.int64)
        rho_np = self._get_result_final_density(result).detach().cpu().numpy()

        pv_faces = np.empty((faces_np.shape[0], 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = faces_np
        mesh = pv.PolyData(points_np, pv_faces.reshape(-1))
        mesh["rho"] = rho_np.astype(np.float32)

        face_rho = np.mean(rho_np[faces_np], axis=1) if faces_np.size > 0 else np.empty((0,))
        solid_faces = faces_np[face_rho >= float(thr)]
        if solid_faces.size > 0:
            pv_solid_faces = np.empty((solid_faces.shape[0], 4), dtype=np.int64)
            pv_solid_faces[:, 0] = 3
            pv_solid_faces[:, 1:] = solid_faces
            solid = pv.PolyData(points_np, pv_solid_faces.reshape(-1))
        else:
            solid = pv.PolyData()

        if show_solid:
            plotter = pv.Plotter()
            plotter.set_background("white")
            if solid.n_cells > 0:
                plotter.add_mesh(solid, color="#8ecae6", show_edges=False, lighting=False)
            seed_xyz_parts = []
            for pred in result.get("best_pred", []):
                seeds_xyz_i = pred.get("seeds_xyz", None) if isinstance(pred, dict) else None
                if isinstance(seeds_xyz_i, torch.Tensor):
                    seed_xyz_parts.append(seeds_xyz_i.detach().cpu().numpy())
            if seed_xyz_parts:
                seed_xyz = np.concatenate(seed_xyz_parts, axis=0).astype(np.float32)
                plotter.add_mesh(
                    pv.PolyData(seed_xyz),
                    color="red",
                    render_points_as_spheres=True,
                    point_size=12,
                )
            if show_rho_overlay:
                plotter.add_mesh(
                    mesh,
                    scalars="rho",
                    cmap="viridis",
                    opacity=0.20,
                    show_edges=False,
                    lighting=False,
                )
            plotter.show_axes()
            plotter.show()

        return solid, float(thr)
    def sample_face_field_for_visualization(
        self,
        ft: dict,
        decoder,
        pred: dict,
        shape_or_path,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_boundary_attachment: bool = True,
        trim_tol: float = 1e-7,
    ):
        """
        Dense CAD-native field sampling on one face for smooth visualization.

        This version:
        - builds a dense UV grid in normalized face UV
        - optionally prefilters points by proximity to sampled UV cloud
        - evaluates xyz, Xu, Xv on the actual CAD face
        - keeps only trim-valid points
        - evaluates decoder on those dense query points

        Returns:
            {
                "uv_dense": (Nd,2),
                "uv_raw_dense": (Nd,2),
                "xyz_dense": (Nd,3),
                "Xu_dense": (Nd,3),
                "Xv_dense": (Nd,3),
                "rho_dense": (Nd,),
                "rho_v_dense": (Nd,),
                "rho_b_dense": (Nd,),
                "fiber3d_dense": (Nd,3),
                "edge_field_dense": (Nd,),
                "mask_dense_prefilter": (Nu*Nv,),
                "grid_shape": (Nu, Nv),
            }
        """
        device = ft["uv"].device
        dtype = ft["uv"].dtype

        uv_face = ft["uv"]
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        # ------------------------------------------------------------
        # 1) Dense UV grid in normalized face UV coordinates
        # ------------------------------------------------------------
        uv_grid, _u_lin, _v_lin = self._build_face_uv_grid(ft, grid_res_u, grid_res_v)

        # ------------------------------------------------------------
        # 2) Optional UV-cloud prefilter
        #    Helps avoid querying huge empty regions on trimmed faces.
        # ------------------------------------------------------------
        if uv_mask_tol is None:
            uv_mask_tol = self._estimate_uv_mask_tol(
                uv_face=uv_face,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )

        dmin = self._periodic_uv_min_dist(
            uv_grid,
            uv_face,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        mask_dense_prefilter = dmin <= uv_mask_tol
        uv_query = uv_grid[mask_dense_prefilter]

        if uv_query.numel() == 0:
            raise ValueError(
                f"No dense UV query points survived prefilter on face {ft.get('face_id', 'unknown')}. "
                f"Try increasing uv_mask_tol."
            )

        # ------------------------------------------------------------
        # 3) CAD-native geometry evaluation
        # ------------------------------------------------------------
        geom = self.generator.eval_face_uv_from_face_tensor(
            shape_or_path=shape_or_path,
            face_tensor=ft,
            uv_norm=uv_query,
            metric_tol=getattr(self.generator, "metric_tol", 1e-9),
            trim_tol=trim_tol,
            as_torch=True,
        )

        valid_mask = geom["valid_mask"]
        if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):
            raise ValueError(
                f"No valid CAD-evaluable dense points on face {ft.get('face_id', 'unknown')}."
            )

        uv_dense = geom["uv_norm"][valid_mask]
        uv_raw_dense = geom["uv_raw"][valid_mask]
        xyz_dense = geom["points_xyz"][valid_mask]
        Xu_dense = geom["Xu"][valid_mask]
        Xv_dense = geom["Xv"][valid_mask]
        mask_dense_valid = torch.zeros_like(mask_dense_prefilter, dtype=torch.bool)
        mask_dense_valid[mask_dense_prefilter] = valid_mask



        # ------------------------------------------------------------
        # 5) Recover trained parameters
        # ------------------------------------------------------------
        seeds_raw = pred["seeds_raw"]
        h_raw = pred.get("h_raw", None)

        theta = pred.get("theta", None)
        a_raw = pred.get("a_raw", None)

        boundary_width_raw = pred.get("boundary_width_raw", None)
        boundary_alpha_raw = pred.get("boundary_alpha_raw", None)
        boundary_beta_raw = pred.get("boundary_beta_raw", None)

        # ------------------------------------------------------------
        # 6) Evaluate decoder on CAD-native dense query points
        # ------------------------------------------------------------
        decoder_out = decoder.build_swept_tube_fields(
            points_uv=uv_dense,
            points_3d=xyz_dense,
            seeds_uv=seeds_raw,
            Xu=Xu_dense,
            Xv=Xv_dense,
            cad_domain=self.Cad_domain,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
            return_xyz=True,
            generate_density_fiber=getattr(self.cfg, "generate_decoder_density_fiber", True),
        )

        self._require_decoder_keys(
            decoder_out,
            ["rho", "fiber3d"],
        )

        full_indices = -torch.ones(
            mask_dense_valid.shape[0],
            dtype=torch.long,
            device=device,
        )
        full_indices[mask_dense_valid] = torch.arange(
            int(mask_dense_valid.sum().item()),
            dtype=torch.long,
            device=device,
        )
        faces_dense = []
        for i in range(grid_res_u - 1):
            for j in range(grid_res_v - 1):
                k00 = i * grid_res_v + j
                k01 = i * grid_res_v + j + 1
                k10 = (i + 1) * grid_res_v + j
                k11 = (i + 1) * grid_res_v + j + 1
                ids = full_indices[torch.tensor([k00, k01, k10, k11], device=device)]
                if bool((ids >= 0).all().item()):
                    faces_dense.append(ids[[0, 1, 2]])
                    faces_dense.append(ids[[2, 1, 3]])
        if faces_dense:
            faces_dense = torch.stack(faces_dense, dim=0)
        else:
            faces_dense = torch.empty((0, 3), dtype=torch.long, device=device)

        dense_face_tensor = {
            "points_xyz": xyz_dense,
            "faces_ijk": faces_dense,
            "Xu": Xu_dense,
            "Xv": Xv_dense,
        }
        decoder_out = apply_density_postprocess_to_output(
            decoder_out,
            dense_face_tensor,
            self.cfg,
            return_debug=False,
        )

        return {
            "face_id": self._face_id_key(ft.get("face_id", 0)),
            "uv_dense": uv_dense,
            "uv_raw_dense": uv_raw_dense,
            "xyz_dense": xyz_dense,
            "Xu_dense": Xu_dense,
            "Xv_dense": Xv_dense,
            "rho_dense": decoder_out["rho"],
            "rho_raw_decoder_dense": decoder_out["rho_raw_decoder"],
            "rho_postprocessed_dense": decoder_out["rho_postprocessed"],
            "fiber3d_dense": decoder_out["fiber3d"],
            "seeds_uv": decoder_out.get("seeds_uv", decoder_out.get("seeds", None)),
            "seeds_xyz": decoder_out.get("seeds_xyz", None),
            "edge_curves_uv": decoder_out.get("edge_curves_uv", None),
            "edge_curves_xyz": decoder_out.get("edge_curves_xyz", None),
            "mask_dense_prefilter": mask_dense_prefilter,
            "mask_dense_valid": mask_dense_valid,
            "grid_shape": (grid_res_u, grid_res_v),
        }
   
    def sample_result_field_dense_for_visualization(
        self,
        result: dict,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_best_pred: bool = True,
    ):
        """
        Dense CAD-native field sampling over all faces for smooth visualization.
        """
        face_tensors = result["face_tensors"]
        decoders = result["decoders"]

        if use_best_pred:
            pred_list = result["best_pred"]
        else:
            raise ValueError("Only use_best_pred=True is currently supported.")

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)

        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for CAD-native dense sampling. "
                "Pass it explicitly or store 'shape_path' in result."
            )

        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in pred_list}

        xyz_parts = []
        rho_parts = []
        fiber_parts = []
        face_ranges = []
        per_face = []

        start = 0
        for ft, decoder in zip(face_tensors, decoders):
            face_id = self._face_id_key(ft.get("face_id", 0))
            if face_id not in pred_by_face_id:
                raise KeyError(f"Missing best_pred for face_id={face_id}")

            pred = pred_by_face_id[face_id]

            sampled = self.sample_face_field_for_visualization(
                ft=ft,
                decoder=decoder,
                pred=pred,
                shape_or_path=shape_or_path,
                grid_res_u=grid_res_u,
                grid_res_v=grid_res_v,
                uv_mask_tol=uv_mask_tol,
            )

            n = sampled["xyz_dense"].shape[0]
            end = start + n

            xyz_parts.append(sampled["xyz_dense"])
            rho_parts.append(sampled["rho_dense"])
            fiber_parts.append(sampled["fiber3d_dense"])

            face_ranges.append((start, end, face_id))
            per_face.append(sampled)
            start = end

        return {
            "points_xyz": torch.cat(xyz_parts, dim=0),
            "rho": torch.cat(rho_parts, dim=0),
            "fiber3d": torch.cat(fiber_parts, dim=0),
            "face_ranges": face_ranges,
            "per_face": per_face,
        }

    @staticmethod
    def _resolve_visualization_grid_resolution(
        grid_res_u: int,
        grid_res_v: int,
        dense_factor: float = 1.0,
        min_res: int = 8,
        max_res: int = 1024,
    ) -> tuple[int, int]:
        dense_factor = float(max(dense_factor, 1e-3))
        res_u = int(round(float(grid_res_u) * dense_factor))
        res_v = int(round(float(grid_res_v) * dense_factor))
        res_u = max(int(min_res), min(int(max_res), res_u))
        res_v = max(int(min_res), min(int(max_res), res_v))
        return res_u, res_v

    @staticmethod
    def _dense_face_triangles(mask_dense_valid, grid_shape):
        mask = np.asarray(mask_dense_valid, dtype=bool).reshape(-1)
        Nu, Nv = (int(grid_shape[0]), int(grid_shape[1]))
        full_indices = -np.ones(mask.shape[0], dtype=np.int64)
        full_indices[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
        triangles = []

        def idx(i, j):
            return i * Nv + j

        for i in range(Nu - 1):
            for j in range(Nv - 1):
                ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                mapped = [full_indices[k] for k in ids]
                if any(m < 0 for m in mapped):
                    continue
                i0, i1, i2, i3 = mapped
                triangles.append([i0, i1, i2])
                triangles.append([i2, i1, i3])

        return np.asarray(triangles, dtype=np.int64)

    def visualize_result_final_edge_field(
        self,
        result,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
        cmap: str = "viridis",
        show_seeds: bool = True,
        show_uv: bool = True,
        show_3d: bool = True,
    ):
        """Plot the decoder's geometric Voronoi edge field in UV and on the CAD surface."""
        if not show_uv and not show_3d:
            raise ValueError("At least one of show_uv or show_3d must be True.")

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )
        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in result["best_pred"]}
        face_plots = []
        for face_data in dense["per_face"]:
            mask = face_data["mask_dense_valid"].detach().cpu().numpy()
            triangles = self._dense_face_triangles(mask, face_data["grid_shape"])
            face_plots.append(
                {
                    "face_id": face_data["face_id"],
                    "uv": face_data["uv_dense"].detach().cpu().numpy().astype(np.float32),
                    "xyz": face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32),
                    "edge_field": face_data["edge_field_dense"].detach().cpu().numpy().astype(np.float32),
                    "triangles": triangles,
                }
            )

        uv_fig = None
        if show_uv:
            n_faces = len(face_plots)
            ncols = min(3, max(1, n_faces))
            nrows = int(np.ceil(float(n_faces) / float(ncols)))
            uv_fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.6 * ncols, 5.0 * nrows),
                squeeze=False,
                constrained_layout=True,
            )
            color_artist = None
            for ax, face_plot in zip(axes.ravel(), face_plots):
                uv = face_plot["uv"]
                triangles = face_plot["triangles"]
                edge_field = face_plot["edge_field"]
                if triangles.size > 0:
                    color_artist = ax.tripcolor(
                        uv[:, 0],
                        uv[:, 1],
                        triangles,
                        edge_field,
                        shading="gouraud",
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                    )
                else:
                    color_artist = ax.scatter(
                        uv[:, 0],
                        uv[:, 1],
                        c=edge_field,
                        s=6,
                        linewidths=0,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                    )

                if show_seeds:
                    pred = pred_by_face_id.get(face_plot["face_id"])
                    if pred is not None:
                        seeds_uv = pred["seeds_raw"].detach().cpu().numpy()
                        ax.scatter(
                            seeds_uv[:, 0],
                            seeds_uv[:, 1],
                            s=38,
                            c="#e04b3f",
                            edgecolors="white",
                            linewidths=1.0,
                            zorder=3,
                        )
                ax.set_title(f"Face {face_plot['face_id']} | Edge Field")
                ax.set_xlabel("u")
                ax.set_ylabel("v")
                ax.set_aspect("equal", adjustable="box")

            for ax in axes.ravel()[len(face_plots):]:
                ax.axis("off")
            if color_artist is not None:
                uv_fig.colorbar(color_artist, ax=axes.ravel().tolist(), label="edge_field")
            uv_fig.suptitle("Geometric Voronoi Edge Field in UV", y=1.02)
            plt.show()

        plotter = None
        if show_3d:
            plotter = pv.Plotter()
            for face_plot in face_plots:
                triangles = face_plot["triangles"]
                if triangles.size == 0:
                    continue
                pv_faces = np.empty((triangles.shape[0], 4), dtype=np.int64)
                pv_faces[:, 0] = 3
                pv_faces[:, 1:] = triangles
                mesh = pv.PolyData(face_plot["xyz"], pv_faces.reshape(-1))
                mesh["edge_field"] = face_plot["edge_field"]
                plotter.add_mesh(
                    mesh,
                    scalars="edge_field",
                    cmap=cmap,
                    clim=[0.0, 1.0],
                    show_edges=False,
                    scalar_bar_args={"title": "edge_field"},
                )

            seed_points_final = result.get("seed_points_final")
            if show_seeds and seed_points_final is not None:
                plotter.add_mesh(
                    seed_points_final,
                    render_points_as_spheres=True,
                    point_size=6,
                    color="#e04b3f",
                )
            plotter.add_text("Geometric Voronoi Edge Field", font_size=11)
            plotter.show_axes()
            plotter.show()

        return {
            "uv_fig": uv_fig,
            "plotter": plotter,
            "dense": dense,
            "per_face": face_plots,
            "grid_shape": (grid_res_u, grid_res_v),
        }

    def visualize_result_final_smooth_points(
        self,
        result,
        shape_or_path=None,
        thr: float = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
    ):
        """
        Smooth point-cloud style threshold visualization from dense CAD-native decoder sampling.

        `dense_factor` scales the internal UV sampling density used for visualization.
        Larger values produce a denser point cloud and finer visual detail.
        """
        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        points_xyz = dense["points_xyz"].detach().cpu().numpy()
        rho = dense["rho"].detach().cpu().numpy()

        keep = rho >= thr
        solid_points = points_xyz[keep]

        print(
            f"Smooth CAD-native visualization: kept {keep.sum()} / {keep.shape[0]} dense points "
            f"with threshold {thr:.3f} on grid ({grid_res_u} x {grid_res_v})"
        )


        cloud = pv.PolyData(solid_points)

        plotter = pv.Plotter()
        plotter.add_points(
            cloud,
            render_points_as_spheres=True,
            point_size=6,
        )

        plotter.show()

        return {
            "solid_points": solid_points,
            "points_xyz": points_xyz,
            "rho": rho,
            "keep_mask": keep,
            "dense": dense,
        }

    def Visualize_fresult_final_fiber_Direction(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr: float = 0.5,
    ):
        import numpy as np
        import pyvista as pv

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density = self._get_result_final_density(result).detach().cpu()
        fiber = result["Final_shape_fiber_direction"].detach().cpu()
        points_xyz_cpu = points_xyz.detach().cpu()
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        keep = torch.isfinite(density)
        keep = keep & torch.isfinite(fiber).all(dim=1)
        keep = keep & (density >= float(thr))
        keep = keep & (torch.linalg.norm(fiber, dim=1) > 1e-10)

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            print(f"No fiber arrows to display for threshold {thr:.3f}.")
            return {
                "points_xyz": points_xyz_cpu.numpy(),
                "rho": density.numpy(),
                "fiber3d": fiber.numpy(),
                "keep_mask": keep.numpy(),
            }

        max_arrows = 2000
        if keep_idx.numel() > max_arrows:
            step = int(np.ceil(float(keep_idx.numel()) / float(max_arrows)))
            keep_idx = keep_idx[::step]

        pts_np = points_xyz_cpu[keep_idx].numpy().astype(np.float32)
        fiber_np = fiber[keep_idx].numpy().astype(np.float32)
        rho_np = density[keep_idx].numpy().astype(np.float32)

        bbox = points_xyz_cpu.amax(dim=0) - points_xyz_cpu.amin(dim=0)
        diag = float(torch.linalg.norm(bbox).item())
        arrow_scale_used = 0.03 * diag

        plotter = pv.Plotter()

        surface = pv.PolyData(
            points_xyz_cpu.numpy().astype(np.float32),
            pv_faces_fixed,
        )
        surface["rho"] = density.numpy().astype(np.float32)
        plotter.add_mesh(
            surface,
            scalars="rho",
            cmap="Greys",
            opacity=0.20,
            show_edges=False,
        )

        arrow_cloud = pv.PolyData(pts_np)
        arrow_cloud["vectors"] = fiber_np
        arrow_cloud["rho"] = rho_np

        glyphs = arrow_cloud.glyph(
            orient="vectors",
            scale=False,
            factor=arrow_scale_used,
            geom=pv.Arrow(),
        )
        plotter.add_mesh(glyphs, color="royalblue")
        plotter.show_axes()
        plotter.show()

        print(
            f"Fiber-direction visualization: showing {pts_np.shape[0]} arrows "
            f"with threshold {thr:.3f}"
        )

        return {
            "arrow_points": pts_np,
            "arrow_vectors": fiber_np,
            "arrow_rho": rho_np,
            "points_xyz": points_xyz_cpu.numpy(),
            "rho": density.numpy(),
            "fiber3d": fiber.numpy(),
            "keep_mask": keep.numpy(),
            "arrow_scale_used": float(arrow_scale_used),
        }

    def visualize_result_final_fiber_direction(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def Visualize_fresult_final_fiber_Direction_3D(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def visualize_result_final_fiber_direction_3d(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    @staticmethod
    def _fiber3d_to_uv_direction(Xu_np, Xv_np, fiber_np, eps=1e-12):
        a11 = np.sum(Xu_np * Xu_np, axis=1)
        a12 = np.sum(Xu_np * Xv_np, axis=1)
        a22 = np.sum(Xv_np * Xv_np, axis=1)
        b1 = np.sum(Xu_np * fiber_np, axis=1)
        b2 = np.sum(Xv_np * fiber_np, axis=1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < eps, np.nan, det)

        du = (a22 * b1 - a12 * b2) / det
        dv = (-a12 * b1 + a11 * b2) / det
        tuv = np.stack([du, dv], axis=1)
        nrm = np.linalg.norm(tuv, axis=1, keepdims=True)
        ok = np.isfinite(tuv).all(axis=1, keepdims=True) & (nrm > eps)
        tuv = np.where(ok, tuv / np.clip(nrm, eps, None), 0.0)
        return tuv.astype(np.float32)

    def visualize_result_final_smooth_surface_pyvista(
        self,
        result,
        points_xyz=None,
        faces_ijk=None,
        shape_or_path=None,
        thr: float | str | None = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        show_density: bool = True,
        auto_target_volfrac: float | None = None,
        dense_factor: float = 1.0,
    ):
        import pyvista as pv
        import numpy as np

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)
        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for smooth CAD-native visualization. "
                "Pass it explicitly or store it in result['shape_path']."
            )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        rho_all = []
        area_w_all = []
        for face_data in dense["per_face"]:
            rho_i = face_data["rho_dense"]
            Xu_i = face_data["Xu_dense"]
            Xv_i = face_data["Xv_dense"]
            area_w_i = torch.linalg.norm(torch.cross(Xu_i, Xv_i, dim=1), dim=1).clamp_min(self.cfg.eps)
            rho_all.append(rho_i.detach().cpu().numpy())
            area_w_all.append(area_w_i.detach().cpu().numpy())

        rho_all = np.concatenate(rho_all, axis=0)
        area_w_all = np.concatenate(area_w_all, axis=0)
        area_w_sum = float(area_w_all.sum()) + float(self.cfg.eps)
        volfrac_cont = float((rho_all * area_w_all).sum() / area_w_sum)

        thr_used = thr
        target_text = "none"
        if thr is None or (isinstance(thr, str) and str(thr).lower() == "auto"):
            if auto_target_volfrac is None:
                raise ValueError(
                    "thr='auto' requires auto_target_volfrac because target_volfrac "
                    "is no longer part of the final optimization config."
                )
            target = float(auto_target_volfrac)
            target = float(np.clip(target, 0.0, 1.0))
            target_text = f"{target:.4f}"

            # Weighted quantile so that area fraction above threshold ~= target.
            q = 1.0 - target
            order = np.argsort(rho_all)
            rho_s = rho_all[order]
            w_s = area_w_all[order]
            cdf = np.cumsum(w_s) / (np.sum(w_s) + float(self.cfg.eps))
            thr_used = float(np.interp(q, cdf, rho_s))
        else:
            thr_used = float(thr)

        volfrac_thr = float(area_w_all[rho_all >= thr_used].sum() / area_w_sum)
        print(
            f"[smooth_surface] thr={thr_used:.4f} | "
            f"volfrac_cont(rho)={volfrac_cont:.4f} | "
            f"volfrac_thr(binary)={volfrac_thr:.4f} | "
            f"target={target_text} | "
            f"grid=({grid_res_u} x {grid_res_v})"
        )

        plotter = pv.Plotter()
        per_face = []

        for face_data in dense["per_face"]:
            xyz = face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32)
            rho = face_data["rho_dense"].detach().cpu().numpy().astype(np.float32)
            face_id = face_data["face_id"]
            Nu, Nv = face_data["grid_shape"]
            mask = face_data["mask_dense_valid"].detach().cpu().numpy()
            uv = face_data["uv_dense"].detach().cpu().numpy().astype(np.float32)

            full_indices = -np.ones(mask.shape[0], dtype=np.int64)
            full_indices[mask] = np.arange(mask.sum(), dtype=np.int64)

            faces_keep = []

            def idx(i, j):
                return i * Nv + j

            for i in range(Nu - 1):
                for j in range(Nv - 1):
                    ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                    mapped = [full_indices[k] for k in ids]
                    if any(m < 0 for m in mapped):
                        continue

                    i0, i1, i2, i3 = mapped
                    if rho[i0] >= thr_used and rho[i1] >= thr_used and rho[i2] >= thr_used:
                        faces_keep.append([i0, i1, i2])
                    if rho[i2] >= thr_used and rho[i1] >= thr_used and rho[i3] >= thr_used:
                        faces_keep.append([i2, i1, i3])

            faces_keep = np.asarray(faces_keep, dtype=np.int64)
            if faces_keep.size == 0:
                per_face.append({
                    "face_id": face_id,
                    "uv": uv,
                    "xyz": xyz,
                    "rho": rho,
                    "faces_keep": faces_keep,
                })
                continue

            pv_faces = np.empty((faces_keep.shape[0], 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces_keep
            mesh = pv.PolyData(xyz, pv_faces.reshape(-1))

            if show_density:
                mesh["rho"] = rho
                plotter.add_mesh(mesh, scalars="rho", cmap="viridis", clim=[0, 1])
            else:
                plotter.add_mesh(mesh, color="lightblue")

            per_face.append({
                "face_id": face_id,
                "uv": uv,
                "xyz": xyz,
                "rho": rho,
                "faces_keep": faces_keep,
            })
        plotter.show()
        return {
            "thr_used": float(thr_used),
            "volfrac_cont": float(volfrac_cont),
            "volfrac_thr": float(volfrac_thr),
            "dense": dense,
            "per_face": per_face,
        }
  
    def train(self, shape_path, face_tensors):
        cfg = self.cfg
        # time.perf_counter() returns the value (in fractional seconds) of a performance counter, i.e., a clock with the highest available resolution to measure a short duration.
        train_start_time = time.perf_counter()

        # Always train one face. If multiple faces are provided, use cfg.training_face_index.
        # List of CAD-face mesh tensors containing UV coordinates, 3D geometry, connectivity, and surface information.
        face_tensors = self._select_single_training_face(face_tensors)
        face_tensor = face_tensors[0]

        # validate the selected face tensor before training
        self._validate_face_tensors(face_tensors)

        # Assign device and data type used during training process
        ref_uv = face_tensor["uv"]
        device = ref_uv.device
        dtype = ref_uv.dtype

        # Total number of points used for training on the selected face
        gidx = face_tensor["global_vertex_idx"]
        vertices_number = int(gidx.max().item()) + 1
        # ------------------------------------------------------------
        # Build global vertex areas
        A_v = torch.zeros((vertices_number,), dtype=dtype, device=device)
        A_local = self.generator.vertex_area_lumped(
            face_tensor["uv"].shape[0],
            face_tensor["faces_ijk"],
            face_tensor["face_areas"],
        ).to(device=device, dtype=dtype)
        face_weight = A_local.sum().clamp_min(cfg.eps)
        A_v[gidx] += A_local

        # ------------------------------------------------------------
        # Build models / optimizer / scheduler
        # ------------------------------------------------------------
        decoder, ppnet = self._build_face_model(face_tensor=face_tensor, device=device)
        decoders = [decoder]
        ppnets = [ppnet]
        named_trainable_modules = [("ppnet", ppnet), ("decoder", decoder)]
        trainable_modules = [module for _, module in named_trainable_modules]
        # Build initial seeds from the selected face tensor, which will be optimized during training.
        uv_init = self._init_face_seed(face_tensor)
        uv_anchor = uv_init.clone()
        uv_init_list = [uv_init]

        # Build the optimizer for the initial stage trainable parameters.
        opt = self._build_optimizer(ppnet, decoder)
        validate_optimizer_parameter_coverage(
            named_trainable_modules=named_trainable_modules,
            optimizer=opt,
        )
        consecutive_invalid_fem_steps = 0
        latest_valid_model_state = self._clone_modules_state_dict(trainable_modules)
        latest_valid_optimizer_state = self._clone_optimizer_state_dict(opt)
        # Invalid FEM attempts consume the global attempt budget.
        # They do not advance optimizer, scheduler, stage-local step,
        # patience, or checkpoint selection.
        decoder_trainable_names = [
            name
            for name, parameter in decoder.named_parameters()
            if parameter.requires_grad
        ]
        tqdm.write(
            "Invalid FEM attempts consume global budget: "
            f"{bool(cfg.invalid_fem_consumes_budget)}"
        )
        tqdm.write(
            "Decoder trainable parameters: "
            + (", ".join(decoder_trainable_names) if decoder_trainable_names else "none")
        )
        
        
        # Stage 1 is the warm-up stage; Stage 2 is the main physical optimization.
        stage_specs = self._adaptive_stage_specs()
        total_step_budget = sum(int(spec.max_steps) for spec in stage_specs)
        first_physical_stage = self._first_physical_stage_id(stage_specs)
        stage_runtimes = [StageRuntime(spec=spec) for spec in stage_specs]
        current_stage_index = 0
        current_stage_runtime = stage_runtimes[0]
        stage_end_summaries: list[dict[str, Any]] = []
        best_feasible_checkpoint = None
        best_feasible_key = (float("inf"), float("inf"), float("inf"))
        best_infeasible_checkpoint = None
        best_infeasible_key = (float("inf"), float("inf"), float("inf"))
        last_valid_checkpoint = None
        selected_checkpoint_source = "global"
        selected_checkpoint_stage_loss = float("nan")
        returned_best_source = "unavailable"
        stage1_transition_step = None
        stage1_transition_score = float("nan")
        stage1_transition_checkpoint_source = None
        physical_checkpoint_trackers_reset = first_physical_stage <= 1

        stage_settings_initial = self._stage_settings_for_stage_id(current_stage_runtime.spec.stage_id)
        self._apply_stage_trainability(ppnet, stage_settings_initial)
        opt = self._build_optimizer(ppnet, decoder)
        validate_optimizer_parameter_coverage(
            named_trainable_modules=named_trainable_modules,
            optimizer=opt,
        )
        scheduler = self._build_scheduler(
            opt,
            self._stage_scheduler_milestones(current_stage_runtime.spec.stage_id),
        )
        latest_valid_model_state = self._clone_modules_state_dict(trainable_modules)
        latest_valid_optimizer_state = self._clone_optimizer_state_dict(opt)

        # ------------------------------------------------------------
        # Optional timelapse setup
        # ------------------------------------------------------------
        recorder = None
        render_cache = None
        timelapse_output_folder = None
        if cfg.MakeTimelaps:
            case_name = shape_path.stem
            timelapse_output_folder = getattr(cfg, "timelapse_output_folder", None)
            if timelapse_output_folder:
                timelapse_output_folder = os.path.normpath(str(timelapse_output_folder))

                base_folder = timelapse_output_folder
                counter = 1

                while os.path.exists(timelapse_output_folder):
                    timelapse_output_folder = f"{base_folder}{counter}"
                    counter += 1

                os.makedirs(timelapse_output_folder)

                frame_out_dir = os.path.join(timelapse_output_folder, "timelapse_frames")
                video_path = os.path.join(
                    timelapse_output_folder,
                    case_name + "_timelapse.avi"
                )
            else:
                frame_out_dir = "timelapse_frames"
                video_path = case_name + "_timelapse.avi"
            # defining the timelapse recorder, which will save the training progress as a video. 
            # The output directory for the frames is "timelapse_frames", 
            # the video will be saved with the name "{case_name}_timelapse.avi". 
            # The frames per second (fps) for the video is set to 8.
            if self.shell_problem is not None and getattr(self.shell_problem, "mesh", None) is not None:
                fem_mesh = self.shell_problem.mesh
                fem_elems = int(fem_mesh["nelx"]) * int(fem_mesh["nely"]) * int(fem_mesh["nelz"])
            else:
                fem_elems = 0
            

            load_value = (
            float(getattr(self.shell_problem, "Load_magnitude", 0.0))
            if self.shell_problem is not None
            else 0.0
        )
            geometry_summary = self._timelapse_geometry_summary(face_tensors)
            recorder = TimelapseRecorder(
                out_dir=frame_out_dir,
                video_path=video_path,
                fps=8,
                header_title=(
                    f"{shape_path.name} ({geometry_summary}) | "
                    f"BC: {cfg.LoadingCase} (F = {load_value:.3f} , FEM elements: {fem_elems})"
                ),
                header_subtitle=self._timelapse_optimized_parameter_summary(),
            )
            # building a cache for rendering the timelapse, which likely includes precomputing certain data or settings that will be used 
            # repeatedly during the rendering of each frame in the timelapse video. 
            render_cache = self.build_timelapse_render_cache(
                face_tensors=face_tensors,
            )
            if self.timelapse_loading_img is None and self.shell_problem is not None:
                try:
                    self.timelapse_loading_img = self.shell_problem.show_voxels_surface_and_bc(
                        return_img=True,
                        off_screen=True,
                        window_size=(520, 280),
                    )
                    self.timelapse_loading_img = self._composite_to_white(self.timelapse_loading_img)
                except Exception as e:
                    tqdm.write(f"Failed to render timelapse loading panel: {e}")

        # ------------------------------------------------------------
        # Loss normalizers
        # ------------------------------------------------------------
        # These RunningNorm instances are used to keep track of the running mean and standard deviation of various loss components during training.
        # if on , it will normalize the loss components to have a more stable training process, especially when the scales of different loss terms vary significantly.
        norm_rep = RunningNorm()
        norm_cvt = RunningNorm()
        norm_l_seed = RunningNorm()
        norm_total_fiber_length = RunningNorm()
        norm_l_curve_cell = RunningNorm()
        adaptive_lambda_fem = float(
            getattr(
                cfg,
                "fem_lambda_initial",
                max(float(getattr(cfg, "stage2_lam_fem", 1.0)), 1.0),
            )
        )
        target_active_seeds = (
            cfg.seed_number
            if cfg.min_active_seeds is None
            else cfg.min_active_seeds
        )

        # ------------------------------------------------------------
        # Best-state tracking
        # ------------------------------------------------------------
        best_score = float("inf")
        best_step = -1
        best_active_count = None
        best_inactive_count = None
        best_raw_seed_count = None
        best_seed_active_mask = None
        best_active_seed_ids = None
        best_rho = None
        best_fiber_surface = None
        best_seeds = None
        best_pred = None
        best_fem_density_field = None
        best_fem_stress_field = None
        best_fem_displacement_field = None


 
        # ------------------------------------------------------------

        steps_since_improve = 0

        final_shape_density = None
        final_shape_fiber_direction = None
        seed_points_init = None
        seed_points_final = None
        seed_points_init_uv = None
        seed_points_final_uv = None
        rho0 = None
        seeds0 = None
        anchor_update_allowed = True
        history = []

        def reset_physical_checkpoint_trackers(reset_stage: int) -> None:
            nonlocal best_feasible_checkpoint
            nonlocal best_feasible_key
            nonlocal best_infeasible_checkpoint
            nonlocal best_infeasible_key
            nonlocal last_valid_checkpoint
            nonlocal selected_checkpoint_source
            nonlocal selected_checkpoint_stage_loss
            nonlocal returned_best_source
            nonlocal best_score
            nonlocal best_step
            nonlocal best_active_count
            nonlocal best_inactive_count
            nonlocal best_raw_seed_count
            nonlocal best_seed_active_mask
            nonlocal best_active_seed_ids
            nonlocal best_rho
            nonlocal best_fiber_surface
            nonlocal best_seeds
            nonlocal best_pred
            nonlocal best_fem_density_field
            nonlocal best_fem_stress_field
            nonlocal best_fem_displacement_field
            nonlocal steps_since_improve
            nonlocal physical_checkpoint_trackers_reset
            nonlocal stage2_loss_references

            best_feasible_checkpoint = None
            best_feasible_key = (float("inf"), float("inf"), float("inf"))
            best_infeasible_checkpoint = None
            best_infeasible_key = (float("inf"), float("inf"), float("inf"))
            last_valid_checkpoint = None
            selected_checkpoint_source = "unavailable"
            selected_checkpoint_stage_loss = float("nan")
            returned_best_source = "unavailable"
            best_score = float("inf")
            best_step = -1
            best_active_count = None
            best_inactive_count = None
            best_raw_seed_count = None
            best_seed_active_mask = None
            best_active_seed_ids = None
            best_rho = None
            best_fiber_surface = None
            best_seeds = None
            best_pred = None
            best_fem_density_field = None
            best_fem_stress_field = None
            best_fem_displacement_field = None
            steps_since_improve = 0
            if int(reset_stage) == 2:
                stage2_loss_references = {}
            physical_checkpoint_trackers_reset = True
            tqdm.write(
                "[Checkpoint reset] Reset physical checkpoint trackers "
                f"for Stage {int(reset_stage)}"
            )

        self.current_face_tensors = face_tensors
        debug_anomaly_detection = bool(getattr(cfg, "debug_anomaly_detection", False))
        if debug_anomaly_detection:
            torch.autograd.set_detect_anomaly(True, check_nan=True)

        stage2_loss_references: dict[str, torch.Tensor] = {}
        previous_stage_id: int | None = None

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------
        # The progress bar budget is derived from the staged schedule. Stage-local
        # counters control warm-up, main-stage stopping, and per-stage ramps.
        with tqdm(
            range(total_step_budget),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for step in pbar:
                # Logging Sequence
                should_log = (
                    step == 0
                    or step % cfg.log_every == 0
                    or step == total_step_budget - 1
                )
                # Timelapse  Sequence
                should_record_timelapse = (
                    bool(cfg.MakeTimelaps)
                    and step % int(cfg.timelapse_frame_step) == 0
                )
    
                stage_settings = self._stage_settings_for_stage_id(current_stage_runtime.spec.stage_id)
                stage_id = int(stage_settings["stage"])
                stage_local_step = int(current_stage_runtime.local_step)
                stage_max_steps = int(current_stage_runtime.spec.max_steps)

                # Stage-local warmup for allowing seeds outside the domain.
                allow_seed_outside_domain_step = self.allow_seed_outside_domain_for_step(
                    stage_local_step,
                    stage_max_steps,
                    stage_allow_seed_outside_domain=bool(stage_settings["allow_seed_outside_domain"]),
                )
                ppnet.allow_seed_outside_domain = allow_seed_outside_domain_step
                rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
                fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                seeds_list = []
                seed_xyz_list = []
                seed_active_mask_list = []
                pred_list = []
                density_post_stats_acc = {
                    "filter_delta_mean": 0.0,
                    "filter_delta_max": 0.0,
                    "projection_delta_mean": 0.0,
                    "projection_delta_max": 0.0,
                    "raw_mean": 0.0,
                    "filtered_mean": 0.0,
                    "projected_mean": 0.0,
                    "final_mean": 0.0,
                }
                density_post_stats_weight = 0.0

                rep_terms = []
                cvt_terms = []
                total_fiber_length_terms = []
                curve_length_values = []
                l_curve_cell_terms = []
                cell_area_values = []
                l_seed_terms = []
                h_terms = []
                participating_count_total = 0
                participating_frac_sum = 0.0
                inactive_count_total = 0
                inactive_frac_sum = 0.0
                raw_seed_count_total = 0
                topology_seed_count_total = 0
                soft_active_total = 0.0
                hard_active_total = 0.0

                # Activate losses based on the current stage lambdas. If a lambda is
                # zero, the corresponding loss is not computed and remains zero.
                if previous_stage_id != stage_id:
                    if stage_id == 2:
                        stage2_loss_references = {}
                    previous_stage_id = stage_id
                freeze_seed_motion_step = bool(stage_settings["freeze_seeds"])
                self._apply_stage_trainability(ppnet, stage_settings)
                lam_fem_base_step = float(stage_settings["lam_fem"])
                lam_fem_step = lam_fem_base_step
                if bool(getattr(cfg, "adaptive_fem_penalty", True)) and lam_fem_base_step != 0.0:
                    lam_fem_step = lam_fem_base_step * adaptive_lambda_fem
                lam_cvt_step = float(stage_settings["lam_cvt"])
                lam_rep_step = float(stage_settings["lam_rep"])
                lam_l_seed_step = float(stage_settings["lam_l_seed"])
                lam_total_fiber_length_step = float(stage_settings["lam_total_fiber_length"])
                lam_l_curve_cell_step = float(stage_settings["lam_l_curve_cell"])
                if (
                    not physical_checkpoint_trackers_reset
                    and int(stage_id) >= int(first_physical_stage)
                ):
                    scheduled_transition_checkpoint = last_valid_checkpoint
                    scheduled_transition_source = "last_valid"
                    if scheduled_transition_checkpoint is not None:
                        uv_anchor = self._restore_stage_checkpoint(
                            scheduled_transition_checkpoint,
                            ppnet,
                            decoder,
                        )
                        stage1_transition_step = int(
                            scheduled_transition_checkpoint.get("global_step", -1)
                        )
                        stage1_transition_score = float(
                            scheduled_transition_checkpoint.get("stage_monitor_raw", float("nan"))
                        )
                        stage1_transition_checkpoint_source = scheduled_transition_source
                        tqdm.write(
                            "[Stage transition] Restored Stage "
                            f"{int(scheduled_transition_checkpoint.get('stage_id', 1))} "
                            "topology checkpoint at "
                            f"global_step={stage1_transition_step}"
                        )
                    reset_physical_checkpoint_trackers(first_physical_stage)

                compute_rep_loss = lam_rep_step != 0.0
                compute_cvt_loss = lam_cvt_step != 0.0
                compute_l_seed_loss = lam_l_seed_step != 0.0
                compute_total_fiber_length_loss = lam_total_fiber_length_step != 0.0
                compute_l_curve_cell_loss = lam_l_curve_cell_step != 0.0
                # Checkpoint rows feed adaptive topology control and final timelapse
                # summaries, so keep topology diagnostics available on every step.
                collect_topology_metrics = True
                collect_curve_metrics = (
                    should_log
                    or should_record_timelapse
                    or compute_total_fiber_length_loss
                    or collect_topology_metrics
                )

                # Determine whether to update seed anchors based on the configuration and current step, seed anchors are reference points used in the training process.
                # if it is on, it will update the seed anchors after a certain warmup period, and the update is allowed based on the configuration settings.
                update_seed_anchors = (
                    cfg.use_rolling_seed_anchors 
                    # Stage-local anchor warmup is used for rolling seed anchors.
                    and (anchor_update_allowed or not cfg.anchor_guard_updates)
                    and not freeze_seed_motion_step
                )
                
                # print(f"use_rolling_seed_anchors: {cfg.use_rolling_seed_anchors}")
                # print(f"ancher_updated_allowed:{anchor_update_allowed}")
                # print(f"anchor_see_guard_updates:{cfg.anchor_guard_updates}")
                # print(f"freeze_seed_motion_step:{freeze_seed_motion_step}")
                # print(f"update_seed_anchers:{update_seed_anchors}")
                update_seed_anchors= True

                seed_offset_scale_step = self.seed_offset_scale_for_step(
                    stage_local_step,
                    stage_max_steps,
                )
                #seed_offset_scale_step=cfg.Offset_scale
                uv_anchor_next = None

                ft = face_tensor
                uv_anchor_i = uv_anchor
                if True:
                    pred_i = ppnet(uv_anchor_i, offset_scale=seed_offset_scale_step)
                    seeds_raw_i = pred_i["seeds_raw"]
                    if freeze_seed_motion_step:
                        seeds_raw_i = seeds_raw_i.detach()
                        pred_i["seeds_raw"] = seeds_raw_i
                    Gen_den_fiber= getattr(cfg, "generate_decoder_density_fiber", True)
                    if(current_stage_runtime.spec.stage_id==1):
                        Gen_den_fiber =False
                    decoder_out = decoder(
                        seeds_uv=seeds_raw_i,
                        generate_density_fiber=Gen_den_fiber,
                    )
                    seed_activity_weight_live_i = decoder_out["seed_activity_weight"]
                    seed_activity_weight_i = (
                        seed_activity_weight_live_i
                        if cfg.use_smooth_seed_activity_in_losses
                        else None
                    )
                    self._require_decoder_keys(
                        decoder_out,
                        [
                            "seeds",
                        ],
                    )
                    need_curve_geometry = needs_shared_curve_geometry(
                        compute_total_fiber_length_loss=compute_total_fiber_length_loss,
                        compute_l_curve_cell_loss=compute_l_curve_cell_loss,
                        collect_curve_metrics=collect_curve_metrics,
                        collect_topology_metrics=collect_topology_metrics,
                    )
                    # Builds a shared geometry container containing all edge-curve data,
                    # including edge lengths, graph connectivity, and validity masks,
                    # for reuse across multiple curve-based loss functions.
                    curve_geometry_i = (
                        build_shared_curve_geometry(
                            decoder_out,
                            require_edge_type=True,
                            require_edge_seed_pair=compute_l_curve_cell_loss,
                        )
                        if need_curve_geometry
                        else None
                    )
                    if need_curve_geometry and curve_geometry_i is None:
                        raise ValueError(
                            "Shared curve geometry was required, but decoder_out "
                            "does not contain edge_curves_xyz."
                        )
                    reported_curve_lengths_i = None
                    topology_metrics_i = None
                    if collect_curve_metrics:
                        reported_curve_lengths_i = self.curve_3d_edge_lengths(
                            curve_geometry_i,
                        ).detach()
                        curve_length_values.append(reported_curve_lengths_i)
                    if collect_topology_metrics:
                        topology_metrics_i = self.solution_topology_metrics(
                            decoder_out,
                            curve_lengths=reported_curve_lengths_i,
                        )
                        cell_areas_i, _ = self.cell_boundary_areas(curve_geometry_i)
                        if cell_areas_i.numel() > 0:
                            cell_area_values.append(cell_areas_i.detach())
                    if compute_total_fiber_length_loss:
                        # Activity-dependent participation coefficients remain graph-connected
                        # here, so they can contribute positional gradients through activity.
                        total_fiber_length_terms.append(
                            self.curve_network_length_loss(
                                curve_geometry_i,
                                seed_active_weights=seed_activity_weight_i,
                            )
                        )
                    if compute_l_curve_cell_loss:
                        # Activity-dependent participation coefficients remain graph-connected
                        # here, so they can contribute positional gradients through activity.
                        l_curve_cell_terms.append(
                            self.cell_edge_uniformity_loss(
                                curve_geometry_i,
                                seed_active_weights=seed_activity_weight_i,
                            )
                        )
                    if compute_l_seed_loss:
                        # Activity weights are differentiable functions of seed locations, not
                        # independent trainable parameters. Gradients through these weights update
                        # seed coordinates through the decoder's activity computation.
                        l_seed_terms.append(
                            self.loss_l_seed(
                                seed_active_weights=seed_activity_weight_live_i,
                                minimum_active=float(target_active_seeds),
                                possible_min_active=cfg.possible_min_active,
                                eps=cfg.eps,
                            )
                        )
                    if compute_cvt_loss:
                        seed_xyz_cvt = self._eval_uv_to_xyz_differentiable(seeds_raw_i)
                        cvt_terms.append(
                            self.loss_cvt(
                                seeds_uv=seeds_raw_i,
                                sample_uv=ft["uv"],
                                seed_xyz=seed_xyz_cvt,
                                sample_xyz=ft["points_xyz"],
                                sample_area_weights=A_local,
                                importance=None,
                                seed_active_weights=seed_activity_weight_i,
                                temperature=float(cfg.cvt_temperature),
                                activity_floor=cfg.activity_weight_floor,
                                activity_power=cfg.activity_weight_power,
                                activity_log_floor=cfg.cvt_activity_log_floor,
                                activity_temperature=cfg.cvt_activity_temperature,
                                eps=float(cfg.eps),
                            )
                        )

                    seeds_i = decoder_out["seeds"]
                    if getattr(cfg, "generate_decoder_density_fiber", True):
                        decoder_out, density_post_stats_i = apply_density_postprocess_to_output(
                            decoder_out,
                            ft,
                            cfg,
                            return_debug=True,
                        )
                        self._require_decoder_keys(
                            decoder_out,
                            [
                                "rho",
                                "fiber3d",
                            ],
                        )
                        rho_i = decoder_out["rho"]
                        fiber3d_i = decoder_out["fiber3d"]
                    else:
                        rho_i, fiber3d_i, density_post_stats_i = self.neutral_density_fiber_fields(
                            ft["uv"],
                            ft.get("Xu", None),
                        )
                    h_i = decoder_out.get("h", torch.zeros((), dtype=dtype, device=device))

                    activation_counts_i = self._decoder_seed_activation_counts(decoder_out)
                    active_count_i = int(activation_counts_i["active"])
                    inactive_count_i = int(activation_counts_i["inactive"])
                    total_seed_i = int(activation_counts_i["raw"])
                    topology_seed_count_i = int(activation_counts_i["topology"])
                    soft_active_i = float(activation_counts_i["soft_active"])
                    raw_seed_count_total += total_seed_i
                    topology_seed_count_total += topology_seed_count_i
                    if math.isfinite(soft_active_i):
                        soft_active_total += soft_active_i
                    hard_active_total += float(decoder_out["seed_active_mask"].sum().detach().item())
                    total_seed_i_for_frac = max(total_seed_i, 1)
                    participating_count_total += active_count_i
                    participating_frac_sum += active_count_i / float(total_seed_i_for_frac)
                    inactive_count_total += inactive_count_i
                    inactive_frac_sum += inactive_count_i / float(total_seed_i_for_frac)

                    for name, t in {
                        "seeds_i": seeds_i,
                        "rho_i": rho_i,
                        "fiber3d_i": fiber3d_i,
                    }.items():
                        if not torch.isfinite(t).all():
                            tqdm.write(f"[step {step}] face {ft['face_id']} invalid tensor: {name}")
                            raise RuntimeError(
                                f"Invalid decoder output on face {ft['face_id']} at step {step}"
                            )

                    gidx = ft["global_vertex_idx"]
                    w_local = A_local.clamp_min(cfg.eps)
                    stats_weight_i = float(w_local.detach().sum().item())
                    density_post_stats_weight += stats_weight_i
                    for key, value in density_post_stats_i.items():
                        if key.endswith("_max"):
                            density_post_stats_acc[key] = max(
                                density_post_stats_acc[key],
                                float(value),
                            )
                        else:
                            density_post_stats_acc[key] += float(value) * stats_weight_i

                    rho_acc[gidx] += rho_i * w_local
                    rho_wgt[gidx] += w_local

                    fiber_acc[gidx] += fiber3d_i * w_local[:, None]
                    fiber_wgt[gidx] += w_local

                    seeds_list.append(seeds_i)
                    seeds_xyz_i = decoder_out.get("seeds_xyz")
                    active_mask_i = decoder_out.get("seed_active_mask")
                    if isinstance(seeds_xyz_i, torch.Tensor):
                        if not isinstance(active_mask_i, torch.Tensor):
                            raise RuntimeError(
                                "Decoder returned seeds_xyz without seed_active_mask."
                            )
                        seed_xyz_list.append(seeds_xyz_i)
                        seed_active_mask_list.append(active_mask_i)
                    if update_seed_anchors:
                        anchor_alpha = float(cfg.seed_anchor_momentum)
                        uv_anchor_next_i = (
                            (1.0 - anchor_alpha) * uv_anchor_i + anchor_alpha * seeds_i.detach()
                        )
                    else:
                        uv_anchor_next_i = uv_anchor_i.detach().clone()
                    uv_anchor_next = uv_anchor_next_i

                    pred_list.append({
                        "face_id": self._face_id_key(ft.get("face_id", 0)),
                        "seeds_raw": seeds_raw_i.detach().clone(),
                        "h": h_i.detach().clone() if isinstance(h_i, torch.Tensor) else h_i,
                        "strut_thickness": float(cfg.strut_thickness),
                        "centerline_radius": decoder_out.get("centerline_radius", None).detach().clone() if isinstance(decoder_out.get("centerline_radius", None), torch.Tensor) else None,
                        "seeds_uv": decoder_out["seeds_uv"].detach().clone(),
                        "seed_active_mask": decoder_out["seed_active_mask"].detach().clone(),
                        "active_seed_ids": decoder_out["active_seed_ids"].detach().clone(),
                        "seed_activity_weight": decoder_out["seed_activity_weight"].detach().clone(),
                        "seed_box_activity_weight": decoder_out.get("seed_box_activity_weight", None).detach().clone() if isinstance(decoder_out.get("seed_box_activity_weight", None), torch.Tensor) else None,
                        "seed_duplicate_activity_weight": decoder_out.get("seed_duplicate_activity_weight", None).detach().clone() if isinstance(decoder_out.get("seed_duplicate_activity_weight", None), torch.Tensor) else None,
                        "topology_seeds_uv": decoder_out["topology_seeds_uv"].detach().clone(),
                        "seeds_xyz": decoder_out["seeds_xyz"].detach().clone() if isinstance(decoder_out.get("seeds_xyz"), torch.Tensor) else None,
                        "edge_curves_uv": decoder_out["edge_curves_uv"].detach().clone() if isinstance(decoder_out.get("edge_curves_uv"), torch.Tensor) else None,
                        "edge_curves_xyz": decoder_out["edge_curves_xyz"].detach().clone() if isinstance(decoder_out.get("edge_curves_xyz"), torch.Tensor) else None,
                        "edge_index": decoder_out["graph"]["edge_index"].detach().clone() if isinstance(decoder_out.get("graph"), dict) and isinstance(decoder_out["graph"].get("edge_index"), torch.Tensor) else None,
                        "edge_seed_pair": decoder_out["graph"]["edge_seed_pair"].detach().clone() if isinstance(decoder_out.get("graph"), dict) and isinstance(decoder_out["graph"].get("edge_seed_pair"), torch.Tensor) else None,
                        "edge_type": decoder_out["graph"]["edge_type"].detach().clone() if isinstance(decoder_out.get("graph"), dict) and isinstance(decoder_out["graph"].get("edge_type"), torch.Tensor) else None,
                        "graph": self._clone_detached_tree(decoder_out.get("graph")),
                        "number_of_edges": (
                            int(topology_metrics_i["number_of_selected_edges"])
                            if topology_metrics_i is not None
                            else None
                        ),
                        "number_of_total_edges": (
                            int(topology_metrics_i["number_of_total_edges"])
                            if topology_metrics_i is not None
                            else None
                        ),
                        "topology_identifier": (
                            topology_metrics_i["topology_identifier"]
                            if topology_metrics_i is not None
                            else ""
                        ),
                    })

                    if compute_rep_loss:
                        # Activity-dependent participation coefficients remain graph-connected
                        # here, so they can contribute positional gradients through activity.
                        rep_terms.append(
                            self.loss_rep(
                                seed_positions=decoder_out["seeds_xyz"],
                                seed_active_weights=seed_activity_weight_i,
                                target_dist=1.5 * float(cfg.strut_thickness),
                                activity_floor=cfg.activity_weight_floor,
                                activity_power=cfg.activity_weight_power,
                                recovery_floor=cfg.activity_recovery_floor,
                                recovery_power=cfg.activity_recovery_power,
                                duplicate_recovery_strength=cfg.repulsion_inactive_recovery_strength,
                                eps=cfg.eps,
                            )
                        )

                    h_terms.append(h_i.reshape(()))


                uv_anchor = uv_anchor_next

                # ----------------------------------------------------
                # Selected-face outputs
                # ----------------------------------------------------
                participating_count_mean = participating_count_total
                participating_frac_mean = participating_frac_sum
                inactive_count_mean = inactive_count_total
                inactive_frac_mean = inactive_frac_sum

                rho = rho_acc / rho_wgt.clamp_min(cfg.eps)
                density_post_stats = dict(density_post_stats_acc)
                if density_post_stats_weight > 0.0:
                    for key in [
                        "filter_delta_mean",
                        "projection_delta_mean",
                        "raw_mean",
                        "filtered_mean",
                        "projected_mean",
                        "final_mean",
                    ]:
                        density_post_stats[key] = (
                            density_post_stats_acc[key] / density_post_stats_weight
                        )

                fiber_surface = fiber_acc / fiber_wgt.clamp_min(cfg.eps)[:, None]
                fiber_norm = fiber_surface.norm(dim=1, keepdim=True).clamp_min(cfg.eps)
                fiber_surface = fiber_surface / fiber_norm

                zero = self._trainable_zero(ppnets, dtype=dtype, device=device)

                loss_rep = rep_terms[0] if compute_rep_loss and rep_terms else zero
                loss_total_fiber_length = (
                    total_fiber_length_terms[0]
                    if compute_total_fiber_length_loss and total_fiber_length_terms
                    else zero
                )
                loss_l_curve_cell = (
                    l_curve_cell_terms[0]
                    if compute_l_curve_cell_loss and l_curve_cell_terms
                    else zero
                )
                loss_cvt = (
                    cvt_terms[0]
                    if compute_cvt_loss and cvt_terms
                    else zero
                )
                loss_l_seed = (
                    l_seed_terms[0]
                    if compute_l_seed_loss and l_seed_terms
                    else zero
                )

                # ----------------------------------------------------
                # FEM loss
                # ----------------------------------------------------
                fem_out = {
                    "fem_total": torch.zeros((), dtype=dtype, device=device),
                    "fem_valid": True,
                    "failure_reason": None,
                    "density_field": None,
                    "stress_field": None,
                    "displacement_field": None,
                    "loaded_boundary_displacement_field": None,
                    "stress_constraint_loss": torch.zeros((), dtype=dtype, device=device),
                    "displacement_constraint_loss": torch.zeros((), dtype=dtype, device=device),
                    "baseline_fem_loss": torch.zeros((), dtype=dtype, device=device),
                    "violation_fem_loss": torch.zeros((), dtype=dtype, device=device),
                    "stress_constraint_excess": torch.zeros((), dtype=dtype, device=device),
                    "displacement_constraint_excess": torch.zeros((), dtype=dtype, device=device),
                    "stress_ratio": torch.zeros((), dtype=dtype, device=device),
                    "displacement_ratio": torch.zeros((), dtype=dtype, device=device),
                    "training_stress_ratio": torch.zeros((), dtype=dtype, device=device),
                    "training_displacement_ratio": torch.zeros((), dtype=dtype, device=device),
                    "physical_stress_ratio": torch.zeros((), dtype=dtype, device=device),
                    "physical_displacement_ratio": torch.zeros((), dtype=dtype, device=device),
                    "constraint_violation": torch.zeros((), dtype=dtype, device=device),
                    "training_feasible": True,
                    "physical_feasible": True,
                    "stress_max": torch.zeros((), dtype=dtype, device=device),
                    "displacement_max": torch.zeros((), dtype=dtype, device=device),
                }

                fem_was_evaluated = False
                if lam_fem_step != 0.0 and cfg.generate_decoder_density_fiber:
                    fem_was_evaluated = True
                    fem_out = self.loss_fem.evaluate(
                        rho_surface=rho,
                        fiber_surface=fiber_surface,
                        max_displacement=cfg.fem_max_displacement,
                        yield_strength=cfg.fem_yield_strength,
                        constraint_weight=cfg.fem_constraint_weight,
                        baseline_weight=cfg.fem_baseline_weight,
                        violation_power=cfg.fem_violation_power,
                        stress_density_threshold=(
                            cfg.fem_stress_density_threshold
                            if cfg.fem_stress_density_threshold is not None
                            else max(
                                float(getattr(cfg, "vis_thr", 0.5)),
                                float(getattr(cfg, "fem_rho_min_ratio", 1.0e-5)) * 1.05,
                            )
                        ),
                        rho_min_ratio=self._scheduled_fem_rho_min_ratio(
                            stage_local_step,
                            stage_max_steps,
                        ),
                        penal=cfg.fem_penal,
                        eps=cfg.eps,
                        save_debug_history=getattr(cfg, "save_fem_debug_history", True),
                    )

                loss_fem = fem_out["fem_total"]
                fem_violation_loss = fem_out.get("violation_fem_loss", zero)
                loss_fem_stress_constraint = fem_out.get("stress_constraint_loss", zero)
                loss_fem_displacement_constraint = fem_out.get("displacement_constraint_loss", zero)
                fem_is_valid = bool(fem_out["fem_valid"]) and bool(fem_was_evaluated)
                fem_failure_reason = fem_out["failure_reason"]
                fem_density_field = fem_out.get("density_field", None)
                fem_stress_field = fem_out.get("stress_field", None)
                fem_displacement_field = fem_out.get("displacement_field", None)
                fem_loaded_boundary_displacement_field = fem_out.get("loaded_boundary_displacement_field", None)
                stress_density_threshold = float(cfg.fem_stress_density_threshold)
                if isinstance(fem_stress_field, torch.Tensor) and fem_stress_field.numel() > 0:
                    finite_stress = fem_stress_field.detach().reshape(-1)
                    if (
                        isinstance(fem_density_field, torch.Tensor)
                        and fem_density_field.numel() == fem_stress_field.numel()
                    ):
                        finite_stress = finite_stress[
                            fem_density_field.detach().reshape(-1) >= stress_density_threshold
                        ]
                    finite_stress = finite_stress[torch.isfinite(finite_stress)]
                    if finite_stress.numel() > 0:
                        stress_max = self._finite_or_default(fem_out.get("stress_max", zero))
                        stress_p95 = float(torch.quantile(finite_stress, 0.95).item())
                        stress_p99 = float(torch.quantile(finite_stress, 0.99).item())
                    else:
                        stress_max = float("nan")
                        stress_p95 = float("nan")
                        stress_p99 = float("nan")
                else:
                    stress_max = float("nan")
                    stress_p95 = float("nan")
                    stress_p99 = float("nan")
                if isinstance(fem_displacement_field, torch.Tensor) and fem_displacement_field.numel() > 0:
                    finite_disp_field = fem_displacement_field.detach().reshape(-1)
                    finite_disp_field = finite_disp_field[torch.isfinite(finite_disp_field)]
                    if finite_disp_field.numel() > 0:
                        disp_field_max = float(finite_disp_field.max().item())
                        disp_field_p95 = float(torch.quantile(finite_disp_field, 0.95).item())
                        disp_field_p99 = float(torch.quantile(finite_disp_field, 0.99).item())
                    else:
                        disp_field_max = float("nan")
                        disp_field_p95 = float("nan")
                        disp_field_p99 = float("nan")
                else:
                    disp_field_max = float("nan")
                    disp_field_p95 = float("nan")
                    disp_field_p99 = float("nan")

                disp_metric_source = fem_loaded_boundary_displacement_field
                if not isinstance(disp_metric_source, torch.Tensor) or disp_metric_source.numel() <= 0:
                    disp_metric_source = fem_displacement_field
                if isinstance(disp_metric_source, torch.Tensor) and disp_metric_source.numel() > 0:
                    finite_disp_metric = disp_metric_source.detach().reshape(-1)
                    finite_disp_metric = finite_disp_metric[torch.isfinite(finite_disp_metric)]
                    if finite_disp_metric.numel() > 0:
                        disp_mean = float(finite_disp_metric.mean().item())
                        disp_max = self._finite_or_default(fem_out.get("displacement_max", zero))
                        disp_p95 = float(torch.quantile(finite_disp_metric, 0.95).item())
                        disp_p99 = float(torch.quantile(finite_disp_metric, 0.99).item())
                    else:
                        disp_mean = float("nan")
                        disp_max = float("nan")
                        disp_p95 = float("nan")
                        disp_p99 = float("nan")
                else:
                    disp_mean = float("nan")
                    disp_max = float("nan")
                    disp_p95 = float("nan")
                    disp_p99 = float("nan")

                # ----------------------------------------------------
                # Normalize losses
                # ----------------------------------------------------

                stage2_fixed_norm_active = bool(cfg.normalize_losses) and int(stage_id) == 2
                stage2_reference_capture_valid = (
                    stage2_fixed_norm_active
                    and (
                        lam_fem_step == 0.0
                        or not bool(cfg.generate_decoder_density_fiber)
                        or bool(fem_is_valid)
                    )
                )
                if stage2_reference_capture_valid:
                    for _ref_name, _ref_loss, _ref_enabled in (
                        ("total_fiber_length", loss_total_fiber_length, compute_total_fiber_length_loss),
                        ("cvt", loss_cvt, compute_cvt_loss),
                        ("repulsion", loss_rep, compute_rep_loss),
                        ("seed_active", loss_l_seed, compute_l_seed_loss),
                        ("cell_edge_uniformity", loss_l_curve_cell, compute_l_curve_cell_loss),
                    ):
                        self._capture_fixed_stage2_reference(
                            stage2_loss_references,
                            _ref_name,
                            _ref_loss,
                            _ref_enabled,
                            cfg.eps,
                        )

                if stage2_fixed_norm_active:
                    n_cvt = stage2_loss_references.get("cvt", loss_cvt.new_tensor(1.0))
                    n_rep = stage2_loss_references.get("repulsion", loss_rep.new_tensor(1.0))
                    n_l_seed = stage2_loss_references.get("seed_active", loss_l_seed.new_tensor(1.0))
                    n_total_fiber_length = stage2_loss_references.get(
                        "total_fiber_length",
                        loss_total_fiber_length.new_tensor(1.0),
                    )
                    n_l_curve_cell = stage2_loss_references.get(
                        "cell_edge_uniformity",
                        loss_l_curve_cell.new_tensor(1.0),
                    )
                    n_fem = 1.0
                elif cfg.normalize_losses:
                    n_cvt = norm_cvt.update_if_active(
                        loss_cvt.detach().item(),
                        compute_cvt_loss,
                    )
                    n_rep = norm_rep.update_if_active(
                        loss_rep.detach().item(),
                        compute_rep_loss,
                    )
                    n_l_seed = norm_l_seed.update_if_active(
                        loss_l_seed.detach().item(),
                        compute_l_seed_loss,
                    )
                    n_total_fiber_length = norm_total_fiber_length.update_if_active(
                        loss_total_fiber_length.detach().item(),
                        compute_total_fiber_length_loss,
                    )
                    n_l_curve_cell = norm_l_curve_cell.update_if_active(
                        loss_l_curve_cell.detach().item(),
                        compute_l_curve_cell_loss,
                    )
                    # FEM represents hard mechanical constraint violations.
                    # Do not normalize it by its running magnitude because a severely failed
                    # structure must remain strongly penalized.
                    n_fem = 1.0
                    n_l_seed=1.0
                else:
                    n_cvt = n_rep = n_fem = n_l_seed = n_total_fiber_length = n_l_curve_cell = 1.0

                # FEM represents hard mechanical constraint violations.
                # Do not normalize it by its running magnitude because a severely failed
                # structure must remain strongly penalized.
                n_fem = 1.0
                n_l_seed= 1.0

                loss_cvt_normalized = self._fixed_reference_normalized(
                    loss_cvt,
                    n_cvt,
                    cfg.eps,
                )
                loss_rep_normalized = self._fixed_reference_normalized(
                    loss_rep,
                    n_rep,
                    cfg.eps,
                )
                # loss_l_seed_normalized = self._fixed_reference_normalized(
                #     loss_l_seed,
                #     n_l_seed,
                #     cfg.eps,
                # )
                loss_total_fiber_length_normalized = self._fixed_reference_normalized(
                    loss_total_fiber_length,
                    n_total_fiber_length,
                    cfg.eps,
                )
                loss_l_curve_cell_normalized = self._fixed_reference_normalized(
                    loss_l_curve_cell,
                    n_l_curve_cell,
                    cfg.eps,
                )
                loss_fem_normalized = loss_fem
                loss_l_seed_normalized = loss_l_seed

                # ----------------------------------------------------
                # Total loss
                # ----------------------------------------------------
                design_score = (
                    zero
                    + lam_total_fiber_length_step * loss_total_fiber_length_normalized
                    + lam_cvt_step * loss_cvt_normalized
                    + lam_rep_step * loss_rep_normalized
                    + lam_l_curve_cell_step * loss_l_curve_cell_normalized
                    + lam_l_seed_step * loss_l_seed
                )
                stage2_objective_mode = "design_only"
                if int(stage_id) == 2:
                    L_total, design_score, stage2_objective_mode = self._assemble_feasibility_first_stage2_loss(
                        fem_total_loss=loss_fem,
                        fem_violation_loss=fem_violation_loss,
                        loss_total_fiber_length_stage2_norm=loss_total_fiber_length_normalized,
                        loss_cvt_normalized=loss_cvt_normalized,
                        loss_rep_normalized=loss_rep_normalized,
                        loss_seed=loss_l_seed,
                        loss_curve_cell_normalized=loss_l_curve_cell_normalized,
                        lam_fem_step=lam_fem_step,
                        lam_total_fiber_length_step=lam_total_fiber_length_step,
                        lam_cvt_step=lam_cvt_step,
                        lam_rep_step=lam_rep_step,
                        lam_l_seed_step=lam_l_seed_step,
                        lam_l_curve_cell_step=lam_l_curve_cell_step,
                    )

                else:
                    # Stage 1 or a stage without FEM.
                    L_total = (
                        zero
                        + lam_cvt_step * loss_cvt_normalized
                        + lam_rep_step * loss_rep_normalized
                        + lam_l_seed_step * loss_l_seed
                        + lam_total_fiber_length_step
                        * loss_total_fiber_length_normalized
                        + lam_l_curve_cell_step
                        * loss_l_curve_cell_normalized
                    )

                min_active_required = int(cfg.min_active_seeds or 1)
                active_seed_feasible = float(hard_active_total) >= float(min_active_required)
                active_seed_violation_value = max(
                    float(min_active_required) / max(float(hard_active_total), 1.0) - 1.0,
                    0.0,
                )
                active_seed_violation = zero + float(active_seed_violation_value)
                fem_constraint_violation = fem_out.get("constraint_violation", zero)
                overall_constraint_violation = torch.maximum(
                    fem_constraint_violation.reshape(()),
                    active_seed_violation.reshape(()),
                )
                physical_feasible = bool(
                    fem_is_valid
                    and bool(fem_out.get("physical_feasible", False))
                )
                overall_feasible = bool(
                    physical_feasible
                    and active_seed_feasible
                )
                stage_monitor_mode = (
                    "design"
                    if (int(stage_id) != 2 or overall_feasible)
                    else "recovery"
                )

                total_is_finite = self._scalar_tensor_is_finite(L_total)
                loss_debug_terms = [
                    ("L_total", L_total),
                    ("loss_cvt", loss_cvt),
                    ("loss_rep", loss_rep),
                    ("loss_l_seed", loss_l_seed),
                    ("loss_total_fiber_length", loss_total_fiber_length),
                    ("loss_l_curve_cell", loss_l_curve_cell),
                    ("loss_fem", loss_fem),
                    ("loss_fem_stress_constraint", loss_fem_stress_constraint),
                    ("loss_fem_displacement_constraint", loss_fem_displacement_constraint),
                ]

                # ----------------------------------------------------
                # Backprop
                # ----------------------------------------------------
                opt.zero_grad(set_to_none=True)

                optimizer_step_skipped = True
                fem_required = lam_fem_step != 0.0 and bool(cfg.generate_decoder_density_fiber)
                skip_entire_step = fem_required and not bool(fem_is_valid)
                invalid_step_reason = None
                if skip_entire_step:
                    consecutive_invalid_fem_steps += 1
                    invalid_step_reason = fem_failure_reason or "Invalid FEM result"
                    lr_changes = []
                    if consecutive_invalid_fem_steps >= int(cfg.invalid_fem_patience):
                        lr_changes = self._reduce_optimizer_lr(
                            opt,
                            factor=float(cfg.invalid_lr_factor),
                            minimum_lr=float(cfg.minimum_learning_rate),
                        )
                    if lr_changes:
                        lr_text = ", ".join(
                            f"{old_lr:.3e}->{new_lr:.3e}"
                            for old_lr, new_lr in lr_changes
                        )
                    else:
                        lr_text = "unchanged"
                    tqdm.write(
                        f"[step {step}] FEM invalid; optimizer step skipped. "
                        f"reason={invalid_step_reason}; "
                        f"consecutive_invalid={consecutive_invalid_fem_steps}; "
                        f"lr={lr_text}."
                    )
                elif total_is_finite:
                    L_total.backward()

                    bad_grad_info = self._nonfinite_grad_info(trainable_modules)
                    if bad_grad_info:
                        consecutive_invalid_fem_steps += 1
                        cause_desc = self._nonfinite_grad_cause_summary(
                            trainable_modules,
                            bad_grad_info,
                            loss_terms=loss_debug_terms,
                            fem_is_valid=fem_is_valid,
                            fem_failure_reason=fem_failure_reason,
                        )
                        tqdm.write(
                            f"[step {step}] Non-finite gradients detected; optimizer step skipped. "
                            f"{cause_desc}."
                        )
                        for _mi, _pn, p in self._named_trainable_params(trainable_modules):
                            if p.grad is not None:
                                p.grad = None
                        if consecutive_invalid_fem_steps >= int(cfg.invalid_fem_patience):
                            lr_changes = self._reduce_optimizer_lr(
                                opt,
                                factor=float(cfg.invalid_lr_factor),
                                minimum_lr=float(cfg.minimum_learning_rate),
                            )
                            tqdm.write(
                                f"[step {step}] Reduced learning rate after invalid gradients: "
                                + ", ".join(f"{old:.3e}->{new:.3e}" for old, new in lr_changes)
                            )
                    else:
                        pre_step_snapshot = {
                            p: p.detach().clone()
                            for _mi, _pn, p in self._named_trainable_params(trainable_modules)
                        }

                        grad_clip_norm = getattr(cfg, "grad_clip_norm", None)
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            params = [
                                p
                                for _mi, _pn, p in self._named_trainable_params(trainable_modules)
                            ]
                            if params:
                                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

                        bad_grad_info = self._nonfinite_grad_info(trainable_modules)
                        if bad_grad_info:
                            consecutive_invalid_fem_steps += 1
                            cause_desc = self._nonfinite_grad_cause_summary(
                                trainable_modules,
                                bad_grad_info,
                                loss_terms=loss_debug_terms,
                                fem_is_valid=fem_is_valid,
                                fem_failure_reason=fem_failure_reason,
                            )
                            tqdm.write(
                                f"[step {step}] Non-finite gradients remained after clipping, "
                                f"optimizer step skipped. {cause_desc}."
                            )
                            for _mi, _pn, p in self._named_trainable_params(trainable_modules):
                                if p.grad is not None:
                                    p.grad = None
                            if consecutive_invalid_fem_steps >= int(cfg.invalid_fem_patience):
                                lr_changes = self._reduce_optimizer_lr(
                                    opt,
                                    factor=float(cfg.invalid_lr_factor),
                                    minimum_lr=float(cfg.minimum_learning_rate),
                                )
                                tqdm.write(
                                    f"[step {step}] Reduced learning rate after clipped-gradient failure: "
                                    + ", ".join(f"{old:.3e}->{new:.3e}" for old, new in lr_changes)
                                )
                        else:
                            opt.step()

                            bad_param_info = self._nonfinite_param_info(trainable_modules)
                            if bad_param_info:
                                bad_param_desc = ", ".join(
                                    f"face={mi}:{pn}" for mi, pn in bad_param_info[:8]
                                )
                                consecutive_invalid_fem_steps += 1
                                if latest_valid_model_state is not None:
                                    self._restore_modules_state_dict(
                                        trainable_modules,
                                        latest_valid_model_state,
                                    )
                                else:
                                    self._restore_param_snapshot(pre_step_snapshot)
                                if latest_valid_optimizer_state is not None:
                                    opt.load_state_dict(copy.deepcopy(latest_valid_optimizer_state))
                                for _mi, _pn, p in self._named_trainable_params(trainable_modules):
                                    if p.grad is not None:
                                        p.grad = None
                                if consecutive_invalid_fem_steps >= int(cfg.invalid_fem_patience):
                                    lr_changes = self._reduce_optimizer_lr(
                                        opt,
                                        factor=float(cfg.invalid_lr_factor),
                                        minimum_lr=float(cfg.minimum_learning_rate),
                                    )
                                    lr_desc = ", ".join(f"{old:.3e}->{new:.3e}" for old, new in lr_changes)
                                else:
                                    lr_desc = "unchanged"
                                tqdm.write(
                                    f"[step {step}] Non-finite parameters after opt.step(); restored latest "
                                    f"valid model and optimizer state. Examples: {bad_param_desc}; lr={lr_desc}"
                                )
                            else:
                                optimizer_step_skipped = False
                                consecutive_invalid_fem_steps = 0
                                if scheduler is not None:
                                    scheduler.step()
                                latest_valid_model_state = self._clone_modules_state_dict(trainable_modules)
                                latest_valid_optimizer_state = self._clone_optimizer_state_dict(opt)
                else:
                    consecutive_invalid_fem_steps += 1
                    tqdm.write(f"[step {step}] L_total is non-finite, optimizer step skipped.")

                adaptive_lambda_before = adaptive_lambda_fem
                adaptive_lambda_update_reason = "inactive"
                if (
                    bool(getattr(cfg, "adaptive_fem_penalty", True))
                    and lam_fem_base_step != 0.0
                ):
                    fem_constraint_violation = (
                        float(fem_out["constraint_violation"].detach().item())
                        if isinstance(fem_out.get("constraint_violation", None), torch.Tensor)
                        else float("nan")
                    )
                    adaptive_lambda_fem, adaptive_lambda_update_reason = update_adaptive_fem_lambda(
                        lambda_before=adaptive_lambda_fem,
                        fem_is_valid=bool(fem_is_valid),
                        constraint_violation=fem_constraint_violation,
                        tolerance=float(cfg.fem_constraint_tolerance),
                        growth=float(cfg.fem_lambda_growth),
                        decay=float(cfg.fem_lambda_decay),
                        lambda_min=float(cfg.fem_lambda_min),
                        lambda_max=float(cfg.fem_lambda_max),
                    )

                # ----------------------------------------------------
                # Logging / tracking
                # ----------------------------------------------------
                with torch.no_grad():
                    vol_frac = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                    min_active_seed_dist = self.min_pairwise_active_seed_distance(
                        seed_xyz_list,
                        seed_active_mask_list,
                    )

                    score = float(L_total.detach().item()) if total_is_finite else float("inf")
                    if not (total_is_finite and fem_is_valid):
                        score = float("inf")

                    best_candidate_is_valid = (
                        total_is_finite
                        and (
                            int(stage_id) < int(first_physical_stage)
                            or bool(fem_is_valid)
                        )
                    )
                    best_candidate_is_physically_feasible = (
                        best_candidate_is_valid
                        and (
                            lam_fem_step == 0.0
                            or bool(fem_out.get("physical_feasible", False))
                        )
                    )
                    curve_length_summary = self._curve_length_summary(
                        curve_length_values,
                        zero,
                    )
                    curve_length_min = curve_length_summary["min"]
                    curve_length_max = curve_length_summary["max"]
                    curve_length_mean = curve_length_summary["mean"]
                    curve_length_std = curve_length_summary["std"]
                    curve_length_cv = curve_length_summary["cv"]
                    curve_length_ratio = curve_length_summary["ratio"]

                    prev_best_step = best_step
                    improvement_gap = (step - prev_best_step) if prev_best_step >= 0 else None


                    if best_candidate_is_physically_feasible:
                        steps_since_improve += 1

                    best_candidate_improved = (
                        best_candidate_is_physically_feasible
                        and self.is_meaningful_improvement(
                            score,
                            best_score,
                            current_stage_runtime.spec.min_delta_abs,
                            current_stage_runtime.spec.min_delta_rel,
                        )
                    )

                    if best_candidate_improved:
                        best_score = score
                        best_step = step
                        best_active_count = float(participating_count_total)
                        best_inactive_count = float(inactive_count_total)
                        best_raw_seed_count = int(raw_seed_count_total)
                        if pred_list and isinstance(pred_list[0].get("seed_active_mask"), torch.Tensor):
                            best_seed_active_mask = pred_list[0]["seed_active_mask"].detach().clone()
                        if pred_list and isinstance(pred_list[0].get("active_seed_ids"), torch.Tensor):
                            best_active_seed_ids = pred_list[0]["active_seed_ids"].detach().clone()
                        best_rho = rho.detach().clone()
                        best_fiber_surface = fiber_surface.detach().clone()
                        best_seeds = [s.detach().clone() for s in seeds_list]
                        best_pred = self._clone_pred_list(pred_list)
                        best_fem_density_field = (
                            fem_density_field.detach().clone()
                            if isinstance(fem_density_field, torch.Tensor)
                            else None
                        )
                        best_fem_stress_field = (
                            fem_stress_field.detach().clone()
                            if isinstance(fem_stress_field, torch.Tensor)
                            else None
                        )
                        best_fem_displacement_field = (
                            fem_displacement_field.detach().clone()
                            if isinstance(fem_displacement_field, torch.Tensor)
                            else None
                        )

                        if improvement_gap is None or improvement_gap >= 50:
                            best_volfrac_report = float(vol_frac.detach().item())
                            tqdm.write(
                                f"[L-total improvement] New best_step={best_step} | "
                                f"L_total={best_score:.6f} | "
                                f"best_active_units={best_active_count:.1f} | "
                                f"VolFrac={best_volfrac_report:.6f} | "
                                f"L(min/max/ratio)="
                                f"{curve_length_min:.6e}/{curve_length_max:.6e}/{curve_length_ratio:.2f}"
                            )

                    if rho0 is None:
                        rho0 = rho.detach().clone()
                    if seeds0 is None:
                        seeds0 = [s.detach().clone() for s in seeds_list]

                    drho = float((rho - rho0).abs().mean().item())
                    dseed_terms = [float((s - s0).abs().mean().item()) for s, s0 in zip(seeds_list, seeds0)]
                    dseed = sum(dseed_terms) / max(len(dseed_terms), 1)

                    rho_min = float(rho.min().item())
                    rho_mean = float(rho.mean().item())
                    rho_max = float(rho.max().item())
                    solution_metrics = (
                        dict(topology_metrics_i)
                        if topology_metrics_i is not None
                        else {
                            "minimum_length": float("nan"),
                            "maximum_length": float("nan"),
                            "mean_length": float("nan"),
                            "standard_deviation": float("nan"),
                            "coefficient_of_variation": float("nan"),
                            "maximum_minimum_ratio": float("nan"),
                            "number_of_selected_edges": float("nan"),
                            "number_of_total_edges": float("nan"),
                            "number_of_edges": float("nan"),
                            "topology_identifier": "",
                        }
                    )
                    solution_metrics.update({
                        "minimum_length": curve_length_min,
                        "maximum_length": curve_length_max,
                        "mean_length": curve_length_mean,
                        "standard_deviation": curve_length_std,
                        "coefficient_of_variation": curve_length_cv,
                        "maximum_minimum_ratio": curve_length_ratio,
                    })
                    if cell_area_values:
                        finite_cell_areas = torch.cat(cell_area_values).reshape(-1)
                        finite_cell_areas = finite_cell_areas[torch.isfinite(finite_cell_areas)]
                    else:
                        finite_cell_areas = zero.new_empty((0,))
                    if finite_cell_areas.numel() > 0:
                        cell_area_min = float(finite_cell_areas.min().item())
                        cell_area_mean = float(finite_cell_areas.mean().item())
                        cell_area_max = float(finite_cell_areas.max().item())
                    else:
                        cell_area_min = float("nan")
                        cell_area_mean = float("nan")
                        cell_area_max = float("nan")

                    g_mean = 0.0
                    g_count = 0
                    for p in ppnet.parameters():
                        if p.grad is not None:
                            g_mean += float(p.grad.detach().abs().mean().item())
                            g_count += 1
                    g_mean = g_mean / max(g_count, 1)

                    volfrac_scalar = float(vol_frac.detach().item())
                    l_fem_current = self._finite_or_default(loss_fem)
                    l_fem_initial = next(
                        (
                            float(r["loss_fem"])
                            for r in history
                            if math.isfinite(float(r.get("loss_fem", float("nan"))))
                            and float(r.get("loss_fem", float("nan"))) > 0.0
                        ),
                        l_fem_current,
                    )
                    l_fem_candidates = [
                        float(r["loss_fem"])
                        for r in history
                        if math.isfinite(float(r.get("loss_fem", float("nan"))))
                    ]
                    if math.isfinite(float(l_fem_current)):
                        l_fem_candidates.append(l_fem_current)
                    best_l_fem_so_far = min(l_fem_candidates) if l_fem_candidates else float("nan")
                    disp_mean_initial = next(
                        (
                            float(r["disp_mean"])
                            for r in history
                            if math.isfinite(float(r.get("disp_mean", float("nan"))))
                            and float(r.get("disp_mean", float("nan"))) > 0.0
                        ),
                        disp_mean,
                    )
                    stress_p95_initial = next(
                        (
                            float(r["stress_p95"])
                            for r in history
                            if math.isfinite(float(r.get("stress_p95", float("nan"))))
                            and float(r.get("stress_p95", float("nan"))) > 0.0
                        ),
                        stress_p95,
                    )
                    l_fem_norm = (
                        l_fem_current / l_fem_initial
                        if math.isfinite(float(l_fem_initial)) and l_fem_initial > 0.0
                        else float("nan")
                    )
                    disp_norm = (
                        disp_mean / disp_mean_initial
                        if math.isfinite(float(disp_mean_initial)) and disp_mean_initial > 0.0 and math.isfinite(float(disp_mean))
                        else float("nan")
                    )
                    stress_norm = (
                        stress_p95 / stress_p95_initial
                        if math.isfinite(float(stress_p95_initial)) and stress_p95_initial > 0.0 and math.isfinite(float(stress_p95))
                        else float("nan")
                    )

                    effective_lambdas = {
                        "lam_fem": lam_fem_step,
                        "lam_fem_base": lam_fem_base_step,
                        "adaptive_lambda_fem": adaptive_lambda_fem,
                        "lam_cvt": lam_cvt_step,
                        "lam_rep": lam_rep_step,
                        "lam_l_seed": lam_l_seed_step,
                        "lam_total_fiber_length": lam_total_fiber_length_step,
                        "lam_l_curve_cell": lam_l_curve_cell_step,
                    }
                    monitor_loss_values = {
                        "L_total": L_total.detach(),
                        "loss_fem_norm": loss_fem_normalized.detach(),
                        "loss_cvt_norm": loss_cvt_normalized.detach(),
                        "loss_l_curve_cell_norm": loss_l_curve_cell_normalized.detach(),
                        "loss_total_fiber_length_norm": loss_total_fiber_length_normalized.detach(),
                        "loss_rep_norm": loss_rep_normalized.detach(),
                        "loss_l_seed_norm": loss_l_seed_normalized.detach(),
                        "design_score": design_score.detach(),
                        "mechanical_violation": fem_out.get("constraint_violation", zero).detach(),
                        "fem_constraint_violation": fem_constraint_violation.detach(),
                        "active_seed_violation": active_seed_violation.detach(),
                        "overall_constraint_violation": overall_constraint_violation.detach(),
                        "physical_feasible": bool(physical_feasible),
                        "overall_feasible": overall_feasible,
                        "_zero": zero.detach(),
                    }
                    stage_monitor_raw_tensor = self.calculate_stage_monitor(
                        stage_id,
                        monitor_loss_values,
                        effective_lambdas,
                    )
                    stage_monitor_raw = self._finite_or_default(stage_monitor_raw_tensor, default=float("inf"))
                    best_stage_monitor = float(current_stage_runtime.best_raw_monitor)
                    stage_patience_counter = int(current_stage_runtime.patience_counter)
                    stage_patience_limit = int(current_stage_runtime.spec.patience)
                    topology_grace_remaining = int(current_stage_runtime.topology_grace_remaining)
                    stage_local_for_row = int(stage_local_step)
                    stage_max_for_row = int(current_stage_runtime.spec.max_steps)
                    row = {
                        "step": step,
                        "stage": stage_id,
                        "stage_local_step": stage_local_for_row,
                        "stage_max_steps": stage_max_for_row,
                        "stage_monitor_raw": stage_monitor_raw,
                        "best_stage_monitor": best_stage_monitor,
                        "stage_patience_counter": stage_patience_counter,
                        "stage_patience_limit": stage_patience_limit,
                        "topology_grace_remaining": topology_grace_remaining,
                        "topology_grace_resets_used": (
                            int(current_stage_runtime.topology_grace_resets_used)
                            if current_stage_runtime is not None
                            else 0
                        ),
                        "stage_freeze_seeds": 1.0 if freeze_seed_motion_step else 0.0,
                        "stage_allow_seed_outside_domain": (
                            1.0 if bool(stage_settings["allow_seed_outside_domain"]) else 0.0
                        ),
                        "allow_seed_outside_domain": 1.0 if allow_seed_outside_domain_step else 0.0,
                        "lam_fem_eff": lam_fem_step,
                        "lam_fem_base": lam_fem_base_step,
                        "adaptive_lambda_fem": adaptive_lambda_fem,
                        "adaptive_lambda_fem_before": adaptive_lambda_before,
                        "adaptive_lambda_update_reason": adaptive_lambda_update_reason,
                        "lam_cvt_eff": lam_cvt_step,
                        "lam_rep_eff": lam_rep_step,
                        "lam_l_seed_eff": lam_l_seed_step,
                        "lam_total_fiber_length_eff": lam_total_fiber_length_step,
                        "lam_l_curve_cell_eff": lam_l_curve_cell_step,
                        "L_total": self._finite_or_default(L_total),
                        "L_train": self._finite_or_default(L_total),
                        "design_score": self._finite_or_default(design_score),
                        "stage2_objective_mode": stage2_objective_mode,
                        "stage_monitor_mode": stage_monitor_mode,
                        "loss_rep": self._finite_or_default(loss_rep),
                        "loss_fem_norm": self._finite_or_default(monitor_loss_values["loss_fem_norm"]),
                        "loss_cvt_norm": self._finite_or_default(monitor_loss_values["loss_cvt_norm"]),
                        "loss_total_fiber_length_norm": self._finite_or_default(monitor_loss_values["loss_total_fiber_length_norm"]),
                        "loss_rep_norm": self._finite_or_default(monitor_loss_values["loss_rep_norm"]),
                        "loss_l_seed_norm": self._finite_or_default(monitor_loss_values["loss_l_seed_norm"]),
                        "loss_l_curve_cell_norm": self._finite_or_default(monitor_loss_values["loss_l_curve_cell_norm"]),
                        "loss_cvt_reference": self._finite_or_default(n_cvt),
                        "loss_rep_reference": self._finite_or_default(n_rep),
                        "loss_l_seed_reference": self._finite_or_default(n_l_seed),
                        "loss_total_fiber_length_reference": self._finite_or_default(n_total_fiber_length),
                        "loss_l_curve_cell_reference": self._finite_or_default(n_l_curve_cell),
                        "loss_fem_reference": 1.0,
                        "loss_cvt": self._finite_or_default(loss_cvt),
                        "loss_l_seed": self._finite_or_default(loss_l_seed),
                        "loss_total_fiber_length": self._finite_or_default(loss_total_fiber_length),
                        "curve_length_min": curve_length_min,
                        "curve_length_max": curve_length_max,
                        "curve_length_mean": curve_length_mean,
                        "curve_length_std": curve_length_std,
                        "curve_length_cv": curve_length_cv,
                        "curve_length_ratio": curve_length_ratio,
                        "minimum_length": solution_metrics["minimum_length"],
                        "maximum_length": solution_metrics["maximum_length"],
                        "mean_length": solution_metrics["mean_length"],
                        "standard_deviation": solution_metrics["standard_deviation"],
                        "coefficient_of_variation": solution_metrics["coefficient_of_variation"],
                        "maximum_minimum_ratio": solution_metrics["maximum_minimum_ratio"],
                        "minimum_active_seed_distance": min_active_seed_dist,
                        "number_of_edges": solution_metrics["number_of_edges"],
                        "number_of_selected_edges": solution_metrics["number_of_selected_edges"],
                        "number_of_total_edges": solution_metrics["number_of_total_edges"],
                        "topology_identifier": solution_metrics["topology_identifier"],
                        "loss_l_curve_cell": self._finite_or_default(loss_l_curve_cell),
                        "cell_area_min": cell_area_min,
                        "cell_area_mean": cell_area_mean,
                        "cell_area_max": cell_area_max,
                        "loss_fem": l_fem_current,
                        "fem_total_loss": l_fem_current,
                        "loss_fem_stress_constraint": self._finite_or_default(loss_fem_stress_constraint),
                        "loss_fem_displacement_constraint": self._finite_or_default(loss_fem_displacement_constraint),
                        "baseline_fem_loss": self._finite_or_default(fem_out.get("baseline_fem_loss", zero)),
                        "fem_baseline_loss": self._finite_or_default(fem_out.get("baseline_fem_loss", zero)),
                        "violation_fem_loss": self._finite_or_default(fem_violation_loss),
                        "fem_violation_loss": self._finite_or_default(fem_violation_loss),
                        "fem_stress_constraint_excess": self._finite_or_default(fem_out.get("stress_constraint_excess", zero)),
                        "fem_displacement_constraint_excess": self._finite_or_default(fem_out.get("displacement_constraint_excess", zero)),
                        "fem_stress_ratio": self._finite_or_default(fem_out.get("stress_ratio", zero)),
                        "fem_displacement_ratio": self._finite_or_default(fem_out.get("displacement_ratio", zero)),
                        "training_stress_ratio": self._finite_or_default(fem_out.get("training_stress_ratio", zero)),
                        "training_displacement_ratio": self._finite_or_default(fem_out.get("training_displacement_ratio", zero)),
                        "physical_stress_ratio": self._finite_or_default(fem_out.get("physical_stress_ratio", zero)),
                        "physical_displacement_ratio": self._finite_or_default(fem_out.get("physical_displacement_ratio", zero)),
                        "training_feasible": bool(fem_out.get("training_feasible", False)),
                        "physical_feasible": bool(physical_feasible),
                        "active_seed_feasible": bool(active_seed_feasible),
                        "overall_feasible": bool(overall_feasible),
                        "fem_constraint_violation": self._finite_or_default(fem_constraint_violation),
                        "active_seed_violation": self._finite_or_default(active_seed_violation),
                        "overall_constraint_violation": self._finite_or_default(overall_constraint_violation),
                        "mechanical_violation": self._finite_or_default(overall_constraint_violation),
                        "fem_rho_min_ratio": self._scheduled_fem_rho_min_ratio(
                            stage_local_step,
                            stage_max_steps,
                        ),
                        "fem_stress_max": self._finite_or_default(fem_out.get("stress_max", zero)),
                        "fem_displacement_max": self._finite_or_default(fem_out.get("displacement_max", zero)),
                        "stress_max": stress_max,
                        "stress_p95": stress_p95,
                        "stress_p99": stress_p99,
                        "stress_norm": stress_norm,
                        "disp_field_max": disp_field_max,
                        "disp_field_p95": disp_field_p95,
                        "disp_field_p99": disp_field_p99,
                        "disp_mean": disp_mean,
                        "disp_max": disp_max,
                        "disp_p95": disp_p95,
                        "disp_p99": disp_p99,
                        "disp_norm": disp_norm,
                        "VolFrac": volfrac_scalar,
                        "L_FEM_norm": l_fem_norm,
                        "seed_offset_scale": float(seed_offset_scale_step),
                        "rho_min": rho_min,
                        "rho_mean": rho_mean,
                        "rho_max": rho_max,
                        "filter_delta_mean": density_post_stats["filter_delta_mean"],
                        "filter_delta_max": density_post_stats["filter_delta_max"],
                        "projection_delta_mean": density_post_stats["projection_delta_mean"],
                        "projection_delta_max": density_post_stats["projection_delta_max"],
                        "rho_raw_mean": density_post_stats["raw_mean"],
                        "rho_filtered_mean": density_post_stats["filtered_mean"],
                        "rho_projected_mean": density_post_stats["projected_mean"],
                        "rho_final_mean": density_post_stats["final_mean"],
                        "drho": drho,
                        "dseed": dseed,
                        "min_active_seed_dist": min_active_seed_dist,
                        "grad_mean": g_mean,
                        "best_score": best_score,
                        "best_step": best_step,
                        "legacy_l_train_best_score": best_score,
                        "legacy_l_train_best_step": best_step,
                        "best_feasible_design_score": self._best_feasible_design_score(best_feasible_key),
                        "best_feasible_step": self._best_feasible_step(best_feasible_key, best_feasible_checkpoint),
                        "best_infeasible_violation": self._best_infeasible_violation(best_infeasible_key),
                        "best_infeasible_step": self._best_infeasible_step(best_infeasible_key, best_infeasible_checkpoint),
                        "fem_valid": fem_is_valid,
                        "fem_was_evaluated": bool(fem_was_evaluated),
                        "fem_failure_reason": fem_failure_reason,
                        "optimizer_step_skipped": optimizer_step_skipped,
                        "consecutive_invalid_fem_steps": consecutive_invalid_fem_steps,
                        "strut_thickness": float(cfg.strut_thickness),
                        "active_units_total": participating_count_total,
                        "active_units_mean": participating_count_mean,
                        "active_units_frac_mean": participating_frac_mean,
                        "inactive_units_total": inactive_count_total,
                        "inactive_units_mean": inactive_count_mean,
                        "inactive_units_frac_mean": inactive_frac_mean,
                        "raw_seed_units_total": raw_seed_count_total,
                        "topology_seed_units_total": topology_seed_count_total,
                        "soft_active_units_total": soft_active_total,
                        "soft_active_mass": soft_active_total,
                        "hard_active_count": hard_active_total,
                        "hard_active_seed_count": hard_active_total,
                        "min_active_seeds": int(cfg.min_active_seeds or 1),
                        "possible_min_active": (
                            float(cfg.possible_min_active)
                            if cfg.possible_min_active is not None
                            else float("nan")
                        ),
                        "anchor_update_allowed": 1.0 if anchor_update_allowed else 0.0,
                        "collapse_active": (
                            1.0
                            if participating_count_total < float(cfg.min_active_seeds or 1)
                            else 0.0
                        ),
                    }
                    stage_lam_text = self._stage_lambda_summary(row)
                    Logg_stage = f"Stage {int(row.get('stage', 0))} "
                    row["stage_lam_text"] = stage_lam_text

                    trainable_params_finite = not self._nonfinite_param_info(trainable_modules)
                    gradients_finite = not self._nonfinite_grad_info(trainable_modules)
                    finite_diagnostics = all(
                        math.isfinite(float(v))
                        for v in (
                            row.get("minimum_active_seed_distance", float("nan")),
                            row.get("VolFrac", float("nan")),
                            row.get("loss_fem", float("nan")),
                        )
                    )
                    stage_checkpoint_is_valid = (
                        best_candidate_is_valid
                        and not optimizer_step_skipped
                        and math.isfinite(float(stage_monitor_raw))
                        and gradients_finite
                        and trainable_params_finite
                        and finite_diagnostics
                    )
                    spec = current_stage_runtime.spec
                    design_monitor_step = self._stage_monitor_uses_design_mode(stage_id, row)
                    meaningful = (
                        stage_checkpoint_is_valid
                        and design_monitor_step
                        and self.is_meaningful_improvement(
                            stage_monitor_raw,
                            current_stage_runtime.best_raw_monitor,
                            spec.min_delta_abs,
                            spec.min_delta_rel,
                        )
                    )

                    if stage_checkpoint_is_valid:
                        last_valid_checkpoint = self._stage_checkpoint_from_step(
                            source="last_valid",
                            ppnet=ppnet,
                            decoder=decoder,
                            opt=opt,
                            scheduler=scheduler,
                            uv_anchor=uv_anchor,
                            row=row,
                            pred_list=pred_list,
                            seeds_list=seeds_list,
                            rho=rho,
                            fiber_surface=fiber_surface,
                            fem_density_field=fem_density_field,
                            fem_stress_field=fem_stress_field,
                            fem_displacement_field=fem_displacement_field,
                            stage_monitor_raw=stage_monitor_raw,
                            effective_lambdas=effective_lambdas,
                        )
                        if self._is_stage2_physical_checkpoint_candidate(
                            stage_id=stage_id,
                            first_physical_stage=first_physical_stage,
                            fem_was_evaluated=bool(row.get("fem_was_evaluated", False)),
                            fem_is_valid=bool(row.get("fem_valid", False)),
                            total_loss_is_finite=bool(total_is_finite),
                        ):
                            fiber_length_key = float(row.get("loss_total_fiber_length", float("inf")))
                            design_score_key = float(row.get("design_score", float("inf")))
                            mechanical_violation_key = float(row.get("overall_constraint_violation", row.get("mechanical_violation", float("inf"))))
                            candidate_is_feasible, candidate_key = checkpoint_feasibility_key(
                                physical_displacement_ratio=float(row.get("physical_displacement_ratio", float("inf"))),
                                physical_stress_ratio=float(row.get("physical_stress_ratio", float("inf"))),
                                hard_active_seed_count=float(hard_active_total),
                                min_active_seeds=int(cfg.min_active_seeds or 1),
                                design_score=design_score_key,
                                raw_total_fiber_length=fiber_length_key,
                                mechanical_violation=mechanical_violation_key,
                                global_step=int(step),
                                total_loss_is_finite=bool(total_is_finite),
                                fem_is_valid=bool(row.get("fem_valid", False)),
                            )
                            if candidate_is_feasible:
                                feasible_key = candidate_key
                                if feasible_key < best_feasible_key:
                                    best_feasible_key = feasible_key
                                    best_feasible_checkpoint = dict(last_valid_checkpoint)
                                    best_feasible_checkpoint["source"] = "best_feasible"
                                    best_feasible_checkpoint["best_feasible_key"] = tuple(feasible_key)
                                    best_feasible_checkpoint["best_design_score"] = float(feasible_key[0])
                                    best_feasible_checkpoint["best_raw_fiber_length"] = float(feasible_key[1])
                                    row_best_feasible = best_feasible_checkpoint.get("row", {})
                                    if isinstance(row_best_feasible, dict):
                                        row_best_feasible["design_score"] = float(feasible_key[0])
                                    tqdm.write(
                                        "[Best feasible] "
                                        f"step={int(step)} "
                                        f"design_score={float(feasible_key[0]):.6g} "
                                        f"raw_fiber_length={float(feasible_key[1]):.6g} "
                                        f"stress_ratio={float(row.get('physical_stress_ratio', float('nan'))):.6g} "
                                        f"displacement_ratio={float(row.get('physical_displacement_ratio', float('nan'))):.6g} "
                                        f"active_seeds={float(hard_active_total):.0f} "
                                        f"feasible_key=(design_score, raw_fiber_length, step)="
                                        f"({float(feasible_key[0]):.6g}, {float(feasible_key[1]):.6g}, {int(feasible_key[2])})"
                                    )
                                    live_best_output_folder = (
                                        timelapse_output_folder
                                        or getattr(cfg, "timelapse_output_folder", None)
                                    )
                                    try:
                                        saved_live_best_path = self._save_live_best_feasible_checkpoint(
                                            output_folder=live_best_output_folder,
                                            checkpoint=best_feasible_checkpoint,
                                            decoder=decoder,
                                            ppnet=ppnet,
                                            face_tensor=face_tensor,
                                        )
                                        # if saved_live_best_path:
                                        #     tqdm.write(
                                        #         "[Best feasible saved] "
                                        #         f"step={int(step)} | "
                                        #         f"path={saved_live_best_path}"
                                        #     )
                                    except Exception as exc:
                                        tqdm.write(
                                            "[Best feasible save failed] "
                                            f"step={int(step)} | {type(exc).__name__}: {exc}"
                                        )
                            else:
                                if candidate_key < best_infeasible_key:
                                    best_infeasible_key = candidate_key
                                    best_infeasible_checkpoint = dict(last_valid_checkpoint)
                                    best_infeasible_checkpoint["source"] = "best_infeasible"

                        row["best_feasible_design_score"] = self._best_feasible_design_score(best_feasible_key)
                        row["best_feasible_step"] = self._best_feasible_step(best_feasible_key, best_feasible_checkpoint)
                        row["best_infeasible_violation"] = self._best_infeasible_violation(best_infeasible_key)
                        row["best_infeasible_step"] = self._best_infeasible_step(best_infeasible_key, best_infeasible_checkpoint)

                        current_stage_runtime.stage_last_valid_checkpoint = dict(last_valid_checkpoint)
                        self._update_stage_runtime_checkpoint(
                            current_stage_runtime,
                            row=row,
                            checkpoint=last_valid_checkpoint,
                            stage_monitor_raw=stage_monitor_raw,
                            meaningful_improvement=meaningful,
                            stage_id=stage_id,
                        )

                    if not optimizer_step_skipped:
                        self.update_adaptive_stage_controller(
                            current_stage_runtime,
                            row,
                            meaningful_improvement=meaningful,
                            stage_topology_grace_steps=int(cfg.stage_topology_grace_steps),
                            stage_topology_grace_max_resets=int(cfg.stage_topology_grace_max_resets),
                            debug_stage_controller=bool(getattr(cfg, "debug_stage_controller", False)),
                            global_step=step,
                        )

                    history.append(row)

                    pbar.set_postfix(
                        stage=stage_lam_text,
                        loss=f"{row['L_total']:.3e}",
                        vol=f"{row['VolFrac']:.3f}",
                        lfem=f"{row['loss_fem']:.2e}",
                        th=f"{row['strut_thickness']:.3e}",
                        lcvt=f"{row['loss_cvt']:.2e}",
                        lseed=f"{row['loss_l_seed']:.2e}",
                        lcurve=f"{row['loss_total_fiber_length']:.2e}",
                        lcell=f"{row['loss_l_curve_cell']:.2e}",
                        d_active=f"{row['min_active_seed_dist']:.3e}",
                        active=f"{participating_count_mean:.1f}",
                        fem="OK" if fem_is_valid else "BAD",
                        phys="OK" if bool(row.get("overall_feasible", False)) else "BAD",
                        refresh=False,
                    )

                    if should_record_timelapse:
                        if getattr(cfg, "timelapse_show_3d_tubes", True):
                            cad_img = self._render_current_3d_tube_frame_cached(
                                seeds_list=seeds_list,
                                decoders=decoders,
                                pred_list=pred_list,
                                render_cache=render_cache,
                                loading_img=self.timelapse_loading_img,
                                fem_density_field=fem_density_field,
                                fem_stress_field=fem_stress_field,
                                fem_displacement_field=fem_displacement_field,
                                history_rows=history,
                            )
                        else:
                            cad_img = self._render_current_cad_frame_cached(
                                seeds_list=seeds_list,
                                decoders=decoders,
                                pred_list=pred_list,
                                render_cache=render_cache,
                                thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                                loading_img=self.timelapse_loading_img,
                            )

                        recorder.add_frame(
                            step=step,
                            cad_img=cad_img,
                            loss_dict=self._timelapse_loss_chart_dict(row),
                            title_text=(
                                f"S{int(row['stage'])} | "
                                f"Best Step={int(row['best_step'])} | "
                                f"{self._timelapse_geometry_summary_text(row)}"
                            ),
                            stage=row["stage"],
                        )

                    self._tb_log_step(
                        step=step,
                        row=row,
                        rho=rho,
                        fiber_surface=fiber_surface,
                        seeds_list=seeds_list,
                        pred_list=pred_list,
                    )

                    if (not fem_is_valid) and (cfg.skip_bad_fem_steps) and (stage_id > 1):
                        self._print_fem_failure(step)

                    if should_log:
                        tqdm.write(
                            self._format_stage_progress_log(
                                row=row,
                                total_step_budget=total_step_budget,
                                best_feasible_key=best_feasible_key,
                                best_feasible_checkpoint=best_feasible_checkpoint,
                                best_infeasible_key=best_infeasible_key,
                                best_infeasible_checkpoint=best_infeasible_checkpoint,
                            )
                            + " | "
                            f"Active Units/Total={participating_count_total:.0f}/{participating_count_total+inactive_count_total:.0f} | "
                            f"L_cvt={row['loss_cvt']:.3e}(lam={row['lam_cvt_eff']:.2g}) "
                            f"L_total_fiber_length={row['loss_total_fiber_length']:.3e}(lam={row['lam_total_fiber_length_eff']:.2g}) "
                            f"L_curve_cell={row['loss_l_curve_cell']:.3e}(lam={row['lam_l_curve_cell_eff']:.2g}) "
                            f"L_rep={row['loss_rep']:.3e}(lam={row['lam_rep_eff']:.2g}) "
                            f"L_seed={row['loss_l_seed']:.3e}(lam={row['lam_l_seed_eff']:.2g}) | "
                            f"stress_max={row['stress_max']:.3e} "
                            f"disp_max={row['disp_max']:.3e} "
                            f"disp_field_max={row['disp_field_max']:.3e} | "
                            f"L(min/max/mean/ratio)={row['curve_length_min']:.3e}/{row['curve_length_max']:.3e}/{row['curve_length_mean']:.3e}/{row['curve_length_ratio']:.2f} "
                            f"Acell(min/mean/max)={row['cell_area_min']:.3e}/{row['cell_area_mean']:.3e}/{row['cell_area_max']:.3e} |"
                            f"VolFrac={row['VolFrac']:.3f} "
                            f"rho(min/mean/max)={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f} "
                            f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                            f"d_active={min_active_seed_dist:.2e} grad_mean={g_mean:.2e} | "
                            f"Filter Δrho mean={row['filter_delta_mean']:.2e} "
                            f"Filter Δrho max={row['filter_delta_max']:.2e} "
                            f"Projection Δrho mean={row['projection_delta_mean']:.2e} "
                            f"Projection Δrho max={row['projection_delta_max']:.2e} | "
                            f"rho_raw_mean={row['rho_raw_mean']:.3f} "
                            f"rho_filtered_mean={row['rho_filtered_mean']:.3f} "
                            f"rho_final_mean={row['rho_final_mean']:.3f} | "
                            f"fem_solve={'OK' if fem_is_valid else f'BAD({fem_failure_reason})'} "
                            f"train_feas={'OK' if bool(row.get('training_feasible', False)) else 'BAD'} "
                            f"ratios(train s/d)="
                            f"{float(row.get('training_stress_ratio', float('nan'))):.3e}/"
                            f"{float(row.get('training_displacement_ratio', float('nan'))):.3e} "
                            f"fem_violation={float(row.get('fem_constraint_violation', float('nan'))):.3e} "
                            f"active_violation={float(row.get('active_seed_violation', float('nan'))):.3e} "
                            f"grace={int(row.get('topology_grace_remaining', 0))} "
                            f"grace_resets={int(row.get('topology_grace_resets_used', 0))}/{int(cfg.stage_topology_grace_max_resets)} "
                            f"topo_changed={bool(row.get('topology_changed', False))} "
                            f"topo_id={str(row.get('topology_identifier_short', ''))} "
                            f"meaningful={bool(row.get('meaningful_improvement', False))} "
                            f"anchor_update_allowed={bool(row.get('anchor_update_allowed', False))} "
                        )

                    rep_value = float(row["loss_rep"])
                    vol_eff_value = float(row["VolFrac"])
                    min_active_seed_dist_value = float(row["min_active_seed_dist"])
                    min_active_seed_dist_limit = (
                        float(cfg.anchor_guard_min_active_seed_dist_factor)
                        * float(cfg.strut_thickness)
                    )

                    anchor_update_allowed = (
                        rep_value <= float(cfg.anchor_guard_rep_max)
                        and vol_eff_value >= float(cfg.anchor_guard_vol_eff_min)
                        and min_active_seed_dist_value >= min_active_seed_dist_limit
                    )

                    if not optimizer_step_skipped:
                        spec = current_stage_runtime.spec
                        current_stage_runtime.local_step += 1
                        stage_end_reason = None
                        if current_stage_runtime.local_step >= spec.max_steps:
                            stage_end_reason = "max_steps"
                        elif (
                            current_stage_runtime.local_step >= spec.min_steps
                            and current_stage_runtime.patience_counter >= spec.patience
                        ):
                            stage_end_reason = "patience"

                        if stage_end_reason is not None:
                            current_stage_runtime.end_reason = stage_end_reason
                            transition_name = None
                            transition_score = float("inf")
                            selected_transition_checkpoint = None
                            next_stage_id = (
                                stage_specs[current_stage_index + 1].stage_id
                                if current_stage_index + 1 < len(stage_specs)
                                else spec.stage_id
                            )
                            if current_stage_index + 1 < len(stage_specs):
                                transition_name, selected_transition_checkpoint, transition_score = self._select_transition_checkpoint(
                                    current_stage_runtime,
                                    next_stage_id,
                                )
                                tqdm.write(
                                    f"{spec.name} finished at local step {current_stage_runtime.local_step} "
                                    f"(global step {step}) | reason={stage_end_reason}"
                                )
                                for cand_name, cand_ckpt in (
                                    ("best_raw", current_stage_runtime.stage_best_raw_checkpoint),
                                    ("last", current_stage_runtime.stage_last_valid_checkpoint),
                                ):
                                    if cand_ckpt is None:
                                        tqdm.write(f"    {cand_name}: missing")
                                    else:
                                        cand_score = self._checkpoint_stage_monitor_for_stage(cand_ckpt, next_stage_id)
                                        tqdm.write(f"    {cand_name}: next_stage_monitor={cand_score:.6e}")
                            else:
                                tqdm.write(
                                    f"{spec.name} completed | reason={stage_end_reason} | "
                                    f"local_steps={current_stage_runtime.local_step}"
                                )

                            stage_summary = {
                                "stage": int(spec.stage_id),
                                "stage_name": spec.name,
                                "reason": stage_end_reason,
                                "local_steps": int(current_stage_runtime.local_step),
                                "best_local_step": (
                                    int(current_stage_runtime.stage_best_raw_checkpoint.get("stage_local_step", -1))
                                    if current_stage_runtime.stage_best_raw_checkpoint is not None
                                    else -1
                                ),
                                "best_monitor": float(current_stage_runtime.best_raw_monitor),
                                "best_raw_step": (
                                    int(current_stage_runtime.stage_best_raw_checkpoint.get("stage_local_step", -1))
                                    if current_stage_runtime.stage_best_raw_checkpoint is not None
                                    else -1
                                ),
                                "last_valid_step": (
                                    int(current_stage_runtime.stage_last_valid_checkpoint.get("stage_local_step", -1))
                                    if current_stage_runtime.stage_last_valid_checkpoint is not None
                                    else -1
                                ),
                                "transition_checkpoint": transition_name,
                                "transition_score": float(transition_score),
                                "optimizer_reset": bool(cfg.reset_optimizer_between_stages),
                            }
                            stage_end_summaries.append(stage_summary)
                            tqdm.write(
                                f"Stage {spec.stage_id} completed | reason={stage_end_reason} | "
                                f"local_steps={stage_summary['local_steps']} | "
                                f"best_local_step={stage_summary['best_local_step']} | "
                                f"best_monitor={stage_summary['best_monitor']:.6e} | "
                                f"best_raw_step={stage_summary['best_raw_step']} | "
                                f"last_valid_step={stage_summary['last_valid_step']} | "
                                f"transition_checkpoint={transition_name} | "
                                f"transition_score={transition_score:.6e} | "
                                f"optimizer_reset={bool(cfg.reset_optimizer_between_stages)}"
                            )

                            if current_stage_index + 1 >= len(stage_specs):
                                break

                            if bool(cfg.restore_transition_checkpoint) and selected_transition_checkpoint is not None:
                                if bool(cfg.reset_optimizer_between_stages):
                                    uv_anchor = self._restore_stage_checkpoint(
                                        selected_transition_checkpoint,
                                        ppnet,
                                        decoder,
                                    )
                                else:
                                    uv_anchor = self._restore_stage_checkpoint(
                                        selected_transition_checkpoint,
                                        ppnet,
                                        decoder,
                                        opt=opt,
                                        scheduler=scheduler,
                                    )
                                tqdm.write(
                                    "[Stage transition] Restored Stage "
                                    f"{int(selected_transition_checkpoint.get('stage_id', spec.stage_id))} "
                                    "topology checkpoint at "
                                    f"global_step={int(selected_transition_checkpoint.get('global_step', -1))}"
                                )
                                tqdm.write(f"Selected transition checkpoint: {transition_name}")

                            entering_first_physical_stage = (
                                int(spec.stage_id) < int(first_physical_stage)
                                and int(next_stage_id) >= int(first_physical_stage)
                            )
                            if entering_first_physical_stage:
                                if selected_transition_checkpoint is not None:
                                    stage1_transition_step = int(
                                        selected_transition_checkpoint.get("global_step", -1)
                                    )
                                    stage1_transition_score = float(transition_score)
                                    stage1_transition_checkpoint_source = transition_name
                                reset_physical_checkpoint_trackers(first_physical_stage)

                            current_stage_index += 1
                            current_stage_runtime = stage_runtimes[current_stage_index]
                            next_stage_settings = self._stage_settings_for_stage_id(current_stage_runtime.spec.stage_id)
                            self._apply_stage_trainability(ppnet, next_stage_settings)
                            if bool(cfg.reset_optimizer_between_stages) or selected_transition_checkpoint is None:
                                opt = self._build_optimizer(ppnet, decoder)
                                validate_optimizer_parameter_coverage(
                                    named_trainable_modules=named_trainable_modules,
                                    optimizer=opt,
                                )
                                scheduler = self._build_scheduler(
                                    opt,
                                    self._stage_scheduler_milestones(current_stage_runtime.spec.stage_id),
                                )
                                latest_valid_model_state = self._clone_modules_state_dict(trainable_modules)
                                latest_valid_optimizer_state = self._clone_optimizer_state_dict(opt)
                            else:
                                trainable_ids = {
                                    id(p)
                                    for module in trainable_modules
                                    for p in module.parameters()
                                    if p.requires_grad
                                }
                                optimizer_ids = {id(p) for group in opt.param_groups for p in group.get("params", [])}
                                if trainable_ids != optimizer_ids:
                                    opt = self._build_optimizer(ppnet, decoder)
                                    validate_optimizer_parameter_coverage(
                                        named_trainable_modules=named_trainable_modules,
                                        optimizer=opt,
                                    )
                                    scheduler = self._build_scheduler(
                                        opt,
                                        self._stage_scheduler_milestones(current_stage_runtime.spec.stage_id),
                                    )
                                    latest_valid_model_state = self._clone_modules_state_dict(trainable_modules)
                                    latest_valid_optimizer_state = self._clone_optimizer_state_dict(opt)
                                else:
                                    validate_optimizer_parameter_coverage(
                                        named_trainable_modules=named_trainable_modules,
                                        optimizer=opt,
                                    )
                            seeds0 = None
                            continue

        # ------------------------------------------------------------
        # Common final checkpoint selection
        # ------------------------------------------------------------
        selected_final_checkpoint = None
        selected_checkpoint_source = "unavailable"
        selected_final_score = float("inf")
        final_stage_runtime = stage_runtimes[-1]
        (
            selected_checkpoint_source,
            selected_final_checkpoint,
            selected_final_score,
        ) = self._select_final_stage_checkpoint(final_stage_runtime)

        def physical_candidate_or_none(source: str, checkpoint):
            stage_id_for_checkpoint = checkpoint_stage_id(checkpoint)
            if (
                checkpoint is not None
                and stage_id_for_checkpoint is not None
                and int(stage_id_for_checkpoint) < int(first_physical_stage)
            ):
                tqdm.write(
                    "[Checkpoint warning] Ignoring non-physical Stage "
                    f"{int(stage_id_for_checkpoint)} checkpoint from {source}"
                )
                return None
            return checkpoint

        (
            selected_checkpoint_source,
            selected_final_checkpoint,
            selected_final_score,
        ) = select_final_checkpoint(
            proposed_checkpoint=physical_candidate_or_none(
                selected_checkpoint_source,
                selected_final_checkpoint,
            ),
            proposed_source=selected_checkpoint_source,
            proposed_score=selected_final_score,
            best_feasible_checkpoint=physical_candidate_or_none(
                "best_feasible",
                best_feasible_checkpoint,
            ),
            best_feasible_key=best_feasible_key,
            best_infeasible_checkpoint=physical_candidate_or_none(
                "best_infeasible",
                best_infeasible_checkpoint,
            ),
            best_infeasible_key=best_infeasible_key,
            last_valid_checkpoint=physical_candidate_or_none(
                "last_valid",
                last_valid_checkpoint,
            ),
            first_physical_stage=first_physical_stage,
        )

        if selected_final_checkpoint is not None:
            selected_stage = checkpoint_stage_id(selected_final_checkpoint)
            if selected_stage is not None and int(selected_stage) < int(first_physical_stage):
                raise RuntimeError(
                    "Final checkpoint selection chose a non-physical checkpoint: "
                    f"selected_stage={int(selected_stage)}, "
                    f"first_physical_stage={int(first_physical_stage)}, "
                    f"source={selected_checkpoint_source}."
                )
            stage_best_checkpoint_for_log = (
                final_stage_runtime.stage_best_raw_checkpoint
                if final_stage_runtime is not None
                else None
            )
            stage_best_local_step = (
                int(stage_best_checkpoint_for_log.get("stage_local_step", -1))
                if stage_best_checkpoint_for_log is not None
                else -1
            )
            stage_best_global_step = (
                int(stage_best_checkpoint_for_log.get("global_step", -1))
                if stage_best_checkpoint_for_log is not None
                else -1
            )
            best_feasible_global_step = (
                int(best_feasible_checkpoint.get("global_step", -1))
                if best_feasible_checkpoint is not None
                else -1
            )
            selected_row_for_log = selected_final_checkpoint.get("row", {})
            selected_fiber_length_for_log = float(
                selected_row_for_log.get("loss_total_fiber_length", float("nan"))
            )
            selected_design_score_for_log = float(
                selected_row_for_log.get("design_score", float("nan"))
            )
            selected_stress_ratio_for_log = float(
                selected_row_for_log.get("physical_stress_ratio", float("nan"))
            )
            selected_displacement_ratio_for_log = float(
                selected_row_for_log.get("physical_displacement_ratio", float("nan"))
            )
            selected_physical_max_ratio_for_log = max(
                selected_stress_ratio_for_log,
                selected_displacement_ratio_for_log,
            )
            selected_active_count_for_log = float(
                selected_row_for_log.get(
                    "hard_active_count",
                    selected_final_checkpoint.get("active_seed_count", float("nan")),
                )
            )
            best_feasible_key_for_log = (
                f"({float(best_feasible_key[0]):.6g}, "
                f"{float(best_feasible_key[1]):.6g}, "
                f"{int(best_feasible_key[2])})"
                if (
                    best_feasible_key is not None
                    and len(best_feasible_key) >= 3
                    and math.isfinite(float(best_feasible_key[2]))
                )
                else str(best_feasible_key)
            )
            tqdm.write(
                "[Final selection] "
                f"source={selected_checkpoint_source} "
                f"stage={int(selected_stage) if selected_stage is not None else 'unknown'} "
                f"global_step={int(selected_final_checkpoint.get('global_step', -1))} "
                f"design_score={selected_design_score_for_log:.6g} "
                f"raw_fiber_length={selected_fiber_length_for_log:.6g} "
                f"physical_max_ratio={selected_physical_max_ratio_for_log:.6g} "
                f"stress_ratio={selected_stress_ratio_for_log:.6g} "
                f"displacement_ratio={selected_displacement_ratio_for_log:.6g} "
                f"active_seeds={selected_active_count_for_log:.0f} "
                f"stage_best_local_step={stage_best_local_step} "
                f"stage_best_global_step={stage_best_global_step} "
                f"best_feasible_global_step={best_feasible_global_step} "
                f"best_feasible_key=(design_score, raw_fiber_length, step)="
                f"{best_feasible_key_for_log}"
            )
            uv_anchor = self._restore_stage_checkpoint(
                selected_final_checkpoint,
                ppnet,
                decoder,
            )
            selected_checkpoint_stage_loss = float(
                selected_final_checkpoint.get("selected_checkpoint_stage_loss", float("nan"))
            )
            best_step = int(selected_final_checkpoint.get("global_step", -1))
            best_score = (
                selected_final_score
                if math.isfinite(float(selected_final_score))
                else selected_checkpoint_stage_loss
            )
            if selected_checkpoint_source == "best_feasible":
                selected_design_score = checkpoint_design_score(selected_final_checkpoint)
                if math.isfinite(float(selected_design_score)):
                    best_score = float(selected_design_score)
            row_sel = selected_final_checkpoint.get("row", {})
            best_active_count = float(row_sel.get("active_units_total", selected_final_checkpoint.get("active_seed_count", 0.0)))
            best_inactive_count = float(row_sel.get("inactive_units_total", 0.0))
            best_raw_seed_count = int(row_sel.get("raw_seed_units_total", best_raw_seed_count or 0))
            best_rho = selected_final_checkpoint["rho"].detach().clone()
            best_fiber_surface = selected_final_checkpoint["fiber_surface"].detach().clone()
            best_seeds = [s.detach().clone() for s in selected_final_checkpoint.get("seeds", [])]
            best_pred = self._clone_pred_list(selected_final_checkpoint.get("pred_list", []))
            if best_pred and isinstance(best_pred[0].get("seed_active_mask"), torch.Tensor):
                best_seed_active_mask = best_pred[0]["seed_active_mask"].detach().clone()
            if best_pred and isinstance(best_pred[0].get("active_seed_ids"), torch.Tensor):
                best_active_seed_ids = best_pred[0]["active_seed_ids"].detach().clone()
            best_fem_density_field = selected_final_checkpoint.get("fem_density_field", None)
            best_fem_stress_field = selected_final_checkpoint.get("fem_stress_field", None)
            best_fem_displacement_field = selected_final_checkpoint.get("fem_displacement_field", None)
            returned_best_source = selected_checkpoint_source
        else:
            tqdm.write(
                "[Final selection] No valid physical checkpoint was available; "
                "returning current physical fallback state."
            )

        if selected_final_checkpoint is None and first_physical_stage > 1:
            has_physical_history = any(
                int(row.get("stage", 0)) >= int(first_physical_stage)
                for row in history
            )
            if not has_physical_history:
                raise RuntimeError(
                    "No physical-stage checkpoints were evaluated. "
                    f"first_physical_stage={int(first_physical_stage)}; "
                    "refusing to return the Stage 1 topology warm-up as the final "
                    "physical design."
                )

        # ------------------------------------------------------------
        # Fallback best state
        # ------------------------------------------------------------
        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = [s.detach().clone() for s in seeds_list]
                best_pred = self._clone_pred_list(pred_list)
                best_fem_density_field = (
                    fem_density_field.detach().clone()
                    if isinstance(fem_density_field, torch.Tensor)
                    else None
                )
                best_fem_stress_field = (
                    fem_stress_field.detach().clone()
                    if isinstance(fem_stress_field, torch.Tensor)
                    else None
                )
                best_fem_displacement_field = (
                    fem_displacement_field.detach().clone()
                    if isinstance(fem_displacement_field, torch.Tensor)
                    else None
                )
                best_step = step
                best_score = float("inf") if not self._scalar_tensor_is_finite(L_total) else float(L_total.detach().item())
                selected_checkpoint_source = "current_physical_fallback"
                returned_best_source = "current_physical_fallback"

                if best_active_count is None:
                    best_active_count = float(participating_count_total)
                if best_inactive_count is None:
                    best_inactive_count = float(inactive_count_total)
                if best_raw_seed_count is None:
                    best_raw_seed_count = int(raw_seed_count_total)
                if best_seed_active_mask is None and pred_list and isinstance(pred_list[0].get("seed_active_mask"), torch.Tensor):
                    best_seed_active_mask = pred_list[0]["seed_active_mask"].detach().clone()
                if best_active_seed_ids is None and pred_list and isinstance(pred_list[0].get("active_seed_ids"), torch.Tensor):
                    best_active_seed_ids = pred_list[0]["active_seed_ids"].detach().clone()

        if best_rho is not None and selected_checkpoint_source == "global":
            returned_best_source = "global"


        # ------------------------------------------------------------
        # Final outputs
        # ------------------------------------------------------------
        with torch.no_grad():
            hard_rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
            hard_fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            pred_i = best_pred[0] if best_pred else None
            if pred_i is not None:
                decoder_seed_state = self._decoder_seed_state_for_pred(decoder, pred_i, device)
                try:
                    hard_out_i = decoder(
                        seeds_uv=pred_i["seeds_raw"],
                        generate_density_fiber=getattr(cfg, "generate_decoder_density_fiber", True),
                    )
                finally:
                    self._restore_decoder_seed_state(decoder, decoder_seed_state)
                restored_active_count = int(hard_out_i["seed_active_mask"].to(dtype=torch.bool).sum().item())
                restored_topology_count = int(hard_out_i["topology_seeds_uv"].shape[0])
                assert restored_active_count == restored_topology_count

                saved_best_active_count = (
                    int(best_seed_active_mask.to(dtype=torch.bool).sum().item())
                    if isinstance(best_seed_active_mask, torch.Tensor)
                    else int(round(float(best_active_count or 0.0)))
                )
                saved_seed_uv = pred_i.get("seeds_uv", None)
                restored_seed_uv = hard_out_i.get("seeds_uv", None)
                if isinstance(saved_seed_uv, torch.Tensor) and isinstance(restored_seed_uv, torch.Tensor):
                    seed_difference_norm = float(
                        torch.linalg.vector_norm(
                            saved_seed_uv.detach().to(device=restored_seed_uv.device, dtype=restored_seed_uv.dtype)
                            - restored_seed_uv.detach()
                        ).item()
                    )
                else:
                    seed_difference_norm = float("nan")
                if restored_active_count != saved_best_active_count:
                    source = (
                        "topology threshold crossing"
                        if math.isfinite(seed_difference_norm) and seed_difference_norm <= 1e-8
                        else "checkpoint restoration or seed reconstruction"
                    )
                    tqdm.write(
                        "Best-state active count mismatch: "
                        f"saved_active={saved_best_active_count}, "
                        f"restored_active={restored_active_count}, "
                        f"seed_difference_norm={seed_difference_norm:.3e}, "
                        f"likely_source={source}."
                    )
                self._copy_activation_metadata_to_pred(pred_i, hard_out_i)

                if getattr(cfg, "generate_decoder_density_fiber", True):
                    hard_out_i = apply_density_postprocess_to_output(
                        hard_out_i,
                        face_tensor,
                        cfg,
                        return_debug=False,
                    )
                    hard_rho_i = hard_out_i["rho"]
                    hard_fiber_i = hard_out_i["fiber3d"]
                else:
                    hard_rho_i, hard_fiber_i, _ = self.neutral_density_fiber_fields(
                        face_tensor["uv"],
                        face_tensor.get("Xu", None),
                    )

                w_local = A_local.clamp_min(cfg.eps)
                hard_rho_acc[gidx] += hard_rho_i * w_local
                hard_rho_wgt[gidx] += w_local
                hard_fiber_acc[gidx] += hard_fiber_i * w_local[:, None]
                hard_fiber_wgt[gidx] += w_local

            final_shape_density = hard_rho_acc / hard_rho_wgt.clamp_min(cfg.eps)
            final_shape_fiber_direction = hard_fiber_acc / hard_fiber_wgt.clamp_min(cfg.eps)[:, None]
            final_fiber_norm = final_shape_fiber_direction.norm(dim=1, keepdim=True)
            final_shape_fiber_direction = torch.where(
                final_fiber_norm > cfg.eps,
                final_shape_fiber_direction / final_fiber_norm.clamp_min(cfg.eps),
                torch.zeros_like(final_shape_fiber_direction),
            )
            seed_points_final = self._seed_points_xyz(best_seeds[0], face_tensor)
            seed_points_final_uv = best_seeds[0].detach().clone()



        computation_time_sec = time.perf_counter() - train_start_time
        final_centerline_radius = 0.5 * float(cfg.strut_thickness)
        best_score_meaning = "unavailable"
        if selected_checkpoint_source == "best_feasible":
            best_score_meaning = "selected_feasible_design_score"
        elif selected_checkpoint_source == "best_infeasible":
            best_score_meaning = "selected_infeasible_constraint_violation"
        elif selected_checkpoint_source in {"last_valid_fallback", "stage_last_valid", "stage_best_raw"}:
            best_score_meaning = "selected_checkpoint_stage_monitor_or_L_train"
        elif selected_checkpoint_source == "current_physical_fallback":
            best_score_meaning = "current_physical_fallback_L_train"
        best_row = None
        if history and best_step >= 0:
            for hist_row in reversed(history):
                if int(hist_row["step"]) == int(best_step):
                    best_row = hist_row
                    break
        if best_row is not None:
            returned_stage = int(best_row.get("stage", 0))
            if returned_stage < int(first_physical_stage):
                raise RuntimeError(
                    "Returned best_step refers to a non-physical checkpoint: "
                    f"best_step={int(best_step)}, selected_stage={returned_stage}, "
                    f"first_physical_stage={int(first_physical_stage)}, "
                    f"source={returned_best_source}."
                )
        final_curve_length_min = float("nan")
        final_curve_length_max = float("nan")
        final_curve_length_ratio = float("nan")
        final_physical_feasible = False
        final_overall_feasible = False
        final_physical_stress_ratio = float("nan")
        final_physical_displacement_ratio = float("nan")
        final_physical_max_ratio = float("nan")
        final_design_score = float("nan")
        final_physical_fiber_length = float("nan")
        final_hard_active_count = float(best_active_count or 0.0)
        if best_row is not None:
            final_curve_length_min = float(best_row.get("curve_length_min", float("nan")))
            final_curve_length_max = float(best_row.get("curve_length_max", float("nan")))
            final_curve_length_ratio = float(best_row.get("curve_length_ratio", float("nan")))
            final_physical_feasible = bool(best_row.get("physical_feasible", False))
            final_overall_feasible = bool(best_row.get("overall_feasible", final_physical_feasible))
            final_physical_stress_ratio = float(best_row.get("physical_stress_ratio", float("nan")))
            final_physical_displacement_ratio = float(best_row.get("physical_displacement_ratio", float("nan")))
            final_physical_max_ratio = max(final_physical_stress_ratio, final_physical_displacement_ratio)
            final_design_score = float(best_row.get("design_score", float("nan")))
            final_physical_fiber_length = float(best_row.get("loss_total_fiber_length", float("nan")))
            final_hard_active_count = float(best_row.get("hard_active_count", final_hard_active_count))

        tqdm.write(
            f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f} "
            f"({best_score_meaning}) | "
            f"source={returned_best_source}, physical_feasible={final_physical_feasible}, "
            f"overall_feasible={final_overall_feasible} | "
            f"design_score={final_design_score:.6g} | "
            f"physical_ratios(stress/disp)="
            f"{final_physical_stress_ratio:.3e}/{final_physical_displacement_ratio:.3e} | "
            f"physical_max_ratio={final_physical_max_ratio:.3e} | "
            f"physical_fiber_length={final_physical_fiber_length:.3e} | "
            f"hard_active={final_hard_active_count:.0f} | "
            f"centerline_radius={final_centerline_radius:.3e} | "
            f"L(min/max/ratio)="
            f"{final_curve_length_min:.3e}/{final_curve_length_max:.3e}/{final_curve_length_ratio:.2f} | "
            f"active_units={float(best_active_count or 0.0):.0f}, inactive_units={float(best_inactive_count or 0.0):.0f} | "
            f"time={self._format_elapsed_time(computation_time_sec)}"
        )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        if best_row is not None:
            tqdm.write(
                "BEST VOLUME METRIC: "
                f"VolFrac={best_row['VolFrac']:.6g}"
            )

        solution_metric_keys = [
            "minimum_length",
            "maximum_length",
            "mean_length",
            "standard_deviation",
            "coefficient_of_variation",
            "maximum_minimum_ratio",
            "minimum_active_seed_distance",
            "number_of_edges",
            "number_of_selected_edges",
            "number_of_total_edges",
            "topology_identifier",
        ]
        best_solution_metrics = {
            key: best_row.get(key)
            for key in solution_metric_keys
            if best_row is not None and key in best_row
        }

        optimization_log_dir = None
        try:
            optimization_log_dir = self._save_optimization_logs(
                output_folder=timelapse_output_folder or getattr(cfg, "timelapse_output_folder", None),
                history=history,
                best_row=best_row,
                best_score=best_score,
                best_step=best_step,
                computation_time_sec=computation_time_sec,
                returned_best_source=returned_best_source,
            )
            if optimization_log_dir is not None:
                tqdm.write(f"Saved optimization logs: {optimization_log_dir}")
        except Exception as e:
            tqdm.write(f"Failed to save optimization logs: {e}")

        if cfg.MakeTimelaps:
            try:
                def _checkpoint_loss_dict(row, score):
                    return self._timelapse_loss_chart_dict(row or {}, score=float(score))

                def _checkpoint_results_text(row):
                    return (
                        f"{self._timelapse_geometry_summary_text(row)} | "
                        f"compute_time={self._format_elapsed_time(computation_time_sec)}"
                    )

                def _render_checkpoint_timelapse_frame(
                    *,
                    checkpoint: dict[str, Any] | None,
                    pred_list_for_frame: list[dict[str, Any]],
                    seeds_for_frame: list[torch.Tensor],
                    frame_step: int,
                    score: float,
                    output_filename: str | None,
                    chart_title: str,
                    summary_title: str,
                ):
                    if checkpoint is None or recorder is None or not pred_list_for_frame:
                        return None
                    row = checkpoint.get("row", {})
                    total_seed_slots = int(pred_list_for_frame[0]["seeds_raw"].shape[0])
                    active_seed_count = int(round(float(row.get("active_units_total", checkpoint.get("active_seed_count", 0.0)))))
                    title_parts = [
                        f"S{int(row.get('stage', checkpoint.get('stage_id', 0)))}",
                        f"best_step={int(checkpoint.get('global_step', frame_step))}",
                        self._timelapse_geometry_summary_text(row),
                    ]
                    decoder_seed_state = self._decoder_seed_state_for_pred(decoder, pred_list_for_frame[0], device)
                    try:
                        if getattr(cfg, "timelapse_show_3d_tubes", True):
                            cad_img = self._render_current_3d_tube_frame_cached(
                                seeds_list=seeds_for_frame,
                                decoders=decoders,
                                pred_list=pred_list_for_frame,
                                render_cache=render_cache,
                                loading_img=self.timelapse_loading_img,
                                fem_density_field=checkpoint.get("fem_density_field", None),
                                fem_stress_field=checkpoint.get("fem_stress_field", None),
                                fem_displacement_field=checkpoint.get("fem_displacement_field", None),
                                history_rows=history,
                            )
                        else:
                            cad_img = self._render_current_cad_frame_cached(
                                seeds_list=seeds_for_frame,
                                decoders=decoders,
                                pred_list=pred_list_for_frame,
                                render_cache=render_cache,
                                thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                                loading_img=self.timelapse_loading_img,
                            )
                    finally:
                        self._restore_decoder_seed_state(decoder, decoder_seed_state)

                    frame_path = recorder.add_frame(
                        step=frame_step,
                        cad_img=cad_img,
                        loss_dict=_checkpoint_loss_dict(row, score),
                        title_text=" | ".join(title_parts),
                        highlight_best=True,
                        chart_title=chart_title,
                        summary_title=summary_title,
                        prefix_step_in_summary=False,
                        results_title="Results",
                        results_text=_checkpoint_results_text(row),
                        stage=row.get("stage", checkpoint.get("stage_id", None)),
                    )
                    if timelapse_output_folder and output_filename:
                        shutil.copy2(frame_path, os.path.join(timelapse_output_folder, output_filename))
                    return frame_path

                for stage_runtime in stage_runtimes:
                    stage_ckpt = stage_runtime.stage_best_raw_checkpoint
                    if stage_ckpt is None:
                        continue
                    stage_id_for_frame = int(stage_ckpt.get("stage_id", stage_runtime.spec.stage_id))
                    _render_checkpoint_timelapse_frame(
                        checkpoint=stage_ckpt,
                        pred_list_for_frame=stage_ckpt.get("pred_list", []),
                        seeds_for_frame=stage_ckpt.get("seeds", []),
                        frame_step=int(total_step_budget) + stage_id_for_frame + 1,
                        score=float(stage_ckpt.get("stage_monitor_raw", float("inf"))),
                        output_filename=f"stage{stage_id_for_frame}_best_result_frame.png",
                        chart_title=f"Stage {stage_id_for_frame} Best Losses",
                        summary_title="Stage Best Parameters",
                    )

                best_checkpoint_for_frame = {
                    "row": best_row or {},
                    "pred_list": best_pred,
                    "seeds": best_seeds,
                    "stage_id": int(best_row.get("stage", 0)) if best_row is not None else 0,
                    "global_step": int(best_step),
                    "active_seed_count": float(best_active_count or 0.0),
                    "fem_density_field": best_fem_density_field,
                    "fem_stress_field": best_fem_stress_field,
                    "fem_displacement_field": best_fem_displacement_field,
                }
                best_frame_path = _render_checkpoint_timelapse_frame(
                    checkpoint=best_checkpoint_for_frame,
                    pred_list_for_frame=best_pred,
                    seeds_for_frame=best_seeds,
                    frame_step=int(total_step_budget) + 1,
                    score=float(best_score),
                    output_filename="best_result_frame.png",
                    chart_title="Best Result Losses",
                    summary_title="Tuned Parameters",
                )
                if best_frame_path is None:
                    raise RuntimeError("Could not render best result frame.")
                recorder.build_video(hold_last_seconds=10.0)
            except Exception as e:
                tqdm.write(f"Failed to build timelapse video: {e}")

        optimized_function_path = None
        optimized_function_dir = timelapse_output_folder
        if optimized_function_dir is None:
            cfg_output_folder = getattr(cfg, "timelapse_output_folder", None)
            if cfg_output_folder:
                optimized_function_dir = os.path.normpath(str(cfg_output_folder))

        if optimized_function_dir is not None:
            try:
                optimized_function_path = self._save_optimized_shell_function(
                    save_dir=optimized_function_dir,
                    decoder=decoder,
                    ppnet=ppnet,
                    face_tensor=face_tensor,
                    best_pred=best_pred[0] if best_pred else None,
                    best_score=best_score,
                    best_step=best_step,
                    returned_best_source=returned_best_source,
                    final_shape_density=final_shape_density,
                    final_shape_fiber_direction=final_shape_fiber_direction,
                )
                if optimized_function_path is not None:
                    tqdm.write(f"Saved optimized shell function: {optimized_function_path}")
            except Exception as e:
                tqdm.write(f"Failed to save optimized shell function: {e}")

        if debug_anomaly_detection:
            torch.autograd.set_detect_anomaly(False)

        return {
            "decoders": decoders,
            "ppnets": ppnets,
            "optimizer": opt,
            "history": history,
            "best_score": best_score,
            "best_score_meaning": best_score_meaning,
            "best_design_score": (
                float(best_feasible_key[0])
                if (
                    best_feasible_key is not None
                    and len(best_feasible_key) >= 1
                    and math.isfinite(float(best_feasible_key[0]))
                )
                else float("nan")
            ),
            "best_raw_fiber_length": final_physical_fiber_length,
            "best_physical_stress_ratio": final_physical_stress_ratio,
            "best_physical_displacement_ratio": final_physical_displacement_ratio,
            "best_hard_active_count": final_hard_active_count,
            "best_step": best_step,
            "best_feasible_design_score": self._best_feasible_design_score(best_feasible_key),
            "best_feasible_step": self._best_feasible_step(best_feasible_key, best_feasible_checkpoint),
            "best_infeasible_violation": self._best_infeasible_violation(best_infeasible_key),
            "best_infeasible_design_score": (
                float(best_infeasible_key[1])
                if (
                    best_infeasible_key is not None
                    and len(best_infeasible_key) >= 2
                    and math.isfinite(float(best_infeasible_key[1]))
                )
                else float("nan")
            ),
            "best_infeasible_step": self._best_infeasible_step(best_infeasible_key, best_infeasible_checkpoint),
            "best_stage_monitor": (
                float(best_row.get("stage_monitor_raw", float("nan")))
                if best_row is not None
                else float("nan")
            ),
            "selected_checkpoint_source": selected_checkpoint_source,
            "selected_final_source": selected_checkpoint_source,
            "selected_global_step": int(best_step),
            "selected_design_score": final_design_score,
            "selected_raw_total_fiber_length": final_physical_fiber_length,
            "selected_physical_stress_ratio": final_physical_stress_ratio,
            "selected_physical_displacement_ratio": final_physical_displacement_ratio,
            "selected_physical_max_ratio": final_physical_max_ratio,
            "selected_overall_feasible": final_overall_feasible,
            "selected_hard_active_seed_count": final_hard_active_count,
            "selected_stage": (
                int(best_row.get("stage", 0))
                if best_row is not None and best_row.get("stage", None) is not None
                else None
            ),
            "first_physical_stage": int(first_physical_stage),
            "stage1_transition_step": stage1_transition_step,
            "stage1_transition_score": stage1_transition_score,
            "stage1_transition_checkpoint_source": stage1_transition_checkpoint_source,
            "stage_end_summaries": stage_end_summaries,
            "selected_checkpoint_stage_loss": selected_checkpoint_stage_loss,
            "best_solution_metrics": best_solution_metrics,
            "best_raw_seed_count": int(best_raw_seed_count or 0),
            "best_active_units": float(best_active_count or 0.0),
            "best_inactive_units": float(best_inactive_count or 0.0),
            "best_seed_active_mask": best_seed_active_mask,
            "best_active_seed_ids": best_active_seed_ids,
            "returned_best_source": returned_best_source,
            "best_rho": best_rho,
            "best_seeds": best_seeds,
            "best_pred": best_pred,
            "Final_shape_density": final_shape_density,
            "Final_shape_fiber_direction": final_shape_fiber_direction,
            "seed_points_init": seed_points_init,
            "seed_points_final": seed_points_final,
            "seed_points_init_uv": seed_points_init_uv,
            "seed_points_final_uv": seed_points_final_uv,
            "best_seed_points_uv": seed_points_final_uv,
            "best_seed_points_xyz": seed_points_final,
            "best_edge_curves_uv": best_pred[0].get("edge_curves_uv") if best_pred else None,
            "best_edge_curves_xyz": best_pred[0].get("edge_curves_xyz") if best_pred else None,
            "best_graph": best_pred[0].get("graph") if best_pred else None,
            "best_edge_index": best_pred[0].get("edge_index") if best_pred else None,
            "best_edge_seed_pair": best_pred[0].get("edge_seed_pair") if best_pred else None,
            "best_edge_type": best_pred[0].get("edge_type") if best_pred else None,
            "A_v": A_v,
            "uv_init_list": uv_init_list,
            "uv_anchor_list": [uv_anchor],
            "face_tensors": face_tensors,
            "fem_debug_history": self.fem_debug_history,
            "last_fem_debug": self.last_fem_debug,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "optimization_log_dir": optimization_log_dir,
            "shape_path": shape_path,
            "optimized_function_path": optimized_function_path,
        }
