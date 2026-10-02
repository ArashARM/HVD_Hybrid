"""
Phase-1 FEM gradient isolation diagnostic.

Purpose
-------
Identify whether non-finite Phase-1 gradients originate from:
    1) density -> FEM path,
    2) fiber/orientation -> FEM path,
    3) a specific FEM loss component,
    4) the combined FEM loss.

This file is intended to be imported from the existing decoder/FEM notebook after
`face_mesh` and `shell_problem` have been constructed.

Typical notebook use
--------------------
from phase1_fem_gradient_diagnostic import diagnose_phase1_fem_gradients

report = diagnose_phase1_fem_gradients(
    face_mesh=face_mesh,
    shell_problem=shell_problem,
    seeds_uv=test_seed_cases15["handoff_overcomplete"][:40],
    strut_thickness=0.40,
    tau=0.01,
    point_chunk_size=32,
    fem_max_displacement=20.0,
    fem_yield_strength=45000.0,
)
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from Decoder_CLasses import Phase1VoronoiDecoder
from neuraltomo_fem import run_fem_loss
from Training.Loss_FEM import Loss_FEM
from Utils.DifferentiableFilters import smooth_heaviside_projection


# -------------------------------------------------------------------------
# Small trainer stub required by Loss_FEM.
# -------------------------------------------------------------------------

class _DiagnosticConfig:
    """
    Minimal FEM config expected by Training/Loss_FEM.py.

    Keep these values aligned with the user's current training configuration.
    """
    fem_constraint_p_norm = 12.0
    fem_training_safety_factor = 0.90
    fem_safety_margin_weight = 0.05

    # Current Loss_FEM supports different stress constraint aggregations.
    # Use the normal hard maximum path unless explicitly changed.
    fem_stress_constraint_mode = "hard_max"
    fem_stress_ks_rho = 50.0


class _FEMTrainerStub:
    def __init__(self, shell_problem, fem):
        self.shell_problem = shell_problem
        self.fem = fem
        self.cfg = _DiagnosticConfig()
        self.last_fem_debug: dict[str, Any] = {}
        self.fem_debug_history: list[dict[str, Any]] = []


@dataclass
class GradientResult:
    name: str
    fem_valid: bool
    loss_key: str
    loss_value: float
    grad_present: bool
    grad_finite: bool
    grad_norm: float
    grad_abs_max: float
    grad_nan: int
    grad_posinf: int
    grad_neginf: int
    stress_max: float
    displacement_max: float
    physical_stress_ratio: float
    physical_displacement_ratio: float
    note: str = ""


def _scalar(x, default=float("nan")) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            return default
        return float(x.detach().cpu().reshape(()).item())
    try:
        return float(x)
    except Exception:
        return default


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _surface_area_weights(face_mesh: dict[str, Any]) -> torch.Tensor:
    """Lumped physical surface area per surface vertex."""
    points_xyz = face_mesh["points_xyz"]
    faces_ijk = face_mesh["faces_ijk"].long()

    # Accept one-based faces if a mesh happens to contain them.
    if int(faces_ijk.min().detach().cpu().item()) == 1:
        faces_ijk = faces_ijk - 1

    p0 = points_xyz[faces_ijk[:, 0]]
    p1 = points_xyz[faces_ijk[:, 1]]
    p2 = points_xyz[faces_ijk[:, 2]]

    face_areas = 0.5 * torch.linalg.vector_norm(
        torch.cross(p1 - p0, p2 - p0, dim=1),
        dim=1,
    )

    weights = torch.zeros(
        points_xyz.shape[0],
        device=points_xyz.device,
        dtype=points_xyz.dtype,
    )
    third = face_areas / 3.0
    for lv in range(3):
        weights.scatter_add_(0, faces_ijk[:, lv], third)

    return weights


def _optional_tensor(d: dict[str, Any], key: str):
    value = d.get(key, None)
    return value if isinstance(value, torch.Tensor) else None


def _make_phase1_decoder(
    *,
    n_seeds: int,
    device: torch.device,
    dtype: torch.dtype,
    strut_thickness: float,
    tube_beta: float,
    duplicate_merge_distance_factor: float,
    point_chunk_size: int,
    handoff_activity_threshold: float,
    handoff_territory_ratio: float,
) -> Phase1VoronoiDecoder:
    """
    Mirrors the current MainTrain Phase-1 decoder settings.

    duplicate_merge_sigma is a PHYSICAL 3D threshold:
        factor * full strut thickness.
    """
    return Phase1VoronoiDecoder(
        n_seeds=int(n_seeds),
        use_Metric_anisotropy=False,
        fixed_strut_radius=0.5 * float(strut_thickness),
        fixed_height=1.0,
        physical_tube_beta=float(tube_beta),
        duplicate_merge_sigma=(
            float(duplicate_merge_distance_factor)
            * float(strut_thickness)
        ),
        territory_min_ratio=float(handoff_territory_ratio),
        phase2_activity_threshold=float(handoff_activity_threshold),
        phase2_territory_ratio=float(handoff_territory_ratio),
        continuous_length_calibration=1.0,
        use_boundary_attachment=True,
        point_chunk_size=int(point_chunk_size),
        boundary_attach_width=2.0e-5,
        boundary_attach_beta=1.0e-5,
        boundary_attach_alpha=1.0,
        density_projection_strength=0.0,
        eps=1.0e-8,
    ).to(device=device, dtype=dtype)


def _fresh_phase1_forward(
    *,
    face_mesh: dict[str, Any],
    seeds_uv: torch.Tensor,
    strut_thickness: float,
    tau: float,
    tube_beta: float,
    duplicate_merge_distance_factor: float,
    point_chunk_size: int,
    handoff_activity_threshold: float,
    handoff_territory_ratio: float,
    projection_beta: float,
    projection_eta: float,
    projection_strength: float,
):
    """
    Fresh decoder + fresh graph for exactly one backward/autograd.grad call.
    """
    points_uv = face_mesh["uv"]
    points_xyz = face_mesh["points_xyz"]
    Xu = face_mesh["Xu"]
    Xv = face_mesh["Xv"]

    device = points_uv.device
    dtype = points_uv.dtype

    seeds = (
        seeds_uv.detach()
        .clone()
        .to(device=device, dtype=dtype)
        .requires_grad_(True)
    )

    decoder = _make_phase1_decoder(
        n_seeds=int(seeds.shape[0]),
        device=device,
        dtype=dtype,
        strut_thickness=strut_thickness,
        tube_beta=tube_beta,
        duplicate_merge_distance_factor=duplicate_merge_distance_factor,
        point_chunk_size=point_chunk_size,
        handoff_activity_threshold=handoff_activity_threshold,
        handoff_territory_ratio=handoff_territory_ratio,
    )

    area_weights = _surface_area_weights(face_mesh).to(device=device, dtype=dtype)

    # Resolve per-point CAD face ids robustly.
    #
    # Some face_mesh dictionaries store:
    #   - points_face_id: [N]
    # while others store:
    #   - face_id: [N] tensor
    # or a scalar face id.
    points_face_id = _optional_tensor(face_mesh, "points_face_id")
    if points_face_id is None:
        face_id_value = face_mesh.get("face_id", 0)

        if isinstance(face_id_value, torch.Tensor):
            face_id_value = face_id_value.to(device=device, dtype=torch.long)

            if face_id_value.ndim == 0 or face_id_value.numel() == 1:
                points_face_id = torch.full(
                    (int(points_uv.shape[0]),),
                    int(face_id_value.reshape(()).item()),
                    dtype=torch.long,
                    device=device,
                )
            elif face_id_value.numel() == int(points_uv.shape[0]):
                points_face_id = face_id_value.reshape(-1)
            else:
                raise ValueError(
                    "face_mesh['face_id'] is a tensor, but its size does not match "
                    f"the number of query points: face_id shape={tuple(face_id_value.shape)}, "
                    f"N={int(points_uv.shape[0])}"
                )
        else:
            points_face_id = torch.full(
                (int(points_uv.shape[0]),),
                int(face_id_value),
                dtype=torch.long,
                device=device,
            )
    else:
        points_face_id = points_face_id.to(device=device, dtype=torch.long).reshape(-1)

    boundary_uv = _optional_tensor(face_mesh, "boundary_curve_uv")
    if boundary_uv is None:
        boundary_uv = _optional_tensor(face_mesh, "boundary_uv")

    boundary_face_id = _optional_tensor(face_mesh, "boundary_face_id")
    if boundary_uv is not None and boundary_face_id is None:
        # For a single-face case, use the unique point face id.
        unique_point_faces = torch.unique(points_face_id)

        if unique_point_faces.numel() == 1:
            boundary_face_id = torch.full(
                (int(boundary_uv.shape[0]),),
                int(unique_point_faces[0].item()),
                dtype=torch.long,
                device=device,
            )
        else:
            raise ValueError(
                "boundary_uv is present but boundary_face_id is missing, and "
                "points_face_id contains multiple CAD faces. Please provide "
                "face_mesh['boundary_face_id'] explicitly."
            )
    elif boundary_face_id is not None:
        boundary_face_id = boundary_face_id.to(
            device=device,
            dtype=torch.long,
        ).reshape(-1)

    w_raw = torch.zeros(
        (int(seeds.shape[0]), int(seeds.shape[0])),
        dtype=dtype,
        device=device,
    )

    out = decoder(
        points_uv=points_uv,
        Xu=Xu,
        Xv=Xv,
        tau=float(tau),

        seeds_raw=seeds,
        w_raw=w_raw,
        h_raw=None,

        points_face_id=points_face_id,

        boundary_uv=boundary_uv,
        boundary_face_id=boundary_face_id,
        boundary_curve_offsets=_optional_tensor(
            face_mesh,
            "boundary_curve_offsets",
        ),

        seed_domain_mask=_optional_tensor(
            face_mesh,
            "seed_domain_mask_grid",
        ),

        surface_area_weights=area_weights,
    )

    rho_raw = out["rho"]
    rho_projected = smooth_heaviside_projection(
        rho_raw,
        beta=float(projection_beta),
        eta=float(projection_eta),
        strength=float(projection_strength),
        eps=1.0e-8,
        debug=False,
    )

    fiber = out["fiber3d"]

    # Safe fixed tangent field: normalized local Xu.
    fixed_fiber = F.normalize(
        Xu.to(device=device, dtype=dtype),
        dim=1,
        eps=1.0e-8,
    )

    return {
        "seeds": seeds,
        "decoder": decoder,
        "out": out,
        "rho_projected": rho_projected,
        "fiber": fiber,
        "fixed_fiber": fixed_fiber,
        "area_weights": area_weights,
    }


def _make_loss_fem(shell_problem, device):
    fem = run_fem_loss.NeuralTOMOFEM(
        shell_problem,
        device=device,
        isotropic=False,
    )
    stub = _FEMTrainerStub(shell_problem=shell_problem, fem=fem)
    return Loss_FEM(stub), stub


def _evaluate_fem(
    *,
    loss_fem,
    rho_surface: torch.Tensor,
    fiber_surface: torch.Tensor,
    fem_max_displacement: float,
    fem_yield_strength: float,
    fem_constraint_weight: float,
    fem_baseline_weight: float,
    fem_stress_density_threshold: float,
    fem_rho_min_ratio: float,
    fem_penal: float,
    fem_violation_power: float,
    eps: float,
):
    return loss_fem.evaluate(
        rho_surface=rho_surface,
        fiber_surface=fiber_surface,
        max_displacement=float(fem_max_displacement),
        yield_strength=float(fem_yield_strength),
        constraint_weight=float(fem_constraint_weight),
        baseline_weight=float(fem_baseline_weight),
        violation_power=float(fem_violation_power),
        stress_density_threshold=float(fem_stress_density_threshold),
        rho_min_ratio=float(fem_rho_min_ratio),
        penal=float(fem_penal),
        eps=float(eps),
        save_debug_history=False,
    )


def _gradient_stats(
    *,
    name: str,
    loss_key: str,
    loss: torch.Tensor,
    seeds: torch.Tensor,
    fem_out: dict[str, Any],
) -> GradientResult:
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        return GradientResult(
            name=name,
            fem_valid=bool(fem_out.get("fem_valid", False)),
            loss_key=loss_key,
            loss_value=float("nan"),
            grad_present=False,
            grad_finite=False,
            grad_norm=float("nan"),
            grad_abs_max=float("nan"),
            grad_nan=0,
            grad_posinf=0,
            grad_neginf=0,
            stress_max=_scalar(fem_out.get("stress_max")),
            displacement_max=_scalar(fem_out.get("displacement_max")),
            physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
            physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
            note="loss key missing or not scalar",
        )

    if not loss.requires_grad:
        return GradientResult(
            name=name,
            fem_valid=bool(fem_out.get("fem_valid", False)),
            loss_key=loss_key,
            loss_value=_scalar(loss),
            grad_present=False,
            grad_finite=True,
            grad_norm=0.0,
            grad_abs_max=0.0,
            grad_nan=0,
            grad_posinf=0,
            grad_neginf=0,
            stress_max=_scalar(fem_out.get("stress_max")),
            displacement_max=_scalar(fem_out.get("displacement_max")),
            physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
            physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
            note="loss has no gradient (often expected for an inactive zero penalty)",
        )

    grad = torch.autograd.grad(
        loss,
        seeds,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]

    if grad is None:
        return GradientResult(
            name=name,
            fem_valid=bool(fem_out.get("fem_valid", False)),
            loss_key=loss_key,
            loss_value=_scalar(loss),
            grad_present=False,
            grad_finite=True,
            grad_norm=0.0,
            grad_abs_max=0.0,
            grad_nan=0,
            grad_posinf=0,
            grad_neginf=0,
            stress_max=_scalar(fem_out.get("stress_max")),
            displacement_max=_scalar(fem_out.get("displacement_max")),
            physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
            physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
            note="seed gradient unused / detached",
        )

    grad_det = grad.detach()
    finite = torch.isfinite(grad_det)

    return GradientResult(
        name=name,
        fem_valid=bool(fem_out.get("fem_valid", False)),
        loss_key=loss_key,
        loss_value=_scalar(loss),
        grad_present=True,
        grad_finite=bool(finite.all().cpu().item()),
        grad_norm=_scalar(torch.linalg.vector_norm(torch.nan_to_num(grad_det))),
        grad_abs_max=_scalar(torch.nan_to_num(grad_det).abs().max()),
        grad_nan=int(torch.isnan(grad_det).sum().cpu().item()),
        grad_posinf=int(torch.isposinf(grad_det).sum().cpu().item()),
        grad_neginf=int(torch.isneginf(grad_det).sum().cpu().item()),
        stress_max=_scalar(fem_out.get("stress_max")),
        displacement_max=_scalar(fem_out.get("displacement_max")),
        physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
        physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
    )


def _print_result(r: GradientResult) -> None:
    status = "PASS" if r.grad_finite else "FAIL"
    print(
        f"{status:4s} | {r.name:30s} | "
        f"loss={r.loss_value: .6e} | "
        f"|g|={r.grad_norm: .3e} | "
        f"gmax={r.grad_abs_max: .3e} | "
        f"nan={r.grad_nan:4d} inf+={r.grad_posinf:4d} inf-={r.grad_neginf:4d}"
    )
    if r.note:
        print(f"       note: {r.note}")


def diagnose_phase1_fem_gradients(
    *,
    face_mesh: dict[str, Any],
    shell_problem,
    seeds_uv: torch.Tensor,
    strut_thickness: float = 0.40,
    tau: float = 0.01,
    tube_beta: float = 0.01,
    duplicate_merge_distance_factor: float = 1.10,
    point_chunk_size: int = 32,
    handoff_activity_threshold: float = 0.05,
    handoff_territory_ratio: float = 0.20,
    projection_beta: float = 8.0,
    projection_eta: float = 0.50,
    projection_strength: float = 1.0,
    fem_max_displacement: float = 20.0,
    fem_yield_strength: float = 45000.0,
    fem_constraint_weight: float = 2.0,
    fem_baseline_weight: float = 0.001,
    fem_stress_density_threshold: float = 0.50,
    fem_rho_min_ratio: float = 1.0e-3,
    fem_penal: float = 3.0,
    fem_violation_power: float = 3.0,
    eps: float = 1.0e-12,
    run_component_breakdown: bool = True,
    anomaly_case: str | None = None,
):
    """
    Run independent fresh-forward FEM gradient tests.

    Cases
    -----
    full_fem:
        Phase-1 projected rho + Phase-1 fiber.

    density_only_fixed_fiber:
        Phase-1 projected rho + fixed safe tangent fiber.
        If this fails, density/stiffness path is implicated.

    fiber_only_fixed_rho:
        Detached Phase-1 rho + Phase-1 fiber.
        If this fails, orientation/material-transformation path is implicated.

    fixed_rho_fixed_fiber:
        Both inputs detached/fixed. This should have no seed gradient and is a
        forward-solve sanity check.

    Component breakdown
    -------------------
    Each component is tested using a completely fresh decoder/FEM graph.
    This avoids retain_graph=True and is safer for memory.
    """

    device = face_mesh["uv"].device
    dtype = face_mesh["uv"].dtype
    seeds_uv = seeds_uv.to(device=device, dtype=dtype)

    print("\n" + "=" * 110)
    print("PHASE-1 -> FEM GRADIENT ISOLATION")
    print("=" * 110)
    print(f"device                    : {device}")
    print(f"dtype                     : {dtype}")
    print(f"seed count                : {int(seeds_uv.shape[0])}")
    print(f"strut thickness           : {float(strut_thickness):.6g}")
    print(
        "duplicate threshold       : "
        f"{float(duplicate_merge_distance_factor) * float(strut_thickness):.6g}"
    )
    print(f"point chunk size          : {int(point_chunk_size)}")
    print(f"FEM max displacement      : {float(fem_max_displacement):.6g}")
    print(f"FEM yield strength        : {float(fem_yield_strength):.6g}")
    print("=" * 110)

    loss_fem, fem_stub = _make_loss_fem(shell_problem, device=device)

    results: list[GradientResult] = []

    def run_one(
        case_name: str,
        *,
        rho_mode: str,
        fiber_mode: str,
        loss_key: str,
    ):
        _clear_cuda()

        ctx = torch.autograd.detect_anomaly() if anomaly_case == case_name else _nullcontext()

        with ctx:
            fwd = _fresh_phase1_forward(
                face_mesh=face_mesh,
                seeds_uv=seeds_uv,
                strut_thickness=strut_thickness,
                tau=tau,
                tube_beta=tube_beta,
                duplicate_merge_distance_factor=duplicate_merge_distance_factor,
                point_chunk_size=point_chunk_size,
                handoff_activity_threshold=handoff_activity_threshold,
                handoff_territory_ratio=handoff_territory_ratio,
                projection_beta=projection_beta,
                projection_eta=projection_eta,
                projection_strength=projection_strength,
            )

            seeds = fwd["seeds"]
            rho = fwd["rho_projected"]
            fiber = fwd["fiber"]

            if rho_mode == "detached":
                rho_fem = rho.detach()
            elif rho_mode == "uniform":
                rho_fem = torch.full_like(rho, 0.80)
            elif rho_mode == "live":
                rho_fem = rho
            else:
                raise ValueError(f"Unknown rho_mode={rho_mode}")

            if fiber_mode == "fixed":
                fiber_fem = fwd["fixed_fiber"].detach()
            elif fiber_mode == "detached":
                fiber_fem = fiber.detach()
            elif fiber_mode == "live":
                fiber_fem = fiber
            else:
                raise ValueError(f"Unknown fiber_mode={fiber_mode}")

            print(
                f"\n[{case_name}] input diagnostics: "
                f"rho=[{_scalar(rho_fem.min()):.3e}, {_scalar(rho_fem.max()):.3e}], "
                f"fiber finite={bool(torch.isfinite(fiber_fem).all().detach().cpu().item())}, "
                f"fiber norm=[{_scalar(torch.linalg.vector_norm(fiber_fem, dim=1).min()):.3e}, "
                f"{_scalar(torch.linalg.vector_norm(fiber_fem, dim=1).max()):.3e}]"
            )

            fem_out = _evaluate_fem(
                loss_fem=loss_fem,
                rho_surface=rho_fem,
                fiber_surface=fiber_fem,
                fem_max_displacement=fem_max_displacement,
                fem_yield_strength=fem_yield_strength,
                fem_constraint_weight=fem_constraint_weight,
                fem_baseline_weight=fem_baseline_weight,
                fem_stress_density_threshold=fem_stress_density_threshold,
                fem_rho_min_ratio=fem_rho_min_ratio,
                fem_penal=fem_penal,
                fem_violation_power=fem_violation_power,
                eps=eps,
            )

            loss = fem_out.get(loss_key, None)
            if not isinstance(loss, torch.Tensor):
                r = GradientResult(
                    name=case_name,
                    fem_valid=bool(fem_out.get("fem_valid", False)),
                    loss_key=loss_key,
                    loss_value=float("nan"),
                    grad_present=False,
                    grad_finite=False,
                    grad_norm=float("nan"),
                    grad_abs_max=float("nan"),
                    grad_nan=0,
                    grad_posinf=0,
                    grad_neginf=0,
                    stress_max=_scalar(fem_out.get("stress_max")),
                    displacement_max=_scalar(fem_out.get("displacement_max")),
                    physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
                    physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
                    note=f"FEM output has no tensor key '{loss_key}'",
                )
            elif not rho_fem.requires_grad and not fiber_fem.requires_grad:
                # Forward-only sanity case.
                r = GradientResult(
                    name=case_name,
                    fem_valid=bool(fem_out.get("fem_valid", False)),
                    loss_key=loss_key,
                    loss_value=_scalar(loss),
                    grad_present=False,
                    grad_finite=True,
                    grad_norm=0.0,
                    grad_abs_max=0.0,
                    grad_nan=0,
                    grad_posinf=0,
                    grad_neginf=0,
                    stress_max=_scalar(fem_out.get("stress_max")),
                    displacement_max=_scalar(fem_out.get("displacement_max")),
                    physical_stress_ratio=_scalar(fem_out.get("physical_stress_ratio")),
                    physical_displacement_ratio=_scalar(fem_out.get("physical_displacement_ratio")),
                    note="forward-only sanity case; both FEM inputs detached/fixed",
                )
            else:
                r = _gradient_stats(
                    name=case_name,
                    loss_key=loss_key,
                    loss=loss,
                    seeds=seeds,
                    fem_out=fem_out,
                )

            print(
                f"       FEM valid={r.fem_valid} | "
                f"stress_max={r.stress_max:.6e} | "
                f"disp_max={r.displacement_max:.6e} | "
                f"Rstress={r.physical_stress_ratio:.3e} | "
                f"Rdisp={r.physical_displacement_ratio:.3e}"
            )
            _print_result(r)

        results.append(r)

        del fwd
        _clear_cuda()
        return r

    # ------------------------------------------------------------------
    # Main isolation tests.
    # ------------------------------------------------------------------

    run_one(
        "full_fem",
        rho_mode="live",
        fiber_mode="live",
        loss_key="fem_total",
    )

    run_one(
        "density_only_fixed_fiber",
        rho_mode="live",
        fiber_mode="fixed",
        loss_key="fem_total",
    )

    run_one(
        "fiber_only_fixed_rho",
        rho_mode="detached",
        fiber_mode="live",
        loss_key="fem_total",
    )

    run_one(
        "fixed_rho_fixed_fiber",
        rho_mode="detached",
        fiber_mode="fixed",
        loss_key="fem_total",
    )

    # Also test an absolutely uniform safe density with live fiber.
    run_one(
        "fiber_only_uniform_rho",
        rho_mode="uniform",
        fiber_mode="live",
        loss_key="fem_total",
    )

    # ------------------------------------------------------------------
    # FEM component isolation.
    # Fresh graph for each key, no retained graph.
    # ------------------------------------------------------------------

    if run_component_breakdown:
        candidate_keys = [
            "stress_reduction_loss",
            "displacement_reduction_loss",
            "baseline_fem_loss",
            "stress_constraint_loss",
            "displacement_constraint_loss",
            "stress_margin_loss",
            "displacement_margin_loss",
            "stress_hard_violation_loss",
            "displacement_hard_violation_loss",
            "safety_margin_fem_loss",
            "hard_violation_fem_loss",
            "violation_fem_loss",
            "fem_total",
        ]

        print("\n" + "-" * 110)
        print("FEM COMPONENT GRADIENT BREAKDOWN — live Phase-1 rho + live Phase-1 fiber")
        print("-" * 110)

        for key in candidate_keys:
            run_one(
                f"component::{key}",
                rho_mode="live",
                fiber_mode="live",
                loss_key=key,
            )

    # ------------------------------------------------------------------
    # Compact summary.
    # ------------------------------------------------------------------

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    for r in results:
        _print_result(r)

    by_name = {r.name: r for r in results}

    full = by_name.get("full_fem")
    density = by_name.get("density_only_fixed_fiber")
    fiber = by_name.get("fiber_only_fixed_rho")
    fiber_uniform = by_name.get("fiber_only_uniform_rho")

    print("\nInterpretation:")
    if full is not None and not full.grad_finite:
        if density is not None and not density.grad_finite and fiber is not None and fiber.grad_finite:
            print("  -> Density/stiffness path is the leading suspect.")
        elif density is not None and density.grad_finite and fiber is not None and not fiber.grad_finite:
            print("  -> Fiber/orientation/material-transformation path is the leading suspect.")
        elif density is not None and not density.grad_finite and fiber is not None and not fiber.grad_finite:
            print("  -> Both density and fiber paths can independently create non-finite gradients.")
        elif density is not None and density.grad_finite and fiber is not None and fiber.grad_finite:
            print("  -> Individual paths pass; the coupled rho+fiber FEM path is the leading suspect.")
        else:
            print("  -> Inspect component results below.")
    elif full is not None and full.grad_finite:
        print("  -> Direct seed -> decoder -> FEM gradient is finite in this diagnostic state.")
        print("     If training still fails, test the exact PPNet-produced step-0 seeds next.")

    if (
        fiber is not None
        and not fiber.grad_finite
        and fiber_uniform is not None
        and not fiber_uniform.grad_finite
    ):
        print("  -> Fiber failure persists even with uniform rho: strong evidence for orientation/FEM material transform.")

    return {
        "results": results,
        "results_by_name": by_name,
        "fem_debug": dict(fem_stub.last_fem_debug),
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
