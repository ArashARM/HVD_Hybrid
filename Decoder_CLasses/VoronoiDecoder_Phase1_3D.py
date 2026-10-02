from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from dataclasses import dataclass
from typing import Any, Callable
from scipy.spatial import Voronoi, voronoi_plot_2d
from torch.utils.checkpoint import checkpoint



class VoronoiDecoder(nn.Module):
    """
    Phase-1 full-continuous Voronoi decoder.

    This decoder is intended for the first optimization phase: it starts from a
    deliberately over-complete seed set and represents the network by smooth
    fields rather than by an explicit Voronoi graph. Seed participation is
    differentiable and reversible during Phase 1. A hard seed subset is created
    only as a hand-off diagnostic for the graph-based Phase 2.

    Fiber directions are treated as an axial line field:
    t and -t are equivalent. To avoid sign-cancellation artifacts,
    pairwise directions are blended through orientation tensors t t^T.
    """

    def __init__(
        self,
        n_seeds: int,
        eps: float = 1e-8,
        use_Metric_anisotropy: bool = True,
        use_band_weighted_fiber_pairs: bool = True,
        use_boundary_tangent_fibers: bool = True,
        fiber_band_prior_power: float = 2.0,
        fiber_band_prior_floor: float = 0.05,
        pair_boost_strength: float = 0.05,
        pair_boost_enabled: bool = True,
        uv_scale_u: float = 1.0,
        uv_scale_v: float = 1.0, 
        point_chunk_size: int = 2048,   

        # geometric strut half-width lower bound
        w_min: float = 0.05,
        fixed_strut_radius: float | None = None,
        w_max_ratio: float = 0.8,

        # Phase-1 physical tube field. When enabled, density and fiber
        # are generated in physical surface space from the implicit
        # Voronoi bisectors, matching the Phase-2/Hybrid tube concept
        # without constructing a hard graph.
        use_physical_tube_fields: bool = True,
        physical_tube_beta: float | None = None,


        # density transition sharpness
        beta: float = 0.02,
        junction_beta_scale: float = 1.0,
        junction_width_bonus: float = 0.15,

        # effective-number boost for multi-seed zones
        junction_keff_lambda: float = 0.050,
        junction_keff_k0: float = 3.0,
        junction_keff_s: float = 0.35,

        # explicit triple-overlap junction term
        junction_triple_lambda: float = 0,
        junction_triple_power: float = 1.5,

        # raw parameter temperature for bounded maps
        raw_temp: float = 1.25,

        # union sharpness for combining pair bands
        alpha_union: float = 16.0,

        # optional smooth density projection. This keeps gradients continuous
        # while making the density used by FEM closer to a visible strut field.
        density_projection_strength: float = 0.0,
        density_projection_threshold: float = 0.5,
        density_projection_gamma: float = 0.05,

        # duplicate-seed activation. Seeds closer than this radius compete,
        # and one survivor remains effective in each connected duplicate cluster.
        duplicate_merge_sigma: float = 0.05,
        duplicate_effect_temp_ratio: float = 0.20,
        duplicate_effect_strength: float = 6.0,
        duplicate_effect_floor: float = 5e-2,
        seed_activity_sharpness: float = 1.0,
        seed_activity_threshold: float = 0.5,
        domain_effect_floor: float = 1e-8,
        domain_pair_power: float = 2.0,
        duplicate_pair_power: float = 1.0,
        pair_activity_power: float | None = None,
        global_activity_power: float = 2.0,
        invalid_domain_assignment_threshold: float = 1e-6,
        point_domain_floor: float = 0.0,

        # ------------------------------------------------------------------
        # Phase-1 continuous seed participation
        # ------------------------------------------------------------------
        # A small assignment floor keeps every seed differentiably connected
        # to the field, so a seed that is currently weak/outside/redundant can
        # recover later in Phase 1. It is NOT a material-density floor.
        phase1_assignment_floor: float = 1e-4,

        # Territory is measured as the fraction of the physical area of the
        # seed's CAD face that is softly assigned to that seed. Thresholds are
        # scaled by the nominal 1/N share on that face.
        territory_min_ratio: float = 0.15,
        territory_transition_ratio: float = 0.05,

        # One-time Phase-1 -> Phase-2 hand-off criteria. These are diagnostics
        # during Phase 1 and do not hard-mask the continuous field.
        phase2_activity_threshold: float = 0.050,
        phase2_territory_ratio: float = 0.20,

        # Continuous centreline-length estimate. The raw estimator integrates
        # smooth pair bands divided by their local physical band width. The
        # calibration factor is intentionally explicit so it can later be
        # calibrated against exact graph length without changing gradients.
        continuous_length_calibration: float = 1.0,
        continuous_length_min_physical_width: float = 1e-6,
        continuous_length_pair_threshold: float = 0.03,
        continuous_length_pair_softness: float = 0.01,

        # height controls
        h_min: float = 0.50,
        h_max: float = 2.00,
        fixed_height: float | None = None,

        # boundary & periodicity
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,

        # boundary attachment field
        use_boundary_attachment: bool = False,

        # keep these on comparable scales
        boundary_attach_width: float = 2e-5,
        boundary_attach_beta: float = 1e-5,
        boundary_attach_alpha: float = 0.35,

        boundary_attach_width_min: float = 5e-6,
        boundary_attach_width_max: float = 5e-5,

        boundary_attach_alpha_min: float = 0.05,
        boundary_attach_alpha_max: float = 1.00,

        boundary_attach_beta_min: float = 1e-6,
        boundary_attach_beta_max: float = 1e-4,

        # robust boundary-distance evaluation
        boundary_knn_k: int = 8,
        boundary_softmin_tau: float = 2e-3,
        boundary_spacing_blend: float = 0.5,
    ):
        super().__init__()

        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        self.use_Metric_anisotropy = bool(use_Metric_anisotropy)
        self.use_band_weighted_fiber_pairs = bool(use_band_weighted_fiber_pairs)
        self.use_boundary_tangent_fibers = bool(use_boundary_tangent_fibers)
        self.fiber_band_prior_power = float(fiber_band_prior_power)
        self.fiber_band_prior_floor = float(fiber_band_prior_floor)

        self.point_chunk_size = int(point_chunk_size)
        self.w_min = float(w_min)
        self.fixed_strut_radius = (
            None
            if fixed_strut_radius is None
            else float(fixed_strut_radius)
        )
        self.w_max_ratio = float(w_max_ratio)
        self.use_physical_tube_fields = bool(use_physical_tube_fields)
        self.physical_tube_beta = (
            float(beta) if physical_tube_beta is None else float(physical_tube_beta)
        )
        self.beta = float(beta)
        self.junction_beta_scale = float(junction_beta_scale)
        self.junction_width_bonus = float(junction_width_bonus)

        self.junction_keff_lambda = float(junction_keff_lambda)
        self.junction_keff_k0 = float(junction_keff_k0)
        self.junction_keff_s = float(junction_keff_s)

        self.uv_scale_u = float(uv_scale_u)
        self.uv_scale_v = float(uv_scale_v)

        self.junction_triple_lambda = float(junction_triple_lambda)
        self.junction_triple_power = float(junction_triple_power)

        self.raw_temp = float(raw_temp)
        self.alpha_union = float(alpha_union)
        self.density_projection_strength = float(density_projection_strength)
        self.density_projection_threshold = float(density_projection_threshold)
        self.density_projection_gamma = float(density_projection_gamma)
        self.duplicate_merge_sigma = float(duplicate_merge_sigma)
        self.duplicate_effect_temp_ratio = float(duplicate_effect_temp_ratio)
        self.duplicate_effect_strength = float(duplicate_effect_strength)
        self.duplicate_effect_floor = float(duplicate_effect_floor)
        self.seed_activity_sharpness = float(seed_activity_sharpness)
        self.seed_activity_threshold = float(seed_activity_threshold)
        self.domain_effect_floor = float(domain_effect_floor)
        self.domain_pair_power = float(
            domain_pair_power if pair_activity_power is None else pair_activity_power
        )
        self.duplicate_pair_power = float(duplicate_pair_power)
        self.global_activity_power = float(global_activity_power)
        self.invalid_domain_assignment_threshold = float(invalid_domain_assignment_threshold)
        self.point_domain_floor = float(point_domain_floor)

        self.phase1_assignment_floor = float(phase1_assignment_floor)
        self.territory_min_ratio = float(territory_min_ratio)
        self.territory_transition_ratio = float(territory_transition_ratio)
        self.phase2_activity_threshold = float(phase2_activity_threshold)
        self.phase2_territory_ratio = float(phase2_territory_ratio)
        self.continuous_length_calibration = float(continuous_length_calibration)
        self.continuous_length_min_physical_width = float(continuous_length_min_physical_width)
        self.continuous_length_pair_threshold = float(continuous_length_pair_threshold)
        self.continuous_length_pair_softness = float(continuous_length_pair_softness)

        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.fixed_height = float(fixed_height) if fixed_height is not None else None

        self.use_boundary_attachment = bool(use_boundary_attachment)

        self.boundary_attach_width_min = float(boundary_attach_width_min)
        self.boundary_attach_width_max = float(boundary_attach_width_max)
        self.boundary_attach_alpha_min = float(boundary_attach_alpha_min)
        self.boundary_attach_alpha_max = float(boundary_attach_alpha_max)
        self.boundary_attach_beta_min = float(boundary_attach_beta_min)
        self.boundary_attach_beta_max = float(boundary_attach_beta_max)
        self.boundary_knn_k = int(boundary_knn_k)
        self.boundary_softmin_tau = float(boundary_softmin_tau)
        self.boundary_spacing_blend = float(boundary_spacing_blend)

        self.pair_boost_enabled = bool(pair_boost_enabled)
        self.pair_boost_strength = float(pair_boost_strength)

        if not (self.boundary_attach_width_min < self.boundary_attach_width_max):
            raise ValueError(
                f"boundary_attach_width_min must be < boundary_attach_width_max, got "
                f"{self.boundary_attach_width_min} and {self.boundary_attach_width_max}"
            )
        if not (self.boundary_attach_alpha_min < self.boundary_attach_alpha_max):
            raise ValueError(
                f"boundary_attach_alpha_min must be < boundary_attach_alpha_max, got "
                f"{self.boundary_attach_alpha_min} and {self.boundary_attach_alpha_max}"
            )
        if not (self.boundary_attach_beta_min < self.boundary_attach_beta_max):
            raise ValueError(
                f"boundary_attach_beta_min must be < boundary_attach_beta_max, got "
                f"{self.boundary_attach_beta_min} and {self.boundary_attach_beta_max}"
            )
        if self.boundary_knn_k < 1:
            raise ValueError(f"boundary_knn_k must be >= 1, got {self.boundary_knn_k}")
        if self.boundary_softmin_tau <= 0:
            raise ValueError(f"boundary_softmin_tau must be > 0, got {self.boundary_softmin_tau}")
        if self.boundary_spacing_blend < 0:
            raise ValueError(f"boundary_spacing_blend must be >= 0, got {self.boundary_spacing_blend}")
        if self.junction_triple_power <= 0:
            raise ValueError(f"junction_triple_power must be > 0, got {self.junction_triple_power}")
        if self.duplicate_merge_sigma <= 0:
            raise ValueError(f"duplicate_merge_sigma must be > 0, got {self.duplicate_merge_sigma}")
        if self.duplicate_effect_temp_ratio <= 0:
            raise ValueError(
                f"duplicate_effect_temp_ratio must be > 0, got {self.duplicate_effect_temp_ratio}"
            )
        if self.duplicate_effect_strength < 0:
            raise ValueError(
                f"duplicate_effect_strength must be >= 0, got {self.duplicate_effect_strength}"
            )
        if not (0.0 < self.duplicate_effect_floor <= 1.0):
            raise ValueError(
                f"duplicate_effect_floor must be in (0, 1], got {self.duplicate_effect_floor}"
            )
        if self.seed_activity_sharpness <= 0.0:
            raise ValueError(
                f"seed_activity_sharpness must be > 0, got {self.seed_activity_sharpness}"
            )
        if not (0.0 < self.seed_activity_threshold < 1.0):
            raise ValueError(
                "seed_activity_threshold must be in (0, 1), "
                f"got {self.seed_activity_threshold}"
            )
        if not (0.0 < self.domain_effect_floor <= 1.0):
            raise ValueError(
                f"domain_effect_floor must be in (0, 1], got {self.domain_effect_floor}"
            )
        if self.domain_pair_power <= 0.0:
            raise ValueError(f"domain_pair_power must be > 0, got {self.domain_pair_power}")
        if self.duplicate_pair_power <= 0.0:
            raise ValueError(
                f"duplicate_pair_power must be > 0, got {self.duplicate_pair_power}"
            )
        if self.global_activity_power <= 0.0:
            raise ValueError(
                f"global_activity_power must be > 0, got {self.global_activity_power}"
            )
        if self.invalid_domain_assignment_threshold < 0.0:
            raise ValueError(
                "invalid_domain_assignment_threshold must be >= 0, "
                f"got {self.invalid_domain_assignment_threshold}"
            )
        if not (0.0 <= self.point_domain_floor <= 1.0):
            raise ValueError(
                f"point_domain_floor must be in [0, 1], got {self.point_domain_floor}"
            )
        if not (0.0 < self.phase1_assignment_floor < 1.0):
            raise ValueError(
                "phase1_assignment_floor must be in (0, 1), "
                f"got {self.phase1_assignment_floor}"
            )
        if self.territory_min_ratio <= 0.0:
            raise ValueError(
                f"territory_min_ratio must be > 0, got {self.territory_min_ratio}"
            )
        if self.territory_transition_ratio <= 0.0:
            raise ValueError(
                "territory_transition_ratio must be > 0, "
                f"got {self.territory_transition_ratio}"
            )
        if not (0.0 < self.phase2_activity_threshold < 1.0):
            raise ValueError(
                "phase2_activity_threshold must be in (0,1), "
                f"got {self.phase2_activity_threshold}"
            )
        if self.phase2_territory_ratio <= 0.0:
            raise ValueError(
                f"phase2_territory_ratio must be > 0, got {self.phase2_territory_ratio}"
            )
        if self.continuous_length_calibration <= 0.0:
            raise ValueError(
                "continuous_length_calibration must be > 0, "
                f"got {self.continuous_length_calibration}"
            )
        if self.continuous_length_min_physical_width <= 0.0:
            raise ValueError(
                "continuous_length_min_physical_width must be > 0, "
                f"got {self.continuous_length_min_physical_width}"
            )
        if self.continuous_length_pair_softness <= 0.0:
            raise ValueError("continuous_length_pair_softness must be > 0")
        if self.fiber_band_prior_power <= 0.0:
            raise ValueError(f"fiber_band_prior_power must be > 0, got {self.fiber_band_prior_power}")
        if not (0.0 <= self.fiber_band_prior_floor <= 1.0):
            raise ValueError(
                f"fiber_band_prior_floor must be in [0, 1], got {self.fiber_band_prior_floor}"
            )
        if self.alpha_union <= 0.0:
            raise ValueError(f"alpha_union must be > 0, got {self.alpha_union}")
        if not (0.0 <= self.density_projection_strength <= 1.0):
            raise ValueError(
                "density_projection_strength must be in [0,1], "
                f"got {self.density_projection_strength}"
            )
        if self.density_projection_gamma <= 0.0:
            raise ValueError(
                f"density_projection_gamma must be > 0, got {self.density_projection_gamma}"
            )

        if boundary_solid_idx is None:
            boundary_solid_idx = torch.empty(0, dtype=torch.long)
        if face_u_periodic is None:
            face_u_periodic = torch.zeros(1, dtype=torch.bool)
        if face_v_periodic is None:
            face_v_periodic = torch.zeros(1, dtype=torch.bool)
        if seed_face_id is None:
            seed_face_id = torch.zeros(self.n_seeds, dtype=torch.long)

        self.register_buffer("boundary_solid_idx", boundary_solid_idx.to(torch.long))
        self.register_buffer("face_u_periodic", face_u_periodic.to(torch.bool))
        self.register_buffer("face_v_periodic", face_v_periodic.to(torch.bool))
        self.register_buffer("seed_face_id", seed_face_id.to(torch.long))

        self.register_buffer(
            "boundary_attach_width_fixed",
            torch.tensor(float(boundary_attach_width), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_alpha_fixed",
            torch.tensor(float(boundary_attach_alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_beta_fixed",
            torch.tensor(float(boundary_attach_beta), dtype=torch.float32),
        )

    # -------------------- parameter maps --------------------

    def seeds_uv(self, seeds_raw: torch.Tensor) -> torch.Tensor:
        return seeds_raw

    def _seed_face_id_for(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed_face_id is not None:
            return seed_face_id.to(device=seeds.device, dtype=torch.long)
        if self.seed_face_id.shape[0] == seeds.shape[0]:
            return self.seed_face_id.to(device=seeds.device, dtype=torch.long)
        return torch.zeros(seeds.shape[0], device=seeds.device, dtype=torch.long)

    def _pairwise_seed_dist(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        seed_face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same_face = seed_face_id[:, None] == seed_face_id[None, :]

        uper_face = self.face_u_periodic[seed_face_id]
        vper_face = self.face_v_periodic[seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        du = v[..., 0]
        dv = v[..., 1]

        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)

        v[..., 0] = du
        v[..., 1] = dv
        return torch.norm(v, dim=-1)

    def _pairwise_seed_dist_physical(
        self,
        seeds: torch.Tensor,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        duv = seeds[:, None, :] - seeds[None, :, :]
        seed_face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same_face = seed_face_id[:, None] == seed_face_id[None, :]

        uper_face = self.face_u_periodic[seed_face_id]
        vper_face = self.face_v_periodic[seed_face_id]
        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        du = duv[..., 0]
        dv = duv[..., 1]
        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)

        mid_uv = 0.5 * (
            seeds[:, None, :]
            + seeds[None, :, :]
        )
        mid_flat = mid_uv.reshape(-1, 2)
        d_mid = torch.cdist(
            mid_flat.to(device=points_uv.device, dtype=points_uv.dtype),
            points_uv,
        )
        if points_face_id is not None:
            pf = points_face_id.to(device=points_uv.device, dtype=torch.long)
            pair_face = seed_face_id[:, None].expand_as(same_face).reshape(-1)
            same_point_face = pair_face[:, None] == pf[None, :]
            d_mid = torch.where(
                same_point_face,
                d_mid,
                torch.full_like(d_mid, 1.0e6),
            )
        nearest = d_mid.argmin(dim=1)

        xu = Xu.index_select(0, nearest).reshape(seeds.shape[0], seeds.shape[0], 3)
        xv = Xv.index_select(0, nearest).reshape(seeds.shape[0], seeds.shape[0], 3)
        delta_xyz = du[..., None] * xu + dv[..., None] * xv
        dist = torch.linalg.vector_norm(delta_xyz, dim=-1)
        dist = torch.where(same_face, dist, torch.full_like(dist, 1.0e6))
        dist = dist.masked_fill(
            torch.eye(seeds.shape[0], dtype=torch.bool, device=seeds.device),
            0.0,
        )
        return dist.to(device=seeds.device, dtype=seeds.dtype)

    def _seed_xyz_from_surface_samples(
        self,
        seeds: torch.Tensor,
        points_uv: torch.Tensor,
        points_3d: torch.Tensor | None,
    ) -> torch.Tensor:
        if points_3d is None or points_3d.numel() == 0:
            z = torch.zeros((seeds.shape[0], 1), device=seeds.device, dtype=seeds.dtype)
            return torch.cat([seeds, z], dim=1)
        points_uv_local = points_uv.to(device=seeds.device, dtype=seeds.dtype)
        points_xyz_local = points_3d.to(device=seeds.device, dtype=seeds.dtype).reshape(-1, 3)
        nn = torch.cdist(seeds, points_uv_local).argmin(dim=1)
        return points_xyz_local.index_select(0, nn)

    def _sample_seed_domain_values(
        self,
        seeds: torch.Tensor,
        domain: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
        *,
        name: str,
    ) -> torch.Tensor:
        domain_is_callable = callable(domain)
        if callable(domain):
            values = domain(seeds)
            if torch.is_tensor(values):
                values = values.to(device=seeds.device, dtype=seeds.dtype)
            else:
                values = torch.as_tensor(values, device=seeds.device, dtype=seeds.dtype)
        else:
            if torch.is_tensor(domain):
                values = domain.to(device=seeds.device, dtype=seeds.dtype)
            else:
                values = torch.as_tensor(domain, device=seeds.device, dtype=seeds.dtype)

        if values.ndim == 0:
            values = values.expand(seeds.shape[0])
        elif values.shape == (seeds.shape[0],):
            pass
        elif values.shape == (seeds.shape[0], 1):
            values = values.reshape(seeds.shape[0])
        elif values.ndim == 2 and values.shape[-1] == 1 and values.shape[0] == seeds.shape[0]:
            values = values.squeeze(-1)
        elif not domain_is_callable and values.ndim in (2, 3, 4):
            if values.ndim == 2:
                grid_values = values.unsqueeze(0).unsqueeze(0)
            elif values.ndim == 3:
                grid_values = values.unsqueeze(0) if values.shape[0] == 1 else values.unsqueeze(1)
            else:
                grid_values = values

            if grid_values.shape[0] != 1 or grid_values.shape[1] != 1:
                raise ValueError(
                    f"{name} grid must be (H,W), (1,H,W), (1,1,H,W), or per-seed; "
                    f"got {tuple(values.shape)}"
                )

            uv_grid = seeds.reshape(1, -1, 1, 2) * 2.0 - 1.0
            sampled = F.grid_sample(
                grid_values,
                uv_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            values = sampled.reshape(seeds.shape[0])
        else:
            raise ValueError(
                f"{name} must be callable, per-seed, or UV grid; got {tuple(values.shape)}"
            )

        if values.shape != (seeds.shape[0],):
            raise ValueError(f"{name} must evaluate to ({seeds.shape[0]},), got {tuple(values.shape)}")
        return values

    @staticmethod
    def _domain_can_sample_count(
        domain: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
        count: int,
    ) -> bool:
        if domain is None:
            return False
        if callable(domain):
            return True
        values = domain if torch.is_tensor(domain) else torch.as_tensor(domain)
        if values.ndim == 0:
            return True
        if values.shape == (count,) or values.shape == (count, 1):
            return True
        if values.ndim == 2 and values.shape[1] == 1:
            return False
        return values.ndim in (2, 3, 4)

    def _seed_domain_validity_state(
        self,
        seeds: torch.Tensor,
        temp: torch.Tensor,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones((seeds.shape[0],), device=seeds.device, dtype=seeds.dtype)
        active = torch.ones((seeds.shape[0],), device=seeds.device, dtype=torch.bool)
        sdf_values = torch.empty((0,), device=seeds.device, dtype=seeds.dtype)
        mask_values = torch.empty((0,), device=seeds.device, dtype=seeds.dtype)

        if seed_domain_sdf is not None:
            sdf_values = self._sample_seed_domain_values(seeds, seed_domain_sdf, name="seed_domain_sdf")
            sdf_weight = torch.sigmoid(sdf_values / temp.clamp_min(self.eps))
            weight = weight * sdf_weight
            active = active & (sdf_values >= 0.0)

        if seed_domain_mask is not None:
            mask_values = self._sample_seed_domain_values(seeds, seed_domain_mask, name="seed_domain_mask")
            threshold = torch.as_tensor(
                seed_domain_mask_threshold,
                device=seeds.device,
                dtype=seeds.dtype,
            )
            mask_weight = torch.sigmoid((mask_values - threshold) / temp.clamp_min(self.eps))
            weight = weight * mask_weight
            active = active & (mask_values >= threshold)

        return weight.clamp(0.0, 1.0), active, sdf_values, mask_values

    def _sharpen_seed_activity(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp(0.0, 1.0)
        if self.seed_activity_sharpness == 1.0:
            return weights

        sharpness = torch.as_tensor(
            self.seed_activity_sharpness,
            device=weights.device,
            dtype=weights.dtype,
        ).clamp_min(self.eps)
        threshold = torch.as_tensor(
            self.seed_activity_threshold,
            device=weights.device,
            dtype=weights.dtype,
        )
        temp = (0.25 / sharpness).clamp_min(self.eps)
        raw = torch.sigmoid((weights - threshold) / temp)
        lo = torch.sigmoid((torch.zeros_like(threshold) - threshold) / temp)
        hi = torch.sigmoid((torch.ones_like(threshold) - threshold) / temp)
        return ((raw - lo) / (hi - lo).clamp_min(self.eps)).clamp(0.0, 1.0)

    def _point_domain_validity_state(
        self,
        points_uv: torch.Tensor,
        temp: torch.Tensor,
        point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones((points_uv.shape[0],), device=points_uv.device, dtype=points_uv.dtype)
        sdf_values = torch.empty((0,), device=points_uv.device, dtype=points_uv.dtype)
        mask_values = torch.empty((0,), device=points_uv.device, dtype=points_uv.dtype)

        if point_domain_sdf is not None:
            sdf_values = self._sample_seed_domain_values(
                points_uv,
                point_domain_sdf,
                name="point_domain_sdf",
            )
            weight = weight * torch.sigmoid(sdf_values / temp.clamp_min(self.eps))

        if point_domain_mask is not None:
            mask_values = self._sample_seed_domain_values(
                points_uv,
                point_domain_mask,
                name="point_domain_mask",
            )
            threshold = torch.as_tensor(
                point_domain_mask_threshold,
                device=points_uv.device,
                dtype=points_uv.dtype,
            )
            mask_weight = torch.sigmoid((mask_values - threshold) / temp.clamp_min(self.eps))
            weight = weight * mask_weight

        return weight.clamp(0.0, 1.0), sdf_values, mask_values

    def _seed_activation_state(
        self,
        seeds: torch.Tensor,
        hard_seed_mask: bool = True,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp: float | torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        s = seeds.shape[0]
        temp = torch.as_tensor(
            max(float(self.duplicate_merge_sigma) * float(self.duplicate_effect_temp_ratio), self.eps),
            device=seeds.device,
            dtype=seeds.dtype,
        )
        domain_temp = temp if seed_domain_temp is None else torch.as_tensor(
            seed_domain_temp,
            device=seeds.device,
            dtype=seeds.dtype,
        ).clamp_min(self.eps)
        duplicate_floor = torch.as_tensor(
            self.duplicate_effect_floor,
            device=seeds.device,
            dtype=seeds.dtype,
        )
        domain_floor = torch.as_tensor(
            self.domain_effect_floor,
            device=seeds.device,
            dtype=seeds.dtype,
        )

        if s <= 1:
            if s == 0:
                active = torch.ones((s,), device=seeds.device, dtype=torch.bool)
                empty = torch.empty((s,), device=seeds.device, dtype=seeds.dtype)
                ones = torch.ones((s,), device=seeds.device, dtype=seeds.dtype)
                return ones, active, empty, empty, empty, ones, ones
            u = seeds[:, 0]
            v = seeds[:, 1]
            active = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            square_domain_weight = (
                torch.sigmoid(u / temp)
                * torch.sigmoid((1.0 - u) / temp)
                * torch.sigmoid(v / temp)
                * torch.sigmoid((1.0 - v) / temp)
            )
            uv_domain_weight, uv_domain_active, sdf_values, mask_values = self._seed_domain_validity_state(
                seeds=seeds,
                temp=domain_temp,
                seed_domain_sdf=seed_domain_sdf,
                seed_domain_mask=seed_domain_mask,
                seed_domain_mask_threshold=seed_domain_mask_threshold,
            )
            domain_weight = square_domain_weight * uv_domain_weight
            active = active & uv_domain_active
            duplicate_weight = torch.ones_like(domain_weight)
            domain_activity = domain_floor + (1.0 - domain_floor) * domain_weight
            weights = duplicate_weight * domain_activity
            weights = self._sharpen_seed_activity(weights)
            if hard_seed_mask:
                weights = weights * active.to(seeds.dtype)
            return weights, active, domain_weight, sdf_values, mask_values, duplicate_weight, domain_activity

        dist = self._pairwise_seed_dist(seeds).to(device=seeds.device, dtype=seeds.dtype)
        radius = torch.as_tensor(self.duplicate_merge_sigma, device=seeds.device, dtype=seeds.dtype)
        close = dist <= radius
        close.fill_diagonal_(True)

        close_cpu = close.detach().cpu().numpy()
        visited = [False] * s
        active_cpu = np.zeros((s,), dtype=bool)
        for start in range(s):
            if visited[start]:
                continue
            stack = [start]
            component = []
            visited[start] = True
            while stack:
                i = stack.pop()
                component.append(i)
                for j in np.nonzero(close_cpu[i])[0].tolist():
                    if not visited[j]:
                        visited[j] = True
                        stack.append(int(j))
            active_cpu[min(component)] = True

        active = torch.as_tensor(active_cpu, device=seeds.device, dtype=torch.bool)
        temp = (radius * float(self.duplicate_effect_temp_ratio)).clamp_min(self.eps)
        soft_close = torch.sigmoid((radius - dist) / temp)
        soft_close = soft_close.masked_fill(torch.eye(s, dtype=torch.bool, device=seeds.device), 0.0)
        lower_priority = torch.tril(
            torch.ones((s, s), dtype=seeds.dtype, device=seeds.device),
            diagonal=-1,
        )
        earlier_closeness = soft_close * lower_priority
        duplicate_strength = earlier_closeness.max(dim=1).values
        if s > 0:
            duplicate_strength = duplicate_strength.clone()
            duplicate_strength[0] = 0.0
        raw_duplicate_weight = torch.exp(
            -float(self.duplicate_effect_strength) * duplicate_strength
        )
        duplicate_weight = duplicate_floor + (1.0 - duplicate_floor) * raw_duplicate_weight
        u = seeds[:, 0]
        v = seeds[:, 1]
        inside_domain = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        active = active & inside_domain

        square_domain_weight = (
            torch.sigmoid(u / temp)
            * torch.sigmoid((1.0 - u) / temp)
            * torch.sigmoid(v / temp)
            * torch.sigmoid((1.0 - v) / temp)
        )
        uv_domain_weight, uv_domain_active, sdf_values, mask_values = self._seed_domain_validity_state(
            seeds=seeds,
            temp=domain_temp,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
        )
        domain_weight = square_domain_weight * uv_domain_weight
        active = active & uv_domain_active
        domain_activity = domain_floor + (1.0 - domain_floor) * domain_weight
        weights = duplicate_weight * domain_activity
        weights = self._sharpen_seed_activity(weights)
        if hard_seed_mask:
            weights = weights * active.to(seeds.dtype)
        return weights, active, domain_weight, sdf_values, mask_values, duplicate_weight, domain_activity

    def _pair_distinctness(
        self,
        seeds: torch.Tensor,
        device=None,
        dtype=None,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = seeds.device
        if dtype is None:
            dtype = seeds.dtype

        S = seeds.shape[0]
        pair_dist = self._pairwise_seed_dist(seeds, seed_face_id=seed_face_id).to(device=device, dtype=dtype)
        sigma = torch.as_tensor(self.duplicate_merge_sigma, device=device, dtype=dtype)

        distinctness = -torch.expm1(-(pair_dist.pow(2)) / (sigma.pow(2) + self.eps))
        distinctness = distinctness.pow(2)
        distinctness = distinctness.clamp(0.0, 1.0)
        return distinctness * self._strict_upper_tri_mask(S, device, dtype)

    def _pair_distinctness_from_distance(
        self,
        pair_dist: torch.Tensor,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        if device is None:
            device = pair_dist.device
        if dtype is None:
            dtype = pair_dist.dtype

        S = pair_dist.shape[0]
        pair_dist = pair_dist.to(device=device, dtype=dtype)
        sigma = torch.as_tensor(self.duplicate_merge_sigma, device=device, dtype=dtype)
        distinctness = -torch.expm1(-(pair_dist.pow(2)) / (sigma.pow(2) + self.eps))
        distinctness = distinctness.pow(2)
        distinctness = distinctness.clamp(0.0, 1.0)
        return distinctness * self._strict_upper_tri_mask(S, device, dtype)

    def width(
        self,
        w_raw: torch.Tensor,
        seeds: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fixed_strut_radius is not None:
            return torch.full_like(
                w_raw,
                self.fixed_strut_radius,
            )
        T = self.raw_temp
        if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
            raise ValueError(f"w_raw must be square (S,S), got {tuple(w_raw.shape)}")
        if seeds is None:
            raise ValueError("seeds must be provided when w_raw is pairwise")
        if seeds.shape[0] != w_raw.shape[0]:
            raise ValueError(
                f"pairwise w_raw expects seeds with matching S, got {tuple(seeds.shape)} and {tuple(w_raw.shape)}"
            )

        pair_dist = self._pairwise_seed_dist(seeds, seed_face_id=seed_face_id).to(device=w_raw.device, dtype=w_raw.dtype)
        pair_mask = torch.triu(
            torch.ones_like(pair_dist, dtype=torch.bool),
            diagonal=1,
        )
        if bool(pair_mask.any()):
            min_pair_dist = pair_dist[pair_mask].amin()
            width_raw_global = w_raw[pair_mask].mean()
        else:
            min_pair_dist = torch.zeros((), device=w_raw.device, dtype=w_raw.dtype)
            width_raw_global = w_raw.mean()

        w_max = (self.w_max_ratio * min_pair_dist).clamp_min(self.w_min)
        width_frac = 0.5 * (torch.tanh(width_raw_global / max(T, self.eps)) + 1.0)
        w_geo = self.w_min + (w_max - self.w_min) * width_frac
        return w_geo.expand_as(w_raw)

    def height(
        self,
        h_raw: torch.Tensor | None,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fixed_height is not None:
            if ref_tensor is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
            if h_raw is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=h_raw.device,
                    dtype=h_raw.dtype,
                )
            return torch.tensor(float(self.fixed_height))

        if h_raw is None:
            raise ValueError("h_raw must be provided when fixed_height is None")

        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(h_raw)

    def _map_raw_to_range(
        self,
        x_raw: torch.Tensor,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x_raw / temp)

    def raw_from_bounded_value(
        self,
        value: float,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        denom = max(hi - lo, self.eps)
        x = (value - lo) / denom
        x = min(max(x, 1e-6), 1.0 - 1e-6)
        raw = temp * math.log(x / (1.0 - x))
        return torch.tensor(raw, dtype=torch.float32)

    # -------------------- boundary control getters --------------------

    def boundary_width(
        self,
        ref_tensor: torch.Tensor,
        boundary_width_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_width_raw is None:
            return self.boundary_attach_width_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_width_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_width_min,
            self.boundary_attach_width_max,
            temp=1.0,
        )

    def boundary_alpha(
        self,
        ref_tensor: torch.Tensor,
        boundary_alpha_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_alpha_raw is None:
            return self.boundary_attach_alpha_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_alpha_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_alpha_min,
            self.boundary_attach_alpha_max,
            temp=1.0,
        )

    def boundary_beta(
        self,
        ref_tensor: torch.Tensor,
        boundary_beta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_beta_raw is None:
            return self.boundary_attach_beta_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_beta_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_beta_min,
            self.boundary_attach_beta_max,
            temp=1.0,
        )

    # -------------------- anisotropic metric --------------------

    def metric_matrices(
        self,
        theta: torch.Tensor,
        a_raw: torch.Tensor,
        a_min: float = 0.5,
        a_max: float = 2.0,
    ) -> torch.Tensor:
        if theta.ndim != 1 or a_raw.ndim != 1 or theta.shape != a_raw.shape:
            raise ValueError(
                f"metric_matrices expects theta and a_raw of shape (S,), got {theta.shape}, {a_raw.shape}"
            )

        S = theta.shape[0]
        t = torch.tanh(a_raw)
        a = 0.5 * (a_max - a_min) * t + 0.5 * (a_max + a_min)

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            [torch.stack([c, -s], -1), torch.stack([s, c], -1)],
            -2,
        )

        D = torch.zeros((S, 2, 2), device=R.device, dtype=R.dtype)
        D[:, 0, 0] = a
        D[:, 1, 1] = 1.0 / (a + self.eps)

        return R.transpose(1, 2) @ D @ R

    # -------------------- periodic helpers --------------------

    def _wrap_duv_points_to_seeds(
        self,
        diff: torch.Tensor,
        points_face_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if points_face_id is None:
            return diff

        if points_face_id.dtype != torch.long:
            points_face_id = points_face_id.to(torch.long)

        uper = self.face_u_periodic[points_face_id].to(diff.dtype)
        vper = self.face_v_periodic[points_face_id].to(diff.dtype)

        du = diff[..., 0]
        dv = diff[..., 1]

        du = du - torch.round(du) * uper[:, None]
        dv = dv - torch.round(dv) * vper[:, None]

        diff[..., 0] = du
        diff[..., 1] = dv
        return diff

    def _pairwise_uv_dirs(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        seed_face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same_face = seed_face_id[:, None] == seed_face_id[None, :]

        uper_face = self.face_u_periodic[seed_face_id]
        vper_face = self.face_v_periodic[seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        du = v[..., 0]
        dv = v[..., 1]

        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)

        v[..., 0] = du
        v[..., 1] = dv

        t = torch.stack([-v[..., 1], v[..., 0]], dim=-1)
        n = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        return t / n

    # -------------------- fiber helpers --------------------

    def _strict_upper_tri_mask(self, S: int, device, dtype) -> torch.Tensor:
        return torch.triu(torch.ones(S, S, device=device, dtype=dtype), diagonal=1)

    def _soft_pair_weights(
        self,
        weights: torch.Tensor,
        seeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, S = weights.shape
        pair = weights.unsqueeze(2) * weights.unsqueeze(1)
        if seeds is None:
            pair_mask = self._strict_upper_tri_mask(S, weights.device, weights.dtype)
        else:
            pair_mask = self._pair_distinctness(
                seeds=seeds,
                device=weights.device,
                dtype=weights.dtype,
            )
        pair = pair * pair_mask.unsqueeze(0)
        denom = pair.sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        return pair / denom

    def _normalize_upper_tri_pair_weights(self, pair_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_weights.ndim != 3 or pair_weights.shape[1] != pair_weights.shape[2]:
            raise ValueError(f"pair_weights must be (N,S,S), got {tuple(pair_weights.shape)}")

        N, S, _ = pair_weights.shape
        tri = self._strict_upper_tri_mask(S, pair_weights.device, pair_weights.dtype).unsqueeze(0)
        pair = pair_weights * tri
        pair = pair.clamp_min(0.0)

        raw_sum = pair.sum(dim=(1, 2), keepdim=True)
        ok = raw_sum > self.eps
        pair_norm = pair / raw_sum.clamp_min(self.eps)
        return pair_norm, ok.expand(N, S, S)

    def _axial_tensor_from_pair_weights(
        self,
        pair_weights: torch.Tensor,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair_weights = torch.nan_to_num(pair_weights, nan=0.0, posinf=0.0, neginf=0.0)
        t_ij = self._pairwise_uv_dirs(seeds, seed_face_id=seed_face_id)               # (S,S,2)
        t_ij = torch.nan_to_num(t_ij, nan=0.0, posinf=0.0, neginf=0.0)
        Q_ij = t_ij.unsqueeze(-1) * t_ij.unsqueeze(-2)     # (S,S,2,2)
        return (pair_weights.unsqueeze(-1).unsqueeze(-1) * Q_ij.unsqueeze(0)).sum(dim=(1, 2))

    def _axial_tensor_from_local_pair_tangents(
        self,
        pair_weights: torch.Tensor,
        pair_tangent_ij: torch.Tensor,
    ) -> torch.Tensor:
        pair_weights = torch.nan_to_num(pair_weights, nan=0.0, posinf=0.0, neginf=0.0)
        pair_tangent_ij = torch.nan_to_num(pair_tangent_ij, nan=0.0, posinf=0.0, neginf=0.0)
        Q_ij = pair_tangent_ij.unsqueeze(-1) * pair_tangent_ij.unsqueeze(-2)
        return (pair_weights.unsqueeze(-1).unsqueeze(-1) * Q_ij).sum(dim=(1, 2))

    def _principal_axial_direction(self, Q: torch.Tensor) -> torch.Tensor:
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        gap_eps = torch.as_tensor(
            1e-6,
            device=Q.device,
            dtype=Q.dtype,
        )

        q00 = Q[..., 0, 0]
        q01 = Q[..., 0, 1]
        q11 = Q[..., 1, 1]
        trace = q00 + q11
        diff = q00 - q11
        gap = torch.sqrt(diff * diff + 4.0 * q01 * q01 + gap_eps * gap_eps)
        lambda_max = 0.5 * (trace + gap)

        v1 = torch.stack([q01, lambda_max - q00], dim=-1)
        v2 = torch.stack([lambda_max - q11, q01], dim=-1)
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True)
        v2_norm = torch.linalg.vector_norm(v2, dim=-1, keepdim=True)
        use_v1 = v1_norm >= v2_norm
        t_uv = torch.where(use_v1, v1, v2)

        # Isotropic/zero tensors do not have a meaningful principal direction.
        fallback = torch.zeros_like(t_uv)
        fallback[..., 0] = 1.0
        t_norm = torch.linalg.vector_norm(t_uv, dim=-1, keepdim=True)
        t_norm_safe = t_norm.clamp_min(gap_eps)
        t_uv_unit = t_uv / t_norm_safe
        t_uv = torch.where(t_norm > gap_eps, t_uv_unit, fallback)
        has_orientation = trace.reshape(*trace.shape, 1) > self.eps
        return torch.where(has_orientation, t_uv, torch.zeros_like(t_uv))

    def _axial_coherence_from_tensor(self, Q: torch.Tensor) -> torch.Tensor:
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        gap_eps = torch.as_tensor(
            1e-6,
            device=Q.device,
            dtype=Q.dtype,
        )
        q00 = Q[..., 0, 0]
        q01 = Q[..., 0, 1]
        q11 = Q[..., 1, 1]
        trace = (q00 + q11).clamp_min(self.eps)
        diff = q00 - q11
        gap = torch.sqrt(diff * diff + 4.0 * q01 * q01 + gap_eps * gap_eps)
        return (gap / trace).clamp(0.0, 1.0)

    def _blended_uv_fiber_axial(
        self,
        weights: torch.Tensor,
        seeds: torch.Tensor,
        pair_weights: torch.Tensor | None = None,
        normalize_pair_weights: bool = True,
        seed_face_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pair_weights is None:
            pair_weights = self._soft_pair_weights(weights, seeds=seeds)
        elif normalize_pair_weights:
            pair_weights, _ = self._normalize_upper_tri_pair_weights(pair_weights)
        else:
            S = pair_weights.shape[1]
            tri = self._strict_upper_tri_mask(S, pair_weights.device, pair_weights.dtype).unsqueeze(0)
            pair_weights = pair_weights.clamp_min(0.0) * tri

        Q = self._axial_tensor_from_pair_weights(pair_weights, seeds, seed_face_id=seed_face_id)
        t_uv = self._principal_axial_direction(Q)
        return t_uv, Q, pair_weights

    def _blended_uv_fiber(self, weights: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        # Backward-compatible wrapper. Uses axial blending so (-t) and t
        # are treated as the same fiber direction.
        t_uv, _, _ = self._blended_uv_fiber_axial(weights, seeds)
        return t_uv

    def _fiber_pair_weights(
        self,
        w_soft: torch.Tensor,
        seeds: torch.Tensor,
        band_ij: torch.Tensor | None = None,
        pair_relevance: torch.Tensor | None = None,
        seed_active_weights: torch.Tensor | None = None,
        seed_duplicate_weights: torch.Tensor | None = None,
        seed_domain_weights: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed_active_weights is None:
            soft_pair = self._soft_pair_weights(w_soft, seeds=seeds)
            if not self.use_band_weighted_fiber_pairs:
                return soft_pair

            if band_ij is None or pair_relevance is None:
                return soft_pair
        else:
            S = w_soft.shape[1]
            if seed_active_weights.ndim != 1 or seed_active_weights.shape[0] != S:
                raise ValueError(
                    f"seed_active_weights must have shape ({S},), got {tuple(seed_active_weights.shape)}"
                )
            g = seed_active_weights.to(device=w_soft.device, dtype=w_soft.dtype).clamp(0.0, 1.0)
            if seed_duplicate_weights is None:
                duplicate_activity = g
            else:
                if seed_duplicate_weights.ndim != 1 or seed_duplicate_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_duplicate_weights must have shape ({S},), "
                        f"got {tuple(seed_duplicate_weights.shape)}"
                    )
                duplicate_activity = seed_duplicate_weights.to(
                    device=w_soft.device,
                    dtype=w_soft.dtype,
                ).clamp(0.0, 1.0)
            if seed_domain_weights is None:
                domain_activity = g
            else:
                if seed_domain_weights.ndim != 1 or seed_domain_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_domain_weights must have shape ({S},), "
                        f"got {tuple(seed_domain_weights.shape)}"
                    )
                domain_activity = seed_domain_weights.to(
                    device=w_soft.device,
                    dtype=w_soft.dtype,
                ).clamp(0.0, 1.0)
            pair_activity = (
                (domain_activity[:, None] * domain_activity[None, :]).pow(float(self.domain_pair_power))
                * (duplicate_activity[:, None] * duplicate_activity[None, :]).pow(
                    float(self.duplicate_pair_power)
                )
            )
            pair_mask = self._pair_distinctness(
                seeds=seeds,
                device=w_soft.device,
                dtype=w_soft.dtype,
                seed_face_id=seed_face_id,
            )
            raw_pair = (
                w_soft.unsqueeze(2)
                * w_soft.unsqueeze(1)
                * pair_mask.unsqueeze(0)
                * pair_activity.unsqueeze(0)
            )
            if not self.use_band_weighted_fiber_pairs or band_ij is None or pair_relevance is None:
                return raw_pair

        # Prefer pairs whose visible band is present at this point, but keep a
        # small soft-pair floor so clipped ends/junctions do not jump abruptly.
        band_prior = band_ij.clamp(0.0, 1.0).pow(float(self.fiber_band_prior_power))
        floor = torch.as_tensor(
            self.fiber_band_prior_floor,
            device=band_prior.device,
            dtype=band_prior.dtype,
        )
        band_prior = floor + (1.0 - floor) * band_prior
        raw_pair = soft_pair * band_prior if seed_active_weights is None else raw_pair * band_prior
        if seed_active_weights is not None:
            return raw_pair
        pair_norm, ok_mask = self._normalize_upper_tri_pair_weights(raw_pair)
        return torch.where(ok_mask, pair_norm, soft_pair)

    def _estimate_boundary_sample_tangents_uv(
        self,
        boundary_uv: torch.Tensor,
        boundary_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_uv.numel() == 0:
            return torch.zeros_like(boundary_uv)

        B = boundary_uv.shape[0]
        if B < 2:
            return torch.zeros_like(boundary_uv)

        if boundary_face_id is None:
            boundary_face_id = torch.zeros(B, device=boundary_uv.device, dtype=torch.long)
        elif boundary_face_id.dtype != torch.long:
            boundary_face_id = boundary_face_id.to(torch.long)

        diff = boundary_uv.unsqueeze(1) - boundary_uv.unsqueeze(0)
        dmat = torch.norm(diff, dim=-1)

        same_face = boundary_face_id[:, None] == boundary_face_id[None, :]
        eye = torch.eye(B, device=boundary_uv.device, dtype=torch.bool)
        valid_neighbor = same_face & (~eye)
        dmat = torch.where(valid_neighbor, dmat, torch.full_like(dmat, 1e6))

        k = min(max(1, self.boundary_knn_k), max(1, B - 1))
        d_knn, idx_knn = torch.topk(dmat, k=k, dim=1, largest=False)
        valid_knn = d_knn < 1e5

        local_diff = boundary_uv[idx_knn] - boundary_uv.unsqueeze(1)
        sigma = d_knn[..., 0].clamp_min(self.boundary_softmin_tau)
        w = torch.exp(-0.5 * (d_knn / sigma.unsqueeze(1).clamp_min(self.eps)).pow(2))
        w = w * valid_knn.to(w.dtype)

        cov = (
            w.unsqueeze(-1).unsqueeze(-1)
            * (local_diff.unsqueeze(-1) * local_diff.unsqueeze(-2))
        ).sum(dim=1)
        cov = cov / w.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(self.eps)

        tangent = self._principal_axial_direction(cov)
        has_support = valid_knn.any(dim=1, keepdim=True)
        return torch.where(has_support, tangent, torch.zeros_like(tangent))

    def _boundary_tangent_tensor_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_uv.numel() == 0:
            return torch.zeros(
                (points_uv.shape[0], 2, 2),
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        B = boundary_uv.shape[0]
        if points_face_id is None:
            points_face_id = torch.zeros(points_uv.shape[0], device=points_uv.device, dtype=torch.long)
        elif points_face_id.dtype != torch.long:
            points_face_id = points_face_id.to(torch.long)

        if boundary_face_id is None:
            boundary_face_id = torch.zeros(B, device=boundary_uv.device, dtype=torch.long)
        elif boundary_face_id.dtype != torch.long:
            boundary_face_id = boundary_face_id.to(torch.long)

        tangent_uv = self._estimate_boundary_sample_tangents_uv(
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
        )
        sample_Q = tangent_uv.unsqueeze(-1) * tangent_uv.unsqueeze(-2)

        bb_diff = boundary_uv.unsqueeze(1) - boundary_uv.unsqueeze(0)
        bb_dmat = torch.norm(bb_diff, dim=-1)
        same_face_bb = boundary_face_id[:, None] == boundary_face_id[None, :]
        eye = torch.eye(B, device=boundary_uv.device, dtype=torch.bool)
        bb_valid = same_face_bb & (~eye)
        bb_dmat = torch.where(bb_valid, bb_dmat, torch.full_like(bb_dmat, 1e6))
        local_scale = bb_dmat.min(dim=1).values
        local_scale = torch.where(
            local_scale < 1e5,
            local_scale,
            torch.full_like(local_scale, self.boundary_softmin_tau),
        ).clamp_min(self.boundary_softmin_tau)

        pb_dmat = torch.cdist(points_uv, boundary_uv)
        same_face_pb = points_face_id[:, None] == boundary_face_id[None, :]
        pb_dmat = torch.where(same_face_pb, pb_dmat, torch.full_like(pb_dmat, 1e6))

        weights = torch.exp(
            -0.5 * (pb_dmat / local_scale.unsqueeze(0).clamp_min(self.eps)).pow(2)
        )
        weights = weights * same_face_pb.to(weights.dtype)

        weight_sum = weights.sum(dim=1, keepdim=True)
        # Shell boundaries should inject tangent-aligned line directions, but
        # only where the shell-boundary field is active; away from the boundary
        # the Voronoi interior tensor remains in control.
        Q_boundary = (weights.unsqueeze(-1).unsqueeze(-1) * sample_Q.unsqueeze(0)).sum(dim=1)
        Q_boundary = Q_boundary / weight_sum.unsqueeze(-1).clamp_min(self.eps)

        has_support = weight_sum.squeeze(1) > self.eps
        return torch.where(has_support[:, None, None], Q_boundary, torch.zeros_like(Q_boundary))

    def map_to_3d(self, t_uv: torch.Tensor, Xu: torch.Tensor, Xv: torch.Tensor, eps: float = 1e-8):
        T = t_uv[:, 0:1] * Xu + t_uv[:, 1:2] * Xv
        return F.normalize(T, dim=1, eps=eps)

    def _orthonormal_tangent_basis(
        self,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = F.normalize(
            Xu,
            dim=-1,
            eps=self.eps,
        )

        proj = (
            Xv * e1
        ).sum(
            dim=-1,
            keepdim=True,
        )

        e2_raw = Xv - proj * e1

        e2_norm = torch.linalg.vector_norm(
            e2_raw,
            dim=-1,
            keepdim=True,
        )

        abs_e1 = e1.abs()

        use_x = (
            (abs_e1[..., 0] <= abs_e1[..., 1])
            & (abs_e1[..., 0] <= abs_e1[..., 2])
        )

        use_y = (
            (~use_x)
            & (abs_e1[..., 1] <= abs_e1[..., 2])
        )

        axis = torch.zeros_like(e1)
        axis[..., 0] = use_x.to(e1.dtype)
        axis[..., 1] = use_y.to(e1.dtype)
        axis[..., 2] = (~use_x & ~use_y).to(e1.dtype)

        fallback = torch.cross(
            e1,
            axis,
            dim=-1,
        )

        fallback = F.normalize(
            fallback,
            dim=-1,
            eps=self.eps,
        )

        e2 = torch.where(
            e2_norm > self.eps,
            e2_raw / e2_norm.clamp_min(self.eps),
            fallback,
        )

        return e1, e2

    def _local_physical_pair_tangents(
        self,
        pair_tangent_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> torch.Tensor:
        basis_shape = [Xu.shape[0]] + [1] * (pair_tangent_uv.ndim - 2) + [3]

        Xu_b = Xu.reshape(*basis_shape)
        Xv_b = Xv.reshape(*basis_shape)
        e1_b = e1.reshape(*basis_shape)
        e2_b = e2.reshape(*basis_shape)

        T_pair = (
            pair_tangent_uv[..., 0:1] * Xu_b
            + pair_tangent_uv[..., 1:2] * Xv_b
        )

        T_pair_norm = torch.linalg.vector_norm(
            T_pair,
            dim=-1,
            keepdim=True,
        )

        T_pair_unit = torch.where(
            T_pair_norm > self.eps,
            T_pair / T_pair_norm.clamp_min(self.eps),
            e1_b.expand_as(T_pair),
        )

        c1 = (
            T_pair_unit * e1_b
        ).sum(
            dim=-1,
        )

        c2 = (
            T_pair_unit * e2_b
        ).sum(
            dim=-1,
        )

        pair_tangent_local_2d = torch.stack(
            [c1, c2],
            dim=-1,
        )

        return F.normalize(
            pair_tangent_local_2d,
            dim=-1,
            eps=self.eps,
        )

    def _local_direction_to_xyz(
        self,
        t_local: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> torch.Tensor:
        fiber3d_raw = (
            t_local[:, 0:1] * e1
            + t_local[:, 1:2] * e2
        )

        return F.normalize(
            fiber3d_raw,
            dim=-1,
            eps=self.eps,
        )


    # -------------------- physical 3D tube helpers --------------------

    def _pair_tangents_xyz(
        self,
        pair_tangent_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
    ) -> torch.Tensor:
        """Lift local UV line tangents to the physical surface tangent plane.

        This is the differential equivalent of lifting sampled UV centerlines
        through the CAD surface: a UV direction (du,dv) is mapped to
        du*Xu + dv*Xv and normalized in XYZ.
        """
        T = (
            pair_tangent_uv[..., 0:1] * Xu[:, None, None, :]
            + pair_tangent_uv[..., 1:2] * Xv[:, None, None, :]
        )
        norm = torch.linalg.vector_norm(T, dim=-1, keepdim=True)
        fallback = Xu[:, None, None, :].expand_as(T)
        fallback = F.normalize(fallback, dim=-1, eps=self.eps)
        return torch.where(
            norm > self.eps,
            T / norm.clamp_min(self.eps),
            fallback,
        )

    def _principal_axial_direction_3d(
        self,
        Q: torch.Tensor,
        fallback_xyz: torch.Tensor,
    ) -> torch.Tensor:

        Q = torch.nan_to_num(
            Q,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        Q = 0.5 * (
            Q + Q.transpose(-1, -2)
        )

        # Move only the tiny 3x3 eigensystem to CPU.
        # Keep gradients through the copy.
        Q_cpu = Q.to("cpu")

        evals_cpu, evecs_cpu = torch.linalg.eigh(
            Q_cpu
        )

        principal = evecs_cpu[..., -1].to(
            device=Q.device,
            dtype=Q.dtype,
        )

        trace = torch.diagonal(
            Q,
            dim1=-2,
            dim2=-1,
        ).sum(
            dim=-1,
            keepdim=True,
        )

        fallback = F.normalize(
            fallback_xyz,
            dim=-1,
            eps=self.eps,
        )

        principal = F.normalize(
            principal,
            dim=-1,
            eps=self.eps,
        )

        return torch.where(
            trace > self.eps,
            principal,
            fallback,
        )

    def _axial_coherence_from_tensor_3d(self, Q: torch.Tensor) -> torch.Tensor:
        """Return a [0,1] coherence measure from a 3D axial tensor."""
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(
    Q.to("cpu")
).to(
    device=Q.device,
    dtype=Q.dtype,
).clamp_min(0.0)
        l1 = evals[..., -1]
        l2 = evals[..., -2]
        return ((l1 - l2) / l1.clamp_min(self.eps)).clamp(0.0, 1.0)

    def _physical_boundary_tube_field(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        radius: torch.Tensor | float,
        beta: float | torch.Tensor,
        boundary_curve_offsets: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physical tube around the CAD boundary using local surface metric.

        Boundary segments stay in normalized UV, but point-to-segment distance is
        evaluated with the first fundamental form at each query point, yielding a
        physical distance in the CAD model's length unit. Tangents are lifted to
        the physical tangent plane, expressed in the local orthonormal basis, and
        blended axially with the same distance weights.
        """

        N = int(points_uv.shape[0])
        if chunk_size is None:
            chunk_size = self.point_chunk_size

        chunk_size = max(
            1,
            min(
                int(chunk_size),
                N,
            ),
        )
        if boundary_uv is None or boundary_uv.numel() < 4:
            z = points_uv.new_zeros((N,))
            Qz = points_uv.new_zeros((N, 2, 2))
            inf = points_uv.new_full((N,), float("inf"))
            return z, Qz, inf

        B = int(boundary_uv.shape[0])
        if boundary_curve_offsets is not None:
            offsets = torch.as_tensor(
                boundary_curve_offsets,
                device=boundary_uv.device,
                dtype=torch.long,
            ).reshape(-1)
            starts, ends = [], []
            for k in range(max(int(offsets.numel()) - 1, 0)):
                a = int(offsets[k].item())
                b = int(offsets[k + 1].item())
                if b - a >= 2:
                    starts.append(torch.arange(a, b - 1, device=boundary_uv.device))
                    ends.append(torch.arange(a + 1, b, device=boundary_uv.device))
            if starts:
                idx_a = torch.cat(starts)
                idx_b = torch.cat(ends)
            else:
                idx_a = torch.arange(0, B - 1, device=boundary_uv.device)
                idx_b = idx_a + 1
        else:
            idx_a = torch.arange(0, B - 1, device=boundary_uv.device)
            idx_b = idx_a + 1

        seg_a = boundary_uv.index_select(0, idx_a)
        seg_b = boundary_uv.index_select(0, idx_b)
        seg_uv = seg_b - seg_a
        Gs = int(seg_a.shape[0])
        if Gs == 0:
            z = points_uv.new_zeros((N,))
            Qz = points_uv.new_zeros((N, 2, 2))
            inf = points_uv.new_full((N,), float("inf"))
            return z, Qz, inf

        if boundary_face_id is not None:
            bfid = torch.as_tensor(boundary_face_id, device=points_uv.device, dtype=torch.long)
            seg_face = bfid.index_select(0, idx_a.to(device=bfid.device))
        else:
            seg_face = torch.zeros((Gs,), device=points_uv.device, dtype=torch.long)

        radius_t = torch.as_tensor(radius, device=points_uv.device, dtype=points_uv.dtype)
        beta_t = torch.as_tensor(beta, device=points_uv.device, dtype=points_uv.dtype).clamp_min(self.eps)

        rho_chunks = []
        Q2_chunks = []
        dist_chunks = []

        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            q = points_uv[start:end]
            xu = Xu[start:end]
            xv = Xv[start:end]
            e1, e2 = self._orthonormal_tangent_basis(
                xu,
                xv,
            )

            E = (xu * xu).sum(dim=-1)
            Fm = (xu * xv).sum(dim=-1)
            Gm = (xv * xv).sum(dim=-1)

            aq = q[:, None, :] - seg_a[None, :, :]
            abu = seg_uv[None, :, 0]
            abv = seg_uv[None, :, 1]
            aqu = aq[..., 0]
            aqv = aq[..., 1]

            denom = (
                E[:, None] * abu * abu
                + 2.0 * Fm[:, None] * abu * abv
                + Gm[:, None] * abv * abv
            ).clamp_min(self.eps)
            numer = (
                E[:, None] * aqu * abu
                + Fm[:, None] * (aqu * abv + aqv * abu)
                + Gm[:, None] * aqv * abv
            )
            tt = (numer / denom).clamp(0.0, 1.0)
            du = aqu - tt * abu
            dv = aqv - tt * abv
            dist2 = (
                E[:, None] * du * du
                + 2.0 * Fm[:, None] * du * dv
                + Gm[:, None] * dv * dv
            ).clamp_min(0.0)
            dist = torch.sqrt(dist2 + self.eps)

            if points_face_id is not None:
                pf = points_face_id[start:end].to(device=points_uv.device, dtype=torch.long)
                same = pf[:, None] == seg_face[None, :]
                dist = torch.where(same, dist, torch.full_like(dist, 1e6))

            # Smooth nearest-segment selection without pair-count bias.
            weights = torch.softmax(-dist / beta_t, dim=1)
            d_soft = (weights * dist).sum(dim=1)
            rho_b = torch.sigmoid((radius_t - d_soft) / beta_t)

            T_boundary = (
                seg_uv[None, :, 0:1] * xu[:, None, :]
                + seg_uv[None, :, 1:2] * xv[:, None, :]
            )

            T_boundary_norm = torch.linalg.vector_norm(
                T_boundary,
                dim=-1,
                keepdim=True,
            )

            T_boundary = torch.where(
                T_boundary_norm > self.eps,
                T_boundary / T_boundary_norm.clamp_min(self.eps),
                e1[:, None, :].expand_as(T_boundary),
            )

            c1 = (
                T_boundary * e1[:, None, :]
            ).sum(
                dim=-1,
            )

            c2 = (
                T_boundary * e2[:, None, :]
            ).sum(
                dim=-1,
            )

            tangent_local_2d = torch.stack(
                [c1, c2],
                dim=-1,
            )

            tangent_local_2d = F.normalize(
                tangent_local_2d,
                dim=-1,
                eps=self.eps,
            )

            Qb2 = (
                weights[..., None, None]
                * tangent_local_2d[..., :, None]
                * tangent_local_2d[..., None, :]
            ).sum(dim=1)

            rho_chunks.append(rho_b)
            Q2_chunks.append(Qb2)
            dist_chunks.append(d_soft)

        return (
            torch.cat(rho_chunks, dim=0),
            torch.cat(Q2_chunks, dim=0),
            torch.cat(dist_chunks, dim=0),
        )

    def _boundary_curve_length_physical(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        boundary_curve_offsets: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_curve_xyz: torch.Tensor | None = None,
        boundary_curve_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_curve_length is not None and boundary_curve_length.numel() > 0:
            return boundary_curve_length.to(
                device=points_uv.device,
                dtype=points_uv.dtype,
            ).reshape(-1).sum().detach()

        if boundary_uv is None or boundary_uv.numel() == 0:
            return points_uv.new_zeros(())

        if boundary_curve_xyz is not None and boundary_curve_xyz.numel() > 0:
            packed = boundary_curve_xyz.to(
                device=points_uv.device,
                dtype=points_uv.dtype,
            ).reshape(-1, 3)
            if packed.shape[0] < 2:
                return points_uv.new_zeros(())
            if boundary_curve_offsets is None or boundary_curve_offsets.numel() < 2:
                return torch.linalg.vector_norm(
                    packed[1:] - packed[:-1],
                    dim=-1,
                ).sum().detach()
            offsets = boundary_curve_offsets.to(
                device=points_uv.device,
                dtype=torch.long,
            ).reshape(-1)
            total = points_uv.new_zeros(())
            for k in range(max(int(offsets.numel()) - 1, 0)):
                a = int(offsets[k].item())
                b = int(offsets[k + 1].item())
                if b - a >= 2:
                    piece = packed[a:b]
                    total = total + torch.linalg.vector_norm(
                        piece[1:] - piece[:-1],
                        dim=-1,
                    ).sum()
            return total.detach()

        boundary_uv = boundary_uv.to(
            device=points_uv.device,
            dtype=points_uv.dtype,
        ).reshape(-1, 2)
        B = int(boundary_uv.shape[0])
        if B < 2:
            return points_uv.new_zeros(())

        if boundary_curve_offsets is not None and boundary_curve_offsets.numel() >= 2:
            offsets = boundary_curve_offsets.to(
                device=points_uv.device,
                dtype=torch.long,
            ).reshape(-1)
            idx_a_parts = []
            idx_b_parts = []
            for k in range(max(int(offsets.numel()) - 1, 0)):
                a = int(offsets[k].item())
                b = int(offsets[k + 1].item())
                loop_count = b - a
                if loop_count >= 3:
                    local_a = torch.arange(a, b, device=points_uv.device)
                    local_b = torch.roll(local_a, shifts=-1)
                    idx_a_parts.append(local_a)
                    idx_b_parts.append(local_b)
                elif loop_count == 2:
                    idx_a_parts.append(torch.tensor([a], device=points_uv.device))
                    idx_b_parts.append(torch.tensor([a + 1], device=points_uv.device))
            if not idx_a_parts:
                return points_uv.new_zeros(())
            idx_a = torch.cat(idx_a_parts)
            idx_b = torch.cat(idx_b_parts)
        else:
            if B >= 3:
                idx_a = torch.arange(0, B, device=points_uv.device)
                idx_b = torch.roll(idx_a, shifts=-1)
            else:
                idx_a = torch.arange(0, B - 1, device=points_uv.device)
                idx_b = idx_a + 1

        seg_a = boundary_uv.index_select(0, idx_a)
        seg_b = boundary_uv.index_select(0, idx_b)
        seg_uv = seg_b - seg_a
        mid_uv = 0.5 * (seg_a + seg_b)

        d_mid = torch.cdist(mid_uv, points_uv)
        if boundary_face_id is not None and points_face_id is not None:
            boundary_face_id = boundary_face_id.to(
                device=points_uv.device,
                dtype=torch.long,
            ).reshape(-1)
            if int(boundary_face_id.numel()) == B:
                seg_face = boundary_face_id.index_select(0, idx_a)
                point_face = points_face_id.to(
                    device=points_uv.device,
                    dtype=torch.long,
                ).reshape(-1)
                d_mid = torch.where(
                    seg_face[:, None] == point_face[None, :],
                    d_mid,
                    torch.full_like(d_mid, 1.0e6),
                )
        nearest = d_mid.argmin(dim=1)
        xu = Xu.index_select(0, nearest)
        xv = Xv.index_select(0, nearest)

        tangent_xyz = (
            seg_uv[:, 0:1] * xu
            + seg_uv[:, 1:2] * xv
        )
        return torch.linalg.vector_norm(
            tangent_xyz,
            dim=-1,
        ).sum().detach()

    # -------------------- boundary band --------------------

    def boundary_attachment_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        alpha_union: float = 8.0,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros(
                points_uv.shape[0],
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        dmat = torch.cdist(points_uv, boundary_uv)
        if boundary_face_id is not None and points_face_id is not None:
            if boundary_face_id.dtype != torch.long:
                boundary_face_id = boundary_face_id.to(torch.long)
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)

            cross_face = points_face_id[:, None] != boundary_face_id[None, :]
            dmat = dmat + cross_face.to(dmat.dtype) * 1e6

        k = min(self.boundary_knn_k, int(dmat.shape[1]))
        d_knn = torch.topk(dmat, k=k, dim=1, largest=False).values

        tau = torch.as_tensor(self.boundary_softmin_tau, device=dmat.device, dtype=dmat.dtype)
        dmin = -tau * torch.logsumexp(-d_knn / (tau + self.eps), dim=1) + tau * math.log(k)

        tb = self.boundary_width(points_uv, boundary_width_raw=boundary_width_raw)
        bb = self.boundary_beta(points_uv, boundary_beta_raw=boundary_beta_raw)
        if k > 1 and self.boundary_spacing_blend > 0.0 and boundary_uv.shape[0] > 1:
            b2b = torch.cdist(boundary_uv, boundary_uv)
            big = torch.eye(boundary_uv.shape[0], device=b2b.device, dtype=b2b.dtype) * 1e6
            b2b = b2b + big
            h_boundary = b2b.min(dim=1).values.median()
            bb = bb + self.boundary_spacing_blend * h_boundary

        rho_b_raw = torch.sigmoid((tb - dmin) / (bb + self.eps))
        norm = torch.sigmoid(tb / (bb + self.eps))
        rho_b_norm = (rho_b_raw / (norm + self.eps)).clamp(0.0, 1.0)

        rho_b = 1.0 - torch.exp(-alpha_union * rho_b_norm)
        return rho_b.clamp(0.0, 1.0)

    def smooth_union(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
        alpha_b: float | torch.Tensor,
    ) -> torch.Tensor:
        alpha_b = torch.as_tensor(alpha_b, device=rho_a.device, dtype=rho_a.dtype)
        rho = 1.0 - (1.0 - rho_a) * (1.0 - alpha_b * rho_b)
        return rho.clamp(0.0, 1.0)

    def soft_project_density(self, rho: torch.Tensor) -> torch.Tensor:
        strength = float(self.density_projection_strength)
        if strength <= 0.0:
            return rho

        threshold = torch.as_tensor(
            self.density_projection_threshold,
            device=rho.device,
            dtype=rho.dtype,
        )
        gamma = torch.as_tensor(
            self.density_projection_gamma,
            device=rho.device,
            dtype=rho.dtype,
        ).clamp_min(self.eps)
        rho_proj = torch.sigmoid((rho - threshold) / gamma)
        rho_blend = (1.0 - strength) * rho + strength * rho_proj
        return rho_blend.clamp(0.0, 1.0)

    # -------------------- higher-order helpers --------------------

    def _triple_junction_score(self, w_soft: torch.Tensor) -> torch.Tensor:
        N, S = w_soft.shape
        if S < 3:
            return torch.zeros(N, device=w_soft.device, dtype=w_soft.dtype)

        # Sum over i<j<k of (w_i w_j w_k)^p without forming an N x S x S x S tensor.
        # Let a_i = w_i^p. Then e3(a) = sum_{i<j<k} a_i a_j a_k
        # and Newton's identity gives:
        # e3 = (p1^3 - 3 p1 p2 + 2 p3) / 6,
        # where p1=sum(a_i), p2=sum(a_i^2), p3=sum(a_i^3).
        power = float(self.junction_triple_power)
        a = w_soft if power == 1.0 else w_soft.pow(power)

        p1 = a.sum(dim=1)
        p2 = (a * a).sum(dim=1)
        p3 = (a * a * a).sum(dim=1)
        e3 = (p1 * p1 * p1 - 3.0 * p1 * p2 + 2.0 * p3) / 6.0
        return e3.clamp_min(0.0)
    
    # -------------------- bisector band density --------------------
    def _bisector_band_density(
        self,
        points,
        seeds,
        d,
        w_soft,
        w_geo,
        beta,
        M,
        Xu,
        Xv,
        seed_active_weights=None,
        seed_duplicate_weights=None,
        seed_domain_weights=None,
        hard_seed_mask=True,
        seed_face_id=None,
        pair_distinctness_override=None,
    ):
        N, S = d.shape

        # ==================================================
        # 1. Structural seed weights
        # ==================================================
        # Math:
        #   w_struct_ni = w_soft_ni * a_i
        #
        # Then normalize:
        #   w_struct_ni =
        #   w_struct_ni / sum_j w_struct_nj
        #
        # Role:
        #   If seeds are inactive, duplicated, or outside domain,
        #   reduce their influence in pair relevance.
        #
        # If no seed_active_weights is given:
        #   w_struct = w_soft

        w_struct = w_soft

        seed_activity = torch.ones(S, device=d.device, dtype=d.dtype)
        duplicate_activity = torch.ones(S, device=d.device, dtype=d.dtype)
        domain_activity = torch.ones(S, device=d.device, dtype=d.dtype)
        active_seed = torch.ones(S, device=d.device, dtype=torch.bool)

        if seed_active_weights is not None:
            if seed_active_weights.ndim != 1 or seed_active_weights.shape[0] != S:
                raise ValueError(
                    f"seed_active_weights must have shape ({S},), "
                    f"got {tuple(seed_active_weights.shape)}"
                )

            seed_activity = seed_active_weights.to(
                device=d.device,
                dtype=d.dtype,
            ).clamp(0.0, 1.0)

            if seed_duplicate_weights is None:
                duplicate_activity = seed_activity
            else:
                if seed_duplicate_weights.ndim != 1 or seed_duplicate_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_duplicate_weights must have shape ({S},), "
                        f"got {tuple(seed_duplicate_weights.shape)}"
                    )
                duplicate_activity = seed_duplicate_weights.to(
                    device=d.device,
                    dtype=d.dtype,
                ).clamp(0.0, 1.0)

            if seed_domain_weights is None:
                domain_activity = seed_activity
            else:
                if seed_domain_weights.ndim != 1 or seed_domain_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_domain_weights must have shape ({S},), "
                        f"got {tuple(seed_domain_weights.shape)}"
                    )
                domain_activity = seed_domain_weights.to(
                    device=d.device,
                    dtype=d.dtype,
                ).clamp(0.0, 1.0)

            if hard_seed_mask:
                active_seed = seed_activity > 0.0

            w_struct = w_soft * seed_activity.unsqueeze(0)
            w_struct_sum = w_struct.sum(dim=1, keepdim=True)

            w_struct = torch.where(
                w_struct_sum > self.eps,
                w_struct / w_struct_sum.clamp_min(self.eps),
                w_soft,
            )

        # ==================================================
        # 2. Pair activity
        # ==================================================
        # Math:
        #   A_ij =
        #   (domain_i domain_j)^p_domain
        #   (duplicate_i duplicate_j)^p_duplicate
        #
        # Role:
        #   Pair (i,j) is active only if both seeds are valid.

        pair_activity = (
            (domain_activity[:, None] * domain_activity[None, :]).pow(
                float(self.domain_pair_power)
            )
            *
            (duplicate_activity[:, None] * duplicate_activity[None, :]).pow(
                float(self.duplicate_pair_power)
            )
        )

        if hard_seed_mask and seed_active_weights is not None:
            active_count = active_seed.to(dtype=d.dtype).sum()

            if bool(active_seed.any()):
                global_activity = seed_activity[active_seed].amax().clamp(0.0, 1.0)
            else:
                global_activity = torch.zeros((), device=d.device, dtype=d.dtype)
        else:
            active_count = torch.as_tensor(float(S), device=d.device, dtype=d.dtype)
            global_activity = seed_activity.amax().clamp(0.0, 1.0)

        global_activity = global_activity.pow(float(self.global_activity_power))

        # ==================================================
        # 3. Pairwise distance difference
        # ==================================================
        # Math:
        #   delta_nij = d_ni - d_nj
        #
        # Voronoi edge condition:
        #   d_ni = d_nj
        #   delta_nij = 0
        #
        # Role:
        #   The zero level set of delta_nij is the bisector
        #   between seed i and seed j.

        d_i = d.unsqueeze(2)  # (N, S, 1)
        d_j = d.unsqueeze(1)  # (N, 1, S)

        delta = d_i - d_j
        abs_delta = torch.sqrt(delta * delta + self.eps)

        # ==================================================
        # 4. Metric gradient and true distance approximation
        # ==================================================
        # Metric distance:
        #
        #   d_i(x) = sqrt((x - s_i)^T M_i (x - s_i))
        #
        # Define:
        #   v_ni = x_n - s_i
        #
        # Gradient:
        #
        #   grad d_i(x_n) = M_i v_ni / d_ni
        #
        # Then:
        #
        #   grad(delta_ij) = grad d_i - grad d_j
        #
        # Approximate geometric distance to bisector:
        #
        #   D_nij =
        #   |d_ni - d_nj| / ||grad d_i - grad d_j||
        #
        # Role:
        #   Converts distance difference into approximate perpendicular
        #   geometric distance to the bisector.
        #
        # In Euclidean case:
        #   M_i = I
        # so:
        #   grad d_i = (x - s_i) / d_i

        if M.ndim != 3 or M.shape != (S, 2, 2):
            raise ValueError(f"M must be (S, 2, 2), got {tuple(M.shape)}")

        M = M.to(device=d.device, dtype=d.dtype)

        x_minus_s = points.unsqueeze(1) - seeds.unsqueeze(0)  # (N, S, 2)

        grad_d = torch.einsum(
            "sij,nsj->nsi",
            M,
            x_minus_s,
        )

        grad_d = grad_d / d.unsqueeze(2).clamp_min(self.eps)



        grad_vec = (
            grad_d.unsqueeze(2)
            - grad_d.unsqueeze(1)
        )  # [N,S,S,2]


        # ============================================================
        # Physical surface metric
        # ============================================================

        E = (Xu * Xu).sum(dim=-1)
        F_metric = (Xu * Xv).sum(dim=-1)
        G = (Xv * Xv).sum(dim=-1)

        det_g = (
            E * G
            - F_metric * F_metric
        ).clamp_min(self.eps)


        g_uu = G / det_g
        g_uv = -F_metric / det_g
        g_vv = E / det_g


        phi_u = grad_vec[..., 0]
        phi_v = grad_vec[..., 1]


        grad_norm_phys_sq = (
            g_uu[:, None, None] * phi_u.pow(2)
            + 2.0
            * g_uv[:, None, None]
            * phi_u
            * phi_v
            + g_vv[:, None, None] * phi_v.pow(2)
        )

        grad_norm_phys = torch.sqrt(
            grad_norm_phys_sq.clamp_min(self.eps)
        )


        # Physical distance from point to the local Voronoi bisector
        true_dist = (
            abs_delta
            / grad_norm_phys.clamp_min(self.eps)
)


        pair_tangent_ij = torch.stack(
            [-grad_vec[..., 1], grad_vec[..., 0]],
            dim=-1,
        )
        pair_tangent_ij = pair_tangent_ij / torch.norm(
            pair_tangent_ij,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        beta_t = torch.as_tensor(beta, device=d.device, dtype=d.dtype)
        beta_eff = beta_t
        w_geo_eff = w_geo

        # ==================================================
        # 5. Pair distinctness
        # ==================================================
        # Math:
        #   D_ij approx 0 when ||s_i - s_j|| is very small
        #   D_ij approx 1 when seeds are well separated
        #
        # Role:
        #   Prevents duplicate or near-duplicate seeds from creating
        #   meaningless bisector bands.

        if pair_distinctness_override is None:
            pair_distinctness = self._pair_distinctness(
                seeds=seeds,
                device=d.device,
                dtype=d.dtype,
                seed_face_id=seed_face_id,
            )
        else:
            pair_distinctness = pair_distinctness_override.to(
                device=d.device,
                dtype=d.dtype,
            )

        if hard_seed_mask and seed_active_weights is not None:
            active_pair = active_seed[:, None] & active_seed[None, :]
            pair_distinctness = pair_distinctness * active_pair.to(dtype=d.dtype)

        # ==================================================
        # 6. Smooth bisector band
        # ==================================================
        # Math:
        #
        #   B_raw_nij =
        #   sigmoid((w_ij - D_nij) / beta)
        #
        # where:
        #   D_nij = true distance to bisector
        #   w_ij = geometric half-width
        #
        # Peak normalization:
        #
        #   B_nij =
        #   sigmoid((w_ij - D_nij) / beta)
        #   /
        #   sigmoid(w_ij / beta)
        #
        # Role:
        #   Makes the center of the band equal to 1:
        #   if D_nij = 0, then B_nij = 1.
        #
        # beta:
        #   controls edge softness.
        #
        # w_geo:
        #   controls strut half-width.

        band_raw = torch.sigmoid(
            (w_geo_eff - true_dist) / (beta_eff + self.eps)
        )

        band_peak = torch.sigmoid(
            w_geo_eff / (beta_eff + self.eps)
        )

        band_ij = (band_raw / (band_peak + self.eps)).clamp(0.0, 1.0)

        # Apply validity:
        #
        #   B_nij <- B_nij * pair_distinctness_ij * pair_activity_ij

        band_ij = band_ij * pair_distinctness
        band_ij = band_ij * pair_activity.unsqueeze(0)

        # ==================================================
        # 7. Pair relevance from soft Voronoi weights
        # ==================================================
        # Math:
        #   P_nij = w_struct_ni * w_struct_nj
        #
        # Role:
        #   If two seeds do not both influence point x_n,
        #   their pair should not strongly contribute.
        #
        # This suppresses non-neighbor seed-pair bands.

        pair_prod = w_struct.unsqueeze(2) * w_struct.unsqueeze(1)

        # ==================================================
        # 8. Effective seed count and junction boost
        # ==================================================
        # Math:
        #   k_eff_n = 1 / sum_i w_ni^2
        #
        # Meaning:
        #   k_eff approx 1 inside one cell
        #   k_eff approx 2 near normal edge
        #   k_eff approx 3 near triple junction
        #
        # Junction boost:
        #
        #   J_n =
        #   1 + lambda * sigmoid((k_eff_n - threshold) / sharpness)
        #
        # Role:
        #   Slightly boosts multi-seed competition regions.

        sum_w2 = w_struct.pow(2).sum(dim=1).clamp_min(self.eps)
        k_eff = 1.0 / sum_w2

        lambda_junc = 0.15
        sharp_junc = 0.25
        junction_threshold = 1.5

        junction_boost = 1.0 + lambda_junc * torch.sigmoid(
            (k_eff - junction_threshold) / sharp_junc
        )

        # ==================================================
        # 9. Pair gate and final pair strength
        # ==================================================
        # Pair gate:
        #
        #   G_nij =
        #   sigmoid((P_nij - t) / s)
        #
        # Role:
        #   Kills very weak non-neighbor pairs.
        #
        # Soft pair power:
        #
        #   P_nij^p
        #
        # Role:
        #   p < 1 makes weak but valid pairs stronger,
        #   improving strut uniformity.
        #
        # Final pair strength:
        #
        #   S_nij =
        #   B_nij * P_nij^p * G_nij * J_n

        pair_power = 0.3
        pair_threshold = 0.03
        pair_softness = 0.01

        pair_gate = torch.sigmoid(
            (pair_prod - pair_threshold) / pair_softness
        )

        # Pair relevance is deliberately kept separate from geometric band
        # occupancy. It is useful for Phase-1 length integration because the
        # junction/global density boosts should not artificially increase the
        # estimated centreline length.
        pair_relevance = (
            pair_prod.clamp_min(self.eps).pow(pair_power)
            * pair_gate
        )

        pair_strength = (
            band_ij
            * pair_relevance
            * junction_boost[:, None, None]
        )

        # ==================================================
        # 10. Optional global pair-count boost
        # ==================================================
        # Math:
        #   pair_boost =
        #   1 + c * sigmoid(
        #       (valid_pair_count - reference_pair_count)
        #       / reference_pair_count
        #   )
        #
        # Role:
        #   Small global boost when there are many valid pairs.

        if self.pair_boost_enabled:
            active_pair_distinctness = pair_distinctness * pair_activity

            valid_pair_count = active_pair_distinctness.sum().clamp_min(1.0)
            reference_pair_count = (active_count - 1.0).clamp_min(1.0)

            pair_boost = 1.0 + self.pair_boost_strength * torch.sigmoid(
                (valid_pair_count - reference_pair_count)
                / (reference_pair_count + self.eps)
            )

            pair_strength = pair_strength * pair_boost

        # ==================================================
        # 11. Physical tube density (Hybrid-compatible concept)
        # ==================================================
        # The implicit pair bisectors play the role of continuous centerlines.
        # Instead of constructing a graph, form a smooth nearest-bisector
        # physical distance and apply the same tube occupancy idea as Phase 2:
        #
        #   rho = sigmoid((radius - d_phys) / tau_rho)
        #
        # Pair relevance localizes the infinite analytical bisectors to the
        # genuine soft Voronoi interfaces. This keeps the entire mapping
        # differentiable with respect to seed motion.

        tri = self._strict_upper_tri_mask(S, d.device, d.dtype).unsqueeze(0)
        pair_presence_weight = (
            pair_relevance
            * pair_distinctness.unsqueeze(0)
            * pair_activity.unsqueeze(0)
            * tri
        ).clamp_min(0.0)

        tau_phys = torch.as_tensor(
            self.physical_tube_beta if self.use_physical_tube_fields else beta,
            device=d.device,
            dtype=d.dtype,
        ).clamp_min(self.eps)

        # Soft nearest-centerline weights. Using normalized exponential weights
        # avoids the number-of-pairs bias of a raw log-sum-exp soft minimum.

        
        # ==========================================================
        # Physical continuous tube field
        # ==========================================================
        #
        # Each analytical bisector defines a physical tube:
        #
        #   rho_ij = sigmoid((r_ij - d_ij_phys) / beta_phys)
        #
        # but it contributes only where seeds i and j genuinely
        # compete in the soft Voronoi partition.
        #
        # This prevents the infinite extension of non-neighbour
        # bisectors from creating material far away from a real edge.

        tube_occupancy_ij = torch.sigmoid(
            (
                w_geo.unsqueeze(0)
                - true_dist
            )
            / tau_phys
        )

        tube_strength_ij = (
            tube_occupancy_ij
            * pair_presence_weight
            * tri
        )

        # Smooth union of all valid physical tubes.
        #
        # 1 - exp(-alpha * sum(...))
        #
        # keeps the field continuous and naturally merges tubes
        # at Voronoi junctions.
        tube_mass = tube_strength_ij.sum(
            dim=(1, 2)
        )

        junction_gate = torch.sigmoid(
            (k_eff - 2.2) / 0.20
        )

        junction_attenuation = (
            1.0 - 0.30 * junction_gate
        )

        tube_mass = (
            tube_mass * junction_attenuation
        )

        rho = (
            1.0
            - torch.exp(
                -self.alpha_union * tube_mass
            )
        )

        rho = (
            rho
            * global_activity
        ).clamp(0.0, 1.0)

        # ==========================================================
        # Normalized physical tube weights for 3D fibre orientation
        # ==========================================================

        tube_weight_sum = tube_strength_ij.sum(
            dim=(1, 2),
            keepdim=True,
        )

        pair_tube_weights = (
            tube_strength_ij
            / tube_weight_sum.clamp_min(self.eps)
        )

        # Diagnostic effective distance only.
        # This is NOT used to generate density anymore.
        tube_distance_phys = (
            pair_tube_weights
            * true_dist
        ).sum(dim=(1, 2))



        # ==================================================
        # 12. Pure geometric edge field
        # ==================================================
        # Math:
        #   edge_field_n =
        #   1 - product_ij (1 - B_nij)
        #
        # Role:
        #   Shows geometric union of all bisector bands before
        #   pair relevance weighting.
        #   Useful for debugging geometry separately from density.

        band_soft = band_ij.clamp(0.0, 1.0)

        eye = torch.eye(S, dtype=torch.bool, device=band_soft.device).unsqueeze(0)

        one_minus = torch.where(
            eye,
            torch.ones_like(band_soft),
            1.0 - band_soft,
        )

        edge_field = 1.0 - one_minus.prod(dim=2).prod(dim=1)
        edge_field = edge_field.clamp(0.0, 1.0)

        return (
            rho,
            pair_strength,
            band_ij,
            pair_relevance,
            edge_field,
            pair_tangent_ij,
            true_dist,
            pair_tube_weights,
            tube_distance_phys,
            k_eff,
        )
    
    
    # ======================================================================
    # Phase-1 continuous activity / integration helpers
    # ======================================================================

    def _soft_domain_activity(
        self,
        seeds: torch.Tensor,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp: float | torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Smooth domain participation plus a hard *diagnostic* validity mask.

        The smooth activity is used by the continuous field. The hard mask is
        used only for reporting/Phase-2 hand-off; it never deletes a seed from
        the Phase-1 computational graph.
        """
        temp_default = max(
            float(self.duplicate_merge_sigma) * float(self.duplicate_effect_temp_ratio),
            self.eps,
        )
        temp = torch.as_tensor(
            temp_default if seed_domain_temp is None else seed_domain_temp,
            device=seeds.device,
            dtype=seeds.dtype,
        ).clamp_min(self.eps)

        u = seeds[:, 0]
        v = seeds[:, 1]

        # Positive or zero while the seed is inside [0,1]^2.
        # Negative once it leaves the normalized square.
        signed_margin = torch.minimum(
            torch.minimum(u, 1.0 - u),
            torch.minimum(v, 1.0 - v),
        )

        # No penalty anywhere inside the valid square.
        # Penalty starts only after crossing the boundary.
        outside_distance = torch.relu(-signed_margin)

        square_weight = 1.0 / (
            1.0
            + (outside_distance / temp).pow(2)
        )

        square_hard = (
            (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & (v <= 1.0)
        )

        uv_weight, uv_hard, sdf_values, mask_values = self._seed_domain_validity_state(
            seeds=seeds,
            temp=temp,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
        )

        domain_activity = (square_weight * uv_weight).clamp(0.0, 1.0)
        domain_hard = square_hard & uv_hard
        return domain_activity, domain_hard, square_weight, sdf_values, mask_values

    def _soft_duplicate_activity(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
        pair_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reversible differentiable suppression of near-duplicate seeds.

        A deterministic lower-index priority breaks the otherwise symmetric
        duplicate state, but there is no hard connected-component selection.
        If two seeds separate later, the suppressed seed smoothly recovers.
        """
        S = int(seeds.shape[0])
        if S <= 1:
            return torch.ones((S,), device=seeds.device, dtype=seeds.dtype)

        if pair_dist is None:
            dist = self._pairwise_seed_dist(seeds, seed_face_id=seed_face_id).to(
                device=seeds.device,
                dtype=seeds.dtype,
            )
        else:
            dist = pair_dist.to(device=seeds.device, dtype=seeds.dtype)
        radius = torch.as_tensor(
            self.duplicate_merge_sigma,
            device=seeds.device,
            dtype=seeds.dtype,
        ).clamp_min(self.eps)
        temp = (radius * float(self.duplicate_effect_temp_ratio)).clamp_min(self.eps)

        # Smooth closeness ~1 for near duplicates, ~0 for separated seeds.
        closeness = torch.sigmoid((radius - dist) / temp)
        closeness = closeness.masked_fill(
            torch.eye(S, dtype=torch.bool, device=seeds.device),
            0.0,
        )

        face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same_face = face_id[:, None] == face_id[None, :]
        closeness = closeness * same_face.to(closeness.dtype)

        # Row i is suppressed only by the strongest earlier duplicate j < i.
        # The asymmetry is intentional and deterministic, but distant seeds do
        # not accumulate into an artificial index-dependent suppression.
        lower_priority = torch.tril(
            torch.ones((S, S), device=seeds.device, dtype=seeds.dtype),
            diagonal=-1,
        )
        earlier_closeness = closeness * lower_priority
        duplicate_strength = earlier_closeness.max(dim=1).values
        if S > 0:
            duplicate_strength = duplicate_strength.clone()
            duplicate_strength[0] = 0.0
        raw = torch.exp(-float(self.duplicate_effect_strength) * duplicate_strength)

        floor = torch.as_tensor(
            self.duplicate_effect_floor,
            device=seeds.device,
            dtype=seeds.dtype,
        )
        return (floor + (1.0 - floor) * raw).clamp(0.0, 1.0)

    def _assignment_from_activity(
        self,
        d: torch.Tensor,
        tau: float,
        seed_activity: torch.Tensor,
    ) -> torch.Tensor:
        """Stable soft Voronoi assignment with a nonzero recovery floor."""
        floor = torch.as_tensor(
            self.phase1_assignment_floor,
            device=d.device,
            dtype=d.dtype,
        )
        a = seed_activity.to(device=d.device, dtype=d.dtype).clamp(0.0, 1.0)
        a_eff = floor + (1.0 - floor) * a

        logits = -d / float(tau)
        logits = logits + torch.log(a_eff.clamp_min(self.eps)).unsqueeze(0)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = logits.clamp(min=-80.0, max=0.0)
        w = torch.softmax(logits, dim=-1)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _surface_area_weights(
        self,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        surface_area_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical per-sample area weights and an explicit/fallback flag.

        Preferred mode: caller supplies lumped/quadrature physical area weights
        (e.g. one-third adjacent triangle area per mesh vertex).

        Fallback mode: assumes points are approximately uniform samples of the
        normalized UV square and uses ||Xu x Xv|| / N. This has physical area
        units when Xu/Xv are physical derivatives, but it is only a quadrature
        approximation and should be replaced by explicit weights in training.
        """
        N = int(Xu.shape[0])
        if surface_area_weights is not None:
            A = torch.as_tensor(
                surface_area_weights,
                device=Xu.device,
                dtype=Xu.dtype,
            ).reshape(-1)
            if A.shape != (N,):
                raise ValueError(
                    f"surface_area_weights must have shape ({N},), got {tuple(A.shape)}"
                )
            if bool((A < 0).any().detach().cpu().item()):
                raise ValueError("surface_area_weights must be non-negative")
            explicit = torch.ones((), device=Xu.device, dtype=Xu.dtype)
            return A, explicit

        jac = torch.linalg.norm(torch.cross(Xu, Xv, dim=1), dim=1)
        A = jac / max(N, 1)
        explicit = torch.zeros((), device=Xu.device, dtype=Xu.dtype)
        return A, explicit

    def _seed_face_counts(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same = face_id[:, None] == face_id[None, :]
        return same.to(seeds.dtype).sum(dim=1).clamp_min(1.0)

    def _territory_statistics(
        self,
        w_soft: torch.Tensor,
        area_weights: torch.Tensor,
        seeds: torch.Tensor,
        points_face_id: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physical territory area and fraction of each seed's own CAD face."""
        A = area_weights.to(device=w_soft.device, dtype=w_soft.dtype).reshape(-1)
        if A.shape[0] != w_soft.shape[0]:
            raise ValueError("area_weights and w_soft must have the same point count")

        territory_area = (A[:, None] * w_soft).sum(dim=0)
        seed_faces = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)

        if points_face_id is None:
            total = A.sum().clamp_min(self.eps)
            face_area_for_seed = torch.full_like(territory_area, total)
        else:
            pfaces = points_face_id.to(device=w_soft.device, dtype=torch.long).reshape(-1)
            if pfaces.shape[0] != w_soft.shape[0]:
                raise ValueError("points_face_id must have one entry per query point")
            face_area_for_seed = torch.empty_like(territory_area)
            for fid in seed_faces.unique().tolist():
                fid_int = int(fid)
                area_f = A[pfaces == fid_int].sum().clamp_min(self.eps)
                face_area_for_seed[seed_faces == fid_int] = area_f

        territory_fraction = territory_area / face_area_for_seed.clamp_min(self.eps)
        return territory_area, territory_fraction, face_area_for_seed

    def _territory_activity(
        self,
        territory_fraction: torch.Tensor,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Smoothly suppress seeds with negligible Voronoi territory."""
        count_on_face = self._seed_face_counts(seeds, seed_face_id=seed_face_id)
        nominal_share = 1.0 / count_on_face
        threshold = float(self.territory_min_ratio) * nominal_share
        transition = (
            float(self.territory_transition_ratio) * nominal_share
        ).clamp_min(self.eps)
        activity = torch.sigmoid((territory_fraction - threshold) / transition)
        return activity.clamp(0.0, 1.0), threshold, transition

    def _phase2_handoff_state(
        self,
        seeds: torch.Tensor,
        seed_activity: torch.Tensor,
        territory_fraction: torch.Tensor,
        domain_hard_mask: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-time hard selection criteria for the graph-based Phase 2."""
        count_on_face = self._seed_face_counts(seeds, seed_face_id=seed_face_id)
        territory_threshold = float(self.phase2_territory_ratio) / count_on_face
        activity_ok = seed_activity >= float(self.phase2_activity_threshold)
        territory_ok = territory_fraction >= territory_threshold
        mask = domain_hard_mask & activity_ok & territory_ok
        ids = torch.nonzero(mask, as_tuple=False).flatten().to(torch.long)
        return mask, ids, territory_threshold


    def _continuous_partition_length(
        self,
        w_soft: torch.Tensor,          # (N, S)
        diff_uv: torch.Tensor,         # (N, S, 2), x - seed, already wrapped
        d_metric: torch.Tensor,        # (N, S), metric distance before face penalty
        M: torch.Tensor,               # (S, 2, 2)
        Xu: torch.Tensor,              # (N, 3)
        Xv: torch.Tensor,              # (N, 3)
        area_weights: torch.Tensor,    # (N,)
        tau: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Graph-free differentiable estimate of total Voronoi-interface length.

        For a soft partition {w_i(x)}, the interface measure is approximated by

            L = 1/2 * integral_S sum_i ||grad_S w_i|| dA.

        Each internal interface is counted twice by the individual partition
        functions, hence the factor 1/2.

        This avoids summing over all O(S^2) seed pairs and is therefore much more
        suitable for Phase 1 with a large initial seed set.
        """

        if w_soft.ndim != 2:
            raise ValueError(
                f"w_soft must have shape (N,S), got {tuple(w_soft.shape)}"
            )

        N, S = w_soft.shape

        tau_t = torch.as_tensor(
            tau,
            device=w_soft.device,
            dtype=w_soft.dtype,
        ).clamp_min(self.eps)

        M_t = M.to(
            device=w_soft.device,
            dtype=w_soft.dtype,
        )

        # ----------------------------------------------------------
        # 1. UV gradient of metric distance d_i(x)
        #
        # d_i = sqrt((x-s_i)^T M_i (x-s_i))
        #
        # grad_uv d_i = M_i (x-s_i) / d_i
        # ----------------------------------------------------------

        grad_d_uv = torch.einsum(
            "sij,nsj->nsi",
            M_t,
            diff_uv,
        )

        grad_d_uv = grad_d_uv / (
            d_metric.unsqueeze(-1).clamp_min(self.eps)
        )

        # ----------------------------------------------------------
        # 2. Gradient of softmax logits
        #
        # z_i = -d_i/tau + log(activity_i)
        #
        # activity does not depend on spatial query x, therefore:
        #
        # grad z_i = -grad d_i / tau
        # ----------------------------------------------------------

        grad_logits_uv = -grad_d_uv / tau_t

        # ----------------------------------------------------------
        # 3. Analytic softmax gradient
        #
        # grad w_i =
        # w_i [grad z_i - sum_j w_j grad z_j]
        # ----------------------------------------------------------

        mean_grad_logits_uv = (
            w_soft.unsqueeze(-1) * grad_logits_uv
        ).sum(
            dim=1,
            keepdim=True,
        )

        grad_w_uv = (
            w_soft.unsqueeze(-1)
            * (
                grad_logits_uv
                - mean_grad_logits_uv
            )
        )

        # ----------------------------------------------------------
        # 4. Convert UV covector gradient to physical surface
        #    gradient using the first fundamental form.
        #
        # G =
        # [ Xu.Xu   Xu.Xv ]
        # [ Xu.Xv   Xv.Xv ]
        # ----------------------------------------------------------

        Xu_metric = Xu * self.uv_scale_u
        Xv_metric = Xv * self.uv_scale_v

        E = (Xu_metric * Xu_metric).sum(dim=-1)
        Fuv = (Xu_metric * Xv_metric).sum(dim=-1)
        G = (Xv_metric * Xv_metric).sum(dim=-1)

        detG = (
            E * G - Fuv * Fuv
        ).clamp_min(self.eps)

        inv00 = G / detG
        inv01 = -Fuv / detG
        inv11 = E / detG

        grad_u = grad_w_uv[..., 0]
        grad_v = grad_w_uv[..., 1]

        # ||grad_S w||^2 = grad_uv^T G^{-1} grad_uv
        grad_norm_sq = (
            inv00[:, None] * grad_u.pow(2)
            + 2.0 * inv01[:, None] * grad_u * grad_v
            + inv11[:, None] * grad_v.pow(2)
        ).clamp_min(0.0)

        # Smooth norm with zero baseline.
        # Subtracting delta avoids accumulating a tiny artificial
        # positive length for every point/seed combination.
        delta = torch.as_tensor(
            1e-12,
            device=w_soft.device,
            dtype=w_soft.dtype,
        )

        grad_norm = (
            torch.sqrt(grad_norm_sq + delta * delta)
            - delta
        ).clamp_min(0.0)

        # ----------------------------------------------------------
        # 5. Partition-interface length
        #
        # Each interface occurs in two partition functions,
        # therefore divide by 2.
        # ----------------------------------------------------------

        local_length_density = (
            0.5 * grad_norm.sum(dim=1)
        )

        A = area_weights.to(
            device=w_soft.device,
            dtype=w_soft.dtype,
        )

        raw_length = (
            A * local_length_density
        ).sum()

        calibrated_length = (
            raw_length
            * float(self.continuous_length_calibration)
        )

        return (
            calibrated_length,
            raw_length,
            local_length_density,
        )
    def _continuous_curve_length(
        self,
        pair_band: torch.Tensor,
        w_soft: torch.Tensor,
        pair_tangent_uv: torch.Tensor,
        w_geo: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        area_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable physical centreline-length estimate.

        Infinite geometric pair bisectors are localized by a smooth pointwise
        co-membership gate. The gate stays close to unit amplitude across a valid
        soft edge but decays away from regions where the two seeds genuinely
        compete, avoiding the large bias caused by integrating full bisectors.

        Density-specific junction/global boosts are intentionally excluded so
        they cannot masquerade as extra fibre length.
        """
        pair_prod = w_soft.unsqueeze(2) * w_soft.unsqueeze(1)

        # Localize the infinite pair bisector to the actual soft Voronoi edge.
        # The gate is close to one where two seeds genuinely compete and decays
        # away from that region. Unlike multiplying by pair_prod**p, this gate
        # preserves nearly unit amplitude across the geometric band and therefore
        # does not artificially shrink its physical width.
# Localize each pair bisector using a zero-baseline relevance gate.
#
# Important:
# pair_prod = 0 must give pair_gate = 0 exactly.
# Otherwise thousands of irrelevant seed pairs accumulate artificial length
# when many seeds are present.

        pair_scale = torch.as_tensor(
            self.continuous_length_pair_threshold,
            device=pair_prod.device,
            dtype=pair_prod.dtype,
        ).clamp_min(self.eps)

        # Zero-baseline saturating relevance gate:
        #
        # pair_prod = 0  -> gate = 0 exactly
        # strong genuine Voronoi competition -> gate -> 1
        #
        # expm1 is used for numerical accuracy near zero.
        pair_gate = -torch.expm1(
            -pair_prod / pair_scale
        )

        pair_gate = pair_gate.clamp(0.0, 1.0)

        tri = self._strict_upper_tri_mask(
            w_soft.shape[1],
            w_soft.device,
            w_soft.dtype,
        )

        pair_gate = pair_gate * tri.unsqueeze(0)

        
        length_pair_field = pair_band * pair_gate

        # Pair-level presence is diagnostic only.
        pair_presence = pair_gate.amax(dim=0)

        beta = torch.as_tensor(self.beta, device=w_geo.device, dtype=w_geo.dtype).clamp_min(self.eps)
        w = w_geo.clamp_min(self.eps)

        # Integral of peak-normalized sigmoid((w-|x|)/beta) over x in R.
        peak = torch.sigmoid(w / beta).clamp_min(self.eps)
        effective_full_width_uv = 2.0 * beta * F.softplus(w / beta) / peak

        # Convert UV-normal width to local physical width using the CAD metric.
        t = pair_tangent_uv
        n_u = -t[..., 1]
        n_v = t[..., 0]
        n_xyz = (
            n_u.unsqueeze(-1) * Xu[:, None, None, :]
            + n_v.unsqueeze(-1) * Xv[:, None, None, :]
        )
        uv_normal_to_xyz_scale = torch.linalg.norm(n_xyz, dim=-1)
        width_phys = (
            effective_full_width_uv.unsqueeze(0) * uv_normal_to_xyz_scale
        ).clamp_min(float(self.continuous_length_min_physical_width))

        A = area_weights.to(device=pair_band.device, dtype=pair_band.dtype).reshape(-1, 1, 1)
        local_length_density = length_pair_field / width_phys
        raw = (A * local_length_density).sum()
        calibrated = raw * float(self.continuous_length_calibration)
        return calibrated, raw, width_phys, pair_presence

    # -------------------- validation --------------------

    def _validate_inputs(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        theta: torch.Tensor | None,
        a_raw: torch.Tensor | None,
    ) -> None:
        if points_uv.ndim != 2 or points_uv.shape[1] != 2:
            raise ValueError(f"points_uv must be (N,2), got {tuple(points_uv.shape)}")
        if Xu.ndim != 2 or Xu.shape[1] != 3:
            raise ValueError(f"Xu must be (N,3), got {tuple(Xu.shape)}")
        if Xv.ndim != 2 or Xv.shape[1] != 3:
            raise ValueError(f"Xv must be (N,3), got {tuple(Xv.shape)}")
        if Xu.shape[0] != points_uv.shape[0] or Xv.shape[0] != points_uv.shape[0]:
            raise ValueError("points_uv, Xu, and Xv must have the same first dimension")
        if seeds_raw.shape != (self.n_seeds, 2):
            raise ValueError(
                f"seeds_raw must be (S,2) with S={self.n_seeds}, got {tuple(seeds_raw.shape)}"
            )
        if w_raw.shape != (self.n_seeds, self.n_seeds):
            raise ValueError(
                f"w_raw must be (S,S) with S={self.n_seeds}, got {tuple(w_raw.shape)}"
            )
        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if self.use_Metric_anisotropy:
            if theta is None or a_raw is None:
                raise ValueError("use_Metric_anisotropy=True requires theta and a_raw.")
            if theta.shape != (self.n_seeds,) or a_raw.shape != (self.n_seeds,):
                raise ValueError(
                    f"theta/a_raw must be (S,) with S={self.n_seeds}, got {theta.shape}, {a_raw.shape}"
                )

    # -------------------- Phase-1 field evaluation --------------------
    def evaluate_at_uv(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        h_raw: torch.Tensor | None,
        theta: torch.Tensor | None = None,
        a_raw: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_uv: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_curve_offsets: torch.Tensor | None = None,
        boundary_curve_xyz: torch.Tensor | None = None,
        boundary_curve_length: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_alpha_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        hard_seed_mask: bool = False,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp: float | torch.Tensor | None = None,
        point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp: float | torch.Tensor | None = None,
        surface_area_weights: torch.Tensor | None = None,
        points_3d: torch.Tensor | None = None,
    ) -> dict[str, Any]:

        """
        Evaluate the fully continuous Phase-1 representation.

        Heavy pairwise fields are evaluated in chunks over surface points
        to avoid materializing full [N, S, S, ...] tensors.

        Hard pruning is still not applied during Phase 1.
        """

        # ============================================================
        # Validation
        # ============================================================

        self._validate_inputs(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            theta=theta,
            a_raw=a_raw,
        )

        seeds = self.seeds_uv(seeds_raw)

        N = int(points_uv.shape[0])
        S = int(seeds.shape[0])

        seed_face_id = self._seed_face_id_for(seeds)
        seed_xyz = self._seed_xyz_from_surface_samples(
            seeds=seeds,
            points_uv=points_uv,
            points_3d=points_3d,
        )

        chunk_size = max(
            1,
            min(
                int(self.point_chunk_size),
                N,
            ),
        )


        # ============================================================
        # Metric
        # ============================================================

        if self.use_Metric_anisotropy:

            M = self.metric_matrices(
                theta,
                a_raw,
            )

        else:

            I = torch.eye(
                2,
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

            M = I.unsqueeze(0).expand(
                S,
                2,
                2,
            )


        # ============================================================
        # Point-to-seed distances
        #
        # [N,S] is acceptable and much smaller than [N,S,S].
        # Keep these globally because assignment/territory/length
        # calculations depend on all points.
        # ============================================================

        diff = (
            points_uv.unsqueeze(1)
            - seeds.unsqueeze(0)
        )

        diff = self._wrap_duv_points_to_seeds(
            diff,
            points_face_id,
        )

        d2 = torch.einsum(
            "nsi,sij,nsj->ns",
            diff,
            M,
            diff,
        )

        d_metric = torch.sqrt(
            d2.clamp_min(self.eps)
        )

        d = d_metric

        if points_face_id is not None:

            pface = points_face_id.to(
                device=points_uv.device,
                dtype=torch.long,
            )

            cross_face = (
                pface[:, None]
                != seed_face_id[None, :]
            )

            d = (
                d
                + cross_face.to(d.dtype) * 1e6
            )


        # ============================================================
        # Point-domain activity
        # ============================================================

        if (
            point_domain_sdf is None
            and self._domain_can_sample_count(
                seed_domain_sdf,
                N,
            )
        ):
            point_domain_sdf = seed_domain_sdf

        if (
            point_domain_mask is None
            and self._domain_can_sample_count(
                seed_domain_mask,
                N,
            )
        ):
            point_domain_mask = seed_domain_mask


        point_temp_source = (
            seed_domain_temp
            if point_domain_temp is None
            else point_domain_temp
        )

        point_temp = torch.as_tensor(
            (
                max(
                    float(self.duplicate_merge_sigma)
                    * float(self.duplicate_effect_temp_ratio),
                    self.eps,
                )
                if point_temp_source is None
                else point_temp_source
            ),
            device=points_uv.device,
            dtype=points_uv.dtype,
        ).clamp_min(self.eps)


        (
            point_domain_weight,
            point_domain_sdf_values,
            point_domain_mask_values,
        ) = self._point_domain_validity_state(
            points_uv=points_uv,
            temp=point_temp,
            point_domain_sdf=point_domain_sdf,
            point_domain_mask=point_domain_mask,
            point_domain_mask_threshold=(
                seed_domain_mask_threshold
                if point_domain_mask_threshold is None
                else point_domain_mask_threshold
            ),
        )


        point_domain_floor = torch.as_tensor(
            self.point_domain_floor,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )


        point_domain_activity = (
            point_domain_floor
            + (
                1.0 - point_domain_floor
            ) * point_domain_weight
        ).clamp(
            0.0,
            1.0,
        )


        # ============================================================
        # Physical integration weights
        # ============================================================

        (
            area_weights_raw,
            area_weights_explicit,
        ) = self._surface_area_weights(
            Xu=Xu,
            Xv=Xv,
            surface_area_weights=surface_area_weights,
        )

        area_weights_valid = (
            area_weights_raw
            * point_domain_activity
        )


        # ============================================================
        # Pass 1:
        # domain + duplicate activity
        # ============================================================

        (
            seed_domain_activity,
            seed_domain_hard_mask,
            seed_square_domain_weight,
            seed_domain_sdf_values,
            seed_domain_mask_values,
        ) = self._soft_domain_activity(
            seeds=seeds,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
        )


        if points_3d is not None and torch.is_tensor(points_3d) and points_3d.numel() > 0:
            seed_duplicate_pair_dist = torch.cdist(seed_xyz, seed_xyz)
            same_seed_face = seed_face_id[:, None] == seed_face_id[None, :]
            seed_duplicate_pair_dist = torch.where(
                same_seed_face,
                seed_duplicate_pair_dist,
                torch.full_like(seed_duplicate_pair_dist, 1.0e6),
            )
        else:
            seed_duplicate_pair_dist = self._pairwise_seed_dist_physical(
                seeds=seeds,
                points_uv=points_uv,
                Xu=Xu,
                Xv=Xv,
                seed_face_id=seed_face_id,
                points_face_id=points_face_id,
            )

        seed_duplicate_activity = (
            self._soft_duplicate_activity(
                seeds=seeds,
                seed_face_id=seed_face_id,
                pair_dist=seed_duplicate_pair_dist,
            )
        )

        pair_distinctness_physical = self._pair_distinctness_from_distance(
            pair_dist=seed_duplicate_pair_dist,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )


        seed_pre_activity = (
            seed_domain_activity
            * seed_duplicate_activity
        ).clamp(
            0.0,
            1.0,
        )


        w_pre = self._assignment_from_activity(
            d=d,
            tau=tau,
            seed_activity=seed_pre_activity,
        )


        (
            pre_territory_area,
            pre_territory_fraction,
            _,
        ) = self._territory_statistics(
            w_soft=w_pre,
            area_weights=area_weights_valid,
            seeds=seeds,
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
        )


        (
            seed_territory_activity,
            territory_activity_threshold,
            territory_activity_transition,
        ) = self._territory_activity(
            territory_fraction=pre_territory_fraction,
            seeds=seeds,
            seed_face_id=seed_face_id,
        )


        # ============================================================
        # Final smooth/reversible activity
        # ============================================================

        seed_active_weights = (
            seed_domain_activity
            * seed_duplicate_activity
            * seed_territory_activity
        ).clamp(
            0.0,
            1.0,
        )

        seed_active_weights = (
            self._sharpen_seed_activity(
                seed_active_weights
            )
        )


        # ============================================================
        # Final soft Voronoi assignment
        # ============================================================

        w_soft = self._assignment_from_activity(
            d=d,
            tau=tau,
            seed_activity=seed_active_weights,
        )


        # ============================================================
        # Territory after final activity
        # ============================================================

        (
            seed_effective_mass,
            seed_territory_fraction,
            seed_face_area,
        ) = self._territory_statistics(
            w_soft=w_soft,
            area_weights=area_weights_valid,
            seeds=seeds,
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
        )


        # ============================================================
        # Phase-2 handoff state
        # ============================================================

        (
            phase2_seed_mask,
            phase2_seed_ids,
            phase2_territory_threshold,
        ) = self._phase2_handoff_state(
            seeds=seeds,
            seed_activity=seed_active_weights,
            territory_fraction=seed_territory_fraction,
            domain_hard_mask=seed_domain_hard_mask,
            seed_face_id=seed_face_id,
        )

        phase2_seeds_uv = (
            seeds[phase2_seed_mask]
        )

        phase2_seeds_xyz = (
            seed_xyz[phase2_seed_mask]
        )


        # ============================================================
        # Pair geometry
        # ============================================================

        w_geo = self.width(
            w_raw,
            seeds=seeds,
            seed_face_id=seed_face_id,
        )


        ones_activity = torch.ones_like(
            seed_active_weights
        )




        # ============================================================
        # CHUNKED IMPLICIT PAIR FIELD
        # ============================================================

        rho_v_chunks = []
        edge_field_chunks = []
        tube_distance_chunks = []
        k_eff_chunks = []

        fiber_Q2_chunks = []


        # pair-level diagnostic only:
        # [S,S], therefore inexpensive.
        continuous_length_pair_presence = torch.zeros(
            S,
            S,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )

        # Do not keep [N,S,S] diagnostics globally.
        continuous_pair_width_phys = None
        continuous_length_pair_presence = torch.empty(
            0,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )

        def _checkpointed_pair_chunk(
            points_c,
            d_c,
            w_soft_c,
            Xu_c,
            Xv_c,
            area_c,
            seeds_c,
            w_geo_c,
            M_c,
            pair_distinctness_c,
        ):
            """
            Heavy differentiable Phase-1 pair computation for one point chunk.

            Used through torch.utils.checkpoint so large pairwise intermediate
            tensors are recomputed during backward instead of being retained for
            every surface chunk.
            """

            ones = torch.ones(
                seeds_c.shape[0],
                device=seeds_c.device,
                dtype=seeds_c.dtype,
            )

            (
                rho_v_c,
                pair_strength_c,
                band_ij_c,
                pair_relevance_c,
                edge_field_c,
                pair_tangent_ij_c,
                pair_distance_phys_c,
                pair_tube_weights_c,
                tube_distance_phys_c,
                k_eff_c,
            ) = self._bisector_band_density(
                points=points_c,
                seeds=seeds_c,
                d=d_c,
                w_soft=w_soft_c,
                w_geo=w_geo_c,
                beta=self.beta,
                M=M_c,
                Xu=Xu_c,
                Xv=Xv_c,

                # Activity is already represented through w_soft.
                seed_active_weights=ones,
                seed_duplicate_weights=ones,
                seed_domain_weights=ones,

                hard_seed_mask=False,
                seed_face_id=seed_face_id,
                pair_distinctness_override=pair_distinctness_c,
            )

            # ------------------------------------------------------------
            # Compact local physical 2D axial orientation tensor
            # ------------------------------------------------------------

            e1_c, e2_c = self._orthonormal_tangent_basis(
                Xu_c,
                Xv_c,
            )

            pair_tangent_local_2d_c = self._local_physical_pair_tangents(
                pair_tangent_uv=pair_tangent_ij_c,
                Xu=Xu_c,
                Xv=Xv_c,
                e1=e1_c,
                e2=e2_c,
            )

            fiber_Q2_c = self._axial_tensor_from_local_pair_tangents(
                pair_tube_weights_c,
                pair_tangent_local_2d_c,
            )

            # ------------------------------------------------------------
            # Pair-band diagnostic
            # ------------------------------------------------------------


            # IMPORTANT:
            # only return compact quantities.
            # Do NOT return pair_tangent_local_2d_c, band_ij_c,
            # pair_tube_weights_c, etc.
            return (
                rho_v_c,
                edge_field_c,
                tube_distance_phys_c,
                k_eff_c,
                fiber_Q2_c,
            )


        for start in range(
            0,
            N,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                N,
            )

            sl = slice(
                start,
                end,
            )

            (
                rho_v_c,
                edge_field_c,
                tube_distance_phys_c,
                k_eff_c,
                fiber_Q2_c,
            ) = checkpoint(
                _checkpointed_pair_chunk,

                points_uv[sl],
                d[sl],
                w_soft[sl],
                Xu[sl],
                Xv[sl],
                area_weights_valid[sl],

                # These carry the dependence on the optimized seeds.
                seeds,
                w_geo,
                M,
                pair_distinctness_physical,

                # Recommended checkpoint implementation.
                use_reentrant=False,
            )

            # ------------------------------------------------------------
            # Compact outputs only
            # ------------------------------------------------------------

            rho_v_chunks.append(
                rho_v_c
            )

            edge_field_chunks.append(
                edge_field_c
            )

            tube_distance_chunks.append(
                tube_distance_phys_c
            )

            k_eff_chunks.append(
                k_eff_c
            )

            fiber_Q2_chunks.append(
                fiber_Q2_c
            )


                


        # ============================================================
        # Concatenate compact outputs
        # ============================================================

        rho_v = torch.cat(
            rho_v_chunks,
            dim=0,
        )

        edge_field = torch.cat(
            edge_field_chunks,
            dim=0,
        )

        tube_distance_phys = torch.cat(
            tube_distance_chunks,
            dim=0,
        )

        k_eff = torch.cat(
            k_eff_chunks,
            dim=0,
        )

        fiber_tensor_Q = torch.cat(
            fiber_Q2_chunks,
            dim=0,
        )


        # ============================================================
        # Local physical tangent basis
        # ============================================================

        e1, e2 = self._orthonormal_tangent_basis(
            Xu,
            Xv,
        )

        fallback_xyz = e1


        # ============================================================
        # Interior fibre from local physical Q2
        # ============================================================

        fiber_tensor_Q_interior = (
            fiber_tensor_Q
        )

        t_local_interior = self._principal_axial_direction(
            fiber_tensor_Q_interior
        )

        fiber3d_interior = (
            self._local_direction_to_xyz(
                t_local_interior,
                e1,
                e2,
            )
        )


        # ============================================================
        # Physical boundary field
        # ============================================================

        physical_radius = (
            points_uv.new_tensor(
                float(self.fixed_strut_radius)
            )
            if self.fixed_strut_radius is not None
            else w_geo.mean()
        )


        physical_beta = (
            points_uv.new_tensor(
                float(self.physical_tube_beta)
            )
            .clamp_min(self.eps)
        )


        if (
            self.use_boundary_attachment
            and boundary_uv is not None
            and boundary_uv.numel() > 0
        ):

            (
                rho_b,
                fiber_tensor_Q_boundary,
                boundary_distance_phys,
            ) = self._physical_boundary_tube_field(
                points_uv=points_uv,
                Xu=Xu,
                Xv=Xv,
                boundary_uv=boundary_uv,
                radius=physical_radius,
                beta=physical_beta,
                boundary_curve_offsets=boundary_curve_offsets,
                points_face_id=points_face_id,
                boundary_face_id=boundary_face_id,
            )


            alpha_b = self.boundary_alpha(
                points_uv,
                boundary_alpha_raw=boundary_alpha_raw,
            )


            rho = self.smooth_union(
                rho_a=rho_v,
                rho_b=rho_b,
                alpha_b=alpha_b,
            )


            boundary_tangent_weight = (
                alpha_b * rho_b
            ).clamp(
                0.0,
                1.0,
            )


            lam_b = (
                boundary_tangent_weight[
                    :,
                    None,
                    None,
                ]
            )


            fiber_tensor_Q_final = (
                (
                    1.0 - lam_b
                )
                * fiber_tensor_Q_interior
                +
                lam_b
                * fiber_tensor_Q_boundary
            )

            t_local_boundary = self._principal_axial_direction(
                fiber_tensor_Q_boundary
            )

            fiber3d_boundary = self._local_direction_to_xyz(
                t_local_boundary,
                e1,
                e2,
            )

            trace_boundary = (
                fiber_tensor_Q_boundary[..., 0, 0]
                + fiber_tensor_Q_boundary[..., 1, 1]
            )

            fiber3d_boundary = torch.where(
                trace_boundary[:, None] > self.eps,
                fiber3d_boundary,
                e1,
            )

            fiber3d_boundary = F.normalize(
                fiber3d_boundary,
                dim=-1,
                eps=self.eps,
            )

        else:

            rho_b = torch.zeros_like(
                rho_v
            )

            rho = rho_v

            alpha_b = torch.zeros(
                (),
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

            boundary_distance_phys = torch.full_like(
                rho_v,
                float("inf"),
            )

            boundary_tangent_weight = torch.zeros_like(
                rho_v
            )

            fiber_tensor_Q_boundary = torch.zeros_like(
                fiber_tensor_Q_interior
            )

            fiber3d_boundary = fallback_xyz

            fiber_tensor_Q_final = (
                fiber_tensor_Q_interior
            )


        # ============================================================
        # Final fibre from blended local physical Q2
        # ============================================================

        t_local = self._principal_axial_direction(
            fiber_tensor_Q_final
        )

        fiber3d = self._local_direction_to_xyz(
            t_local,
            e1,
            e2,
        )

        trace_Q = (
            fiber_tensor_Q_final[..., 0, 0]
            + fiber_tensor_Q_final[..., 1, 1]
        )

        fiber3d = torch.where(
            trace_Q[:, None] > self.eps,
            fiber3d,
            e1,
        )

        fiber3d = F.normalize(
            fiber3d,
            dim=-1,
            eps=self.eps,
        )


        # ============================================================
        # Domain masking + density projection
        # ============================================================

        rho_v = (
            rho_v
            * point_domain_activity
        )

        rho_b = (
            rho_b
            * point_domain_activity
        )

        rho = (
            rho
            * point_domain_activity
        )


        rho = self.soft_project_density(
            rho
        )


        rho = (
            rho
            * point_domain_activity
        ).clamp(
            0.0,
            1.0,
        )


        # ============================================================
        # Solid indicator
        # ============================================================

        eps_rho = 1e-3
        rho0_solid = 0.55
        gamma_solid = 0.02


        rho_s = (
            eps_rho
            + (
                1.0 - eps_rho
            )
            * torch.sigmoid(
                (
                    rho
                    - rho0_solid
                )
                / gamma_solid
            )
        )


        # ============================================================
        # Local physical 2D fibre diagnostics
        # ============================================================

        t_uv_raw = t_local


        fiber_coherence = (
            self._axial_coherence_from_tensor(
                fiber_tensor_Q_final
            )
        )


        rho0 = 0.5
        gamma = 0.05

        fiber_strength = torch.sigmoid(
            (
                rho - rho0
            )
            / gamma
        )


        t_uv = t_uv_raw


        h = self.height(
            h_raw,
            ref_tensor=points_uv,
        )


        # ============================================================
        # PRIMARY Phase-1 continuous length
        #
        # This uses only [N,S] fields and is therefore kept unchanged.
        # ============================================================

        (
            continuous_curve_length,
            continuous_curve_length_raw,
            continuous_length_density,
        ) = self._continuous_partition_length(
            w_soft=w_soft,
            diff_uv=diff,
            d_metric=d_metric,
            M=M,
            Xu=Xu,
            Xv=Xv,
            area_weights=area_weights_valid,
            tau=tau,
        )

        boundary_curve_length_total = self._boundary_curve_length_physical(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            boundary_uv=boundary_uv,
            boundary_curve_offsets=boundary_curve_offsets,
            boundary_face_id=boundary_face_id,
            points_face_id=points_face_id,
            boundary_curve_xyz=boundary_curve_xyz,
            boundary_curve_length=boundary_curve_length,
        )

        continuous_total_curve_length = (
            continuous_curve_length
            + boundary_curve_length_total
        )


        # ============================================================
        # Pair-band diagnostic length
        #
        # Each chunk's raw integral is additive.
        # ============================================================

        pairband_curve_length = torch.zeros(
            (),
            device=points_uv.device,
            dtype=points_uv.dtype,
        )


        pairband_curve_length_raw = torch.zeros_like(
            pairband_curve_length
        )


        # ============================================================
        # Seed diagnostics
        # ============================================================

        seed_visual_outside_domain_mask = (
            ~seed_domain_hard_mask
        )

        seed_visual_participates_mask = (
            phase2_seed_mask
        )

        seed_visual_inactive_mask = (
            ~phase2_seed_mask
        )


        seed_visual_inactive_ids = torch.nonzero(
            seed_visual_inactive_mask,
            as_tuple=False,
        ).flatten().to(
            torch.long
        )


        soft_active_seed_count = (
            seed_active_weights.sum()
        )

        active_seed_count = (
            phase2_seed_mask
            .to(seeds.dtype)
            .sum()
        )

        inactive_seed_count = (
            (~phase2_seed_mask)
            .to(seeds.dtype)
            .sum()
        )


        # ============================================================
        # Return
        # ============================================================

        return {

            # --------------------------------------------------------
            # Stable training / FEM outputs
            # --------------------------------------------------------

            "rho": rho,
            "density": rho,
            "rho_surface": rho,

            "rho_s": rho_s,
            "rho_v": rho_v,
            "rho_b": rho_b,

            "fiber3d": fiber3d,
            "fiber": fiber3d,
            "fiber_direction": fiber3d,


            # --------------------------------------------------------
            # Seeds / assignments
            # --------------------------------------------------------

            "seeds": seeds,
            "seeds_uv": seeds,
            "original_seeds_uv": seeds,
            "seeds_xyz": seed_xyz,

            "w_soft": w_soft,
            "w_soft_pre_territory": w_pre,

            "d": d,
            "M": M,

            "k_eff": k_eff,


            # --------------------------------------------------------
            # Smooth activity
            # --------------------------------------------------------

            "seed_active_weights": seed_active_weights,
            "seed_pre_activity": seed_pre_activity,

            "seed_domain_activity_weights": seed_domain_activity,
            "seed_domain_weight": seed_domain_activity,

            "seed_duplicate_weights": seed_duplicate_activity,
            "seed_territory_weights": seed_territory_activity,

            "seed_effective_mass": seed_effective_mass,
            "seed_territory_fraction": seed_territory_fraction,

            "seed_territory_fraction_pre": pre_territory_fraction,
            "seed_territory_area_pre": pre_territory_area,

            "seed_face_area": seed_face_area,

            "territory_activity_threshold": territory_activity_threshold,
            "territory_activity_transition": territory_activity_transition,

            "seed_square_domain_weight": seed_square_domain_weight,

            "seed_domain_hard_mask": seed_domain_hard_mask,
            "seed_domain_sdf_values": seed_domain_sdf_values,
            "seed_domain_mask_values": seed_domain_mask_values,


            # --------------------------------------------------------
            # Point domain
            # --------------------------------------------------------

            "point_domain_weight": point_domain_weight,
            "point_domain_activity": point_domain_activity,

            "point_domain_sdf_values": point_domain_sdf_values,
            "point_domain_mask_values": point_domain_mask_values,


            # --------------------------------------------------------
            # Phase-2 handoff
            # --------------------------------------------------------

            "phase2_seed_mask": phase2_seed_mask,
            "phase2_seed_ids": phase2_seed_ids,
            "phase2_seeds_uv": phase2_seeds_uv,
            "phase2_seeds_xyz": phase2_seeds_xyz,

            "phase2_territory_threshold": phase2_territory_threshold,

            "topology_seeds_uv": phase2_seeds_uv,

            "seed_active_mask": phase2_seed_mask,

            "active_seed_count": active_seed_count,
            "soft_active_seed_count": soft_active_seed_count,
            "inactive_seed_count": inactive_seed_count,
            "inactive_seed_indices": seed_visual_inactive_ids,


            # --------------------------------------------------------
            # Visualization aliases
            # --------------------------------------------------------

            "seed_visual_outside_domain_mask":
                seed_visual_outside_domain_mask.detach(),

            "seed_visual_participates_in_domain_vd_mask":
                seed_visual_participates_mask.detach(),

            "seed_visual_inactive_mask":
                seed_visual_inactive_mask.detach(),

            "seed_visual_inactive_ids":
                seed_visual_inactive_ids.detach(),

            "seed_visual_inactive_count":
                torch.as_tensor(
                    int(
                        seed_visual_inactive_ids.numel()
                    ),
                    dtype=torch.long,
                    device=seeds.device,
                ),


            # --------------------------------------------------------
            # Compact fibre diagnostics
            # --------------------------------------------------------

            "t_uv_raw": t_uv_raw,
            "t_uv": t_uv,

            "fiber_strength": fiber_strength,
            "fiber_coherence": fiber_coherence,

            "fiber_tensor_Q": fiber_tensor_Q_final,
            "fiber_tensor_Q_interior":
                fiber_tensor_Q_interior,

            "fiber_tensor_Q_boundary":
                fiber_tensor_Q_boundary,

            "fiber_tensor_Q_final":
                fiber_tensor_Q_final,

            "fiber3d_interior":
                fiber3d_interior,

            "fiber3d_boundary":
                fiber3d_boundary,

            "boundary_tangent_weight":
                boundary_tangent_weight,

            "tube_distance_phys":
                tube_distance_phys,

            "edge_field":
                edge_field,

            "h": h,

            "w_geo": w_geo,


            # --------------------------------------------------------
            # IMPORTANT:
            # large pairwise diagnostics intentionally not retained
            # --------------------------------------------------------

            "pair_strength": None,
            "band_ij": None,
            "pair_relevance": None,
            "pair_tangent_ij": None,
            "pair_tangent_xyz": None,
            "pair_distance_phys": None,
            "pair_tube_weights": None,
            "fiber_pair_weights": None,


            # --------------------------------------------------------
            # Length
            # --------------------------------------------------------

            "continuous_voronoi_length":
                continuous_curve_length,

            "continuous_voronoi_length_raw":
                continuous_curve_length_raw,

            "boundary_curve_length":
                boundary_curve_length_total,

            "continuous_total_curve_length":
                continuous_total_curve_length,

            "total_curve_length":
                continuous_total_curve_length,

            "total_voronoi_curve_length":
                continuous_curve_length,

            "continuous_length_density":
                continuous_length_density,

            "continuous_length_calibration":
                torch.as_tensor(
                    self.continuous_length_calibration,
                    device=seeds.device,
                    dtype=seeds.dtype,
                ),

            "pairband_curve_length":
                pairband_curve_length,

            "pairband_curve_length_raw":
                pairband_curve_length_raw,

            "continuous_pair_width_phys":
                continuous_pair_width_phys,

            "continuous_length_pair_presence":
                continuous_length_pair_presence,

            "surface_area_weights":
                area_weights_raw,

            "surface_area_weights_valid":
                area_weights_valid,

            "surface_area_weights_explicit":
                area_weights_explicit,


            # --------------------------------------------------------
            # Boundary
            # --------------------------------------------------------

            "boundary_alpha":
                alpha_b,

            "boundary_width":
                (
                    physical_radius
                    if self.use_boundary_attachment
                    else torch.zeros(
                        (),
                        device=points_uv.device,
                        dtype=points_uv.dtype,
                    )
                ),

            "boundary_beta":
                (
                    physical_beta
                    if self.use_boundary_attachment
                    else torch.zeros(
                        (),
                        device=points_uv.device,
                        dtype=points_uv.dtype,
                    )
                ),
        }


    def forward(
        self,
        points_uv,
        Xu,
        Xv,
        tau,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_curve_offsets=None,
        boundary_curve_xyz=None,
        boundary_curve_length=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        hard_seed_mask=False,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=0.5,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        surface_area_weights=None,
        points_3d=None,
    ):
        return self.evaluate_at_uv(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            points_face_id=points_face_id,
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            boundary_curve_offsets=boundary_curve_offsets,
            boundary_curve_xyz=boundary_curve_xyz,
            boundary_curve_length=boundary_curve_length,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=boundary_alpha_raw,
            boundary_beta_raw=boundary_beta_raw,
            hard_seed_mask=hard_seed_mask,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            point_domain_sdf=point_domain_sdf,
            point_domain_mask=point_domain_mask,
            point_domain_mask_threshold=point_domain_mask_threshold,
            point_domain_temp=point_domain_temp,
            surface_area_weights=surface_area_weights,
            points_3d=points_3d,
        )





@dataclass
class MeshQueryData:
    points_uv: torch.Tensor
    Xu: torch.Tensor
    Xv: torch.Tensor
    points_xyz: torch.Tensor
    faces_ijk: torch.Tensor
    tau: float
    points_face_id: torch.Tensor | None = None
    boundary_uv: torch.Tensor | None = None
    boundary_face_id: torch.Tensor | None = None
    boundary_curve_offsets: torch.Tensor | None = None
    seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    seed_domain_mask_threshold: float = 0.5
    seed_domain_temp: float | torch.Tensor | None = None
    point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    point_domain_mask_threshold: float = 0.5
    point_domain_temp: float | torch.Tensor | None = None
    surface_area_weights: torch.Tensor | None = None

class VoronoiModelVisualizer:
    """
    Helper for evaluating a VoronoiDecoder on a fixed mesh/query set and
    visualizing results in UV and 3D.

    Boundary data can be supplied either:
    - at initialization as defaults
    - or per evaluation call to override defaults
    """

    def __init__(
        self,
        *,
        points_uv,
        Xu,
        Xv,
        points_xyz,
        faces_ijk,
        tau: float,
        n_seeds: int,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp=None,
        surface_area_weights=None,
        eps: float = 1e-8,
        use_metric_anisotropy: bool = False,
        w_min: float = 0.005,
        fixed_height: float | None = None,
        use_boundary_attachment: bool = False,
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        density_projection_strength: float = 0.0,
        density_projection_threshold: float = 0.5,
        density_projection_gamma: float = 0.05,
        seed_activity_sharpness: float = 1.0,
        **decoder_kwargs,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        point_count = int(points_uv.shape[0])

        self.query = MeshQueryData(
            points_uv=self._to_tensor(points_uv, dtype=self.dtype),
            Xu=self._to_tensor(Xu, dtype=self.dtype),
            Xv=self._to_tensor(Xv, dtype=self.dtype),
            points_xyz=self._to_tensor(points_xyz, dtype=self.dtype),
            faces_ijk=self._to_tensor(faces_ijk, dtype=torch.long),
            tau=float(tau),
            points_face_id=self._to_tensor(points_face_id, dtype=torch.long),
            boundary_uv=self._to_tensor(boundary_uv, dtype=self.dtype),
            boundary_face_id=self._to_tensor(boundary_face_id, dtype=torch.long),
            seed_domain_sdf=self._to_domain_input(seed_domain_sdf),
            seed_domain_mask=self._to_domain_input(seed_domain_mask),
            seed_domain_mask_threshold=float(seed_domain_mask_threshold),
            seed_domain_temp=self._to_tensor(seed_domain_temp, dtype=self.dtype),
            point_domain_sdf=self._to_domain_input(
                seed_domain_sdf
                if (
                    point_domain_sdf is None
                    and VoronoiDecoder._domain_can_sample_count(seed_domain_sdf, point_count)
                )
                else point_domain_sdf
            ),
            point_domain_mask=self._to_domain_input(
                seed_domain_mask
                if (
                    point_domain_mask is None
                    and VoronoiDecoder._domain_can_sample_count(seed_domain_mask, point_count)
                )
                else point_domain_mask
            ),
            point_domain_mask_threshold=float(
                seed_domain_mask_threshold
                if point_domain_mask_threshold is None
                else point_domain_mask_threshold
            ),
            point_domain_temp=self._to_tensor(
                seed_domain_temp if point_domain_temp is None else point_domain_temp,
                dtype=self.dtype,
            ),
            surface_area_weights=self._to_tensor(surface_area_weights, dtype=self.dtype),
        )

        self.decoder = VoronoiDecoder(
            n_seeds=self.n_seeds,
            eps=eps,
            use_Metric_anisotropy=use_metric_anisotropy,
            w_min=w_min,
            fixed_height=fixed_height,
            use_boundary_attachment=use_boundary_attachment,
            boundary_solid_idx=boundary_solid_idx,
            face_u_periodic=face_u_periodic,
            face_v_periodic=face_v_periodic,
            seed_face_id=seed_face_id,
            density_projection_strength = density_projection_strength,
            density_projection_threshold = density_projection_threshold,
            density_projection_gamma = density_projection_gamma,
            seed_activity_sharpness = seed_activity_sharpness,
            **decoder_kwargs,
        ).to(device=self.device, dtype=self.dtype)
        self.decoder.eval()

        try:
            pv.set_jupyter_backend("trame")
        except Exception:
            pass

    # ---------------------------
    # tensor helpers
    # ---------------------------

    def _to_tensor(
        self,
        value,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device or self.device, dtype=dtype or value.dtype)
        return torch.as_tensor(
            value,
            device=device or self.device,
            dtype=dtype or self.dtype,
        )

    def _to_domain_input(self, value):
        if value is None or callable(value):
            return value
        return self._to_tensor(value, dtype=self.dtype)

    def make_query_data(
        self,
        *,
        points_uv=None,
        Xu=None,
        Xv=None,
        points_xyz=None,
        faces_ijk=None,
        tau: float | None = None,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold: float | None = None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp=None,
        surface_area_weights=None,
    ) -> MeshQueryData:
        """
        Create a query object, using stored defaults for omitted values.
        """
        return MeshQueryData(
            points_uv=self._to_tensor(
                self.query.points_uv if points_uv is None else points_uv,
                dtype=self.dtype,
            ),
            Xu=self._to_tensor(
                self.query.Xu if Xu is None else Xu,
                dtype=self.dtype,
            ),
            Xv=self._to_tensor(
                self.query.Xv if Xv is None else Xv,
                dtype=self.dtype,
            ),
            points_xyz=self._to_tensor(
                self.query.points_xyz if points_xyz is None else points_xyz,
                dtype=self.dtype,
            ),
            faces_ijk=self._to_tensor(
                self.query.faces_ijk if faces_ijk is None else faces_ijk,
                dtype=torch.long,
            ),
            tau=float(self.query.tau if tau is None else tau),
            points_face_id=self._to_tensor(
                self.query.points_face_id if points_face_id is None else points_face_id,
                dtype=torch.long,
            ),
            boundary_uv=self._to_tensor(
                self.query.boundary_uv if boundary_uv is None else boundary_uv,
                dtype=self.dtype,
            ),
            boundary_face_id=self._to_tensor(
                self.query.boundary_face_id if boundary_face_id is None else boundary_face_id,
                dtype=torch.long,
            ),
            seed_domain_sdf=self._to_domain_input(
                self.query.seed_domain_sdf if seed_domain_sdf is None else seed_domain_sdf
            ),
            seed_domain_mask=self._to_domain_input(
                self.query.seed_domain_mask if seed_domain_mask is None else seed_domain_mask
            ),
            seed_domain_mask_threshold=float(
                self.query.seed_domain_mask_threshold
                if seed_domain_mask_threshold is None
                else seed_domain_mask_threshold
            ),
            seed_domain_temp=self._to_tensor(
                self.query.seed_domain_temp if seed_domain_temp is None else seed_domain_temp,
                dtype=self.dtype,
            ),
            point_domain_sdf=self._to_domain_input(
                self.query.point_domain_sdf if point_domain_sdf is None else point_domain_sdf
            ),
            point_domain_mask=self._to_domain_input(
                self.query.point_domain_mask if point_domain_mask is None else point_domain_mask
            ),
            point_domain_mask_threshold=float(
                self.query.point_domain_mask_threshold
                if point_domain_mask_threshold is None
                else point_domain_mask_threshold
            ),
            point_domain_temp=self._to_tensor(
                self.query.point_domain_temp if point_domain_temp is None else point_domain_temp,
                dtype=self.dtype,
            ),
            surface_area_weights=self._to_tensor(
                self.query.surface_area_weights if surface_area_weights is None else surface_area_weights,
                dtype=self.dtype,
            ),
        )

    @classmethod
    def from_face_tensor(
        cls,
        face_tensor: dict[str, Any],
        *,
        tau: float,
        n_seeds: int,
        use_face_seed_domain_mask: bool = True,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp=None,
        **kwargs,
    ) -> "VoronoiModelVisualizer":
        points_uv = face_tensor["uv"]
        device = points_uv.device if isinstance(points_uv, torch.Tensor) else None
        boundary_uv = kwargs.pop("boundary_uv", None)
        boundary_face_id = kwargs.pop("boundary_face_id", None)
        if boundary_uv is None and face_tensor.get("boundary_idx_ring1", None) is not None:
            bidx = torch.unique(face_tensor["boundary_idx_ring1"].to(dtype=torch.long))
            if bidx.numel() > 0:
                boundary_uv = face_tensor["uv"][bidx]
                boundary_face_id = torch.zeros(bidx.numel(), dtype=torch.long, device=device)

        seed_domain_mask = kwargs.pop("seed_domain_mask", None)
        if seed_domain_mask is None and use_face_seed_domain_mask:
            seed_domain_mask = face_tensor.get("seed_domain_mask_grid", None)
            if seed_domain_mask is None:
                seed_domain_mask = face_tensor.get("seed_domain_mask", None)

        return cls(
            points_uv=face_tensor["uv"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            points_xyz=face_tensor["points_xyz"],
            faces_ijk=face_tensor["faces_ijk"],
            tau=tau,
            n_seeds=n_seeds,
            points_face_id=torch.zeros(
                face_tensor["uv"].shape[0],
                dtype=torch.long,
                device=device,
            ),
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            face_u_periodic=torch.tensor([bool(face_tensor.get("u_periodic", False))], dtype=torch.bool),
            face_v_periodic=torch.tensor([bool(face_tensor.get("v_periodic", False))], dtype=torch.bool),
            seed_face_id=torch.zeros(n_seeds, dtype=torch.long),
            **kwargs,
        )

    # ---------------------------
    # geometry helpers
    # ---------------------------

    @staticmethod
    def faces_ijk_to_pv_faces(faces_ijk: torch.Tensor) -> np.ndarray:
        f = faces_ijk.detach().cpu().numpy().astype(np.int64)
        pv_faces = np.empty((f.shape[0], 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = f
        return pv_faces.reshape(-1)

    @staticmethod
    def seeds_uv_to_xyz_nearest(
        seeds_uv: torch.Tensor,
        uv: torch.Tensor,
        points_xyz: torch.Tensor,
    ) -> torch.Tensor:
        device = uv.device
        seeds_uv = seeds_uv.to(device=device, dtype=uv.dtype)
        points_xyz = points_xyz.to(device=device, dtype=points_xyz.dtype)
        nn = torch.cdist(seeds_uv, uv).argmin(dim=1)
        return points_xyz[nn]

    # ---------------------------
    # evaluation
    # ---------------------------

    def run_case(
        self,
        *,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        query: MeshQueryData | None = None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        hard_seed_mask = False,
    ) -> dict[str, torch.Tensor]:
        q = self.query if query is None else query

        q_boundary_uv = (
            q.boundary_uv
            if boundary_uv is None
            else self._to_tensor(boundary_uv, dtype=self.dtype)
        )
        q_boundary_face_id = (
            q.boundary_face_id
            if boundary_face_id is None
            else self._to_tensor(boundary_face_id, dtype=torch.long)
        )
        q_seed_domain_sdf = (
            q.seed_domain_sdf
            if seed_domain_sdf is None
            else self._to_domain_input(seed_domain_sdf)
        )
        q_seed_domain_mask = (
            q.seed_domain_mask
            if seed_domain_mask is None
            else self._to_domain_input(seed_domain_mask)
        )
        q_seed_domain_temp = (
            q.seed_domain_temp
            if seed_domain_temp is None
            else self._to_tensor(seed_domain_temp, dtype=self.dtype)
        )
        q_point_domain_sdf = (
            q.point_domain_sdf
            if point_domain_sdf is None
            else self._to_domain_input(point_domain_sdf)
        )
        q_point_domain_mask = (
            q.point_domain_mask
            if point_domain_mask is None
            else self._to_domain_input(point_domain_mask)
        )
        q_point_domain_temp = (
            q.point_domain_temp
            if point_domain_temp is None
            else self._to_tensor(point_domain_temp, dtype=self.dtype)
        )

        with torch.no_grad():
            return self.decoder.evaluate_at_uv(
                points_uv=q.points_uv,
                Xu=q.Xu,
                Xv=q.Xv,
                tau=float(q.tau),
                seeds_raw=self._to_tensor(seeds_raw, dtype=self.dtype),
                w_raw=self._to_tensor(w_raw, dtype=self.dtype),
                h_raw=self._to_tensor(h_raw, dtype=self.dtype),
                theta=self._to_tensor(theta, dtype=self.dtype),
                a_raw=self._to_tensor(a_raw, dtype=self.dtype),
                points_face_id=q.points_face_id,
                boundary_uv=q_boundary_uv,
                boundary_face_id=q_boundary_face_id,
                boundary_width_raw=self._to_tensor(boundary_width_raw, dtype=self.dtype),
                boundary_alpha_raw=self._to_tensor(boundary_alpha_raw, dtype=self.dtype),
                boundary_beta_raw=self._to_tensor(boundary_beta_raw, dtype=self.dtype),
                hard_seed_mask=hard_seed_mask,
                seed_domain_sdf=q_seed_domain_sdf,
                seed_domain_mask=q_seed_domain_mask,
                seed_domain_mask_threshold=(
                    q.seed_domain_mask_threshold
                    if seed_domain_mask_threshold is None
                    else seed_domain_mask_threshold
                ),
                seed_domain_temp=q_seed_domain_temp,
                point_domain_sdf=q_point_domain_sdf,
                point_domain_mask=q_point_domain_mask,
                point_domain_mask_threshold=(
                    q.point_domain_mask_threshold
                    if point_domain_mask_threshold is None
                    else point_domain_mask_threshold
                ),
                point_domain_temp=q_point_domain_temp,
                surface_area_weights=q.surface_area_weights,
            )

    def compute_case_volume(
        self,
        case_or_result: dict[str, Any],
        *,
        query: MeshQueryData | None = None,
        use_sharpened: bool = False,
    ) -> dict[str, float]:
        q = self.query if query is None else query
        case = case_or_result["case"] if "case" in case_or_result else case_or_result

        rho_key = "rho_s" if use_sharpened else "rho"
        rho = self._to_tensor(case[rho_key], dtype=self.dtype, device=q.points_uv.device)
        h = self._to_tensor(case["h"], dtype=self.dtype, device=q.points_uv.device)

        area_w = torch.linalg.norm(torch.cross(q.Xu, q.Xv, dim=1), dim=1).clamp_min(self.eps)
        if h.ndim == 0:
            h = h.expand_as(rho)
        elif h.shape != rho.shape:
            h = h.expand_as(rho)

        surface_area = area_w.sum()
        volume = (rho * h * area_w).sum()
        volume_fraction = (rho * area_w).sum() / surface_area.clamp_min(self.eps)

        rho_cont = self._to_tensor(case["rho"], dtype=self.dtype, device=q.points_uv.device)
        rho_sharp = self._to_tensor(case["rho_s"], dtype=self.dtype, device=q.points_uv.device)
        volume_cont = (rho_cont * h * area_w).sum()
        volume_sharp = (rho_sharp * h * area_w).sum()
        volfrac_cont = (rho_cont * area_w).sum() / surface_area.clamp_min(self.eps)
        volfrac_sharp = (rho_sharp * area_w).sum() / surface_area.clamp_min(self.eps)

        return {
            "surface_area": float(surface_area.detach().cpu().item()),
            "mean_height": float(h.mean().detach().cpu().item()),
            "volume": float(volume.detach().cpu().item()),
            "volume_cont": float(volume_cont.detach().cpu().item()),
            "volume_sharp": float(volume_sharp.detach().cpu().item()),
            "volume_fraction": float(volume_fraction.detach().cpu().item()),
            "volfrac_cont": float(volfrac_cont.detach().cpu().item()),
            "volfrac_sharp": float(volfrac_sharp.detach().cpu().item()),
        }

    # ---------------------------
    # plotting
    # ---------------------------

    def plot_uv_fields(
        self,
        *,
        out: dict[str, torch.Tensor],
        seeds_raw,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (12.0, 5.0),
        fiber_stride: int = 20,
        fiber_scale: float = 0.06,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        query: MeshQueryData | None = None,
    ):
        q = self.query if query is None else query
        uv_plot = q.points_uv.detach().cpu()
        seeds_plot = self._to_tensor(seeds_raw, dtype=self.dtype).detach().cpu()

        active_mask_out = out.get("seed_active_mask")
        if active_mask_out is None:
            active_mask = torch.ones(seeds_plot.shape[0], dtype=torch.bool)
        else:
            active_mask = active_mask_out.detach().cpu().bool()
        seed_weight_out = out.get("seed_active_weights")
        if seed_weight_out is None:
            seed_activity = torch.ones(seeds_plot.shape[0], dtype=torch.float32)
        else:
            seed_activity = seed_weight_out.detach().cpu().to(torch.float32).clamp(0.0, 1.0)

        rho_plot = out["rho"].detach().cpu()
        t_uv_plot = out["t_uv_raw"].detach().cpu()
        fiber_strength = out["fiber_strength"].detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
        ax_rho, ax_fiber = axes[0]

        sc = ax_rho.scatter(
            uv_plot[:, 0],
            uv_plot[:, 1],
            c=rho_plot,
            s=10,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )

        if (~active_mask).any():
            ax_rho.scatter(
                seeds_plot[~active_mask, 0],
                seeds_plot[~active_mask, 1],
                s=90,
                c="lightgray",
                edgecolors="black",
                linewidths=1.0,
                label="inactive seed",
            )

        if active_mask.any():
            if color_seeds_by_activation:
                seed_sc = ax_rho.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=95,
                    c=seed_activity[active_mask],
                    cmap=seed_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="white",
                    linewidths=1.0,
                    label="active seed",
                )
                fig.colorbar(seed_sc, ax=ax_rho, fraction=0.046, pad=0.10, label="seed activity")
            else:
                ax_rho.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=90,
                    c="red",
                    edgecolors="white",
                    linewidths=1.0,
                    label="active seed",
                )

        ax_rho.set_title("Density In UV")
        ax_rho.set_aspect("equal")
        ax_rho.set_xlabel("u")
        ax_rho.set_ylabel("v")
        fig.colorbar(sc, ax=ax_rho, fraction=0.046, pad=0.04, label="rho")

        if show_fiber_density_background:
            ax_fiber.scatter(
                uv_plot[:, 0],
                uv_plot[:, 1],
                c=rho_plot,
                s=8,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.35,
            )

        sample_mask = fiber_strength > fiber_min_strength
        if fiber_stride > 1:
            stride_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
            stride_mask[::fiber_stride] = True
            sample_mask = sample_mask & stride_mask

        if bool(sample_mask.any()):
            uv_s = uv_plot[sample_mask]
            t_uv_s = t_uv_plot[sample_mask]
            strength_s = fiber_strength[sample_mask]

            ax_fiber.quiver(
                uv_s[:, 0].numpy(),
                uv_s[:, 1].numpy(),
                t_uv_s[:, 0].numpy(),
                t_uv_s[:, 1].numpy(),
                strength_s.numpy(),
                cmap=cmap,
                angles="xy",
                scale_units="xy",
                scale=max(fiber_scale, 1e-8) ** -1,
                width=0.003,
                pivot="mid",
            )

        if (~active_mask).any():
            ax_fiber.scatter(
                seeds_plot[~active_mask, 0],
                seeds_plot[~active_mask, 1],
                s=90,
                c="lightgray",
                edgecolors="black",
                linewidths=1.0,
            )

        if active_mask.any():
            if color_seeds_by_activation:
                ax_fiber.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=95,
                    c=seed_activity[active_mask],
                    cmap=seed_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="white",
                    linewidths=1.0,
                )
            else:
                ax_fiber.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=90,
                    c="red",
                    edgecolors="white",
                    linewidths=1.0,
                )

        ax_fiber.set_title("Fiber Directions In UV")
        ax_fiber.set_aspect("equal")
        ax_fiber.set_xlabel("u")
        ax_fiber.set_ylabel("v")

        handles, labels = ax_rho.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
            fig.subplots_adjust(top=0.85, wspace=0.25)
        else:
            fig.subplots_adjust(wspace=0.25)

        return fig

    def plot_3d_fields(
        self,
        *,
        out: dict[str, torch.Tensor],
        seeds_raw,
        cmap: str = "viridis",
        window_size: tuple[int, int] = (1500, 700),
        clim: tuple[float, float] = (0.0, 1.0),
        show_edges: bool = False,
        fiber_stride: int = 20,
        fiber_scale: float = 0.08,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        query: MeshQueryData | None = None,
    ):
        q = self.query if query is None else query

        seed_xyz = self.seeds_uv_to_xyz_nearest(
            seeds_uv=self._to_tensor(seeds_raw, dtype=self.dtype),
            uv=q.points_uv,
            points_xyz=q.points_xyz,
        )
        active_mask_out = out.get("seed_active_mask")
        if active_mask_out is None:
            active_mask = torch.ones(seed_xyz.shape[0], dtype=torch.bool)
        else:
            active_mask = active_mask_out.detach().cpu().bool()
        seed_weight_out = out.get("seed_active_weights")
        if seed_weight_out is None:
            seed_activity = torch.ones(seed_xyz.shape[0], dtype=torch.float32)
        else:
            seed_activity = seed_weight_out.detach().cpu().to(torch.float32).clamp(0.0, 1.0)

        pv_faces = self.faces_ijk_to_pv_faces(q.faces_ijk)
        mesh = pv.PolyData(
            q.points_xyz.detach().cpu().numpy(),
            pv_faces,
        )
        mesh["rho"] = out["rho"].detach().cpu().numpy().astype(np.float32)

        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)

        plotter.subplot(0, 0)
        plotter.add_text("Density In 3D", font_size=10)
        plotter.add_mesh(
            mesh.copy(),
            scalars="rho",
            cmap=cmap,
            clim=list(clim),
            show_edges=show_edges,
        )

        if active_mask.any():
            active_cloud = pv.PolyData(seed_xyz[active_mask].detach().cpu().numpy())
            if color_seeds_by_activation:
                active_cloud["seed_activity"] = seed_activity[active_mask].numpy().astype(np.float32)
                plotter.add_mesh(
                    active_cloud,
                    scalars="seed_activity",
                    cmap=seed_cmap,
                    clim=[0.0, 1.0],
                    render_points_as_spheres=True,
                    point_size=14,
                    scalar_bar_args={"title": "seed activity"},
                )
            else:
                plotter.add_mesh(
                    active_cloud,
                    color="red",
                    render_points_as_spheres=True,
                    point_size=14,
                )

        if (~active_mask).any():
            inactive_cloud = pv.PolyData(seed_xyz[~active_mask].detach().cpu().numpy())
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                opacity=0.45,
                render_points_as_spheres=True,
                point_size=12,
            )

        plotter.show_axes()

        plotter.subplot(0, 1)
        plotter.add_text("Fiber Directions In 3D", font_size=10)
        if show_fiber_density_background:
            plotter.add_mesh(
                mesh.copy(),
                scalars="rho",
                cmap=cmap,
                clim=list(clim),
                show_edges=show_edges,
                opacity=0.30,
            )

        fiber_xyz = out["fiber3d"].detach().cpu()
        fiber_strength = out["fiber_strength"].detach().cpu()
        sample_mask = fiber_strength > fiber_min_strength
        if fiber_stride > 1:
            stride_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
            stride_mask[::fiber_stride] = True
            sample_mask = sample_mask & stride_mask

        if bool(sample_mask.any()):
            pts = q.points_xyz.detach().cpu()[sample_mask].numpy()
            vecs = fiber_xyz[sample_mask].numpy()
            mags = fiber_strength[sample_mask].numpy().astype(np.float32)

            fiber_cloud = pv.PolyData(pts)
            fiber_cloud["vectors"] = vecs
            fiber_cloud["strength"] = mags

            glyphs = fiber_cloud.glyph(
                orient="vectors",
                scale="strength",
                factor=fiber_scale,
            )
            plotter.add_mesh(glyphs, scalars="strength", cmap=cmap, clim=list(clim))

        if active_mask.any():
            active_cloud = pv.PolyData(seed_xyz[active_mask].detach().cpu().numpy())
            if color_seeds_by_activation:
                active_cloud["seed_activity"] = seed_activity[active_mask].numpy().astype(np.float32)
                plotter.add_mesh(
                    active_cloud,
                    scalars="seed_activity",
                    cmap=seed_cmap,
                    clim=[0.0, 1.0],
                    render_points_as_spheres=True,
                    point_size=14,
                    show_scalar_bar=False,
                )
            else:
                plotter.add_mesh(
                    active_cloud,
                    color="red",
                    render_points_as_spheres=True,
                    point_size=14,
                )

        if (~active_mask).any():
            inactive_cloud = pv.PolyData(seed_xyz[~active_mask].detach().cpu().numpy())
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                opacity=0.45,
                render_points_as_spheres=True,
                point_size=12,
            )

        plotter.show_axes()
        plotter.link_views()
        return plotter

    def visualize_fields(
        self,
        *,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        query: MeshQueryData | None = None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        show_uv: bool = True,
        show_3d: bool = True,
        cmap: str = "viridis",
        fiber_stride: int = 20,
        fiber_scale_uv: float = 0.06,
        fiber_scale_3d: float = 0.08,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        hard_seed_mask = False,
    ) -> dict[str, Any]:
        out = self.run_case(
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            query=query,
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=boundary_alpha_raw,
            boundary_beta_raw=boundary_beta_raw,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            point_domain_sdf=point_domain_sdf,
            point_domain_mask=point_domain_mask,
            point_domain_mask_threshold=point_domain_mask_threshold,
            point_domain_temp=point_domain_temp,
            hard_seed_mask= hard_seed_mask
        )

        result: dict[str, Any] = {"case": out}

        if show_uv:
            result["uv_fig"] = self.plot_uv_fields(
                out=out,
                seeds_raw=seeds_raw,
                cmap=cmap,
                fiber_stride=fiber_stride,
                fiber_scale=fiber_scale_uv,
                fiber_min_strength=fiber_min_strength,
                show_fiber_density_background=show_fiber_density_background,
                color_seeds_by_activation=color_seeds_by_activation,
                seed_cmap=seed_cmap,
                query=query,
            )

        if show_3d:
            result["plotter"] = self.plot_3d_fields(
                out=out,
                seeds_raw=seeds_raw,
                cmap=cmap,
                fiber_stride=fiber_stride,
                fiber_scale=fiber_scale_3d,
                fiber_min_strength=fiber_min_strength,
                show_fiber_density_background=show_fiber_density_background,
                color_seeds_by_activation=color_seeds_by_activation,
                seed_cmap=seed_cmap,
                query=query,
            )

        return result


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


class ExactUVVoronoi:
    def __init__(self, seeds_all, active_mask=None):
        self.seeds_all = seeds_all
        self.active_mask = active_mask

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def get_seeds(self):
        seeds = self.seeds_all

        if self.active_mask is not None:
            mask = self.active_mask
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().bool()
            seeds = seeds[mask]

        seeds_np = self._to_numpy(seeds).reshape(-1, 2)
        return seeds_np

    def show(self, figsize=(7, 7), title="Exact Voronoi Skeleton in UV"):
        seeds_np = self.get_seeds()

        print("seeds used:", seeds_np.shape[0])

        if self.active_mask is not None:
            mask = self.active_mask
            inactive = (~mask).sum().item() if torch.is_tensor(mask) else np.sum(~mask)
            print("inactive seeds:", inactive)

        if seeds_np.shape[0] < 4:
            raise ValueError("Voronoi needs at least 4 points in 2D.")

        vor = Voronoi(seeds_np, qhull_options="QJ")

        fig, ax = plt.subplots(figsize=figsize)

        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            show_points=True,
            line_width=1.5,
            line_alpha=0.8,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        ax.grid(False)

        return fig, ax


# Explicit semantic alias for new training code.
Phase1ContinuousVoronoiDecoder = VoronoiDecoder
