from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError, cKDTree


class NearUniformHoneycombBaseline:
    """Deterministic non-trainable near-uniform honeycomb seed baseline.

    The baseline generates a triangular lattice in normalized UV space, then
    delegates topology, trimming, CAD lifting, density, and fibre fields to the
    existing ContinuousVoronoiDecoder.
    """

    valid_modes = {"uv_hex", "physical_relax"}

    def __init__(
        self,
        decoder: Any,
        cad_domain: Any,
        face_mesh: dict[str, Any],
        target_seed_count: int | None = 50,
        spacing_uv: float | None = None,
        mode: str = "uv_hex",
        rotation_degrees: float = 0.0,
        phase_uv: tuple[float, float] = (0.0, 0.0),
        relaxation_steps: int = 0,
        relaxation_factor: float = 0.5,
        include_exterior_support_sites: bool = True,
        support_band_cells: float = 2.0,
        relax_support_sites: bool = True,
        support_relaxation_neighbors: int = 6,
        max_relaxation_displacement_cells: float = 0.75,
        preserve_lattice_topology: bool = True,
    ) -> None:
        if decoder is None:
            raise ValueError("decoder must be provided.")
        if face_mesh is None:
            raise ValueError("face_mesh must be provided.")
        if mode not in self.valid_modes:
            raise ValueError(
                f"mode must be one of {sorted(self.valid_modes)}, got {mode!r}."
            )
        target_active = target_seed_count is not None
        spacing_active = spacing_uv is not None
        if target_active == spacing_active:
            raise ValueError(
                "Exactly one of target_seed_count or spacing_uv must be supplied."
            )
        if target_active and int(target_seed_count) <= 0:
            raise ValueError("target_seed_count must be a positive integer.")
        if spacing_active and float(spacing_uv) <= 0.0:
            raise ValueError("spacing_uv must be positive.")
        if int(relaxation_steps) < 0:
            raise ValueError("relaxation_steps must be >= 0.")
        if mode == "physical_relax" and int(relaxation_steps) < 1:
            raise ValueError("relaxation_steps must be >= 1 when mode='physical_relax'.")
        if not 0.0 <= float(relaxation_factor) <= 1.0:
            raise ValueError("relaxation_factor must be in [0, 1].")
        if float(support_band_cells) < 0.0:
            raise ValueError("support_band_cells must be >= 0.")
        if int(support_relaxation_neighbors) < 1:
            raise ValueError("support_relaxation_neighbors must be >= 1.")
        if float(max_relaxation_displacement_cells) < 0.0:
            raise ValueError("max_relaxation_displacement_cells must be >= 0.")
        if len(tuple(phase_uv)) != 2:
            raise ValueError("phase_uv must contain exactly two values.")

        self.decoder = decoder
        self.cad_domain = cad_domain
        self.face_mesh = face_mesh
        self.target_seed_count = None if target_seed_count is None else int(target_seed_count)
        self.spacing_uv = None if spacing_uv is None else float(spacing_uv)
        self.mode = str(mode)
        self.rotation_degrees = float(rotation_degrees)
        self.phase_uv = (float(phase_uv[0]), float(phase_uv[1]))
        self.relaxation_steps = int(relaxation_steps)
        self.relaxation_factor = float(relaxation_factor)
        self.include_exterior_support_sites = bool(include_exterior_support_sites)
        self.support_band_cells = float(support_band_cells)
        self.relax_support_sites = bool(relax_support_sites)
        self.support_relaxation_neighbors = int(support_relaxation_neighbors)
        self.max_relaxation_displacement_cells = float(max_relaxation_displacement_cells)
        self.preserve_lattice_topology = bool(preserve_lattice_topology)
        self.seed_domain_margin = float(getattr(decoder, "seed_domain_margin", 0.0))

        if self._decoder_periodic():
            raise NotImplementedError(
                "NearUniformHoneycombBaseline follows the decoder policy and "
                "does not support periodic surfaces."
            )

    @staticmethod
    def _bool_value(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.detach().reshape(-1)[0].item()) if value.numel() else False
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else False
        return bool(value)

    def _decoder_periodic(self) -> bool:
        return self._bool_value(getattr(self.decoder, "face_u_periodic", False)) or self._bool_value(
            getattr(self.decoder, "face_v_periodic", False)
        )

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _support_samples_uv_np(self) -> np.ndarray:
        uv = self.face_mesh.get("uv", None)
        if uv is None:
            return np.empty((0, 2), dtype=np.float64)
        uv_np = self._as_numpy(uv).astype(np.float64, copy=False).reshape(-1, 2)
        valid = np.isfinite(uv_np).all(axis=1)
        return uv_np[valid]

    def _sample_trim_sdf_np(self, uv_np: np.ndarray) -> np.ndarray | None:
        uv_np = np.asarray(uv_np, dtype=np.float64).reshape(-1, 2)
        if uv_np.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        if self.cad_domain is not None and hasattr(self.cad_domain, "sample_trim_sdf"):
            ref = self.face_mesh.get("uv", None)
            dtype = ref.dtype if isinstance(ref, torch.Tensor) and ref.is_floating_point() else torch.float64
            device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")
            uv_t = torch.as_tensor(uv_np, dtype=dtype, device=device)
            sdf = self.cad_domain.sample_trim_sdf(uv_t)
            return self._as_numpy(sdf).astype(np.float64, copy=False).reshape(-1)
        grid = None
        if isinstance(self.cad_domain, dict):
            grid = self.cad_domain.get("seed_domain_sdf_grid")
        if grid is None:
            grid = self.face_mesh.get("seed_domain_sdf_grid", None)
        if grid is None:
            return None
        grid_np = self._as_numpy(grid).astype(np.float64, copy=False)
        if grid_np.ndim != 2:
            return None
        height, width = grid_np.shape
        u = np.clip(uv_np[:, 0], 0.0, 1.0) * float(width - 1)
        v = np.clip(uv_np[:, 1], 0.0, 1.0) * float(height - 1)
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        sx = u - x0
        sy = v - y0
        v00 = grid_np[y0, x0]
        v10 = grid_np[y0, x1]
        v01 = grid_np[y1, x0]
        v11 = grid_np[y1, x1]
        return (1.0 - sx) * (1.0 - sy) * v00 + sx * (1.0 - sy) * v10 + (1.0 - sx) * sy * v01 + sx * sy * v11

    def _inside_domain_np(self, uv_np: np.ndarray) -> np.ndarray:
        uv_np = np.asarray(uv_np, dtype=np.float64).reshape(-1, 2)
        inside_box = (
            (uv_np[:, 0] >= -1e-12)
            & (uv_np[:, 0] <= 1.0 + 1e-12)
            & (uv_np[:, 1] >= -1e-12)
            & (uv_np[:, 1] <= 1.0 + 1e-12)
        )
        sdf = self._sample_trim_sdf_np(uv_np)
        if sdf is not None and sdf.shape[0] == uv_np.shape[0]:
            return inside_box & (sdf >= -max(float(getattr(self.decoder, "clip_tol", 1e-10)), 1e-10))
        if self.cad_domain is not None and hasattr(self.cad_domain, "smooth_inside_activity"):
            ref = self.face_mesh.get("uv", None)
            dtype = ref.dtype if isinstance(ref, torch.Tensor) and ref.is_floating_point() else torch.float64
            device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")
            uv_t = torch.as_tensor(uv_np, dtype=dtype, device=device)
            activity = self.cad_domain.smooth_inside_activity(
                uv_t, tau=float(getattr(self.decoder, "tau_trim", 0.01))
            )
            act_np = self._as_numpy(activity).reshape(-1)
            threshold = float(getattr(self.decoder, "seed_domain_mask_threshold", 0.5))
            return inside_box & (act_np >= threshold)
        return inside_box

    def _distance_to_valid_domain_np(self, uv_np: np.ndarray) -> np.ndarray:
        uv_np = np.asarray(uv_np, dtype=np.float64).reshape(-1, 2)
        distances = np.zeros((uv_np.shape[0],), dtype=np.float64)

        sdf = self._sample_trim_sdf_np(uv_np)
        if sdf is not None and sdf.shape[0] == uv_np.shape[0]:
            inside_box = (
                (uv_np[:, 0] >= -1e-12)
                & (uv_np[:, 0] <= 1.0 + 1e-12)
                & (uv_np[:, 1] >= -1e-12)
                & (uv_np[:, 1] <= 1.0 + 1e-12)
            )
            distances[inside_box] = np.maximum(-sdf[inside_box], 0.0)
            outside_box = ~inside_box
            if np.any(outside_box):
                samples = self._support_samples_uv_np()
                if samples.shape[0] > 0:
                    valid_samples = samples[self._inside_domain_np(samples)]
                    if valid_samples.shape[0] > 0:
                        distances[outside_box] = cKDTree(valid_samples).query(uv_np[outside_box], k=1)[0]
                    else:
                        distances[outside_box] = np.linalg.norm(
                            uv_np[outside_box] - np.clip(uv_np[outside_box], 0.0, 1.0),
                            axis=1,
                        )
                else:
                    distances[outside_box] = np.linalg.norm(
                        uv_np[outside_box] - np.clip(uv_np[outside_box], 0.0, 1.0),
                        axis=1,
                    )
            return distances

        valid_query = self._inside_domain_np(uv_np)
        invalid_query = ~valid_query
        if not np.any(invalid_query):
            return distances

        samples = self._support_samples_uv_np()
        valid_samples = samples[self._inside_domain_np(samples)] if samples.shape[0] > 0 else samples
        if valid_samples.shape[0] == 0:
            raise ValueError(
                "Cannot compute distance to trimmed valid domain without a trim SDF: "
                "at least one invalid query point was supplied, but no valid auxiliary "
                "mesh UV samples are available."
            )
        distances[invalid_query] = cKDTree(valid_samples).query(uv_np[invalid_query], k=1)[0]
        return distances

    def _triangular_lattice_np(self, spacing: float) -> np.ndarray:
        h = float(spacing)
        if h <= 0.0:
            raise ValueError("spacing must be positive.")
        angle = math.radians(self.rotation_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        inv_rotation = rotation.T

        center = np.asarray([0.5, 0.5], dtype=np.float64)
        support_pad = self.support_band_cells * h if self.include_exterior_support_sites else 0.0
        pad = max(2.5 * h + support_pad, self.seed_domain_margin + support_pad + h)
        box = np.asarray(
            [
                [-pad, -pad],
                [1.0 + pad, -pad],
                [1.0 + pad, 1.0 + pad],
                [-pad, 1.0 + pad],
            ],
            dtype=np.float64,
        )
        local = (box - center) @ inv_rotation.T + center
        min_xy = local.min(axis=0) - h
        max_xy = local.max(axis=0) + h
        row_step = math.sqrt(3.0) * h / 2.0
        i_min = math.floor(min_xy[0] / h) - 1
        i_max = math.ceil(max_xy[0] / h) + 1
        j_min = math.floor(min_xy[1] / row_step) - 1
        j_max = math.ceil(max_xy[1] / row_step) + 1

        points = []
        for j in range(j_min, j_max + 1):
            y = j * row_step
            offset = 0.5 * h if (j % 2) else 0.0
            for i in range(i_min, i_max + 1):
                points.append((i * h + offset + self.phase_uv[0], y + self.phase_uv[1]))
        lattice = np.asarray(points, dtype=np.float64)
        lattice = (lattice - center) @ rotation.T + center
        return lattice
    def _interior_seed_count_np(self, spacing: float) -> int:
        points = self._triangular_lattice_np(spacing)
        interior = self._inside_domain_np(points)
        return int(interior.sum())
    def _retained_lattice_np(
        self,
        spacing: float,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Generate one continuous triangular lattice across and beyond the shell.
        points = self._triangular_lattice_np(spacing)

        interior = self._inside_domain_np(points)

        if not self.include_exterior_support_sites:
            raise ValueError(
                "A clipped periodic honeycomb requires exterior support sites."
            )

        # Two complete exterior rows are sufficient to preserve boundary cells.
        support_band = max(
            float(self.support_band_cells),
            2.0,
        ) * float(spacing)

        if self.seed_domain_margin + 1e-12 < support_band:
            raise ValueError(
                "decoder.seed_domain_margin is too small for a clipped "
                f"honeycomb. Required >= {support_band:.6g}, "
                f"received {self.seed_domain_margin:.6g}."
            )

        # Keep every lattice point within the decoder search margin.
        in_search = (
            (points[:, 0] >= -self.seed_domain_margin - 1e-12)
            & (points[:, 0] <= 1.0 + self.seed_domain_margin + 1e-12)
            & (points[:, 1] >= -self.seed_domain_margin - 1e-12)
            & (points[:, 1] <= 1.0 + self.seed_domain_margin + 1e-12)
        )

        distance_to_domain = self._distance_to_valid_domain_np(points)

        support = (
            (~interior)
            & in_search
            & (distance_to_domain <= support_band + 1e-12)
        )

        keep = interior | support
        retained = points[keep]
        interior_retained = interior[keep]

        order = np.lexsort((retained[:, 0], retained[:, 1]))

        return retained[order], interior_retained[order]

    def _choose_spacing(self) -> float:
        if self.spacing_uv is not None:
            return float(self.spacing_uv)
        assert self.target_seed_count is not None
        target = int(self.target_seed_count)
        area_guess = max(float(target), 1.0) * math.sqrt(3.0) / 2.0
        nominal = math.sqrt(1.0 / area_guess)
        low = max(nominal * 0.25, 1e-4)
        high = min(max(nominal * 4.0, low * 2.0), 2.0)

        best_spacing = nominal
        best_score = (float("inf"), float("inf"))
        for _ in range(36):
            spacing = 0.5 * (low + high)
            count = self._interior_seed_count_np(spacing)
            score = (abs(count - target), abs(spacing - nominal))
            if score < best_score:
                best_score = score
                best_spacing = spacing
            if count > target:
                low = spacing
            elif count < target:
                high = spacing
            else:
                best_spacing = spacing
                break
        for spacing in np.linspace(max(low, 1e-4), high, 25):
            count = self._interior_seed_count_np(float(spacing))
            score = (abs(count - target), abs(float(spacing) - nominal))
            if score < best_score:
                best_score = score
                best_spacing = float(spacing)
        return float(best_spacing)

    def _seed_xyz_np(self, uv_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
        if uv_np.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        uv_t = torch.as_tensor(uv_np, dtype=dtype, device=device)
        with torch.no_grad():
            xyz = self.decoder.seed_xyz_from_uv(self.cad_domain, uv_t)
        return xyz.detach().cpu().numpy().astype(np.float64, copy=False)

    def _vertex_area_weights_np(self) -> np.ndarray:
        uv_np = self._support_samples_uv_np()
        n = int(uv_np.shape[0])
        faces = self.face_mesh.get("faces_ijk", None)
        areas = self.face_mesh.get("face_areas", None)
        if faces is None or areas is None:
            return np.ones((n,), dtype=np.float64) / max(n, 1)
        faces_np = self._as_numpy(faces).astype(np.int64, copy=False).reshape(-1, 3)
        areas_np = self._as_numpy(areas).astype(np.float64, copy=False).reshape(-1)
        if faces_np.shape[0] != areas_np.shape[0] or n == 0:
            return np.ones((n,), dtype=np.float64) / max(n, 1)
        weights = np.zeros((n,), dtype=np.float64)
        valid = (faces_np >= 0).all(axis=1) & (faces_np < n).all(axis=1)
        for tri, area in zip(faces_np[valid], areas_np[valid]):
            weights[tri] += max(float(area), 0.0) / 3.0
        if not np.any(weights > 0.0):
            weights[:] = 1.0
        return weights

    def _valid_sample_data_np(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples_uv_all = self._support_samples_uv_np()
        if samples_uv_all.shape[0] == 0:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
        valid_mask = self._inside_domain_np(samples_uv_all)
        samples_uv = samples_uv_all[valid_mask]
        points_xyz = self.face_mesh.get("points_xyz", None)
        if points_xyz is None:
            raise ValueError("face_mesh['points_xyz'] is required for physical relaxation.")
        points_xyz_np = self._as_numpy(points_xyz).astype(np.float64, copy=False).reshape(-1, 3)
        if points_xyz_np.shape[0] != samples_uv_all.shape[0]:
            raise ValueError("face_mesh['points_xyz'] must have the same length as face_mesh['uv'].")
        samples_xyz = points_xyz_np[valid_mask]
        weights = self._vertex_area_weights_np()[valid_mask]
        return samples_uv, samples_xyz, weights

    def _physical_assignment_np(
        self,
        seeds_np: np.ndarray,
        interior_mask: np.ndarray,
        samples_xyz: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[np.ndarray, np.ndarray]:
        movable_ids = np.flatnonzero(np.asarray(interior_mask, dtype=bool))
        if movable_ids.size == 0:
            raise ValueError("physical_relax requires at least one interior movable seed.")
        # Only valid interior sites are evaluated on the CAD surface.  Exterior
        # support sites may lie outside the normalized surface bounds and are
        # moved later by coherent displacement extrapolation.
        movable_xyz = self._seed_xyz_np(
            seeds_np[movable_ids],
            device=device,
            dtype=dtype,
        )
        distances, nearest_local = cKDTree(movable_xyz).query(samples_xyz, k=1)
        nearest_global = movable_ids[np.asarray(nearest_local, dtype=np.int64)]
        return np.asarray(distances, dtype=np.float64), nearest_global

    def _weighted_physical_energy_np(
        self,
        seeds_np: np.ndarray,
        interior_mask: np.ndarray,
        samples_xyz: np.ndarray,
        weights: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        distances, _ = self._physical_assignment_np(
            seeds_np,
            interior_mask,
            samples_xyz,
            device=device,
            dtype=dtype,
        )
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != distances.shape[0] or float(weights.sum()) <= 0.0:
            return float(np.mean(distances * distances))
        return float(np.sum(weights * distances * distances) / weights.sum())

    def _collision_tolerance(self, spacing: float) -> float:
        return max(
            float(getattr(self.decoder, "eps", 1e-8)) * 16.0,
            min(0.25 * float(spacing), max(float(getattr(self.decoder, "min_tube_spacing", 1e-3)), 1e-6)),
        )

    @staticmethod
    def _minimum_pairwise_distance_np(seeds_np: np.ndarray) -> float:
        seeds_np = np.asarray(seeds_np, dtype=np.float64).reshape(-1, 2)
        if seeds_np.shape[0] <= 1:
            return float("inf")
        delta = seeds_np[:, None, :] - seeds_np[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        distances[np.eye(seeds_np.shape[0], dtype=bool)] = np.inf
        return float(np.min(distances))

    @staticmethod
    def _delaunay_edges_np(seeds_np: np.ndarray) -> frozenset[tuple[int, int]]:
        """Return the undirected Delaunay graph used to guard lattice topology."""
        seeds_np = np.asarray(seeds_np, dtype=np.float64).reshape(-1, 2)
        if seeds_np.shape[0] < 3:
            return frozenset()
        try:
            triangulation = Delaunay(seeds_np)
        except QhullError as error:
            raise ValueError(
                "Cannot construct a Delaunay graph for topology-preserving "
                "physical relaxation."
            ) from error
        edges: set[tuple[int, int]] = set()
        for triangle in np.asarray(triangulation.simplices, dtype=np.int64):
            for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
                a = int(triangle[local_a])
                b = int(triangle[local_b])
                edges.add((a, b) if a < b else (b, a))
        return frozenset(edges)

    def _cap_displacements_np(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
        seed_ids: np.ndarray,
        spacing: float,
    ) -> np.ndarray:
        """Limit cumulative motion so the relaxed layout remains lattice-like."""
        seed_ids = np.asarray(seed_ids, dtype=np.int64).reshape(-1)
        if seed_ids.size == 0:
            return candidate
        cap = self.max_relaxation_displacement_cells * float(spacing)
        if cap <= 0.0:
            candidate[seed_ids] = reference[seed_ids]
            return candidate
        displacement = candidate[seed_ids] - reference[seed_ids]
        norm = np.linalg.norm(displacement, axis=1)
        scale = np.minimum(1.0, cap / np.maximum(norm, 1e-30))
        candidate[seed_ids] = reference[seed_ids] + displacement * scale[:, None]
        return candidate

    def _extended_lattice_candidate_is_valid_np(
        self,
        candidate: np.ndarray,
        interior_mask: np.ndarray,
        spacing: float,
        reference_edges: frozenset[tuple[int, int]],
    ) -> bool:
        """Check margin, role, separation, and optional topology invariants."""
        candidate = np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
        interior_mask = np.asarray(interior_mask, dtype=bool).reshape(-1)
        margin = float(self.seed_domain_margin) + 1e-12
        in_margin = (
            (candidate[:, 0] >= -margin)
            & (candidate[:, 0] <= 1.0 + margin)
            & (candidate[:, 1] >= -margin)
            & (candidate[:, 1] <= 1.0 + margin)
        )
        if not bool(np.all(in_margin)):
            return False
        if not bool(np.all(self._inside_domain_np(candidate[interior_mask]))):
            return False
        support_mask = ~interior_mask
        if np.any(support_mask) and bool(np.any(self._inside_domain_np(candidate[support_mask]))):
            return False
        if self._minimum_pairwise_distance_np(candidate) < self._collision_tolerance(spacing):
            return False
        if self.preserve_lattice_topology:
            try:
                if self._delaunay_edges_np(candidate) != reference_edges:
                    return False
            except ValueError:
                return False
        return True

    def _extrapolate_support_motion_np(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        interior_mask: np.ndarray,
        spacing: float,
        reference_edges: frozenset[tuple[int, int]],
    ) -> tuple[np.ndarray | None, float]:
        """Move ghost/support sites with the interior lattice deformation.

        The physical relaxation is computed only where the trimmed CAD surface
        is valid.  Its cumulative UV displacement is extended to nearby support
        sites by inverse-distance interpolation.  A line search keeps support
        sites outside the trim and preserves the retained lattice graph.
        """
        reference = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
        candidate = np.asarray(candidate, dtype=np.float64).reshape(-1, 2).copy()
        interior_mask = np.asarray(interior_mask, dtype=bool).reshape(-1)
        interior_ids = np.flatnonzero(interior_mask)
        support_ids = np.flatnonzero(~interior_mask)

        if support_ids.size == 0 or not self.relax_support_sites:
            candidate[support_ids] = reference[support_ids]
            valid = self._extended_lattice_candidate_is_valid_np(
                candidate,
                interior_mask,
                spacing,
                reference_edges,
            )
            return (candidate if valid else None), 0.0

        k = min(self.support_relaxation_neighbors, int(interior_ids.size))
        distances, nearest_local = cKDTree(reference[interior_ids]).query(
            reference[support_ids],
            k=k,
        )
        distances = np.asarray(distances, dtype=np.float64)
        nearest_local = np.asarray(nearest_local, dtype=np.int64)
        if k == 1:
            distances = distances[:, None]
            nearest_local = nearest_local[:, None]

        length_eps = max(float(spacing) * 1e-6, 1e-12)
        weights = 1.0 / np.maximum(distances, length_eps) ** 2
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
        interior_displacement = (
            candidate[interior_ids] - reference[interior_ids]
        )
        support_displacement = np.sum(
            weights[..., None] * interior_displacement[nearest_local],
            axis=1,
        )

        cap = self.max_relaxation_displacement_cells * float(spacing)
        if cap <= 0.0:
            support_displacement[:] = 0.0
        else:
            displacement_norm = np.linalg.norm(support_displacement, axis=1)
            displacement_scale = np.minimum(
                1.0,
                cap / np.maximum(displacement_norm, 1e-30),
            )
            support_displacement *= displacement_scale[:, None]

        support_scale = 1.0
        for _ in range(16):
            trial = candidate.copy()
            trial[support_ids] = (
                reference[support_ids]
                + support_scale * support_displacement
            )
            if self._extended_lattice_candidate_is_valid_np(
                trial,
                interior_mask,
                spacing,
                reference_edges,
            ):
                return trial, float(support_scale)
            support_scale *= 0.5

        # Zero support motion is the final safe candidate.  This can still be
        # rejected when the proposed interior motion changes lattice topology.
        trial = candidate.copy()
        trial[support_ids] = reference[support_ids]
        if self._extended_lattice_candidate_is_valid_np(
            trial,
            interior_mask,
            spacing,
            reference_edges,
        ):
            return trial, 0.0
        return None, 0.0

    def _validate_pairwise_separation_np(self, seeds_np: np.ndarray, spacing: float | None = None) -> None:
        spacing_value = float(self.spacing_uv if spacing is None and self.spacing_uv is not None else (spacing or 1.0))
        tol = self._collision_tolerance(spacing_value)
        min_distance = self._minimum_pairwise_distance_np(seeds_np)
        if min_distance < tol:
            raise ValueError(
                "Retained baseline sites contain coincident or excessively close "
                f"pairs: minimum distance {min_distance:.6g} is below collision "
                f"tolerance {tol:.6g}."
            )

    def _resolve_seed_collisions_np(
        self,
        seeds_np: np.ndarray,
        interior_mask: np.ndarray,
        valid_samples_uv: np.ndarray,
        spacing: float,
    ) -> np.ndarray:
        if seeds_np.shape[0] <= 1:
            return seeds_np
        tol = self._collision_tolerance(spacing)
        resolved = seeds_np.copy()
        movable = set(int(i) for i in np.flatnonzero(np.asarray(interior_mask, dtype=bool)))
        if not movable:
            return resolved

        def collides(seed_id: int) -> bool:
            delta = resolved - resolved[seed_id]
            dist = np.linalg.norm(delta, axis=1)
            dist[seed_id] = np.inf
            return bool(np.any(dist < tol))

        for seed_id in sorted(movable):
            if not collides(seed_id):
                continue
            original_position = resolved[seed_id].copy()
            if valid_samples_uv.shape[0] == 0:
                raise ValueError("Unable to resolve close seeds: no valid auxiliary mesh UV samples are available.")
            sample_order = np.lexsort((valid_samples_uv[:, 0], valid_samples_uv[:, 1]))
            best_id = None
            best_score = (np.inf, np.inf, np.inf)
            for sample_id in sample_order:
                candidate = valid_samples_uv[int(sample_id)]
                distances = np.linalg.norm(resolved - candidate, axis=1)
                distances[seed_id] = np.inf
                if float(distances.min()) < tol:
                    continue
                original_distance = float(np.linalg.norm(candidate - original_position))
                score = (original_distance, float(candidate[1]), float(candidate[0]))
                if score < best_score:
                    best_id = int(sample_id)
                    best_score = score
            if best_id is None:
                raise ValueError(
                    "Unable to resolve close seeds: no feasible valid auxiliary "
                    "mesh UV sample is separated from every other retained seed."
                )
            resolved[seed_id] = valid_samples_uv[best_id]
        min_distance = self._minimum_pairwise_distance_np(resolved)
        if min_distance < tol:
            raise ValueError(
                "Unable to resolve close seeds into a separated valid configuration: "
                f"minimum distance {min_distance:.6g} is below collision tolerance {tol:.6g}."
            )
        return resolved

    def _relax_physical_np(
        self,
        seeds_np: np.ndarray,
        interior_mask: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
        spacing: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        steps = self.relaxation_steps if self.mode == "physical_relax" else 0
        if steps <= 0 or seeds_np.shape[0] == 0:
            return seeds_np, {
                "relaxation_accepted_iterations": 0,
                "support_relaxation_enabled": False,
                "minimum_support_motion_scale": 0.0,
            }
        samples_uv, samples_xyz, weights = self._valid_sample_data_np()
        if samples_uv.shape[0] == 0:
            raise ValueError("physical_relax requires finite valid auxiliary mesh samples.")
        movable_ids = np.flatnonzero(np.asarray(interior_mask, dtype=bool))
        if movable_ids.size == 0:
            raise ValueError("physical_relax requires at least one interior movable seed.")

        reference = np.asarray(seeds_np, dtype=np.float64).reshape(-1, 2).copy()
        reference_edges = self._delaunay_edges_np(reference)
        relaxed = self._resolve_seed_collisions_np(reference, interior_mask, samples_uv, spacing)
        accepted_iterations = 0
        accepted_support_scales: list[float] = []
        for _ in range(steps):
            before_energy = self._weighted_physical_energy_np(
                relaxed,
                interior_mask,
                samples_xyz,
                weights,
                device=device,
                dtype=dtype,
            )
            _, nearest_global = self._physical_assignment_np(
                relaxed,
                interior_mask,
                samples_xyz,
                device=device,
                dtype=dtype,
            )
            target_uv_by_seed: dict[int, np.ndarray] = {}
            for seed_id in movable_ids.tolist():
                assigned = nearest_global == int(seed_id)
                if not np.any(assigned):
                    continue
                cluster_weights = weights[assigned]
                weight_sum = float(cluster_weights.sum())
                if weight_sum <= 0.0:
                    continue
                target_xyz = (cluster_weights[:, None] * samples_xyz[assigned]).sum(axis=0) / weight_sum
                assigned_xyz = samples_xyz[assigned]
                assigned_uv = samples_uv[assigned]
                best_local = int(np.argmin(np.linalg.norm(assigned_xyz - target_xyz, axis=1)))
                target_uv_by_seed[int(seed_id)] = assigned_uv[best_local]

            accepted = False
            factor = float(self.relaxation_factor)
            for _attempt in range(12):
                candidate = relaxed.copy()
                # Support sites are derived from the cumulative deformation of
                # the original lattice, rather than being independently relaxed
                # outside the valid CAD trim.
                candidate[~interior_mask] = reference[~interior_mask]
                for seed_id, target_uv in target_uv_by_seed.items():
                    new_uv = (1.0 - factor) * relaxed[seed_id] + factor * target_uv
                    if not bool(self._inside_domain_np(new_uv.reshape(1, 2))[0]):
                        best_sample = int(np.argmin(np.linalg.norm(samples_uv - new_uv, axis=1)))
                        new_uv = samples_uv[best_sample]
                    candidate[seed_id] = new_uv
                candidate = self._cap_displacements_np(
                    candidate,
                    reference,
                    movable_ids,
                    spacing,
                )
                candidate = self._resolve_seed_collisions_np(candidate, interior_mask, samples_uv, spacing)
                candidate, support_scale = self._extrapolate_support_motion_np(
                    reference,
                    candidate,
                    interior_mask,
                    spacing,
                    reference_edges,
                )
                if candidate is None:
                    factor *= 0.5
                    continue
                after_energy = self._weighted_physical_energy_np(
                    candidate,
                    interior_mask,
                    samples_xyz,
                    weights,
                    device=device,
                    dtype=dtype,
                )
                if after_energy <= before_energy + max(1e-14, 1e-10 * abs(before_energy)):
                    relaxed = candidate
                    accepted = True
                    accepted_iterations += 1
                    accepted_support_scales.append(float(support_scale))
                    break
                factor *= 0.5
            if not accepted:
                break

        support_ids = np.flatnonzero(~np.asarray(interior_mask, dtype=bool))
        support_displacement = (
            relaxed[support_ids] - reference[support_ids]
            if support_ids.size > 0
            else np.empty((0, 2), dtype=np.float64)
        )
        support_displacement_norm = (
            np.linalg.norm(support_displacement, axis=1)
            if support_displacement.shape[0] > 0
            else np.empty((0,), dtype=np.float64)
        )
        relaxation_diagnostics: dict[str, Any] = {
            "relaxation_accepted_iterations": int(accepted_iterations),
            "support_relaxation_enabled": bool(
                self.relax_support_sites and support_ids.size > 0
            ),
            "minimum_support_motion_scale": float(
                min(accepted_support_scales)
                if accepted_support_scales
                else 0.0
            ),
            "mean_support_uv_displacement": float(
                support_displacement_norm.mean()
                if support_displacement_norm.size > 0
                else 0.0
            ),
            "max_support_uv_displacement": float(
                support_displacement_norm.max()
                if support_displacement_norm.size > 0
                else 0.0
            ),
            "lattice_topology_preserved": bool(
                self._delaunay_edges_np(relaxed) == reference_edges
            ),
        }
        return relaxed, relaxation_diagnostics

    def _coverage_diagnostics(self, seeds_np: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
        samples_uv = self._support_samples_uv_np()
        valid_sample_mask = self._inside_domain_np(samples_uv)
        if seeds_np.shape[0] == 0 or not np.any(interior_mask) or not np.any(valid_sample_mask):
            return {}
        samples_uv = samples_uv[valid_sample_mask]
        all_xyz = self._as_numpy(self.face_mesh["points_xyz"]).astype(np.float64, copy=False).reshape(-1, 3)
        samples_xyz = all_xyz[valid_sample_mask]
        seed_xyz = self._seed_xyz_np(
            seeds_np[interior_mask],
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        distances, nearest = cKDTree(seed_xyz).query(samples_xyz, k=1)
        diagnostics = {
            "mean_nearest_sample_distance": float(np.mean(distances)),
            "rms_nearest_sample_distance": float(np.sqrt(np.mean(distances * distances))),
        }
        weights = self._vertex_area_weights_np()[valid_sample_mask]
        if weights.shape[0] == nearest.shape[0] and weights.sum() > 0.0:
            cluster_area = np.bincount(nearest, weights=weights, minlength=seed_xyz.shape[0])
            occupied = cluster_area[cluster_area > 0.0]
            if occupied.shape[0] > 1 and occupied.mean() > 0.0:
                diagnostics["assignment_area_cv"] = float(occupied.std() / occupied.mean())
        return diagnostics

    @staticmethod
    def _edge_length_cv(out: dict[str, Any]) -> float | None:
        curves = out.get("edge_curves_xyz")
        graph = out.get("graph", {})
        edge_type = graph.get("edge_type") if isinstance(graph, dict) else None
        if not isinstance(curves, torch.Tensor) or curves.numel() == 0:
            return None
        lengths = torch.linalg.vector_norm(curves[:, 1:] - curves[:, :-1], dim=-1).sum(dim=1)
        if isinstance(edge_type, torch.Tensor) and edge_type.numel() == lengths.numel():
            mask = (edge_type == 0) | (edge_type == 1) | (edge_type == 3)
            lengths = lengths[mask]
        finite = lengths[torch.isfinite(lengths) & (lengths > 0)]
        if finite.numel() <= 1:
            return None
        return float((finite.std(unbiased=False) / finite.mean().clamp_min(1e-12)).detach().cpu().item())

    @staticmethod
    def _curve_length_fields(out: dict[str, Any]) -> dict[str, torch.Tensor]:
        curves = out.get("edge_curves_xyz")
        if not isinstance(curves, torch.Tensor):
            return {}
        if curves.ndim != 3 or curves.shape[-1] != 3:
            raise ValueError(
                "edge_curves_xyz must have shape [E, K, 3], "
                f"got {tuple(curves.shape)}."
            )

        if curves.shape[0] == 0:
            edge_lengths = curves.new_empty((0,))
        elif curves.shape[1] < 2:
            edge_lengths = curves.new_zeros((curves.shape[0],))
        else:
            edge_lengths = torch.linalg.vector_norm(
                curves[:, 1:, :] - curves[:, :-1, :],
                dim=-1,
            ).sum(dim=1)

        finite = torch.isfinite(edge_lengths)
        all_total = edge_lengths[finite].sum() if bool(finite.any().detach().cpu().item()) else curves.sum() * 0.0

        graph = out.get("graph", {})
        edge_type = graph.get("edge_type") if isinstance(graph, dict) else None
        vd_total = all_total
        if isinstance(edge_type, torch.Tensor) and edge_type.numel() == edge_lengths.numel():
            edge_type = edge_type.to(device=edge_lengths.device, dtype=torch.long).reshape(-1)
            vd_mask = (edge_type == 0) | (edge_type == 1) | (edge_type == 3)
            keep = finite & vd_mask
            vd_total = edge_lengths[keep].sum() if bool(keep.any().detach().cpu().item()) else curves.sum() * 0.0

        return {
            "edge_curve_lengths_xyz": edge_lengths,
            "total_curve_length": all_total,
            "total_voronoi_curve_length": vd_total,
        }

    @staticmethod
    def _hexagonal_cell_fraction(out: dict[str, Any]) -> float | None:
        graph = out.get("graph", {})
        if not isinstance(graph, dict):
            return None
        seed_ids = graph.get("cell_boundary_seed_ids")
        cell_edges = graph.get("cell_boundary_edge_indices")
        edge_type = graph.get("edge_type")
        if not isinstance(seed_ids, torch.Tensor) or not isinstance(cell_edges, list) or not isinstance(edge_type, torch.Tensor):
            return None
        counts = []
        for ids in cell_edges:
            if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
                continue
            types = edge_type[ids]
            if bool((types == 4).any().detach().cpu().item()):
                continue
            counts.append(int(ids.numel()))
        if not counts:
            return None
        counts_np = np.asarray(counts, dtype=np.int64)
        return float(np.mean(counts_np == 6))

    def _validate_sites_for_decoder(self, seeds_np: np.ndarray, spacing: float | None = None) -> None:
        seeds_np = np.asarray(seeds_np, dtype=np.float64).reshape(-1, 2)
        if seeds_np.shape[0] < 3:
            raise ValueError(
                "Near-uniform honeycomb baseline requires at least three finite, "
                "non-collinear retained sites before decoder topology construction."
            )
        if not np.isfinite(seeds_np).all():
            raise ValueError("Retained baseline sites must all be finite.")
        in_margin = (
            (seeds_np[:, 0] >= -self.seed_domain_margin - 1e-12)
            & (seeds_np[:, 0] <= 1.0 + self.seed_domain_margin + 1e-12)
            & (seeds_np[:, 1] >= -self.seed_domain_margin - 1e-12)
            & (seeds_np[:, 1] <= 1.0 + self.seed_domain_margin + 1e-12)
        )
        if not np.all(in_margin):
            raise ValueError(
                "Retained baseline sites exceed decoder.seed_domain_margin; "
                "increase seed_domain_margin or reduce support_band_cells."
            )
        centered = seeds_np - seeds_np.mean(axis=0, keepdims=True)
        if int(np.linalg.matrix_rank(centered, tol=1e-12)) < 2:
            raise ValueError(
                "Retained baseline sites are collinear; choose a smaller spacing, "
                "different phase_uv, or a larger valid domain."
            )
        self._validate_pairwise_separation_np(seeds_np, spacing=spacing)

    def generate(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        generate_density_fiber: bool = True,
    ) -> dict[str, Any]:
        device = torch.device("cpu") if device is None else torch.device(device)
        if self._decoder_periodic():
            raise NotImplementedError(
                "NearUniformHoneycombBaseline follows the decoder policy and "
                "does not support periodic surfaces."
            )
        spacing = self._choose_spacing()
        seeds_np, interior_mask = self._retained_lattice_np(spacing)
        interior_mask = np.asarray(interior_mask, dtype=bool)
        relaxation_initial_energy = None
        relaxation_final_energy = None
        relaxation_diagnostics: dict[str, Any] = {}
        if self.mode == "physical_relax":
            samples_uv_diag, samples_xyz_diag, weights_diag = self._valid_sample_data_np()
            if samples_uv_diag.shape[0] > 0 and np.any(interior_mask):
                relaxation_initial_energy = self._weighted_physical_energy_np(
                    seeds_np,
                    interior_mask,
                    samples_xyz_diag,
                    weights_diag,
                    device=device,
                    dtype=dtype,
                )
        with torch.no_grad():
            seeds_np, relaxation_diagnostics = self._relax_physical_np(
                seeds_np,
                interior_mask,
                device=device,
                dtype=dtype,
                spacing=spacing,
            )
            self._validate_sites_for_decoder(seeds_np, spacing=spacing)
            if self.mode == "physical_relax" and relaxation_initial_energy is not None:
                relaxation_final_energy = self._weighted_physical_energy_np(
                    seeds_np,
                    interior_mask,
                    samples_xyz_diag,
                    weights_diag,
                    device=device,
                    dtype=dtype,
                )
            seeds_uv = torch.as_tensor(seeds_np, dtype=dtype, device=device)
            seeds_uv.requires_grad_(False)
            out = self.decoder(
                seeds_uv=seeds_uv,
                generate_density_fiber=bool(generate_density_fiber),
            )

        diagnostics: dict[str, Any] = {}
        diagnostics.update(out.get("diagnostics", {}))
        diagnostics.update(
            {
                "requested_seed_count": self.target_seed_count,
                "spacing_uv": float(spacing),
                "generated_interior_seed_count": int(interior_mask.sum()),
                "generated_support_seed_count": int(seeds_np.shape[0] - interior_mask.sum()),
                "relaxation_iterations": int(self.relaxation_steps if self.mode == "physical_relax" else 0),
                "relax_support_sites": bool(self.relax_support_sites),
                "support_relaxation_neighbors": int(self.support_relaxation_neighbors),
                "max_relaxation_displacement_cells": float(self.max_relaxation_displacement_cells),
                "preserve_lattice_topology": bool(self.preserve_lattice_topology),
            }
        )
        diagnostics.update(relaxation_diagnostics)
        diagnostics.update(self._coverage_diagnostics(seeds_np, interior_mask))
        if relaxation_initial_energy is not None:
            diagnostics["relaxation_initial_energy"] = float(relaxation_initial_energy)
        if relaxation_final_energy is not None:
            diagnostics["relaxation_final_energy"] = float(relaxation_final_energy)
        edge_cv = self._edge_length_cv(out)
        if edge_cv is not None:
            diagnostics["edge_length_cv"] = edge_cv
        curve_length_fields = self._curve_length_fields(out)
        if curve_length_fields:
            out.update(curve_length_fields)
            diagnostics["total_curve_length"] = float(
                curve_length_fields["total_curve_length"].detach().cpu().item()
            )
            diagnostics["total_voronoi_curve_length"] = float(
                curve_length_fields["total_voronoi_curve_length"].detach().cpu().item()
            )
        hex_fraction = self._hexagonal_cell_fraction(out)
        if hex_fraction is not None:
            diagnostics["hexagonal_cell_fraction"] = hex_fraction

        # Preserve the decoder mode before identifying this result
        # as a near-uniform honeycomb baseline.
        out["decoder_mode"] = out.get("mode", "unknown")
        out["mode"] = "near_uniform_honeycomb"
        out["baseline_mode"] = self.mode

        # Store the combined decoder and baseline diagnostics.
        out["diagnostics"] = diagnostics

        graph = out.get("graph")
        if isinstance(graph, dict):
            graph["diagnostics"] = diagnostics
            if curve_length_fields:
                graph["edge_curve_lengths_xyz"] = curve_length_fields["edge_curve_lengths_xyz"]
                graph["total_curve_length"] = curve_length_fields["total_curve_length"]
                graph["total_voronoi_curve_length"] = curve_length_fields["total_voronoi_curve_length"]

        # Ensure consistent device and dtype, and keep both aliases
        # synchronized.
        final_seeds_uv = out.get("seeds_uv", seeds_uv).to(
            device=device,
            dtype=dtype,
        )

        out["seeds_uv"] = final_seeds_uv
        out["seeds"] = final_seeds_uv

        # Identify interior design seeds and exterior support seeds.
        interior_seed_mask = torch.as_tensor(
            interior_mask,
            dtype=torch.bool,
            device=device,
        )

        if interior_seed_mask.numel() != final_seeds_uv.shape[0]:
            raise RuntimeError(
                "The interior seed mask is inconsistent with the "
                "number of seeds returned by the decoder: "
                f"{interior_seed_mask.numel()} mask entries for "
                f"{final_seeds_uv.shape[0]} seeds."
            )

        out["interior_seed_mask"] = interior_seed_mask
        out["support_seed_mask"] = ~interior_seed_mask

        # Reproduce the field-data contract used by the FEM,
        # visualization, and Abaqus-export utilities.
        out["face_tensor"] = self.face_mesh

        return out
    def visualize(
        self,
        result: dict[str, Any] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        tube_radius: float | None = None,
        show_surface: bool = True,
        show_seeds: bool = False,
        show_support_seeds: bool = False,
        distinguish_boundary_struts: bool = True,
        surface_opacity: float = 0.35,
        surface_color: str = "lightgray",
        strut_color: str = "darkorange",
        boundary_strut_color: str = "deepskyblue",
        background_color: str = "white",
        window_size: tuple[int, int] = (1400, 900),
        screenshot: str | None = None,
        show: bool = True,
    ) -> Any:
        """Visualize the generated honeycomb network using PyVista.

        Parameters
        ----------
        result
            Output returned by ``generate()``. If omitted, a new result is
            generated automatically.
        tube_radius
            Visualization radius of the struts in physical CAD units. If omitted,
            half of ``decoder.strut_thickness`` is used.
        show_seeds
            Show the interior seed positions.
        show_support_seeds
            Show the exterior support seeds.
        distinguish_boundary_struts
            Plot mandatory type-4 boundary struts using a separate colour.
        screenshot
            Optional path for saving a PNG image.
        show
            If False, construct and return the plotter without opening it.

        Returns
        -------
        pyvista.Plotter
            The constructed PyVista plotter.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "PyVista is required for visualization. Install it using "
                "`pip install pyvista`."
            ) from exc

        if result is None:
            result = self.generate(
                device=device,
                dtype=dtype,
                generate_density_fiber=True,
            )

        # ---------------------------------------------------------
        # Surface mesh
        # ---------------------------------------------------------
        points_value = self.face_mesh.get("points_xyz")
        faces_value = self.face_mesh.get("faces_ijk")

        if points_value is None:
            raise KeyError("face_mesh must contain 'points_xyz'.")
        if faces_value is None:
            raise KeyError("face_mesh must contain 'faces_ijk'.")

        points = self._as_numpy(points_value).astype(
            np.float64,
            copy=False,
        )
        faces = self._as_numpy(faces_value).astype(
            np.int64,
            copy=False,
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "face_mesh['points_xyz'] must have shape [N, 3]."
            )

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                "face_mesh['faces_ijk'] must have shape [F, 3]."
            )

        # PyVista triangle format:
        # [3, i0, i1, i2, 3, i0, i1, i2, ...]
        pyvista_faces = np.column_stack(
            (
                np.full(faces.shape[0], 3, dtype=np.int64),
                faces,
            )
        ).ravel()

        surface = pv.PolyData(points, pyvista_faces)

        plotter = pv.Plotter(window_size=window_size)
        plotter.set_background(background_color)

        if show_surface:
            plotter.add_mesh(
                surface,
                color=surface_color,
                opacity=float(surface_opacity),
                show_edges=True,
                edge_color="gray",
                line_width=0.5,
                label="Shell surface",
            )

        # ---------------------------------------------------------
        # Honeycomb struts
        # ---------------------------------------------------------
        curves_value = result.get("edge_curves_xyz")

        if curves_value is None:
            raise KeyError(
                "The generated result does not contain 'edge_curves_xyz'."
            )

        curves_xyz = self._as_numpy(curves_value).astype(
            np.float64,
            copy=False,
        )

        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError(
                "'edge_curves_xyz' must have shape [E, K, 3]."
            )

        if tube_radius is None:
            tube_radius = 0.5 * float(
                getattr(self.decoder, "strut_thickness", 0.01)
            )

        if tube_radius <= 0.0:
            raise ValueError("tube_radius must be positive.")

        graph = result.get("graph", {})
        edge_types_value = (
            graph.get("edge_type")
            if isinstance(graph, dict)
            else None
        )

        if edge_types_value is not None:
            edge_types = self._as_numpy(
                edge_types_value
            ).astype(np.int64, copy=False).reshape(-1)
        else:
            edge_types = np.full(
                curves_xyz.shape[0],
                -1,
                dtype=np.int64,
            )

        for edge_id, curve in enumerate(curves_xyz):
            finite_mask = np.isfinite(curve).all(axis=1)
            curve = curve[finite_mask]

            if curve.shape[0] < 2:
                continue

            # Remove consecutive coincident samples.
            segment_lengths = np.linalg.norm(
                curve[1:] - curve[:-1],
                axis=1,
            )

            keep = np.concatenate(
                (
                    np.array([True]),
                    segment_lengths > 1e-12,
                )
            )
            curve = curve[keep]

            if curve.shape[0] < 2:
                continue

            polyline = pv.lines_from_points(
                curve,
                close=False,
            )

            tube = polyline.tube(
                radius=float(tube_radius),
                n_sides=12,
                capping=True,
            )

            edge_type = (
                int(edge_types[edge_id])
                if edge_id < edge_types.size
                else -1
            )

            is_boundary_strut = (
                distinguish_boundary_struts
                and edge_type == 4
            )

            color = (
                boundary_strut_color
                if is_boundary_strut
                else strut_color
            )

            plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
            )

        # ---------------------------------------------------------
        # Seed positions
        # ---------------------------------------------------------
        seeds_value = result.get("seeds_xyz")

        if seeds_value is not None and (
            show_seeds or show_support_seeds
        ):
            seeds_xyz = self._as_numpy(seeds_value).astype(
                np.float64,
                copy=False,
            )

            interior_mask = self._as_numpy(
                result["interior_seed_mask"]
            ).astype(bool)

            support_mask = self._as_numpy(
                result["support_seed_mask"]
            ).astype(bool)

            if show_seeds and np.any(interior_mask):
                plotter.add_points(
                    seeds_xyz[interior_mask],
                    color="red",
                    point_size=12,
                    render_points_as_spheres=True,
                    label="Interior seeds",
                )

            if show_support_seeds and np.any(support_mask):
                plotter.add_points(
                    seeds_xyz[support_mask],
                    color="royalblue",
                    point_size=10,
                    render_points_as_spheres=True,
                    label="Support seeds",
                )

        plotter.add_axes()
        plotter.view_isometric()

        if show_seeds or show_support_seeds:
            plotter.add_legend()

        if show:
            plotter.show(
                screenshot=screenshot,
            )
        elif screenshot is not None:
            plotter.show(
                screenshot=screenshot,
                auto_close=False,
            )

        return plotter
