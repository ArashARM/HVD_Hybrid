from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


class IntrinsicSurfaceHoneycombBaseline:
    """Near-uniform intrinsic Voronoi honeycomb on a triangular shell mesh.

    The surface mesh, rather than normalized UV space, defines distances and
    topology.  Mesh-edge graph distances approximate surface geodesics.  The
    generated three-dimensional curves are passed to the existing decoder's
    3D tube-field routine; the decoder's UV Voronoi construction is not used.

    Supported patterns
    ------------------
    ``uniform``
        Isotropic geodesic cells with constant target density.
    ``graded``
        Cell scale varies according to ``vertex_density``.
    ``anisotropic``
        Cells elongate along ``orientation_field`` according to
        ``anisotropy_ratio``.
    ``graded_anisotropic``
        Combines graded scale and directional anisotropy.
    ``stochastic``
        Isotropic initialization with deterministic FPS score jitter.
    """

    valid_patterns = {
        "uniform",
        "graded",
        "anisotropic",
        "graded_anisotropic",
        "stochastic",
    }

    def __init__(
        self,
        decoder: Any,
        face_mesh: dict[str, Any],
        target_seed_count: int,
        *,
        pattern: str = "uniform",
        vertex_density: Any | None = None,
        orientation_field: Any | None = None,
        anisotropy_ratio: float = 1.0,
        stochastic_jitter: float = 0.15,
        random_seed: int = 0,
        relaxation_steps: int = 10,
        centroid_candidate_count: int = 8,
        include_boundary_struts: bool = True,
        curve_samples: int = 32,
        density_tau_physical: float | None = None,
        fiber_tau_physical: float | None = None,
    ) -> None:
        if decoder is None:
            raise ValueError("decoder must be provided.")
        if not isinstance(face_mesh, dict):
            raise TypeError("face_mesh must be a dictionary.")
        if int(target_seed_count) < 2:
            raise ValueError("target_seed_count must be at least 2.")
        if pattern not in self.valid_patterns:
            raise ValueError(
                f"pattern must be one of {sorted(self.valid_patterns)}, got {pattern!r}."
            )
        if float(anisotropy_ratio) < 1.0:
            raise ValueError("anisotropy_ratio must be >= 1.")
        if not 0.0 <= float(stochastic_jitter) < 1.0:
            raise ValueError("stochastic_jitter must lie in [0, 1).")
        if int(relaxation_steps) < 0:
            raise ValueError("relaxation_steps must be >= 0.")
        if int(centroid_candidate_count) < 1:
            raise ValueError("centroid_candidate_count must be >= 1.")
        if int(curve_samples) < 2:
            raise ValueError("curve_samples must be >= 2.")
        if density_tau_physical is not None and float(density_tau_physical) <= 0.0:
            raise ValueError("density_tau_physical must be positive.")
        if fiber_tau_physical is not None and float(fiber_tau_physical) <= 0.0:
            raise ValueError("fiber_tau_physical must be positive.")

        self.decoder = decoder
        self.face_mesh = face_mesh
        self.target_seed_count = int(target_seed_count)
        self.pattern = str(pattern)
        self.vertex_density_input = vertex_density
        self.orientation_field_input = orientation_field
        self.anisotropy_ratio = float(anisotropy_ratio)
        self.stochastic_jitter = float(stochastic_jitter)
        self.random_seed = int(random_seed)
        self.relaxation_steps = int(relaxation_steps)
        self.centroid_candidate_count = int(centroid_candidate_count)
        self.include_boundary_struts = bool(include_boundary_struts)
        self.curve_samples = int(curve_samples)
        self.density_tau_physical = (
            None if density_tau_physical is None else float(density_tau_physical)
        )
        self.fiber_tau_physical = (
            None if fiber_tau_physical is None else float(fiber_tau_physical)
        )

        self.points_xyz = self._required_numpy("points_xyz", np.float64)
        self.faces = self._required_numpy("faces_ijk", np.int64)
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError("face_mesh['points_xyz'] must have shape [V, 3].")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("face_mesh['faces_ijk'] must have shape [F, 3].")
        if self.faces.size == 0:
            raise ValueError("The triangular surface mesh is empty.")
        if self.faces.min() == 1 and self.faces.max() == self.points_xyz.shape[0]:
            self.faces = self.faces - 1
        if self.faces.min() < 0 or self.faces.max() >= self.points_xyz.shape[0]:
            raise ValueError("face_mesh['faces_ijk'] contains invalid vertex indices.")
        if self.target_seed_count > self.points_xyz.shape[0]:
            raise ValueError("target_seed_count cannot exceed the number of mesh vertices.")

        self.uv = self._optional_numpy("uv", np.float64)
        if self.uv is not None:
            self.uv = self.uv.reshape(-1, 2)
            if self.uv.shape[0] != self.points_xyz.shape[0]:
                raise ValueError("face_mesh['uv'] must have one row per mesh vertex.")

        self.face_areas = self._face_areas_np()
        self.vertex_areas = self._vertex_areas_np()
        self.vertex_density = self._vertex_density_np()
        self.orientation_field = self._orientation_field_np()
        (
            self.mesh_edges,
            self.boundary_edges,
            self.geodesic_graph,
        ) = self._build_geodesic_graph()

        component_count, component_labels = connected_components(
            self.geodesic_graph,
            directed=False,
            return_labels=True,
        )
        if component_count != 1:
            raise ValueError(
                "The intrinsic honeycomb currently requires one connected surface "
                f"mesh; found {component_count} connected components."
            )
        self.component_labels = component_labels

    @staticmethod
    def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def _required_numpy(self, key: str, dtype: Any) -> np.ndarray:
        if key not in self.face_mesh:
            raise KeyError(f"face_mesh must contain {key!r}.")
        return self._as_numpy(self.face_mesh[key], dtype=dtype)

    def _optional_numpy(self, key: str, dtype: Any) -> np.ndarray | None:
        value = self.face_mesh.get(key)
        return None if value is None else self._as_numpy(value, dtype=dtype)

    def _face_areas_np(self) -> np.ndarray:
        supplied = self._optional_numpy("face_areas", np.float64)
        if supplied is not None:
            supplied = supplied.reshape(-1)
            if supplied.shape[0] == self.faces.shape[0] and np.all(supplied > 0.0):
                return supplied
        p0 = self.points_xyz[self.faces[:, 0]]
        p1 = self.points_xyz[self.faces[:, 1]]
        p2 = self.points_xyz[self.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("The surface mesh contains degenerate triangles.")
        return areas

    def _vertex_areas_np(self) -> np.ndarray:
        weights = np.zeros(self.points_xyz.shape[0], dtype=np.float64)
        for local in range(3):
            np.add.at(weights, self.faces[:, local], self.face_areas / 3.0)
        if np.any(weights <= 0.0):
            raise ValueError("Every surface vertex must belong to a positive-area face.")
        return weights

    def _vertex_density_np(self) -> np.ndarray:
        needs_density = self.pattern in {"graded", "graded_anisotropic"}
        if self.vertex_density_input is None:
            if needs_density:
                raise ValueError(
                    f"pattern={self.pattern!r} requires vertex_density with one value per vertex."
                )
            return np.ones(self.points_xyz.shape[0], dtype=np.float64)
        density = self._as_numpy(self.vertex_density_input, np.float64).reshape(-1)
        if density.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("vertex_density must have one value per surface vertex.")
        if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
            raise ValueError("vertex_density values must be finite and positive.")
        return density / float(np.average(density, weights=self.vertex_areas))

    def _orientation_field_np(self) -> np.ndarray | None:
        needs_orientation = self.pattern in {"anisotropic", "graded_anisotropic"}
        if self.orientation_field_input is None:
            if needs_orientation:
                raise ValueError(
                    f"pattern={self.pattern!r} requires orientation_field."
                )
            return None
        orientation = self._as_numpy(self.orientation_field_input, np.float64)
        if orientation.ndim == 1:
            if orientation.shape[0] != 3:
                raise ValueError("A constant orientation_field must have shape [3].")
            orientation = np.broadcast_to(
                orientation[None, :],
                (self.points_xyz.shape[0], 3),
            ).copy()
        if orientation.shape != self.points_xyz.shape:
            raise ValueError("orientation_field must have shape [3] or [V, 3].")
        norm = np.linalg.norm(orientation, axis=1)
        if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-12):
            raise ValueError("orientation_field must contain finite nonzero vectors.")
        orientation = orientation / norm[:, None]

        # Project supplied directions onto the discrete vertex tangent planes.
        p0 = self.points_xyz[self.faces[:, 0]]
        p1 = self.points_xyz[self.faces[:, 1]]
        p2 = self.points_xyz[self.faces[:, 2]]
        face_normal = np.cross(p1 - p0, p2 - p0)
        face_normal /= np.maximum(np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-30)
        vertex_normal = np.zeros_like(self.points_xyz)
        for local in range(3):
            np.add.at(
                vertex_normal,
                self.faces[:, local],
                self.face_areas[:, None] * face_normal,
            )
        vertex_normal /= np.maximum(
            np.linalg.norm(vertex_normal, axis=1, keepdims=True),
            1e-30,
        )
        tangent = orientation - np.sum(orientation * vertex_normal, axis=1, keepdims=True) * vertex_normal
        tangent_norm = np.linalg.norm(tangent, axis=1)
        if np.any(tangent_norm <= 1e-12):
            raise ValueError(
                "orientation_field is normal to the surface at one or more vertices."
            )
        return tangent / tangent_norm[:, None]

    def _build_geodesic_graph(
        self,
    ) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
        """Build a weighted mesh-edge graph for approximate geodesic distance."""
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for triangle in self.faces:
            for a, b in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                edge = (a, b) if a < b else (b, a)
                edge_count[edge] += 1

        edges = np.asarray(sorted(edge_count), dtype=np.int64)
        if edges.shape[0] == 0:
            raise ValueError("The triangular mesh has no edges.")
        boundary_edges = np.asarray(
            [edge for edge in sorted(edge_count) if edge_count[edge] == 1],
            dtype=np.int64,
        ).reshape(-1, 2)

        delta = self.points_xyz[edges[:, 1]] - self.points_xyz[edges[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        if np.any(~np.isfinite(length)) or np.any(length <= 0.0):
            raise ValueError("The triangular mesh contains zero-length or invalid edges.")
        direction = delta / length[:, None]

        density_factor = np.sqrt(
            0.5 * (
                self.vertex_density[edges[:, 0]]
                + self.vertex_density[edges[:, 1]]
            )
        )
        directional_factor = np.ones_like(length)
        if self.pattern in {"anisotropic", "graded_anisotropic"}:
            assert self.orientation_field is not None
            orientation0 = self.orientation_field[edges[:, 0]]
            orientation1 = self.orientation_field[edges[:, 1]].copy()
            # Treat orientation as an unsigned line field: q and -q describe
            # the same preferred cell direction.
            flip = np.sum(orientation0 * orientation1, axis=1) < 0.0
            orientation1[flip] *= -1.0
            orientation = orientation0 + orientation1
            orientation /= np.maximum(
                np.linalg.norm(orientation, axis=1, keepdims=True),
                1e-30,
            )
            parallel = np.abs(np.sum(direction * orientation, axis=1))
            perpendicular_sq = np.maximum(1.0 - parallel * parallel, 0.0)
            # Motion along the preferred direction is cheaper, producing cells
            # elongated along that tangent direction.
            directional_factor = np.sqrt(
                parallel * parallel / (self.anisotropy_ratio ** 2)
                + perpendicular_sq
            )

        cost = length * density_factor * directional_factor
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
        data = np.concatenate((cost, cost))
        graph = coo_matrix(
            (data, (rows, cols)),
            shape=(self.points_xyz.shape[0], self.points_xyz.shape[0]),
        ).tocsr()
        return edges, boundary_edges, graph

    def _initial_seed_vertex_ids(self) -> np.ndarray:
        """Deterministic geodesic farthest-point sampling on the shell mesh."""
        weighted_centroid = np.average(
            self.points_xyz,
            axis=0,
            weights=self.vertex_areas * self.vertex_density,
        )
        first = int(
            np.argmin(np.linalg.norm(self.points_xyz - weighted_centroid, axis=1))
        )
        selected = [first]
        minimum_distance = np.asarray(
            dijkstra(self.geodesic_graph, directed=False, indices=first),
            dtype=np.float64,
        ).reshape(-1)

        rng = np.random.default_rng(self.random_seed)
        fixed_noise = rng.uniform(-1.0, 1.0, self.points_xyz.shape[0])
        selected_mask = np.zeros(self.points_xyz.shape[0], dtype=bool)
        selected_mask[first] = True

        while len(selected) < self.target_seed_count:
            score = minimum_distance.copy()
            if self.pattern == "stochastic" and self.stochastic_jitter > 0.0:
                score *= 1.0 + self.stochastic_jitter * fixed_noise
            score[selected_mask] = -np.inf
            next_seed = int(np.argmax(score))
            if not np.isfinite(score[next_seed]):
                raise RuntimeError("Unable to place the requested number of geodesic seeds.")
            selected.append(next_seed)
            selected_mask[next_seed] = True
            new_distance = np.asarray(
                dijkstra(self.geodesic_graph, directed=False, indices=next_seed),
                dtype=np.float64,
            ).reshape(-1)
            minimum_distance = np.minimum(minimum_distance, new_distance)
        return np.asarray(selected, dtype=np.int64)

    def _distance_fields_and_labels(
        self,
        seed_vertex_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.asarray(
            dijkstra(
                self.geodesic_graph,
                directed=False,
                indices=np.asarray(seed_vertex_ids, dtype=np.int64),
            ),
            dtype=np.float64,
        )
        if distances.ndim == 1:
            distances = distances[None, :]
        if np.any(~np.isfinite(distances.min(axis=0))):
            raise RuntimeError("Some surface vertices are unreachable from all seeds.")
        labels = np.argmin(distances, axis=0).astype(np.int64)
        return distances, labels

    def _cell_centroid_candidates(
        self,
        cell_vertices: np.ndarray,
        current_seed: int,
    ) -> np.ndarray:
        weights = self.vertex_areas[cell_vertices] * self.vertex_density[cell_vertices]
        centroid = np.average(self.points_xyz[cell_vertices], axis=0, weights=weights)
        order = np.argsort(
            np.linalg.norm(self.points_xyz[cell_vertices] - centroid, axis=1),
            kind="stable",
        )
        candidates = cell_vertices[order[: self.centroid_candidate_count]]
        if current_seed not in candidates:
            candidates = np.concatenate((np.asarray([current_seed]), candidates))
        return np.unique(candidates).astype(np.int64)

    def _relax_seed_vertex_ids(
        self,
        initial_seed_vertex_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Discrete geodesic CVT using area-weighted candidate medoids."""
        seeds = np.asarray(initial_seed_vertex_ids, dtype=np.int64).copy()
        accepted_iterations = 0
        initial_energy = None
        final_energy = None

        for iteration in range(self.relaxation_steps + 1):
            distance_fields, labels = self._distance_fields_and_labels(seeds)
            nearest_distance = distance_fields[labels, np.arange(labels.size)]
            sample_weight = self.vertex_areas * self.vertex_density
            energy = float(np.sum(sample_weight * nearest_distance ** 2) / sample_weight.sum())
            if initial_energy is None:
                initial_energy = energy
            final_energy = energy
            if iteration == self.relaxation_steps:
                break

            proposed = seeds.copy()
            for cell_id in range(seeds.size):
                cell_vertices = np.flatnonzero(labels == cell_id)
                if cell_vertices.size == 0:
                    continue
                candidates = self._cell_centroid_candidates(
                    cell_vertices,
                    int(seeds[cell_id]),
                )
                candidate_distances = np.asarray(
                    dijkstra(
                        self.geodesic_graph,
                        directed=False,
                        indices=candidates,
                    ),
                    dtype=np.float64,
                )
                if candidate_distances.ndim == 1:
                    candidate_distances = candidate_distances[None, :]
                cell_weight = sample_weight[cell_vertices]
                candidate_energy = np.sum(
                    cell_weight[None, :]
                    * candidate_distances[:, cell_vertices] ** 2,
                    axis=1,
                )
                proposed[cell_id] = int(candidates[int(np.argmin(candidate_energy))])

            # Prevent two cells from choosing the same mesh vertex.
            if np.unique(proposed).size != proposed.size:
                occupied: set[int] = set()
                for cell_id in range(proposed.size):
                    choice = int(proposed[cell_id])
                    if choice in occupied:
                        choice = int(seeds[cell_id])
                    proposed[cell_id] = choice
                    occupied.add(choice)
            if np.array_equal(proposed, seeds):
                break

            proposed_distances, proposed_labels = self._distance_fields_and_labels(proposed)
            proposed_nearest = proposed_distances[
                proposed_labels,
                np.arange(proposed_labels.size),
            ]
            proposed_energy = float(
                np.sum(sample_weight * proposed_nearest ** 2) / sample_weight.sum()
            )
            if proposed_energy > energy + max(1e-14, 1e-10 * abs(energy)):
                break
            seeds = proposed
            accepted_iterations += 1

        distance_fields, labels = self._distance_fields_and_labels(seeds)
        diagnostics = {
            "relaxation_requested_iterations": int(self.relaxation_steps),
            "relaxation_accepted_iterations": int(accepted_iterations),
            "geodesic_cvt_initial_energy": float(initial_energy or 0.0),
            "geodesic_cvt_final_energy": float(final_energy or 0.0),
        }
        return seeds, distance_fields, labels, diagnostics

    @staticmethod
    def _trace_segment_graph(
        node_count: int,
        segments: list[tuple[int, int]],
    ) -> list[list[int]]:
        """Trace an undirected segment graph into branch-to-branch polylines."""
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
        clean_segments: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for a, b in segments:
            a = int(a)
            b = int(b)
            if a == b:
                continue
            pair = (a, b) if a < b else (b, a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            segment_id = len(clean_segments)
            clean_segments.append((a, b))
            adjacency[a].append((b, segment_id))
            adjacency[b].append((a, segment_id))

        used = np.zeros(len(clean_segments), dtype=bool)
        paths: list[list[int]] = []

        def walk(start: int, first_neighbor: int, first_segment: int) -> list[int]:
            path = [start, first_neighbor]
            used[first_segment] = True
            previous = start
            current = first_neighbor
            while len(adjacency[current]) == 2:
                choices = [item for item in adjacency[current] if not used[item[1]]]
                if not choices:
                    break
                next_node, segment_id = choices[0]
                if next_node == previous and len(choices) > 1:
                    next_node, segment_id = choices[1]
                used[segment_id] = True
                path.append(next_node)
                previous, current = current, next_node
                if current == start:
                    break
            return path

        # Open ends and junctions define natural polyline endpoints.
        for node_id, neighbors in enumerate(adjacency):
            if len(neighbors) == 2:
                continue
            for neighbor, segment_id in neighbors:
                if not used[segment_id]:
                    paths.append(walk(node_id, neighbor, segment_id))

        # Any unused segments form degree-two closed loops.
        for segment_id, (a, b) in enumerate(clean_segments):
            if not used[segment_id]:
                paths.append(walk(a, b, segment_id))
        return paths

    @staticmethod
    def _resample_polyline(
        points: np.ndarray,
        sample_count: int,
        auxiliary: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Resample a physical polyline by 3D arclength."""
        points = np.asarray(points, dtype=np.float64)
        keep = np.ones(points.shape[0], dtype=bool)
        if points.shape[0] > 1:
            keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-14
        points = points[keep]
        if auxiliary is not None:
            auxiliary = np.asarray(auxiliary, dtype=np.float64)[keep]
        if points.shape[0] < 2:
            raise ValueError("A curve must contain at least two distinct points.")
        cumulative = np.concatenate(
            (np.asarray([0.0]), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )
        total = float(cumulative[-1])
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("A curve has zero or invalid physical length.")
        targets = np.linspace(0.0, total, int(sample_count))
        result = np.column_stack(
            [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)]
        )
        auxiliary_result = None
        if auxiliary is not None:
            auxiliary_result = np.column_stack(
                [
                    np.interp(targets, cumulative, auxiliary[:, axis])
                    for axis in range(auxiliary.shape[1])
                ]
            )
        return result, auxiliary_result

    def _extract_intrinsic_voronoi_curves(
        self,
        distance_fields: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
        """Extract the discrete intrinsic Voronoi network on the triangle mesh."""
        boundary_edge_set = {
            (int(a), int(b)) if a < b else (int(b), int(a))
            for a, b in self.boundary_edges
        }
        node_lookup: dict[tuple[Any, ...], int] = {}
        node_xyz: list[np.ndarray] = []
        node_uv: list[np.ndarray] = []
        node_is_boundary: list[bool] = []
        segments: list[tuple[int, int]] = []

        def crossing_node(v0: int, v1: int) -> int | None:
            label0 = int(labels[v0])
            label1 = int(labels[v1])
            if label0 == label1:
                return None
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            key = ("edge", edge[0], edge[1])
            if key in node_lookup:
                return node_lookup[key]
            f0 = float(distance_fields[label0, v0] - distance_fields[label1, v0])
            f1 = float(distance_fields[label0, v1] - distance_fields[label1, v1])
            denominator = f0 - f1
            t = 0.5 if abs(denominator) <= 1e-14 else f0 / denominator
            t = float(np.clip(t, 0.0, 1.0))
            node_id = len(node_xyz)
            node_lookup[key] = node_id
            node_xyz.append((1.0 - t) * self.points_xyz[v0] + t * self.points_xyz[v1])
            if self.uv is not None:
                node_uv.append((1.0 - t) * self.uv[v0] + t * self.uv[v1])
            node_is_boundary.append(edge in boundary_edge_set)
            return node_id

        for face_id, triangle in enumerate(self.faces):
            changed_nodes: list[int] = []
            for v0, v1 in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                node_id = crossing_node(v0, v1)
                if node_id is not None and node_id not in changed_nodes:
                    changed_nodes.append(node_id)
            if len(changed_nodes) == 2:
                segments.append((changed_nodes[0], changed_nodes[1]))
            elif len(changed_nodes) == 3:
                # Three labels meet in this triangle.  A triangle-local junction
                # keeps all branches on the piecewise-planar surface.
                key = ("face", int(face_id))
                junction = len(node_xyz)
                node_lookup[key] = junction
                node_xyz.append(np.mean(np.asarray([node_xyz[i] for i in changed_nodes]), axis=0))
                if self.uv is not None:
                    node_uv.append(np.mean(np.asarray([node_uv[i] for i in changed_nodes]), axis=0))
                node_is_boundary.append(False)
                segments.extend((junction, node_id) for node_id in changed_nodes)

        paths = self._trace_segment_graph(len(node_xyz), segments)
        curves_xyz: list[np.ndarray] = []
        curves_uv: list[np.ndarray] = []
        edge_types: list[int] = []
        for path in paths:
            try:
                curve_xyz, curve_uv = self._resample_polyline(
                    np.asarray([node_xyz[i] for i in path]),
                    self.curve_samples,
                    None if self.uv is None else np.asarray([node_uv[i] for i in path]),
                )
            except ValueError:
                continue
            boundary_endpoint_count = int(node_is_boundary[path[0]]) + int(
                node_is_boundary[path[-1]]
            )
            curves_xyz.append(curve_xyz)
            if curve_uv is not None:
                curves_uv.append(curve_uv)
            edge_types.append((0, 1, 3)[min(boundary_endpoint_count, 2)])

        boundary_paths: list[list[int]] = []
        if self.include_boundary_struts and self.boundary_edges.shape[0] > 0:
            boundary_segments = [tuple(map(int, edge)) for edge in self.boundary_edges]
            boundary_paths = self._trace_segment_graph(
                self.points_xyz.shape[0],
                boundary_segments,
            )
            for path in boundary_paths:
                try:
                    curve_xyz, curve_uv = self._resample_polyline(
                        self.points_xyz[path],
                        self.curve_samples,
                        None if self.uv is None else self.uv[path],
                    )
                except ValueError:
                    continue
                curves_xyz.append(curve_xyz)
                if curve_uv is not None:
                    curves_uv.append(curve_uv)
                edge_types.append(4)

        if not curves_xyz:
            raise RuntimeError("The intrinsic Voronoi extraction produced no valid curves.")
        curves_xyz_array = np.stack(curves_xyz, axis=0)
        curves_uv_array = None
        if self.uv is not None:
            curves_uv_array = np.stack(curves_uv, axis=0)
        edge_types_array = np.asarray(edge_types, dtype=np.int64)
        extraction_diagnostics = {
            "num_voronoi_nodes": int(len(node_xyz)),
            "num_voronoi_segments": int(len(segments)),
            "num_network_curves": int(np.sum(edge_types_array != 4)),
            "num_boundary_struts": int(np.sum(edge_types_array == 4)),
            "num_boundary_components": int(len(boundary_paths)),
            "num_edge_type_0": int(np.sum(edge_types_array == 0)),
            "num_edge_type_1": int(np.sum(edge_types_array == 1)),
            "num_edge_type_3": int(np.sum(edge_types_array == 3)),
            "num_edge_type_4": int(np.sum(edge_types_array == 4)),
        }
        return curves_xyz_array, curves_uv_array, edge_types_array, extraction_diagnostics

    def _pattern_diagnostics(
        self,
        labels: np.ndarray,
        curves_xyz: np.ndarray,
        edge_types: np.ndarray,
    ) -> dict[str, Any]:
        cell_areas = np.bincount(
            labels,
            weights=self.vertex_areas,
            minlength=self.target_seed_count,
        )
        area_mean = float(np.mean(cell_areas))
        cell_area_cv = float(np.std(cell_areas) / area_mean) if area_mean > 0.0 else math.inf
        variable_mask = np.isin(edge_types, np.asarray([0, 1, 3]))
        curve_lengths = np.sum(
            np.linalg.norm(np.diff(curves_xyz[variable_mask], axis=1), axis=2),
            axis=1,
        )
        length_mean = float(np.mean(curve_lengths)) if curve_lengths.size else 0.0
        length_cv = (
            float(np.std(curve_lengths) / length_mean)
            if length_mean > 0.0
            else math.inf
        )

        boundary_vertices = np.unique(self.boundary_edges.reshape(-1)) if self.boundary_edges.size else np.empty(0, dtype=np.int64)
        boundary_cells = set(map(int, labels[boundary_vertices]))
        neighbors: list[set[int]] = [set() for _ in range(self.target_seed_count)]
        for v0, v1 in self.mesh_edges:
            c0 = int(labels[int(v0)])
            c1 = int(labels[int(v1)])
            if c0 != c1:
                neighbors[c0].add(c1)
                neighbors[c1].add(c0)
        interior_cells = [i for i in range(self.target_seed_count) if i not in boundary_cells]
        hex_fraction = (
            float(np.mean([len(neighbors[i]) == 6 for i in interior_cells]))
            if interior_cells
            else float("nan")
        )
        return {
            "cell_area_mean": area_mean,
            "cell_area_cv": cell_area_cv,
            "variable_curve_length_mean": length_mean,
            "variable_curve_length_cv": length_cv,
            "num_boundary_adjacent_cells": int(len(boundary_cells)),
            "num_interior_cells": int(len(interior_cells)),
            "interior_six_neighbor_fraction": hex_fraction,
        }

    def _reference_tensor(self) -> torch.Tensor:
        source = self.face_mesh.get("points_xyz")
        if isinstance(source, torch.Tensor):
            return source
        decoder_points = getattr(self.decoder, "points_3d", None)
        if isinstance(decoder_points, torch.Tensor):
            return decoder_points
        return torch.as_tensor(self.points_xyz, dtype=torch.float32)

    def _physical_temperature_scale(self, reference: torch.Tensor) -> torch.Tensor:
        xu_value = self.face_mesh.get("Xu", getattr(self.decoder, "Xu", None))
        xv_value = self.face_mesh.get("Xv", getattr(self.decoder, "Xv", None))
        if xu_value is not None and xv_value is not None and hasattr(
            self.decoder, "_local_uv_to_xyz_scale"
        ):
            xu = torch.as_tensor(xu_value, dtype=reference.dtype, device=reference.device)
            xv = torch.as_tensor(xv_value, dtype=reference.dtype, device=reference.device)
            return self.decoder._local_uv_to_xyz_scale(xu, xv, reference).clamp_min(
                float(getattr(self.decoder, "eps", 1e-8))
            )
        edge_length = np.linalg.norm(
            self.points_xyz[self.mesh_edges[:, 1]] - self.points_xyz[self.mesh_edges[:, 0]],
            axis=1,
        )
        return reference.new_tensor(float(np.median(edge_length)))

    def forward(self, *, generate_density_fiber: bool = True) -> dict[str, Any]:
        """Generate intrinsic 3D curves and decoder-compatible physical fields."""
        initial_seeds = self._initial_seed_vertex_ids()
        seeds, distance_fields, labels, relaxation_diagnostics = self._relax_seed_vertex_ids(
            initial_seeds
        )
        curves_xyz, curves_uv, edge_types, extraction_diagnostics = (
            self._extract_intrinsic_voronoi_curves(distance_fields, labels)
        )
        diagnostics = {
            "mode": "intrinsic_surface_honeycomb",
            "pattern": self.pattern,
            "requested_seed_count": int(self.target_seed_count),
            "generated_seed_count": int(seeds.size),
            "geodesic_approximation": "weighted_mesh_edge_graph",
            "anisotropy_ratio": float(self.anisotropy_ratio),
            "include_boundary_struts": bool(self.include_boundary_struts),
            **relaxation_diagnostics,
            **extraction_diagnostics,
            **self._pattern_diagnostics(labels, curves_xyz, edge_types),
        }

        reference = self._reference_tensor()
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        points = torch.as_tensor(self.points_xyz, dtype=dtype, device=device)
        seeds_xyz = torch.as_tensor(self.points_xyz[seeds], dtype=dtype, device=device)
        curve_tensor = torch.as_tensor(curves_xyz, dtype=dtype, device=device)
        edge_type_tensor = torch.as_tensor(edge_types, dtype=torch.long, device=device)
        seed_uv_tensor = (
            torch.as_tensor(self.uv[seeds], dtype=dtype, device=device)
            if self.uv is not None
            else points.new_empty((seeds.size, 0))
        )
        curve_uv_tensor = (
            torch.as_tensor(curves_uv, dtype=dtype, device=device)
            if curves_uv is not None
            else points.new_empty((curves_xyz.shape[0], self.curve_samples, 0))
        )

        eps = float(getattr(self.decoder, "eps", 1e-8))
        radius = self.decoder.centerline_radius(points).clamp_min(0.0)
        physical_scale = self._physical_temperature_scale(points)
        tau_distance = max(float(getattr(self.decoder, "tube_distance_tau", 0.01)), eps) * physical_scale
        tau_density = (
            points.new_tensor(self.density_tau_physical)
            if self.density_tau_physical is not None
            else max(float(getattr(self.decoder, "tube_density_tau", 0.01)), eps) * physical_scale
        )
        tau_fiber = (
            points.new_tensor(self.fiber_tau_physical)
            if self.fiber_tau_physical is not None
            else max(float(getattr(self.decoder, "tube_fiber_tau", 0.01)), eps) * physical_scale
        )
        fallback_value = self.face_mesh.get("Xu", getattr(self.decoder, "Xu", None))
        fallback_fiber = None
        if fallback_value is not None:
            fallback_fiber = torch.as_tensor(fallback_value, dtype=dtype, device=device)

        if generate_density_fiber:
            field = self.decoder.soft_tube_density_and_fiber_to_elements(
                elem_centers_xyz=points,
                curves_xyz=curve_tensor,
                radius=radius,
                tau_distance=float(tau_distance.detach().item()),
                tau_density=float(tau_density.detach().item()),
                tau_fiber=float(tau_fiber.detach().item()),
                rho_min=float(getattr(self.decoder, "rho_min", 0.0)),
                fallback_fiber=fallback_fiber,
            )
        else:
            fallback = points.new_tensor([1.0, 0.0, 0.0]).expand(points.shape[0], 3)
            if fallback_fiber is not None:
                fallback = torch.nn.functional.normalize(fallback_fiber, dim=-1, eps=eps)
            field = {
                "density": points.new_zeros(points.shape[0]),
                "fiber": fallback,
                "distance": points.new_full((points.shape[0],), float("inf")),
            }

        graph = {
            "edge_type": edge_type_tensor,
            "seed_vertex_ids": torch.as_tensor(seeds, dtype=torch.long, device=device),
            "vertex_cell_labels": torch.as_tensor(labels, dtype=torch.long, device=device),
            "diagnostics": diagnostics,
        }
        output = {
            "mode": "intrinsic_surface_honeycomb",
            "decoder_mode": "curve_fields_only",
            "baseline_mode": self.pattern,
            "seeds": seed_uv_tensor,
            "seeds_uv": seed_uv_tensor,
            "seeds_xyz": seeds_xyz,
            "interior_seed_mask": torch.ones(seeds.size, dtype=torch.bool, device=device),
            "support_seed_mask": torch.zeros(seeds.size, dtype=torch.bool, device=device),
            "rho": field["density"],
            "density": field["density"],
            "fiber3d": field["fiber"],
            "fiber": field["fiber"],
            "tube_distance": field["distance"],
            "strut_thickness": points.new_tensor(float(self.decoder.strut_thickness)),
            "centerline_radius": radius,
            "edge_curves_uv": curve_uv_tensor,
            "edge_curves_xyz": curve_tensor,
            "graph": graph,
            "diagnostics": diagnostics,
            "face_tensor": self.face_mesh,
        }
        if "phi" in field:
            output["phi"] = field["phi"]
        if "theta" in field:
            output["theta"] = field["theta"]
        return output

    __call__ = forward
