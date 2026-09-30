import csv
import json
import itertools
import warnings
from time import perf_counter

import networkx as nx
import numpy as np
try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected in the project env
    cKDTree = None


class ManufacturingInfeasibilityError(RuntimeError):
    """Raised when requested hard manufacturing limits have no feasible route."""

class ManufacturingGeometryError(ValueError):
    """Raised when requested deposition lanes do not fit the strut geometry."""


def calculate_uniform_even_fibre_count(
    strut_thickness,
    fibre_line_width,
    fibre_gap=0.0,
    boundary_margin=0.0,
    fibres_per_edge=None,
    boundary_edge_mask=None,
):
    """Return or validate the common even number of fibre lanes per edge."""
    widths = np.asarray(strut_thickness, dtype=np.float64)
    scalar_width = widths.ndim == 0
    widths_1d = np.asarray([float(widths)]) if scalar_width else widths.reshape(-1)
    fibre_line_width = float(fibre_line_width)
    fibre_gap = float(fibre_gap)
    boundary_margin = float(boundary_margin)
    if fibre_line_width <= 0.0:
        raise ManufacturingGeometryError("fibre_line_width must be positive.")
    if fibre_gap < 0.0:
        raise ManufacturingGeometryError("fibre_gap must be nonnegative.")
    if boundary_margin < 0.0:
        raise ManufacturingGeometryError("boundary_margin must be nonnegative.")
    interior_usable = widths_1d - 2.0 * boundary_margin
    available_usable = interior_usable.copy()
    if boundary_edge_mask is not None:
        mask = np.asarray(boundary_edge_mask, dtype=bool).reshape(-1)
        if mask.shape != widths_1d.shape:
            if scalar_width:
                widths_1d = np.full(mask.shape, float(widths), dtype=np.float64)
                interior_usable = widths_1d - 2.0 * boundary_margin
                available_usable = interior_usable.copy()
            else:
                raise ManufacturingGeometryError(
                    "boundary_edge_mask must contain one value per physical edge."
                )
        available_usable[mask] = 0.5 * widths_1d[mask] - 2.0 * boundary_margin
    usable = float(np.min(available_usable))
    if usable <= 0.0:
        raise ManufacturingGeometryError(
            f"No usable strut width remains after boundary margins: {usable:.6g}."
        )
    if fibres_per_edge is None:
        n_max = int(np.floor((usable + fibre_gap) / (fibre_line_width + fibre_gap)))
        n = 2 * (n_max // 2)
        if n < 2:
            raise ManufacturingGeometryError(
                "Insufficient strut width for an even number of fibre lanes: "
                f"usable={usable:.6g}, fibre_line_width={fibre_line_width:.6g}, "
                f"fibre_gap={fibre_gap:.6g}."
            )
        return int(n)
    n = int(fibres_per_edge)
    if n <= 0 or n % 2 != 0:
        raise ManufacturingGeometryError("fibres_per_edge must be a positive even integer.")
    required = n * fibre_line_width + (n - 1) * fibre_gap
    if required > usable + 1.0e-10:
        raise ManufacturingGeometryError(
            "Requested fibres_per_edge does not fit in the strut: "
            f"required={required:.6g}, usable={usable:.6g}."
        )
    return n


def symmetric_lane_offsets(fibres_per_edge, fibre_line_width, fibre_gap=0.0):
    n = int(fibres_per_edge)
    if n <= 0 or n % 2 != 0:
        raise ManufacturingGeometryError("fibres_per_edge must be a positive even integer.")
    pitch = float(fibre_line_width) + float(fibre_gap)
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * pitch


def boundary_lane_offset_distances(
    fibres_per_edge,
    fibre_line_width,
    fibre_gap=0.0,
    boundary_margin=0.0,
):
    """Unsigned one-sided centre offsets for lanes inside a boundary strut."""
    n = int(fibres_per_edge)
    if n <= 0 or n % 2 != 0:
        raise ManufacturingGeometryError("fibres_per_edge must be a positive even integer.")
    if float(fibre_line_width) <= 0.0:
        raise ManufacturingGeometryError("fibre_line_width must be positive.")
    if float(fibre_gap) < 0.0 or float(boundary_margin) < 0.0:
        raise ManufacturingGeometryError("fibre_gap and boundary_margin must be nonnegative.")
    pitch = float(fibre_line_width) + float(fibre_gap)
    return (
        float(boundary_margin)
        + 0.5 * float(fibre_line_width)
        + np.arange(n, dtype=np.float64) * pitch
    )


def _curve_pair_distance(first, second):
    first = _to_numpy_array(first)
    second = _to_numpy_array(second)
    if first.shape != second.shape:
        return np.inf, False
    direct = np.linalg.norm(first - second, axis=1)
    reversed_dist = np.linalg.norm(first - second[::-1], axis=1)
    direct_max = float(direct.max()) if direct.size else np.inf
    reversed_max = float(reversed_dist.max()) if reversed_dist.size else np.inf
    if direct_max <= reversed_max:
        return direct_max, False
    return reversed_max, True


def _normalize_edge_index_array(edge_index, *, error_type=ManufacturingGeometryError):
    edges = _to_numpy_array(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise error_type("edge_index must have shape [E, 2] or [2, E].")
    if edges.shape[1] == 2:
        return edges.copy()
    if edges.shape[0] == 2:
        return edges.T.copy()
    raise error_type("edge_index must have shape [E, 2] or [2, E].")


def _filter_parallel_array(values, keep_indices, original_edge_count, name):
    if values is None:
        return None
    if np.isscalar(values):
        return values
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values
        if values.ndim > 0 and values.shape[0] == int(original_edge_count):
            return values[np.asarray(keep_indices, dtype=np.int64)]
        raise ManufacturingGeometryError(
            f"{name} must be scalar or contain one value per original physical edge; "
            f"expected first dimension {int(original_edge_count)}, got {values.shape}."
        )
    if len(values) != int(original_edge_count):
        raise ManufacturingGeometryError(
            f"{name} must contain one value per original physical edge; "
            f"expected {int(original_edge_count)}, got {len(values)}."
        )
    return [values[int(index)] for index in keep_indices]


def _coerce_edge_mask(mask, edge_types, edge_count):
    if mask is not None:
        result = np.asarray(mask, dtype=bool).reshape(-1)
        if result.shape != (edge_count,):
            raise ManufacturingGeometryError("boundary_edge_mask must contain one value per edge.")
        return result
    if edge_types is None:
        return np.zeros((edge_count,), dtype=bool)
    types = _to_numpy_array(edge_types, dtype=np.int64).reshape(-1)
    if types.shape != (edge_count,):
        raise ManufacturingGeometryError("edge_types must contain one value per edge.")
    return types == 4


def _edge_spacing_stats(edge_passes, requested_pitch=None, tolerance=1.0e-6):
    spacings = []
    per_pair_distances = []
    for first, second in zip(edge_passes[:-1], edge_passes[1:]):
        count = min(len(first), len(second))
        distances = np.linalg.norm(first[:count] - second[:count], axis=1)
        if distances.size:
            per_pair_distances.append(distances)
            spacings.extend(float(value) for value in distances)
    if not spacings:
        return {
            "minimum": None,
            "mean": None,
            "maximum": None,
            "raw_minimum_spacing": None,
            "interior_minimum_spacing": None,
            "spacing_5th_percentile": None,
            "compressed_sample_count": 0,
            "compressed_sample_fraction": 0.0,
            "maximum_compressed_run": 0,
            "endpoint_compression": False,
            "sustained_compression": False,
        }
    values = np.asarray(spacings, dtype=np.float64)
    threshold = None if requested_pitch is None else float(requested_pitch) - float(tolerance)
    compressed_count = 0
    compressed_total = 0
    maximum_run = 0
    endpoint_compression = False
    interior_values = []
    for distances in per_pair_distances:
        if distances.size > 2:
            interior_values.extend(float(value) for value in distances[1:-1])
        else:
            interior_values.extend(float(value) for value in distances)
        if threshold is None:
            continue
        compressed = distances < threshold
        compressed_count += int(np.sum(compressed))
        compressed_total += int(compressed.size)
        if compressed.size:
            endpoint_compression = bool(endpoint_compression or compressed[0] or compressed[-1])
        run = 0
        for flag in compressed:
            if bool(flag):
                run += 1
                maximum_run = max(maximum_run, run)
            else:
                run = 0
    interior_array = np.asarray(interior_values, dtype=np.float64)
    compressed_fraction = 0.0 if compressed_total == 0 else float(compressed_count) / float(compressed_total)
    sustained = bool(
        threshold is not None
        and compressed_count > 0
        and (
            compressed_fraction >= 0.10
            or maximum_run >= 3
            or (interior_array.size > 0 and float(np.percentile(interior_array, 5.0)) < threshold)
        )
    )
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "raw_minimum_spacing": float(values.min()),
        "interior_minimum_spacing": float(interior_array.min()) if interior_array.size else float(values.min()),
        "spacing_5th_percentile": float(np.percentile(values, 5.0)),
        "compressed_sample_count": int(compressed_count),
        "compressed_sample_fraction": float(compressed_fraction),
        "maximum_compressed_run": int(maximum_run),
        "endpoint_compression": bool(endpoint_compression),
        "sustained_compression": sustained,
    }


def _select_boundary_inward_sign(
    *,
    edge_id,
    centreline,
    lateral,
    normals,
    unsigned_offsets,
    projector,
    surface_evaluator,
    uv,
    metric_inverse_lateral,
    tolerance,
):
    if unsigned_offsets.size == 0:
        raise ManufacturingGeometryError(f"Edge {edge_id}: no boundary offsets were supplied.")
    requested = float(unsigned_offsets[0])
    candidates = []
    for sign in (-1.0, 1.0):
        if projector is not None:
            displaced = centreline + sign * requested * lateral
            pass_xyz, _, correction = projector.project(displaced)
            realised = np.linalg.norm(pass_xyz - centreline, axis=1)
            method = "trimmed_mesh_projection"
        elif surface_evaluator is not None and uv is not None and metric_inverse_lateral is not None:
            pass_uv = uv + sign * requested * metric_inverse_lateral
            pass_xyz, _, evaluated = _evaluate_surface_xyz_normals(surface_evaluator, pass_uv)
            inside_mask = None
            for key in ("inside", "inside_mask", "is_inside", "trimmed_inside"):
                if key in evaluated:
                    inside_mask = _to_numpy_array(evaluated[key], dtype=bool).reshape(-1)
                    break
            if inside_mask is None or inside_mask.shape != (len(pass_uv),):
                raise ManufacturingGeometryError(
                    f"Edge {edge_id}: CAD-only boundary inward detection requires "
                    "a reliable trimmed-domain inside mask or a supplied surface mesh."
                )
            if not bool(np.all(inside_mask)):
                correction = np.full((len(pass_uv),), np.inf, dtype=np.float64)
                realised = np.zeros((len(pass_uv),), dtype=np.float64)
            else:
                displaced = centreline + sign * requested * lateral
                correction = np.linalg.norm(pass_xyz - displaced, axis=1)
                realised = np.linalg.norm(pass_xyz - centreline, axis=1)
            method = "cad_uv_metric"
        else:
            raise ManufacturingGeometryError(
                "A trimmed surface mesh or CAD evaluator is required to determine boundary inward direction."
            )
        min_realised = float(realised.min()) if realised.size else 0.0
        max_correction = float(correction.max()) if correction.size else 0.0
        candidates.append({
            "sign": sign,
            "min_realised": min_realised,
            "max_correction": max_correction,
            "score": (max_correction, abs(min_realised - requested)),
            "method": method,
        })
    viable = [
        item for item in candidates
        if item["min_realised"] >= requested - max(float(tolerance), 1.0e-9)
    ]
    if not viable:
        raise ManufacturingGeometryError(
            f"Edge {edge_id}: cannot determine an inward boundary offset direction "
            "that preserves the requested displacement."
        )
    viable.sort(key=lambda item: item["score"])
    if (
        len(viable) > 1
        and abs(viable[0]["max_correction"] - viable[1]["max_correction"]) <= max(float(tolerance), 1.0e-9)
        and abs(viable[0]["min_realised"] - viable[1]["min_realised"]) <= max(float(tolerance), 1.0e-9)
    ):
        raise ManufacturingGeometryError(
            f"Edge {edge_id}: boundary inward direction is ambiguous."
        )
    return float(viable[0]["sign"]), candidates


def prepare_manufacturing_physical_graph(
    *,
    edge_index,
    edge_curves_xyz,
    edge_types,
    edge_curves_uv=None,
    edge_normals_xyz=None,
    strut_thickness=None,
    coincident_edge_tolerance,
    minimum_manufacturable_edge_length=None,
    maximum_contracted_uv_spread=None,
    **parallel_arrays,
):
    """
    Remove only coincident type-3 copies of type-4 physical boundary edges.

    Candidate duplicates must share the same unordered graph-node pair and be
    geometrically coincident within ``coincident_edge_tolerance`` in either
    direct or reversed sample order. Ambiguous coincident pairs are rejected.
    """
    if coincident_edge_tolerance is None:
        raise ManufacturingGeometryError("coincident_edge_tolerance must be explicit.")
    tolerance = float(coincident_edge_tolerance)
    if tolerance < 0.0:
        raise ManufacturingGeometryError("coincident_edge_tolerance must be nonnegative.")
    edge_index_np = _normalize_edge_index_array(edge_index)
    edge_types_np = _to_numpy_array(edge_types, dtype=np.int64).reshape(-1)
    curves = [_to_numpy_array(curve) for curve in edge_curves_xyz]
    if len(curves) != len(edge_index_np) or edge_types_np.shape != (len(edge_index_np),):
        raise ManufacturingGeometryError("edge_index, edge_curves_xyz, and edge_types must have matching edge counts.")

    by_nodes = {}
    for edge_id, (u, v) in enumerate(edge_index_np):
        by_nodes.setdefault(tuple(sorted((int(u), int(v)))), []).append(edge_id)

    removed = set()
    merged_pairs = []
    ambiguous = []
    for ids in by_nodes.values():
        for first_pos in range(len(ids)):
            first = ids[first_pos]
            if first in removed:
                continue
            for second in ids[first_pos + 1:]:
                if second in removed:
                    continue
                distance, reversed_order = _curve_pair_distance(curves[first], curves[second])
                if distance > tolerance:
                    continue
                types = {int(edge_types_np[first]), int(edge_types_np[second])}
                if types == {3, 4}:
                    preserved = first if int(edge_types_np[first]) == 4 else second
                    removed_id = second if preserved == first else first
                    removed.add(removed_id)
                    merged_pairs.append({
                        "preserved_original_edge_id": int(preserved),
                        "removed_original_edge_id": int(removed_id),
                        "preserved_edge_type": int(edge_types_np[preserved]),
                        "removed_edge_type": int(edge_types_np[removed_id]),
                        "unordered_nodes": [int(v) for v in sorted(edge_index_np[preserved])],
                        "geometric_distance": float(distance),
                        "reversed_sample_order": bool(reversed_order),
                    })
                else:
                    ambiguous.append({
                        "edge_ids": [int(first), int(second)],
                        "edge_types": [int(edge_types_np[first]), int(edge_types_np[second])],
                        "geometric_distance": float(distance),
                    })
        if ambiguous:
            break
    if ambiguous:
        raise ManufacturingGeometryError(
            "Ambiguous coincident physical edges cannot be merged silently: "
            f"{ambiguous[0]}."
        )

    keep_indices = [edge_id for edge_id in range(len(edge_index_np)) if edge_id not in removed]
    canonical_old_to_new = {int(old): int(new) for new, old in enumerate(keep_indices)}
    old_to_new = dict(canonical_old_to_new)
    for pair in merged_pairs:
        preserved = int(pair["preserved_original_edge_id"])
        removed_id = int(pair["removed_original_edge_id"])
        retained_new_id = canonical_old_to_new[preserved]
        old_to_new[preserved] = retained_new_id
        old_to_new[removed_id] = retained_new_id
        pair["preserved_filtered_edge_id"] = int(retained_new_id)
        pair["removed_maps_to_filtered_edge_id"] = int(retained_new_id)
    new_to_old = {int(new): int(old) for new, old in enumerate(keep_indices)}
    filtered = {
        "edge_index": edge_index_np[np.asarray(keep_indices, dtype=np.int64)],
        "edge_curves_xyz": [curves[index] for index in keep_indices],
        "edge_types": edge_types_np[np.asarray(keep_indices, dtype=np.int64)],
        "kept_original_physical_edge_ids": np.asarray(keep_indices, dtype=np.int64),
        "old_to_new_edge_id": old_to_new,
        "new_to_old_edge_id": new_to_old,
        "diagnostics": {
            "coincident_edge_tolerance": tolerance,
            "merged_pairs": merged_pairs,
            "old_to_new_edge_id": dict(old_to_new),
            "new_to_old_edge_id": dict(new_to_old),
            "preserved_original_edge_ids": [int(item["preserved_original_edge_id"]) for item in merged_pairs],
            "removed_original_edge_ids": [int(item["removed_original_edge_id"]) for item in merged_pairs],
        },
    }
    optional = {
        "edge_curves_uv": edge_curves_uv,
        "edge_normals_xyz": edge_normals_xyz,
        "strut_thickness": strut_thickness,
        **parallel_arrays,
    }
    for name, values in optional.items():
        if values is not None:
            filtered[name] = _filter_parallel_array(values, keep_indices, len(edge_index_np), name)
    if minimum_manufacturable_edge_length is None:
        return filtered

    minimum_length = float(minimum_manufacturable_edge_length)
    if not np.isfinite(minimum_length) or minimum_length <= 0.0:
        raise ManufacturingGeometryError("minimum_manufacturable_edge_length must be positive when supplied.")
    uv_spread_limit = None
    if maximum_contracted_uv_spread is not None:
        uv_spread_limit = float(maximum_contracted_uv_spread)
        if not np.isfinite(uv_spread_limit) or uv_spread_limit < 0.0:
            raise ManufacturingGeometryError("maximum_contracted_uv_spread must be nonnegative when supplied.")

    base_edge_index = np.asarray(filtered["edge_index"], dtype=np.int64)
    base_curves_xyz = [np.asarray(curve, dtype=np.float64).copy() for curve in filtered["edge_curves_xyz"]]
    base_edge_types = np.asarray(filtered["edge_types"], dtype=np.int64)
    base_original_ids = np.asarray(filtered["kept_original_physical_edge_ids"], dtype=np.int64)
    base_edge_count = len(base_edge_index)
    edge_lengths = np.asarray([
        float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))) if len(curve) > 1 else 0.0
        for curve in base_curves_xyz
    ], dtype=np.float64)
    short_filtered_ids = [int(edge_id) for edge_id, length in enumerate(edge_lengths) if length < minimum_length]
    short_filtered_set = set(short_filtered_ids)
    short_original_ids = [int(base_original_ids[edge_id]) for edge_id in short_filtered_ids]
    short_lengths = {int(base_original_ids[edge_id]): float(edge_lengths[edge_id]) for edge_id in short_filtered_ids}

    def graph_is_connected(edge_index_values):
        nodes = set(int(value) for edge in edge_index_values for value in edge)
        if not nodes:
            return False
        adjacency = {node: set() for node in nodes}
        for u, v in edge_index_values:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
        stack = [next(iter(nodes))]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(adjacency[node] - seen))
        return seen == nodes

    if not short_filtered_ids:
        graph_connected = graph_is_connected(base_edge_index)
        if not graph_connected:
            raise ManufacturingGeometryError("The physical graph is disconnected.")
        diagnostics = dict(filtered["diagnostics"])
        diagnostics.update({
            "minimum_manufacturable_edge_length": minimum_length,
            "maximum_contracted_uv_spread": uv_spread_limit,
            "short_edge_original_ids": [],
            "short_edge_lengths": {},
            "contracted_node_map": {},
            "contracted_node_components": [],
            "contracted_component_representatives": [],
            "removed_short_edge_count": 0,
            "removed_short_total_length": 0.0,
            "maximum_endpoint_snap_distance": 0.0,
            "filtered_edge_count_after_contraction": int(base_edge_count),
            "graph_connected_after_contraction": True,
        })
        filtered["diagnostics"] = diagnostics
        return filtered

    parent = {}

    def find(node):
        node = int(node)
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        representative = min(root_first, root_second)
        other = max(root_first, root_second)
        parent[other] = representative

    for edge_id in short_filtered_ids:
        u, v = base_edge_index[edge_id]
        union(int(u), int(v))

    component_groups = {}
    for node in list(parent):
        component_groups.setdefault(find(node), set()).add(int(node))
    components = [sorted(nodes) for nodes in component_groups.values()]
    components.sort(key=lambda item: item[0])
    representatives = [int(component[0]) for component in components]
    node_to_representative = {int(node): int(component[0]) for component in components for node in component}

    base_uv = None
    if "edge_curves_uv" in filtered:
        base_uv = [None if values is None else np.asarray(values, dtype=np.float64).copy() for values in filtered["edge_curves_uv"]]
    base_normals = None
    if "edge_normals_xyz" in filtered:
        base_normals = [None if values is None else np.asarray(values, dtype=np.float64).copy() for values in filtered["edge_normals_xyz"]]

    spread_limit = max(minimum_length, tolerance, 1.0e-9)
    canonical_by_representative = {}
    endpoint_spreads = {}
    maximum_endpoint_snap_distance = 0.0
    for component in components:
        representative = int(component[0])
        component_nodes = set(component)
        records = []
        for edge_id, (u, v) in enumerate(base_edge_index):
            curve = base_curves_xyz[edge_id]
            endpoint_data = ((int(u), 0, curve[0]), (int(v), -1, curve[-1]))
            for node, endpoint_index, xyz in endpoint_data:
                if node not in component_nodes:
                    continue
                uv = None if base_uv is None else base_uv[edge_id][endpoint_index]
                normal = None if base_normals is None else base_normals[edge_id][endpoint_index]
                records.append({
                    "node": int(node),
                    "edge_id": int(edge_id),
                    "original_edge_id": int(base_original_ids[edge_id]),
                    "endpoint_index": int(endpoint_index),
                    "xyz": np.asarray(xyz, dtype=np.float64),
                    "uv": None if uv is None else np.asarray(uv, dtype=np.float64),
                    "normal": None if normal is None else np.asarray(normal, dtype=np.float64),
                })
        if not records:
            raise ManufacturingGeometryError(f"Unsafe contraction for node component {component}: no endpoint samples exist.")
        representative_records = [item for item in records if item["node"] == representative]
        choices = representative_records if representative_records else records
        choices = sorted(choices, key=lambda item: (item["original_edge_id"], item["endpoint_index"]))
        canonical_record = choices[0]
        canonical_xyz = canonical_record["xyz"].copy()
        xyz_distances = np.asarray([np.linalg.norm(item["xyz"] - canonical_xyz) for item in records], dtype=np.float64)
        xyz_spread = float(xyz_distances.max()) if xyz_distances.size else 0.0
        if xyz_spread > spread_limit:
            raise ManufacturingGeometryError(
                "Unsafe contraction would snap widely separated XYZ endpoints: "
                f"component={component}, spread={xyz_spread}, limit={spread_limit}."
            )
        maximum_endpoint_snap_distance = max(maximum_endpoint_snap_distance, xyz_spread)

        canonical_uv = None
        uv_spread = None
        if base_uv is not None:
            uv_records = [item for item in records if item["uv"] is not None]
            if len(uv_records) != len(records):
                raise ManufacturingGeometryError(f"Unsafe contraction for node component {component}: incomplete UV endpoint data.")
            canonical_uv = canonical_record["uv"].copy()
            if canonical_uv.ndim != 1 or canonical_uv.size == 0 or not np.all(np.isfinite(canonical_uv)):
                raise ManufacturingGeometryError(f"Unsafe contraction for node component {component}: invalid canonical UV endpoint.")
            for item in uv_records:
                if item["uv"].shape != canonical_uv.shape or not np.all(np.isfinite(item["uv"])):
                    raise ManufacturingGeometryError(
                        f"Unsafe contraction for node component {component}: UV endpoints must have matching finite coordinates."
                    )
            uv_distances = np.asarray([np.linalg.norm(item["uv"] - canonical_uv) for item in uv_records], dtype=np.float64)
            uv_spread = float(uv_distances.max()) if uv_distances.size else 0.0
            if uv_spread_limit is not None and uv_spread > uv_spread_limit:
                raise ManufacturingGeometryError(
                    "Contracted UV endpoint spread exceeds maximum_contracted_uv_spread: "
                    f"component={component}, spread={uv_spread}, limit={uv_spread_limit}."
                )

        canonical_normal = None
        minimum_normal_dot = None
        if base_normals is not None:
            normal_records = [item for item in records if item["normal"] is not None]
            if len(normal_records) != len(records):
                raise ManufacturingGeometryError(f"Unsafe contraction for node component {component}: incomplete normal endpoint data.")
            canonical_normal = _normalise_rows(canonical_record["normal"].reshape(1, 3), name="contracted endpoint normal")[0]
            normals = _normalise_rows(np.vstack([item["normal"] for item in normal_records]), name="contracted endpoint normals")
            dots = normals @ canonical_normal
            minimum_normal_dot = float(dots.min()) if dots.size else 1.0
            if minimum_normal_dot < 0.0:
                raise ManufacturingGeometryError(
                    "Unsafe contraction would combine oppositely oriented endpoint normals: "
                    f"component={component}, minimum_dot={minimum_normal_dot}."
                )

        canonical_by_representative[representative] = {
            "xyz": canonical_xyz,
            "uv": canonical_uv,
            "normal": canonical_normal,
        }
        endpoint_spreads[representative] = {
            "xyz": xyz_spread,
            "uv": uv_spread,
            "minimum_normal_dot": minimum_normal_dot,
        }

    final_keep_filtered_ids = [edge_id for edge_id in range(base_edge_count) if edge_id not in short_filtered_set]
    remapped_edge_index = []
    for edge_id in final_keep_filtered_ids:
        u, v = base_edge_index[edge_id]
        remapped_u = node_to_representative.get(int(u), int(u))
        remapped_v = node_to_representative.get(int(v), int(v))
        if remapped_u == remapped_v:
            raise ManufacturingGeometryError(
                "Unsafe contraction would turn a retained non-short edge into a self-loop: "
                f"original_edge_id={int(base_original_ids[edge_id])}, node={int(remapped_u)}."
            )
        remapped_edge_index.append([int(remapped_u), int(remapped_v)])
    remapped_edge_index = np.asarray(remapped_edge_index, dtype=np.int64)

    final_position_by_filtered_id = {int(edge_id): int(new_id) for new_id, edge_id in enumerate(final_keep_filtered_ids)}
    contracted_old_to_new = {}
    for original_id in range(len(edge_index_np)):
        base_id = filtered["old_to_new_edge_id"].get(int(original_id))
        if base_id is None or int(base_id) in short_filtered_set:
            contracted_old_to_new[int(original_id)] = None
        else:
            contracted_old_to_new[int(original_id)] = int(final_position_by_filtered_id[int(base_id)])
    contracted_new_to_old = {
        int(new_id): int(base_original_ids[edge_id])
        for new_id, edge_id in enumerate(final_keep_filtered_ids)
    }

    snapped_curves_xyz = []
    snapped_curves_uv = [] if base_uv is not None else None
    snapped_normals = [] if base_normals is not None else None
    for edge_id, (remapped_u, remapped_v) in zip(final_keep_filtered_ids, remapped_edge_index):
        original_u, original_v = base_edge_index[edge_id]
        curve = base_curves_xyz[edge_id].copy()
        if int(original_u) != int(remapped_u) or int(remapped_u) in canonical_by_representative:
            canonical = canonical_by_representative.get(int(remapped_u))
            if canonical is not None:
                curve[0] = canonical["xyz"]
        if int(original_v) != int(remapped_v) or int(remapped_v) in canonical_by_representative:
            canonical = canonical_by_representative.get(int(remapped_v))
            if canonical is not None:
                curve[-1] = canonical["xyz"]
        snapped_curves_xyz.append(curve)

        if base_uv is not None:
            uv = base_uv[edge_id].copy()
            canonical = canonical_by_representative.get(int(remapped_u))
            if canonical is not None:
                uv[0] = canonical["uv"]
            canonical = canonical_by_representative.get(int(remapped_v))
            if canonical is not None:
                uv[-1] = canonical["uv"]
            snapped_curves_uv.append(uv)

        if base_normals is not None:
            normals = base_normals[edge_id].copy()
            canonical = canonical_by_representative.get(int(remapped_u))
            if canonical is not None:
                normals[0] = canonical["normal"]
            canonical = canonical_by_representative.get(int(remapped_v))
            if canonical is not None:
                normals[-1] = canonical["normal"]
            snapped_normals.append(_normalise_rows(normals, name="edge_normals_xyz"))

    graph_connected = graph_is_connected(remapped_edge_index)
    if not graph_connected:
        raise ManufacturingGeometryError("Short-edge contraction disconnected the physical graph.")

    result = dict(filtered)
    result.update({
        "edge_index": remapped_edge_index,
        "edge_curves_xyz": snapped_curves_xyz,
        "edge_types": base_edge_types[np.asarray(final_keep_filtered_ids, dtype=np.int64)],
        "kept_original_physical_edge_ids": base_original_ids[np.asarray(final_keep_filtered_ids, dtype=np.int64)],
        "old_to_new_edge_id": contracted_old_to_new,
        "new_to_old_edge_id": contracted_new_to_old,
    })
    if snapped_curves_uv is not None:
        result["edge_curves_uv"] = snapped_curves_uv
    if snapped_normals is not None:
        result["edge_normals_xyz"] = snapped_normals
    for name in optional:
        if name in ("edge_curves_uv", "edge_normals_xyz"):
            continue
        if name in filtered:
            result[name] = _filter_parallel_array(filtered[name], final_keep_filtered_ids, base_edge_count, name)

    diagnostics = dict(filtered["diagnostics"])
    merged_pairs = [dict(item) for item in diagnostics.get("merged_pairs", [])]
    for pair in merged_pairs:
        preserved_original = int(pair["preserved_original_edge_id"])
        final_id = contracted_old_to_new[preserved_original]
        pair["preserved_filtered_edge_id"] = None if final_id is None else int(final_id)
        pair["removed_maps_to_filtered_edge_id"] = None if final_id is None else int(final_id)
    diagnostics.update({
        "merged_pairs": merged_pairs,
        "old_to_new_edge_id": dict(contracted_old_to_new),
        "new_to_old_edge_id": dict(contracted_new_to_old),
        "minimum_manufacturable_edge_length": minimum_length,
        "maximum_contracted_uv_spread": uv_spread_limit,
        "short_edge_original_ids": short_original_ids,
        "short_edge_lengths": short_lengths,
        "contracted_node_map": {int(node): int(rep) for node, rep in sorted(node_to_representative.items())},
        "contracted_node_components": [[int(node) for node in component] for component in components],
        "contracted_component_representatives": representatives,
        "contracted_component_endpoint_spreads": {
            int(rep): values for rep, values in endpoint_spreads.items()
        },
        "removed_short_edge_count": int(len(short_filtered_ids)),
        "removed_short_total_length": float(sum(edge_lengths[edge_id] for edge_id in short_filtered_ids)),
        "maximum_endpoint_snap_distance": float(maximum_endpoint_snap_distance),
        "filtered_edge_count_after_contraction": int(len(final_keep_filtered_ids)),
        "graph_connected_after_contraction": bool(graph_connected),
    })
    diagnostics["removed_original_edge_ids"] = [
        int(edge_id) for edge_id in diagnostics.get("removed_original_edge_ids", [])
    ] + short_original_ids
    result["diagnostics"] = diagnostics
    return result


def _normalise_rows(values, tolerance=1.0e-12, name="vectors"):
    values = np.asarray(values, dtype=np.float64)
    magnitudes = np.linalg.norm(values, axis=-1)
    if np.any(magnitudes <= tolerance):
        bad = int(np.flatnonzero(magnitudes <= tolerance)[0])
        raise ManufacturingGeometryError(f"{name}[{bad}] has zero magnitude.")
    return values / magnitudes[..., None]


def _curve_tangents_np(points, tolerance=1.0e-12):
    points = np.asarray(points, dtype=np.float64)
    tangents = np.zeros_like(points)
    for index in range(len(points)):
        if index == 0:
            vector = points[1] - points[0]
        elif index == len(points) - 1:
            vector = points[-1] - points[-2]
        else:
            vector = points[index + 1] - points[index - 1]
        norm = np.linalg.norm(vector)
        if norm <= tolerance:
            raise ManufacturingGeometryError(
                f"Cannot calculate a valid tangent at sample {index}."
            )
        tangents[index] = vector / norm
    return tangents


def _evaluate_surface_xyz_normals(surface_evaluator, uv):
    evaluated = surface_evaluator(uv)
    if not isinstance(evaluated, dict):
        raise TypeError("surface_evaluator must return a dict with at least 'xyz'.")
    xyz = _to_numpy_array(evaluated["xyz"])
    normals = None
    if "normal" in evaluated:
        normals = _normalise_rows(evaluated["normal"], name="surface normals")
    elif "Xu" in evaluated and "Xv" in evaluated:
        normals = _normalise_rows(
            np.cross(_to_numpy_array(evaluated["Xu"]), _to_numpy_array(evaluated["Xv"])),
            name="surface normals",
        )
    return xyz, normals, evaluated


def _extract_inside_mask(evaluated, expected_count):
    for key in (
        "inside",
        "inside_mask",
        "is_inside",
        "trimmed_inside",
    ):
        if key in evaluated:
            mask = _to_numpy_array(
                evaluated[key],
                dtype=bool,
            ).reshape(-1)

            if mask.shape != (int(expected_count),):
                raise ManufacturingGeometryError(
                    "CAD inside mask must contain "
                    f"{int(expected_count)} values; "
                    f"got {mask.shape}."
                )

            return mask

    return None


def _smoothstep01(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _arc_lengths(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _classify_outside_samples(inside_by_lane):
    inside_by_lane = np.asarray(inside_by_lane, dtype=bool)
    outside = ~inside_by_lane
    outside_lane_ids, outside_sample_ids = np.nonzero(outside)
    if outside_sample_ids.size == 0:
        return {
            "kind": "inside",
            "start": False,
            "end": False,
            "outside_lane_ids": [],
            "outside_sample_ids": [],
            "start_last_sample": None,
            "end_first_sample": None,
        }
    sample_count = inside_by_lane.shape[1]
    outside_any = np.any(outside, axis=0)
    outside_indices = np.flatnonzero(outside_any)
    start_run = 0
    while start_run < sample_count and bool(outside_any[start_run]):
        start_run += 1
    end_run = sample_count - 1
    while end_run >= 0 and bool(outside_any[end_run]):
        end_run -= 1
    endpoint_mask = np.zeros((sample_count,), dtype=bool)
    if start_run > 0:
        endpoint_mask[:start_run] = True
    if end_run < sample_count - 1:
        endpoint_mask[end_run + 1:] = True
    if not np.all(endpoint_mask[outside_indices]):
        return {
            "kind": "interior",
            "start": False,
            "end": False,
            "outside_lane_ids": sorted({int(v) for v in outside_lane_ids}),
            "outside_sample_ids": sorted({int(v) for v in outside_sample_ids}),
            "start_last_sample": int(start_run - 1) if start_run > 0 else None,
            "end_first_sample": int(end_run + 1) if end_run < sample_count - 1 else None,
        }
    return {
        "kind": "endpoint",
        "start": bool(start_run > 0),
        "end": bool(end_run < sample_count - 1),
        "outside_lane_ids": sorted({int(v) for v in outside_lane_ids}),
        "outside_sample_ids": sorted({int(v) for v in outside_sample_ids}),
        "start_last_sample": int(start_run - 1) if start_run > 0 else None,
        "end_first_sample": int(end_run + 1) if end_run < sample_count - 1 else None,
    }


def _lane_order_preserved(offsets_by_lane, tolerance=1.0e-12):
    offsets_by_lane = np.asarray(offsets_by_lane, dtype=np.float64)
    if offsets_by_lane.shape[0] <= 1:
        return True
    return bool(np.all(np.diff(offsets_by_lane, axis=0) > float(tolerance)))


def _select_endpoint_inward_sign(surface_evaluator, uv, metric_inverse_lateral, unsigned_offsets, endpoint_index):
    candidates = []
    for sign in (-1.0, 1.0):
        endpoint_offsets = np.sort(sign * np.asarray(unsigned_offsets, dtype=np.float64))
        probe_uv = np.asarray([
            uv[int(endpoint_index)] + float(offset) * metric_inverse_lateral[int(endpoint_index)]
            for offset in endpoint_offsets
        ], dtype=np.float64)
        _, _, evaluated = _evaluate_surface_xyz_normals(surface_evaluator, probe_uv)
        inside = _extract_inside_mask(evaluated, len(probe_uv))
        candidates.append({"sign": sign, "all_inside": None if inside is None else bool(np.all(inside))})
    inside_candidates = [item for item in candidates if item["all_inside"]]
    if len(inside_candidates) != 1:
        raise ManufacturingGeometryError(
            "Boundary-incident endpoint has ambiguous or unavailable inward side from the CAD inside mask."
        )
    return float(inside_candidates[0]["sign"]), candidates


def _build_blended_offsets(
    interior_offsets,
    start_offsets,
    end_offsets,
    arc_length,
    start_blend_length,
    end_blend_length,
):
    interior_offsets = np.asarray(interior_offsets, dtype=np.float64)
    offsets = np.tile(interior_offsets[:, None], (1, len(arc_length)))
    total_length = float(arc_length[-1]) if len(arc_length) else 0.0
    if start_offsets is not None:
        length = max(float(start_blend_length), 1.0e-12)
        weight = 1.0 - _smoothstep01(arc_length / length)
        offsets += (np.asarray(start_offsets, dtype=np.float64)[:, None] - interior_offsets[:, None]) * weight[None, :]
    if end_offsets is not None:
        length = max(float(end_blend_length), 1.0e-12)
        distance_from_end = total_length - arc_length
        weight = 1.0 - _smoothstep01(distance_from_end / length)
        offsets += (np.asarray(end_offsets, dtype=np.float64)[:, None] - interior_offsets[:, None]) * weight[None, :]
    return offsets


def _build_endpoint_incident_offsets(
    full_offsets,
    arc_length,
    *,
    start_boundary_incident=False,
    end_boundary_incident=False,
    endpoint_blend_length,
):
    full_offsets = np.asarray(full_offsets, dtype=np.float64)
    arc_length = np.asarray(arc_length, dtype=np.float64)
    total_length = float(arc_length[-1]) if arc_length.size else 0.0
    weight = np.ones_like(arc_length, dtype=np.float64)
    length = max(float(endpoint_blend_length), 1.0e-12)
    if bool(start_boundary_incident):
        weight *= _smoothstep01(arc_length / length)
    if bool(end_boundary_incident):
        weight *= _smoothstep01((total_length - arc_length) / length)
    return full_offsets[:, None] * weight[None, :], weight


def _offsets_to_surface_lanes(surface_evaluator, uv, metric_inverse_lateral, centreline, lateral, offsets_by_lane):
    passes_uv = []
    passes_xyz = []
    passes_normals = []
    inside_masks = []
    max_correction = 0.0
    for lane_offsets in np.asarray(offsets_by_lane, dtype=np.float64):
        pass_uv = uv + lane_offsets[:, None] * metric_inverse_lateral
        pass_xyz, pass_normals, evaluated = _evaluate_surface_xyz_normals(surface_evaluator, pass_uv)
        inside = _extract_inside_mask(evaluated, len(pass_uv))
        displaced = centreline + lane_offsets[:, None] * lateral
        correction = np.linalg.norm(pass_xyz - displaced, axis=1)
        max_correction = max(max_correction, float(correction.max()) if correction.size else 0.0)
        passes_uv.append(pass_uv)
        passes_xyz.append(pass_xyz)
        passes_normals.append(pass_normals)
        inside_masks.append(inside)
    return passes_uv, passes_xyz, passes_normals, inside_masks, max_correction


def _local_curves_have_no_reversal(passes_xyz, tolerance=1.0e-12):
    for lane_curve in passes_xyz:
        if len(lane_curve) > 1 and np.any(np.linalg.norm(np.diff(lane_curve, axis=0), axis=1) <= float(tolerance)):
            return False
    return True


def _endpoint_blend_spacing_diagnostics(edge_passes, full_pitch, endpoint_weight, tolerance=1.0e-6):
    endpoint_weight = np.asarray(endpoint_weight, dtype=np.float64).reshape(-1)
    endpoint_samples = endpoint_weight < 1.0 - max(float(tolerance), 1.0e-12)
    interior_samples = ~endpoint_samples
    endpoint_compressed = 0
    robust_values = []
    all_values = []
    for first, second in zip(edge_passes[:-1], edge_passes[1:]):
        count = min(len(first), len(second), len(endpoint_weight))
        if count == 0:
            continue
        distances = np.linalg.norm(first[:count] - second[:count], axis=1)
        all_values.extend(float(value) for value in distances)
        compressed = distances < float(full_pitch) - float(tolerance)
        endpoint_compressed += int(np.sum(compressed & endpoint_samples[:count]))
        robust_values.extend(float(value) for value in distances[interior_samples[:count]])
    robust = np.asarray(robust_values, dtype=np.float64)
    all_distances = np.asarray(all_values, dtype=np.float64)
    return {
        "endpoint_compressed_sample_count": int(endpoint_compressed),
        "minimum_interior_spacing": None if robust.size == 0 else float(robust.min()),
        "minimum_robust_spacing": None if robust.size == 0 else float(robust.min()),
        "minimum_lane_spacing": None if all_distances.size == 0 else float(all_distances.min()),
    }


def _closest_points_on_triangles(points, vertices, faces):
    """Closest-point projection on a triangular mesh, returning points/normals/distances."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    projector = MeshSurfaceProjector(vertices, faces)
    return projector.project(points)


class MeshSurfaceProjector:
    """Reusable accelerated closest-point and normal queries on a triangle mesh."""

    def __init__(self, vertices, faces, nearest_triangle_candidates=None):
        self.vertices, self.faces = coerce_surface_mesh(vertices, faces)
        if self.faces.size == 0:
            raise ManufacturingGeometryError("surface_faces must contain triangles.")
        self.triangles = self.vertices[self.faces]
        face_normals = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        self.face_normals = _normalise_rows(face_normals, name="mesh face normals")
        self.vertex_normals = _surface_vertex_normals(self.vertices, self.faces)
        self.nearest_triangle_candidates = (
            16
            if nearest_triangle_candidates is None
            else max(1, int(nearest_triangle_candidates))
        )
        self.centroids = self.triangles.mean(axis=1)
        self.triangle_radii = np.linalg.norm(self.triangles - self.centroids[:, None, :], axis=2).max(axis=1)
        self.max_triangle_radius = float(self.triangle_radii.max()) if self.triangle_radii.size else 0.0
        self.centroid_tree = cKDTree(self.centroids) if cKDTree is not None else None
        self.vertex_tree = cKDTree(self.vertices) if cKDTree is not None else None

    @staticmethod
    def _closest_point_on_triangle(p, a, b, c):
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        bp = p - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab
        cp = p - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    def _candidate_triangle_ids(self, point):
        if self.centroid_tree is None:
            return range(len(self.triangles))
        k = min(self.nearest_triangle_candidates, len(self.triangles))
        distances, ids = self.centroid_tree.query(point, k=k)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        best_distance = np.inf
        for tri_id in ids:
            a, b, c = self.triangles[int(tri_id)]
            q = self._closest_point_on_triangle(point, a, b, c)
            best_distance = min(best_distance, float(np.linalg.norm(point - q)))
        if not np.isfinite(best_distance):
            return range(len(self.triangles))
        candidate_radius = best_distance + self.max_triangle_radius + 1.0e-12
        candidates = self.centroid_tree.query_ball_point(point, r=candidate_radius)
        if not candidates:
            return ids
        return np.asarray(candidates, dtype=np.int64).reshape(-1)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        closest = np.empty_like(points)
        closest_normals = np.empty_like(points)
        closest_distances = np.empty((points.shape[0],), dtype=np.float64)
        for point_id, p in enumerate(points):
            best_d2 = np.inf
            best_point = None
            best_normal = None
            for tri_id in self._candidate_triangle_ids(p):
                a, b, c = self.triangles[int(tri_id)]
                q = self._closest_point_on_triangle(p, a, b, c)
                d2_q = float(np.dot(p - q, p - q))
                if d2_q < best_d2:
                    best_d2 = d2_q
                    best_point = q
                    best_normal = self.face_normals[int(tri_id)]
            closest[point_id] = best_point
            closest_normals[point_id] = best_normal
            closest_distances[point_id] = float(np.sqrt(best_d2))
        return closest, closest_normals, closest_distances

    def nearest_normals(self, points):
        points = _to_numpy_array(points, dtype=np.float64).reshape(-1, 3)
        if self.vertex_tree is not None:
            _, indices = self.vertex_tree.query(points, k=1)
            return self.vertex_normals[np.asarray(indices, dtype=np.int64)]
        indices = np.argmin(
            np.linalg.norm(points[:, None, :] - self.vertices[None, :, :], axis=2),
            axis=1,
        )
        return self.vertex_normals[indices]


def _old_closest_points_on_triangles(points, vertices, faces):
    """Legacy exhaustive projector retained only for reference; unused."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices, faces = coerce_surface_mesh(vertices, faces)
    if faces.size == 0:
        raise ManufacturingGeometryError("surface_faces must contain triangles.")
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normals = _normalise_rows(face_normals, name="mesh face normals")
    closest = np.empty_like(points)
    closest_normals = np.empty_like(points)
    closest_distances = np.empty((points.shape[0],), dtype=np.float64)

    # Christer Ericson closest point on triangle, vectorised over triangles per point.
    for point_id, p in enumerate(points):
        best_d2 = np.inf
        best_point = None
        best_normal = None
        for tri_id, (a, b, c) in enumerate(tri):
            ab = b - a
            ac = c - a
            ap = p - a
            d1 = np.dot(ab, ap)
            d2 = np.dot(ac, ap)
            if d1 <= 0.0 and d2 <= 0.0:
                q = a
            else:
                bp = p - b
                d3 = np.dot(ab, bp)
                d4 = np.dot(ac, bp)
                if d3 >= 0.0 and d4 <= d3:
                    q = b
                else:
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        v = d1 / (d1 - d3)
                        q = a + v * ab
                    else:
                        cp = p - c
                        d5 = np.dot(ab, cp)
                        d6 = np.dot(ac, cp)
                        if d6 >= 0.0 and d5 <= d6:
                            q = c
                        else:
                            vb = d5 * d2 - d1 * d6
                            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                w = d2 / (d2 - d6)
                                q = a + w * ac
                            else:
                                va = d3 * d6 - d5 * d4
                                if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                                    q = b + w * (c - b)
                                else:
                                    denom = 1.0 / (va + vb + vc)
                                    v = vb * denom
                                    w = vc * denom
                                    q = a + ab * v + ac * w
            d2_q = float(np.dot(p - q, p - q))
            if d2_q < best_d2:
                best_d2 = d2_q
                best_point = q
                best_normal = face_normals[tri_id]
        closest[point_id] = best_point
        closest_normals[point_id] = best_normal
        closest_distances[point_id] = float(np.sqrt(best_d2))
    return closest, closest_normals, closest_distances


def coerce_pyvista_faces(pv_faces):
    """Convert PyVista packed triangular faces [3, i, j, k, ...] to [F, 3]."""
    packed = _to_numpy_array(pv_faces, dtype=np.int64).reshape(-1)
    if packed.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    faces = []
    cursor = 0
    while cursor < packed.size:
        count = int(packed[cursor])
        cursor += 1
        if count != 3:
            raise ManufacturingGeometryError(
                "Only triangular PyVista faces are supported for manufacturing offsets; "
                f"encountered a face with {count} vertices."
            )
        if cursor + 3 > packed.size:
            raise ManufacturingGeometryError("Malformed PyVista packed face array.")
        faces.append(packed[cursor:cursor + 3])
        cursor += 3
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def coerce_surface_mesh(surface_vertices, surface_faces=None, *, face_mesh=None):
    """Return validated surface vertices [V, 3] and triangular faces [F, 3]."""
    if face_mesh is not None:
        if surface_vertices is None:
            surface_vertices = face_mesh.get("points_xyz")
        if surface_faces is None:
            if "pv_faces" in face_mesh:
                surface_faces = coerce_pyvista_faces(face_mesh["pv_faces"])
            else:
                surface_faces = face_mesh.get("faces", face_mesh.get("triangles"))
    if surface_vertices is None or surface_faces is None:
        raise ManufacturingGeometryError(
            "surface_vertices and triangular surface_faces are required."
        )
    vertices = _to_numpy_array(surface_vertices, dtype=np.float64)
    faces_raw = surface_faces
    if isinstance(face_mesh, dict) and faces_raw is face_mesh.get("pv_faces"):
        faces = coerce_pyvista_faces(faces_raw)
    else:
        faces_array = _to_numpy_array(faces_raw, dtype=np.int64)
        if faces_array.ndim == 1:
            if faces_array.size % 4 == 0 and np.all(faces_array[0::4] == 3):
                faces = coerce_pyvista_faces(faces_array)
            else:
                faces = faces_array.reshape(-1, 3)
        else:
            faces = faces_array.reshape(-1, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ManufacturingGeometryError(
            f"surface_vertices must have shape [V, 3], got {vertices.shape}."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ManufacturingGeometryError(
            f"surface_faces must have shape [F, 3], got {faces.shape}."
        )
    if not np.isfinite(vertices).all():
        raise ManufacturingGeometryError("surface_vertices contains non-finite coordinates.")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ManufacturingGeometryError(
            "surface_faces contains vertex indices outside the surface_vertices range."
        )
    return vertices, faces.astype(np.int64, copy=False)


def _surface_vertex_normals(surface_vertices, surface_faces):
    vertices, faces = coerce_surface_mesh(surface_vertices, surface_faces)
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(face_normals, axis=1)
    valid = areas > 1.0e-12
    face_normals[valid] /= areas[valid, None]
    for face_id, face in enumerate(faces):
        if valid[face_id]:
            normals[face] += face_normals[face_id]
    magnitudes = np.linalg.norm(normals, axis=1)
    missing = magnitudes <= 1.0e-12
    if np.any(missing):
        # Isolated vertices fall back to the nearest nonzero normal.
        nonzero = np.flatnonzero(~missing)
        if nonzero.size == 0:
            raise ManufacturingGeometryError("Surface mesh normals are undefined.")
        normals[missing] = normals[nonzero[0]]
        magnitudes = np.linalg.norm(normals, axis=1)
    return normals / magnitudes[:, None]


def _nearest_mesh_normals(points, surface_vertices, surface_faces):
    projector = MeshSurfaceProjector(surface_vertices, surface_faces)
    points = _to_numpy_array(points, dtype=np.float64).reshape(-1, 3)
    return projector.nearest_normals(points)


def generate_surface_offset_passes(
    edge_curves_xyz,
    fibres_per_edge=None,
    fibre_line_width=None,
    fibre_gap=0.0,
    strut_thickness=None,
    boundary_margin=0.0,
    edge_normals_xyz=None,
    edge_curves_uv=None,
    surface_evaluator=None,
    surface_vertices=None,
    surface_faces=None,
    tolerance=1.0e-10,
    edge_types=None,
    boundary_edge_mask=None,
    boundary_inward_signs=None,
    boundary_strut_is_centered=True,
    boundary_incident_blend_length=None,
    spacing_tolerance=1.0e-6,
    original_physical_edge_ids=None,
    deduplication_diagnostics=None,
):
    """
    Generate uniform even surface-constrained deposition lanes for each edge.

    The structural ``edge_curves_xyz`` are not modified. The returned passes are
    manufacturing geometry only. When UV samples and a CAD evaluator are
    available, offsets are solved in the local surface metric and evaluated back
    onto the CAD surface. Otherwise a tangent-plane displacement followed by
    closest-point projection to the supplied triangle mesh is used as a
    documented fallback.
    """
    # Backward compatibility for the previous positional signature:
    # (edge_curves_xyz, edge_curves_uv, edge_normals_xyz, surface_vertices,
    #  surface_faces, strut_thickness, fibre_line_width, ...)
    if isinstance(fibres_per_edge, (list, tuple, np.ndarray)) and edge_normals_xyz is None:
        old_edge_curves_uv = fibres_per_edge
        old_edge_normals = fibre_line_width
        old_surface_vertices = fibre_gap
        old_surface_faces = strut_thickness
        old_strut_thickness = boundary_margin
        old_fibre_line_width = edge_curves_uv
        edge_curves_uv = old_edge_curves_uv
        edge_normals_xyz = old_edge_normals
        surface_vertices = old_surface_vertices
        surface_faces = old_surface_faces
        strut_thickness = old_strut_thickness
        fibre_line_width = old_fibre_line_width
        fibre_gap = 0.0
        boundary_margin = 0.0
        fibres_per_edge = 2
        warnings.warn(
            "The old generate_surface_offset_passes positional signature is deprecated; "
            "use the keyword-based n-lane signature.",
            DeprecationWarning,
            stacklevel=2,
        )

    if strut_thickness is None or fibre_line_width is None:
        raise ManufacturingGeometryError("strut_thickness and fibre_line_width are required.")
    edge_count = len(edge_curves_xyz)
    boundary_mask = _coerce_edge_mask(boundary_edge_mask, edge_types, edge_count)
    fibres_per_edge = calculate_uniform_even_fibre_count(
        strut_thickness=strut_thickness,
        fibre_line_width=fibre_line_width,
        fibre_gap=fibre_gap,
        boundary_margin=boundary_margin,
        fibres_per_edge=fibres_per_edge,
        boundary_edge_mask=boundary_mask if bool(boundary_strut_is_centered) else None,
    )
    bundle_width = fibres_per_edge * float(fibre_line_width) + (fibres_per_edge - 1) * float(fibre_gap)
    required_width = bundle_width + 2.0 * float(boundary_margin)

    mesh_vertices = None
    mesh_faces = None
    projector = None
    if surface_vertices is not None or surface_faces is not None:
        mesh_vertices, mesh_faces = coerce_surface_mesh(surface_vertices, surface_faces)
        projector = MeshSurfaceProjector(mesh_vertices, mesh_faces)
    curves = [_to_numpy_array(curve) for curve in edge_curves_xyz]
    if edge_normals_xyz is None:
        if projector is None:
            raise ManufacturingGeometryError(
                "edge_normals_xyz or a surface mesh is required to compute lateral offsets."
            )
        normals = [projector.nearest_normals(curve) for curve in curves]
    else:
        normals = [_normalise_rows(values, name=f"edge_normals_xyz[{i}]") for i, values in enumerate(edge_normals_xyz)]
    curves_uv = None if edge_curves_uv is None else [_to_numpy_array(curve) for curve in edge_curves_uv]
    interior_offsets = symmetric_lane_offsets(fibres_per_edge, fibre_line_width, fibre_gap)
    boundary_unsigned_offsets = boundary_lane_offset_distances(
        fibres_per_edge,
        fibre_line_width,
        fibre_gap=fibre_gap,
        boundary_margin=boundary_margin,
    )
    supplied_boundary_signs = None
    if boundary_inward_signs is not None:
        supplied_boundary_signs = np.asarray(boundary_inward_signs, dtype=np.float64).reshape(-1)
        if supplied_boundary_signs.shape != (len(curves),):
            raise ManufacturingGeometryError("boundary_inward_signs must contain one value per edge.")
    edge_pass_curves_xyz = []
    edge_pass_normals_xyz = []
    edge_pass_curves_uv = [] if curves_uv is not None else None
    edge_pass_offsets = []
    max_projection_distance = 0.0
    compressed_edge_ids = []
    per_edge_spacing = []
    boundary_sign_diagnostics = {}
    boundary_incident_edge_diagnostics = {}
    boundary_incident_blended_edge_ids = []
    per_edge_projection_correction = []
    methods_used = set()
    normal_methods_used = set()

    for edge_id, centreline in enumerate(curves):
        if centreline.shape != normals[edge_id].shape:
            raise ManufacturingGeometryError(
                f"Edge {edge_id}: centreline and normals must have matching [K, 3] shapes."
            )
        tangents = _curve_tangents_np(centreline)
        lateral = _normalise_rows(np.cross(normals[edge_id], tangents), name=f"edge {edge_id} lateral")
        passes_xyz = []
        passes_normals = []
        passes_uv = []
        edge_method = "cad_uv_metric" if curves_uv is not None and surface_evaluator is not None else "tangent_plane_mesh_projection"
        if edge_method == "cad_uv_metric":
            uv = curves_uv[edge_id]
            if uv.shape != (centreline.shape[0], 2):
                raise ManufacturingGeometryError(f"edge_curves_uv[{edge_id}] must have shape [K, 2].")
            _, _, evaluated = _evaluate_surface_xyz_normals(surface_evaluator, uv)
            if "Xu" not in evaluated or "Xv" not in evaluated:
                edge_method = "tangent_plane_mesh_projection"
        methods_used.add(edge_method)
        if edge_method == "cad_uv_metric":
            uv = curves_uv[edge_id]
            evaluated = surface_evaluator(uv)
            xu = _to_numpy_array(evaluated["Xu"])
            xv = _to_numpy_array(evaluated["Xv"])
            metric_inverse_lateral = []
            for sample_id in range(len(centreline)):
                jac = np.column_stack((xu[sample_id], xv[sample_id]))
                delta_uv, *_ = np.linalg.lstsq(jac, lateral[sample_id], rcond=None)
                mapped = jac @ delta_uv
                norm = np.linalg.norm(mapped)
                if norm <= tolerance:
                    raise ManufacturingGeometryError(
                        f"Edge {edge_id}, sample {sample_id}: cannot map lateral direction into UV metric."
                    )
                metric_inverse_lateral.append(delta_uv / norm)
            metric_inverse_lateral = np.asarray(metric_inverse_lateral, dtype=np.float64)
            if boundary_mask[edge_id]:
                if supplied_boundary_signs is None:
                    sign, sign_candidates = _select_boundary_inward_sign(
                        edge_id=edge_id,
                        centreline=centreline,
                        lateral=lateral,
                        normals=normals[edge_id],
                        unsigned_offsets=boundary_unsigned_offsets,
                        projector=projector,
                        surface_evaluator=surface_evaluator,
                        uv=uv,
                        metric_inverse_lateral=metric_inverse_lateral,
                        tolerance=spacing_tolerance,
                    )
                else:
                    sign = float(np.sign(supplied_boundary_signs[edge_id]))
                    if sign == 0.0:
                        raise ManufacturingGeometryError(f"boundary_inward_signs[{edge_id}] must be nonzero.")
                    sign_candidates = []
                edge_offsets = sign * boundary_unsigned_offsets
                boundary_sign_diagnostics[int(edge_id)] = {"inward_sign": float(sign), "candidates": sign_candidates}
                offsets_by_lane = np.tile(np.asarray(edge_offsets, dtype=np.float64)[:, None], (1, len(uv)))
            else:
                edge_offsets = interior_offsets
                offsets_by_lane = np.tile(np.asarray(edge_offsets, dtype=np.float64)[:, None], (1, len(uv)))

            (
                candidate_passes_uv,
                candidate_passes_xyz,
                candidate_passes_normals,
                candidate_inside_masks,
                edge_max_projection,
            ) = _offsets_to_surface_lanes(
                surface_evaluator,
                uv,
                metric_inverse_lateral,
                centreline,
                lateral,
                offsets_by_lane,
            )

            if not boundary_mask[edge_id]:
                masks_available = all(mask is not None for mask in candidate_inside_masks)
                if masks_available:
                    original_inside_by_lane = np.asarray(candidate_inside_masks, dtype=bool)
                    outside_class = _classify_outside_samples(original_inside_by_lane)
                    if outside_class["kind"] == "interior":
                        raise ManufacturingGeometryError(
                            f"Edge {edge_id}: CAD-generated lane leaves the trimmed domain in the edge interior; "
                            f"outside_sample_ids={outside_class['outside_sample_ids']}."
                        )
                    if outside_class["kind"] == "endpoint":
                        centreline_inside = _extract_inside_mask(evaluated, len(uv))
                        if centreline_inside is None:
                            raise ManufacturingGeometryError(
                                f"Edge {edge_id}: boundary-incident detection requires a CAD inside mask "
                                "on the centreline evaluation."
                            )
                        start_boundary_incident = bool(outside_class["start"] and centreline_inside[0])
                        end_boundary_incident = bool(outside_class["end"] and centreline_inside[-1])
                        if not (start_boundary_incident or end_boundary_incident):
                            raise ManufacturingGeometryError(
                                f"Edge {edge_id}: endpoint lane exits were found, but the corresponding "
                                "centreline endpoint is not inside the trimmed domain."
                            )
                        arc_length = _arc_lengths(centreline)
                        total_length = float(arc_length[-1]) if arc_length.size else 0.0
                        if total_length <= max(float(tolerance), 1.0e-12):
                            raise ManufacturingGeometryError(
                                f"Edge {edge_id}: boundary-incident blend requires a nondegenerate curve."
                            )
                        affected = []
                        if outside_class["start_last_sample"] is not None:
                            affected.append(float(arc_length[int(outside_class["start_last_sample"])]))
                        if outside_class["end_first_sample"] is not None:
                            affected.append(float(total_length - arc_length[int(outside_class["end_first_sample"])]))
                        requested_pitch = float(fibre_line_width) + float(fibre_gap)
                        local_segments = np.linalg.norm(np.diff(centreline, axis=0), axis=1)
                        local_scale = float(np.median(local_segments)) if local_segments.size else requested_pitch
                        min_length = max([required_width, requested_pitch, 2.0 * local_scale, *affected])
                        max_length = total_length if not (outside_class["start"] and outside_class["end"]) else 0.5 * total_length
                        if boundary_incident_blend_length is None:
                            candidate_lengths = np.unique(np.concatenate((
                                np.linspace(min_length, max_length, 16),
                                np.asarray([min_length, min(2.0 * min_length, max_length), max_length], dtype=np.float64),
                            )))
                        else:
                            candidate_lengths = np.asarray([float(boundary_incident_blend_length)], dtype=np.float64)
                        selected = None
                        for blend_length in candidate_lengths:
                            if not np.isfinite(blend_length) or blend_length <= 0.0 or blend_length > max_length + 1.0e-12:
                                continue
                            blended_offsets, endpoint_weight = _build_endpoint_incident_offsets(
                                interior_offsets,
                                arc_length,
                                start_boundary_incident=start_boundary_incident,
                                end_boundary_incident=end_boundary_incident,
                                endpoint_blend_length=blend_length,
                            )
                            if np.any(np.diff(blended_offsets, axis=0) < -max(float(spacing_tolerance), 1.0e-12)):
                                continue
                            (
                                blended_uv,
                                blended_xyz,
                                blended_normals,
                                blended_inside_masks,
                                blended_projection,
                            ) = _offsets_to_surface_lanes(
                                surface_evaluator,
                                uv,
                                metric_inverse_lateral,
                                centreline,
                                lateral,
                                blended_offsets,
                            )
                            if not all(mask is not None and bool(np.all(mask)) for mask in blended_inside_masks):
                                continue
                            if not _local_curves_have_no_reversal(blended_xyz, tolerance=max(float(tolerance), 1.0e-12)):
                                continue
                            spacing_stats_probe = _endpoint_blend_spacing_diagnostics(
                                blended_xyz,
                                requested_pitch,
                                endpoint_weight,
                                tolerance=spacing_tolerance,
                            )
                            robust_spacing = spacing_stats_probe["minimum_robust_spacing"]
                            if robust_spacing is not None and robust_spacing < requested_pitch - max(float(spacing_tolerance), 1.0e-12):
                                continue
                            selected = (
                                float(blend_length),
                                blended_offsets,
                                endpoint_weight,
                                blended_uv,
                                blended_xyz,
                                blended_normals,
                                blended_projection,
                                spacing_stats_probe,
                            )
                            break
                        if selected is None:
                            raise ManufacturingGeometryError(
                                f"Edge {edge_id}: no boundary-incident blend length keeps all CAD samples inside "
                                "with preserved lane order and spacing."
                            )
                        (
                            blend_length,
                            offsets_by_lane,
                            endpoint_weight,
                            candidate_passes_uv,
                            candidate_passes_xyz,
                            candidate_passes_normals,
                            edge_max_projection,
                            selected_spacing_stats,
                        ) = selected
                        blend_end_sample = None
                        if arc_length.size:
                            start_sample = int(np.searchsorted(arc_length, blend_length, side="left")) if start_boundary_incident else None
                            end_sample = int(np.searchsorted(arc_length, total_length - blend_length, side="right") - 1) if end_boundary_incident else None
                            blend_end_sample = {"start": start_sample, "end": end_sample}
                        boundary_incident_edge_diagnostics[int(edge_id)] = {
                            "boundary_incident_start": bool(outside_class["start"]),
                            "boundary_incident_end": bool(outside_class["end"]),
                            "start_boundary_incident": bool(start_boundary_incident),
                            "end_boundary_incident": bool(end_boundary_incident),
                            "originally_outside_lane_ids": outside_class["outside_lane_ids"],
                            "originally_outside_sample_ids": outside_class["outside_sample_ids"],
                            "selected_inward_sign": None,
                            "blend_length_xyz": float(blend_length),
                            "endpoint_blend_length": float(blend_length),
                            "blend_end_sample": blend_end_sample,
                            "minimum_lane_spacing": selected_spacing_stats["minimum_lane_spacing"],
                            "minimum_interior_spacing": selected_spacing_stats["minimum_interior_spacing"],
                            "minimum_robust_spacing": selected_spacing_stats["minimum_robust_spacing"],
                            "endpoint_compressed_sample_count": selected_spacing_stats["endpoint_compressed_sample_count"],
                            "lane_order_preserved": True,
                            "all_samples_inside": True,
                        }
                        boundary_incident_blended_edge_ids.append(int(edge_id))

            passes_uv = candidate_passes_uv
            passes_xyz = candidate_passes_xyz
            for lane_id, lane_normals in enumerate(candidate_passes_normals):
                if lane_normals is not None:
                    passes_normals.append(lane_normals)
                    normal_methods_used.add("cad_derivatives")
                elif mesh_vertices is not None:
                    passes_normals.append(projector.nearest_normals(passes_xyz[lane_id]))
                    normal_methods_used.add("surface_mesh_vertex_normals")
                else:
                    passes_normals.append(normals[edge_id].copy())
                    normal_methods_used.add("centreline_normals")
            if projector is not None:
                mesh_diagnostics = []
                for pass_xyz in passes_xyz:
                    _, _, trimmed_distances = projector.project(pass_xyz)
                    max_trimmed_distance = float(trimmed_distances.max()) if trimmed_distances.size else 0.0
                    edge_max_projection = max(edge_max_projection, max_trimmed_distance)
                    mesh_diagnostics.append(max_trimmed_distance)
                if boundary_incident_edge_diagnostics.get(int(edge_id)) is not None:
                    boundary_incident_edge_diagnostics[int(edge_id)]["maximum_mesh_correction"] = max(mesh_diagnostics) if mesh_diagnostics else 0.0
        else:
            if mesh_vertices is None or mesh_faces is None:
                raise ManufacturingGeometryError(
                    "surface_vertices and surface_faces are required when CAD UV surface evaluation is unavailable."
                )
            if boundary_mask[edge_id]:
                if supplied_boundary_signs is None:
                    sign, sign_candidates = _select_boundary_inward_sign(
                        edge_id=edge_id,
                        centreline=centreline,
                        lateral=lateral,
                        normals=normals[edge_id],
                        unsigned_offsets=boundary_unsigned_offsets,
                        projector=projector,
                        surface_evaluator=None,
                        uv=None,
                        metric_inverse_lateral=None,
                        tolerance=spacing_tolerance,
                    )
                else:
                    sign = float(np.sign(supplied_boundary_signs[edge_id]))
                    if sign == 0.0:
                        raise ManufacturingGeometryError(f"boundary_inward_signs[{edge_id}] must be nonzero.")
                    sign_candidates = []
                edge_offsets = sign * boundary_unsigned_offsets
                boundary_sign_diagnostics[int(edge_id)] = {"inward_sign": float(sign), "candidates": sign_candidates}
            else:
                edge_offsets = interior_offsets
            edge_max_projection = 0.0
            for offset in edge_offsets:
                displaced = centreline + float(offset) * lateral
                projected, projected_normals, distances = projector.project(displaced)
                edge_max_projection = max(edge_max_projection, float(distances.max()) if distances.size else 0.0)
                passes_xyz.append(projected)
                passes_normals.append(projected_normals)
                normal_methods_used.add("surface_mesh_face_normals")
        max_projection_distance = max(max_projection_distance, edge_max_projection)
        requested_pitch = float(fibre_line_width) + float(fibre_gap)
        spacing_stats = _edge_spacing_stats(
            passes_xyz,
            requested_pitch=requested_pitch,
            tolerance=spacing_tolerance,
        )
        per_edge_spacing.append(spacing_stats)
        per_edge_projection_correction.append(float(edge_max_projection))
        if spacing_stats["minimum"] is not None:
            if int(edge_id) in boundary_incident_edge_diagnostics:
                spacing_stats["accepted_endpoint_compression"] = True
                spacing_stats["sustained_compression"] = False
                spacing_stats["lane_collapsed_to_centreline"] = False
                spacing_stats["minimum_interior_spacing"] = boundary_incident_edge_diagnostics[int(edge_id)]["minimum_interior_spacing"]
                spacing_stats["minimum_robust_spacing"] = boundary_incident_edge_diagnostics[int(edge_id)]["minimum_robust_spacing"]
                spacing_stats["endpoint_compressed_sample_count"] = boundary_incident_edge_diagnostics[int(edge_id)]["endpoint_compressed_sample_count"]
            else:
                min_centreline_distance = min(
                    float(np.linalg.norm(pass_curve - centreline, axis=1).min())
                    for pass_curve in passes_xyz
                )
                if min_centreline_distance <= max(float(spacing_tolerance), 1.0e-12):
                    compressed_edge_ids.append(int(edge_id))
                    spacing_stats["sustained_compression"] = True
                    spacing_stats["lane_collapsed_to_centreline"] = True
                elif spacing_stats["sustained_compression"]:
                    compressed_edge_ids.append(int(edge_id))
                else:
                    spacing_stats["lane_collapsed_to_centreline"] = False
        edge_pass_curves_xyz.append(passes_xyz)
        edge_pass_normals_xyz.append(passes_normals)
        edge_pass_offsets.append(np.asarray(edge_offsets, dtype=np.float64).copy())
        if edge_pass_curves_uv is not None and edge_method == "cad_uv_metric":
            edge_pass_curves_uv.append(passes_uv)
    if compressed_edge_ids:
        raise ManufacturingGeometryError(
            "Projected fibre lanes collapsed or compressed below the requested pitch: "
            f"edge_ids={compressed_edge_ids}, requested_pitch={float(fibre_line_width) + float(fibre_gap):.6g}."
        )

    spacing_values = [value["minimum"] for value in per_edge_spacing if value["minimum"] is not None]
    diagnostics = {
        "method": "+".join(sorted(methods_used)),
        "normal_method": "+".join(sorted(normal_methods_used)),
        "fibres_per_edge": int(fibres_per_edge),
        "fibre_line_width": fibre_line_width,
        "fibre_gap": float(fibre_gap),
        "boundary_margin": boundary_margin,
        "bundle_width": float(bundle_width),
        "required_width": required_width,
        "offsets": interior_offsets,
        "boundary_edge_ids": [int(edge_id) for edge_id in np.flatnonzero(boundary_mask)],
        "boundary_inward_signs": {int(edge_id): data["inward_sign"] for edge_id, data in boundary_sign_diagnostics.items()},
        "boundary_incident_edge_ids": boundary_incident_blended_edge_ids,
        "boundary_incident_blended_edge_ids": boundary_incident_blended_edge_ids,
        "boundary_incident_edge_diagnostics": boundary_incident_edge_diagnostics,
        "edge_offsets": edge_pass_offsets,
        "requested_centreline_pitch": float(fibre_line_width) + float(fibre_gap),
        "per_edge_lane_spacing": per_edge_spacing,
        "minimum_actual_spacing": None if not spacing_values else float(min(spacing_values)),
        "compressed_edge_ids": compressed_edge_ids,
        "per_edge_projection_correction": per_edge_projection_correction,
        "max_projection_distance": float(max_projection_distance),
        "original_physical_edge_ids": (
            None if original_physical_edge_ids is None else
            _to_numpy_array(original_physical_edge_ids, dtype=np.int64).reshape(-1).tolist()
        ),
        "deduplicated_original_physical_mappings": deduplication_diagnostics,
        "boundary_direction_candidates": boundary_sign_diagnostics,
        "note": (
            "Mesh fallback uses tangent-plane displacement plus closest-point projection; "
            "it is surface-constrained but not a full geodesic distance-field isocurve solver."
        ),
    }
    if "tangent_plane_mesh_projection" in methods_used:
        edge_pass_curves_uv = None
    return edge_pass_curves_xyz, edge_pass_normals_xyz, edge_pass_curves_uv, edge_pass_offsets, diagnostics


def generate_manufacturing_offset_passes(
    *,
    offset_method="local_surface_projection",
    voronoi_cell_polygons=None,
    geodesic_distance_fields=None,
    geodesic_validation=None,
    **kwargs,
):
    """
    Dispatch manufacturing offset construction.

    ``local_surface_projection`` is the existing approximate local method.
    ``voronoi_geodesic_isocurve`` is reserved for the article-based heat/isocurve
    workflow and refuses to label output article-based unless physical geodesic
    distance validation data is supplied.
    """
    if offset_method in ("local_surface_projection", "tangent_plane_mesh_projection", "cad_uv_metric"):
        result = generate_surface_offset_passes(**kwargs)
        diagnostics = dict(result[4])
        diagnostics["offset_method"] = "local_surface_projection"
        diagnostics["article_based"] = False
        return result[0], result[1], result[2], result[3], diagnostics
    if offset_method == "voronoi_geodesic_isocurve":
        raise NotImplementedError(
            "offset_method='voronoi_geodesic_isocurve' is out of scope for this "
            "Euler implementation. Use the surface-constrained local offset mode; "
            "the articles are used only as references for offsetting and validation."
        )
    raise ValueError(
        "offset_method must be 'local_surface_projection'."
    )

def generate_edge_pass_curves_xyz(
    edge_curves_xyz,
    edge_normals_xyz,
    strut_thickness,
    fibres_per_edge,
    gap=0.0,
    tolerance=1.0e-12,
):
    """
    Generate laterally offset fibre-pass curves inside each strut.

    Deprecated compatibility helper. Prefer ``generate_surface_offset_passes``,
    which projects lanes back to the CAD surface or triangle mesh and supports
    the final manufacturing validation/export path.

    Parameters
    ----------
    edge_curves_xyz : sequence of [K, 3] arrays
        Optimised centreline curve of every physical strut.

    edge_normals_xyz : sequence of [K, 3] arrays
        CAD surface normal at every centreline sample.

    strut_thickness : float or sequence of length E
        Available in-surface width of each strut.

    fibres_per_edge : int
        Number of fibre passes placed inside every strut.

    gap : float
        Required edge-to-edge gap between adjacent fibre tracks.

    tolerance : float
        Numerical tolerance used for normalisation.

    Returns
    -------
    edge_pass_curves_xyz : list
        edge_pass_curves_xyz[edge_id][fibre_id] is a [K, 3]
        array containing one physical fibre trajectory.

    edge_pass_normals_xyz : list
        CAD normals associated with every generated pass.

    spacing_information : dict
        Fibre width, centre spacing, and lateral offsets for each edge.
    """
    warnings.warn(
        "generate_edge_pass_curves_xyz is deprecated; use generate_surface_offset_passes "
        "for surface-constrained manufacturing geometry.",
        DeprecationWarning,
        stacklevel=2,
    )

    edge_curves_xyz = [
        np.asarray(curve, dtype=np.float64)
        for curve in edge_curves_xyz
    ]

    edge_normals_xyz = [
        np.asarray(normals, dtype=np.float64)
        for normals in edge_normals_xyz
    ]

    number_of_edges = len(edge_curves_xyz)

    if len(edge_normals_xyz) != number_of_edges:
        raise ValueError(
            "edge_normals_xyz must contain one normal array "
            "for every edge curve."
        )

    fibres_per_edge = int(fibres_per_edge)

    if fibres_per_edge <= 0:
        raise ValueError(
            "fibres_per_edge must be positive."
        )

    gap = float(gap)

    if gap < 0.0:
        raise ValueError(
            "gap must be nonnegative."
        )

    # Allow one common thickness or a different thickness per edge.
    if np.isscalar(strut_thickness):

        thicknesses = np.full(
            number_of_edges,
            float(strut_thickness),
            dtype=np.float64,
        )

    else:

        thicknesses = np.asarray(
            strut_thickness,
            dtype=np.float64,
        )

        if thicknesses.shape != (number_of_edges,):
            raise ValueError(
                "strut_thickness must be a scalar or an "
                "array containing one value per edge."
            )

    edge_pass_curves_xyz = []
    edge_pass_normals_xyz = []

    fibre_widths = []
    centre_spacings = []
    edge_offsets = []

    for edge_id, centreline in enumerate(
        edge_curves_xyz
    ):

        normals = edge_normals_xyz[
            edge_id
        ].copy()

        if centreline.shape != normals.shape:
            raise ValueError(
                f"Edge {edge_id}: centreline and normal arrays "
                "must have the same shape."
            )

        if (
            centreline.ndim != 2
            or centreline.shape[1] != 3
        ):
            raise ValueError(
                f"Edge {edge_id}: expected curve shape [K, 3]."
            )

        thickness = float(
            thicknesses[edge_id]
        )

        if thickness <= 0.0:
            raise ValueError(
                f"Edge {edge_id}: strut thickness must be positive."
            )

        total_gap_width = (
            fibres_per_edge - 1
        ) * gap

        available_fibre_width = (
            thickness - total_gap_width
        )

        if available_fibre_width <= 0.0:
            raise ValueError(
                f"Edge {edge_id}: the requested gaps occupy "
                f"{total_gap_width:.6g}, which is not smaller than "
                f"the strut thickness {thickness:.6g}."
            )

        fibre_width = (
            available_fibre_width
            / fibres_per_edge
        )

        centre_spacing = (
            fibre_width + gap
        )

        # Symmetric offsets about the optimised centreline.
        offsets = (
            np.arange(
                fibres_per_edge,
                dtype=np.float64,
            )
            - 0.5 * (
                fibres_per_edge - 1
            )
        ) * centre_spacing

        # -----------------------------------------------------
        # Compute travel tangents along the centreline
        # -----------------------------------------------------

        tangents = np.zeros_like(
            centreline
        )

        for sample_id in range(
            len(centreline)
        ):

            if sample_id == 0:

                tangent = (
                    centreline[1]
                    - centreline[0]
                )

            elif sample_id == len(
                centreline
            ) - 1:

                tangent = (
                    centreline[-1]
                    - centreline[-2]
                )

            else:

                tangent = (
                    centreline[sample_id + 1]
                    - centreline[sample_id - 1]
                )

            tangent_length = np.linalg.norm(
                tangent
            )

            if tangent_length <= tolerance:
                raise RuntimeError(
                    f"Edge {edge_id}, sample {sample_id}: "
                    "cannot calculate a valid tangent."
                )

            tangents[sample_id] = (
                tangent / tangent_length
            )

        # -----------------------------------------------------
        # Normalise CAD surface normals
        # -----------------------------------------------------

        normal_lengths = np.linalg.norm(
            normals,
            axis=1,
        )

        if np.any(
            normal_lengths <= tolerance
        ):
            bad_sample = int(
                np.flatnonzero(
                    normal_lengths <= tolerance
                )[0]
            )

            raise RuntimeError(
                f"Edge {edge_id}, sample {bad_sample}: "
                "surface normal has zero magnitude."
            )

        normals /= normal_lengths[:, None]

        # -----------------------------------------------------
        # In-surface transverse direction
        #
        # tangent × normal lies in the shell tangent plane and
        # is perpendicular to the strut centreline.
        # -----------------------------------------------------

        lateral_directions = np.cross(
            tangents,
            normals,
        )

        lateral_lengths = np.linalg.norm(
            lateral_directions,
            axis=1,
        )

        if np.any(
            lateral_lengths <= tolerance
        ):
            bad_sample = int(
                np.flatnonzero(
                    lateral_lengths <= tolerance
                )[0]
            )

            raise RuntimeError(
                f"Edge {edge_id}, sample {bad_sample}: "
                "tangent and normal do not define a valid "
                "lateral offset direction."
            )

        lateral_directions /= (
            lateral_lengths[:, None]
        )

        # Prevent accidental sign changes along one edge.
        for sample_id in range(
            1,
            len(lateral_directions),
        ):

            if (
                np.dot(
                    lateral_directions[
                        sample_id - 1
                    ],
                    lateral_directions[
                        sample_id
                    ],
                )
                < 0.0
            ):
                lateral_directions[
                    sample_id
                ] *= -1.0

        # -----------------------------------------------------
        # Generate each physical fibre-pass trajectory
        # -----------------------------------------------------

        passes_for_edge = []
        normals_for_edge = []

        for offset in offsets:

            pass_curve = (
                centreline
                + offset
                * lateral_directions
            )

            passes_for_edge.append(
                pass_curve
            )

            # First approximation: use the centreline CAD normals.
            normals_for_edge.append(
                normals.copy()
            )

        edge_pass_curves_xyz.append(
            passes_for_edge
        )

        edge_pass_normals_xyz.append(
            normals_for_edge
        )

        fibre_widths.append(
            fibre_width
        )

        centre_spacings.append(
            centre_spacing
        )

        edge_offsets.append(
            offsets
        )

    spacing_information = {
        "strut_thickness": thicknesses,
        "gap": gap,
        "fibre_width": np.asarray(
            fibre_widths
        ),
        "centre_spacing": np.asarray(
            centre_spacings
        ),
        "offsets": edge_offsets,
    }

    return (
        edge_pass_curves_xyz,
        edge_pass_normals_xyz,
        spacing_information,
    )

def _to_numpy_array(value, dtype=np.float64):
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def compute_edge_normals_from_cad(
    cad_domain,
    fields,
    *,
    surface_vertices=None,
    surface_faces=None,
    xyz_tolerance=1.0e-8,
    normal_tolerance=1.0e-12,
    nozzle_approach="+normal",
    return_diagnostics=True,
):
    """
    Evaluate CAD normals at exactly ``fields["edge_curves_uv"]`` samples.

    Normals are derived from ``Xu x Xv`` returned by the same CAD evaluator
    used to lift UV curves into XYZ. Edge and sample ordering are preserved.
    ``nozzle_approach`` documents whether a downstream nozzle convention
    approaches from ``+normal`` or ``-normal``; this function does not build
    machine-specific tool poses.
    """
    if nozzle_approach not in ("+normal", "-normal"):
        raise ValueError("nozzle_approach must be '+normal' or '-normal'.")
    if cad_domain is None and (surface_vertices is None or surface_faces is None):
        raise ValueError("cad_domain must be supplied unless surface mesh data is supplied.")
    if "edge_curves_uv" not in fields:
        raise KeyError("fields must contain 'edge_curves_uv'.")
    if "edge_curves_xyz" not in fields:
        raise KeyError("fields must contain 'edge_curves_xyz'.")

    curves_uv_value = fields["edge_curves_uv"]
    curves_xyz = _to_numpy_array(fields["edge_curves_xyz"])
    curves_uv_np = _to_numpy_array(curves_uv_value)
    if curves_uv_np.ndim != 3 or curves_uv_np.shape[-1] != 2:
        raise ValueError("fields['edge_curves_uv'] must have shape [E, K, 2].")
    if curves_xyz.ndim != 3 or curves_xyz.shape[:2] != curves_uv_np.shape[:2] or curves_xyz.shape[-1] != 3:
        raise ValueError("fields['edge_curves_xyz'] must have shape [E, K, 3].")

    flat_uv_np = curves_uv_np.reshape(-1, 2)
    torch_evaluator = getattr(cad_domain, "eval_uv_norm_batch_torch", None)
    evaluated = None
    torch_xyz = None
    normal_method = None
    if torch_evaluator is not None and callable(torch_evaluator) and hasattr(curves_uv_value, "reshape"):
        try:
            torch_evaluated = torch_evaluator(curves_uv_value.reshape(-1, 2))
            if isinstance(torch_evaluated, dict) and "xyz" in torch_evaluated:
                torch_xyz = torch_evaluated["xyz"]
        except Exception:
            torch_xyz = None

    batch_evaluator = getattr(cad_domain, "eval_uv_norm_batch", None)
    if batch_evaluator is not None and callable(batch_evaluator):
        batch_evaluated = batch_evaluator(flat_uv_np, return_inside_mask=False)
        if isinstance(batch_evaluated, dict) and all(name in batch_evaluated for name in ("xyz", "Xu", "Xv")):
            evaluated = dict(batch_evaluated)
            if torch_xyz is not None:
                evaluated["xyz"] = torch_xyz
            normal_method = "cad_derivatives"

    if evaluated is None and torch_evaluator is not None and callable(torch_evaluator):
        try:
            import torch
            if isinstance(curves_uv_value, torch.Tensor):
                flat_uv_t = curves_uv_value.reshape(-1, 2).detach().clone().requires_grad_(True)
                reeval = torch_evaluator(flat_uv_t)
                xyz_t = reeval["xyz"] if isinstance(reeval, dict) else reeval
                xu_cols = []
                xv_cols = []
                for component in range(3):
                    grad = torch.autograd.grad(
                        xyz_t[:, component].sum(),
                        flat_uv_t,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if grad is None:
                        raise RuntimeError("CAD XYZ is not differentiable with respect to UV.")
                    xu_cols.append(grad[:, 0])
                    xv_cols.append(grad[:, 1])
                evaluated = {
                    "xyz": xyz_t,
                    "Xu": torch.stack(xu_cols, dim=1),
                    "Xv": torch.stack(xv_cols, dim=1),
                }
                torch_xyz = xyz_t
                normal_method = "torch_autograd"
        except Exception:
            evaluated = None
    if evaluated is None:
        if surface_vertices is not None and surface_faces is not None:
            mesh_normals = _nearest_mesh_normals(curves_xyz.reshape(-1, 3), surface_vertices, surface_faces)
            evaluated = {
                "xyz": curves_xyz.reshape(-1, 3),
                "Xu": np.zeros_like(mesh_normals),
                "Xv": np.zeros_like(mesh_normals),
                "normal": mesh_normals,
            }
            normal_method = "surface_mesh_vertex_normals"
        else:
            raise TypeError(
                "CAD normals require CAD Xu/Xv, differentiable Torch XYZ for autograd, "
                "or surface_vertices/surface_faces for mesh-normal fallback."
            )
    if not isinstance(evaluated, dict):
        raise TypeError("CAD evaluator must return a dict containing 'xyz', 'Xu', and 'Xv'.")

    xyz = _to_numpy_array(evaluated["xyz"]).reshape(curves_xyz.shape)
    if "normal" in evaluated:
        raw_normals = _to_numpy_array(evaluated["normal"]).reshape(-1, 3)
    else:
        missing = [name for name in ("Xu", "Xv") if name not in evaluated]
        if missing:
            raise KeyError(
                "CAD evaluator must return derivatives for normals; missing "
                + ", ".join(missing)
            )
        xu = _to_numpy_array(evaluated["Xu"]).reshape(-1, 3)
        xv = _to_numpy_array(evaluated["Xv"]).reshape(-1, 3)
        raw_normals = np.cross(xu, xv)
    magnitudes = np.linalg.norm(raw_normals, axis=1)
    undefined = np.flatnonzero(~np.isfinite(magnitudes) | (magnitudes <= float(normal_tolerance)))
    normals = np.zeros_like(raw_normals)
    valid = magnitudes > float(normal_tolerance)
    normals[valid] = raw_normals[valid] / magnitudes[valid, None]
    normals = normals.reshape(curves_xyz.shape)

    xyz_errors = np.linalg.norm(xyz - curves_xyz, axis=2)
    max_xyz_error = float(xyz_errors.max()) if xyz_errors.size else 0.0
    inconsistent = []
    flat_normals = normals.reshape(-1, 3)
    for edge_id in range(normals.shape[0]):
        for sample_id in range(1, normals.shape[1]):
            previous = normals[edge_id, sample_id - 1]
            current = normals[edge_id, sample_id]
            if np.linalg.norm(previous) > 0.0 and np.linalg.norm(current) > 0.0:
                dot = float(np.dot(previous, current))
                if dot < 0.0:
                    inconsistent.append({
                        "edge_id": int(edge_id),
                        "sample_ids": [int(sample_id - 1), int(sample_id)],
                        "dot": dot,
                    })

    diagnostics = {
        "nozzle_approach": nozzle_approach,
        "normal_method": normal_method,
        "max_reevaluated_xyz_error": max_xyz_error,
        "xyz_tolerance": float(xyz_tolerance),
        "xyz_matches_supplied": bool(max_xyz_error <= float(xyz_tolerance)),
        "undefined_normal_count": int(undefined.size),
        "undefined_normal_locations": [
            {
                "edge_id": int(index // curves_uv_np.shape[1]),
                "sample_id": int(index % curves_uv_np.shape[1]),
                "normal_magnitude": float(magnitudes[index]),
            }
            for index in undefined[:50]
        ],
        "inconsistent_normal_count": len(inconsistent),
        "inconsistent_normal_locations": inconsistent[:50],
    }
    if undefined.size:
        first = diagnostics["undefined_normal_locations"][0]
        raise RuntimeError(
            "CAD normal is undefined at "
            f"edge {first['edge_id']} sample {first['sample_id']}."
        )
    if max_xyz_error > float(xyz_tolerance):
        raise RuntimeError(
            "Re-evaluated CAD XYZ does not match fields['edge_curves_xyz']: "
            f"max error {max_xyz_error:.8g} exceeds {float(xyz_tolerance):.8g}."
        )
    if inconsistent:
        first = inconsistent[0]
        raise RuntimeError(
            "CAD normals are inconsistently oriented: "
            f"edge {first['edge_id']} samples {first['sample_ids']} "
            f"have dot {first['dot']:.6g}."
        )
    edge_normals = [normals[edge_id].copy() for edge_id in range(normals.shape[0])]
    if return_diagnostics:
        return edge_normals, diagnostics
    return edge_normals


class EulerCCFToolpath:
    """
    Generate and optimise a continuous Euler toolpath for CCF printing.

    Each physical graph edge is represented by an XYZ polyline.
    Every physical edge is duplicated `fibres_per_edge` times.

    If:
        1. the physical graph is connected, and
        2. fibres_per_edge is even,

    then every multigraph node has even degree and an Euler circuit exists.
    """

    def __init__(
        self,
        edge_index,
        edge_curves_xyz,
        fibres_per_edge=2,
        edge_normals_xyz=None,
        edge_curves_uv=None,
        edge_pass_curves_xyz=None,
        edge_pass_normals_xyz=None,
        edge_pass_curves_uv=None,
        edge_pass_offsets=None,
        edge_surface_positions=None,
        edge_stress_directions_xyz=None,
        original_physical_edge_ids=None,
        physical_edge_metadata=None,
    ):
        self.edge_index = _normalize_edge_index_array(edge_index, error_type=ValueError)

        self.edge_curves_xyz = [
            np.asarray(curve, dtype=np.float64)
            for curve in edge_curves_xyz
        ]
        self.edge_normals_xyz = (
            None if edge_normals_xyz is None else
            [np.asarray(normals, dtype=np.float64) for normals in edge_normals_xyz]
        )
        self.edge_curves_uv = (
            None if edge_curves_uv is None else
            [np.asarray(curve, dtype=np.float64) for curve in edge_curves_uv]
        )
        self.edge_pass_curves_xyz = self._coerce_pass_arrays(edge_pass_curves_xyz)
        self.edge_pass_normals_xyz = self._coerce_pass_arrays(edge_pass_normals_xyz)
        self.edge_pass_curves_uv = self._coerce_pass_arrays(edge_pass_curves_uv)
        self.edge_pass_offsets = self._coerce_pass_offsets(edge_pass_offsets)
        self.edge_surface_positions = (
            None if edge_surface_positions is None else
            [np.asarray(values, dtype=np.float64) for values in edge_surface_positions]
        )
        self.edge_stress_directions_xyz = (
            None if edge_stress_directions_xyz is None else
            [np.asarray(values, dtype=np.float64) for values in edge_stress_directions_xyz]
        )
        self.original_physical_edge_ids = (
            np.arange(len(self.edge_index), dtype=np.int64)
            if original_physical_edge_ids is None else
            np.asarray(original_physical_edge_ids, dtype=np.int64).reshape(-1)
        )
        self.physical_edge_metadata = {} if physical_edge_metadata is None else dict(physical_edge_metadata)

        self.fibres_per_edge = int(
            fibres_per_edge
        )

        self._validate_inputs()

        # Geometry inside a physical edge cannot change when an Euler route
        # is reordered. Measure it once, then account for each fibre pass.
        self._internal_turns = self._measure_internal_turns()
        self._internal_angles = np.asarray(
            [entry["angle"] for entry in self._internal_turns], dtype=np.float64
        )
        self._printed_internal_angles = np.tile(
            self._internal_angles, self.fibres_per_edge
        )
        self._internal_max = float(self._internal_angles.max()) if self._internal_angles.size else 0.0
        self._internal_counts = {
            limit: int(np.sum(self._printed_internal_angles > limit))
            for limit in (90, 120, 150)
        }

        self.G = None
        self.euler_edges = None
        self.baseline_euler_edges = None
        self.toolpath_xyz = None
        self.manufacturing_report = None
        self.baseline_manufacturing_report = None
        self.score_breakdown = None
        self.baseline_score_breakdown = None
        self._manufacturing_static_cache = {}
        self._transition_table = None
        self.junction_transitions = None
        self.manufactured_path_report = None

    def _coerce_pass_arrays(self, values):
        if values is None:
            return None
        result = []
        for edge_values in values:
            result.append([np.asarray(pass_values, dtype=np.float64) for pass_values in edge_values])
        return result

    def _coerce_pass_offsets(self, values):
        if values is None:
            return None
        if np.isscalar(values):
            values = [values for _ in range(self.fibres_per_edge)]
        result = []
        if len(values) == len(self.edge_index) and not np.isscalar(values[0]):
            for edge_values in values:
                offsets = np.asarray(edge_values, dtype=np.float64).reshape(-1)
                result.append(offsets)
        else:
            offsets = np.asarray(values, dtype=np.float64).reshape(-1)
            result = [offsets.copy() for _ in range(len(self.edge_index))]
        return result

    # =========================================================
    # Input validation
    # =========================================================

    def _validate_inputs(self):

        if (
            self.edge_index.ndim != 2
            or self.edge_index.shape[1] != 2
        ):
            raise ValueError(
                "edge_index must have shape [E, 2]."
            )

        if len(self.edge_index) == 0:
            raise ValueError("edge_index must contain at least one edge.")

        if np.any(self.edge_index[:, 0] == self.edge_index[:, 1]):
            raise ValueError("Self-loop edges are not supported.")

        if len(self.edge_curves_xyz) != len(
            self.edge_index
        ):
            raise ValueError(
                "The number of edge curves must equal "
                "the number of graph edges."
            )

        for name, values in (
            ("edge_normals_xyz", self.edge_normals_xyz),
            ("edge_curves_uv", self.edge_curves_uv),
            ("edge_pass_curves_xyz", self.edge_pass_curves_xyz),
            ("edge_pass_normals_xyz", self.edge_pass_normals_xyz),
            ("edge_pass_curves_uv", self.edge_pass_curves_uv),
            ("edge_pass_offsets", self.edge_pass_offsets),
            ("edge_surface_positions", self.edge_surface_positions),
            ("edge_stress_directions_xyz", self.edge_stress_directions_xyz),
        ):
            if values is not None and len(values) != len(self.edge_index):
                raise ValueError(f"{name} must contain one array per graph edge.")
        if self.original_physical_edge_ids.shape != (len(self.edge_index),):
            raise ValueError("original_physical_edge_ids must contain one value per graph edge.")

        if self.fibres_per_edge <= 0:
            raise ValueError(
                "fibres_per_edge must be positive."
            )

        if self.fibres_per_edge % 2 != 0:
            raise ValueError(
                "fibres_per_edge must be even."
            )

        for edge_id, curve in enumerate(
            self.edge_curves_xyz
        ):
            if (
                curve.ndim != 2
                or curve.shape[1] != 3
            ):
                raise ValueError(
                    f"Curve {edge_id} must have "
                    f"shape [N, 3], not {curve.shape}."
                )

            if len(curve) < 2:
                raise ValueError(
                    f"Curve {edge_id} must contain "
                    "at least two points."
                )

            if not np.isfinite(curve).all():
                raise ValueError(f"Curve {edge_id} contains non-finite coordinates.")

            for name, values in (
                ("edge_normals_xyz", self.edge_normals_xyz),
                ("edge_surface_positions", self.edge_surface_positions),
                ("edge_stress_directions_xyz", self.edge_stress_directions_xyz),
            ):
                if values is None:
                    continue
                array = values[edge_id]
                if array.shape != curve.shape:
                    raise ValueError(
                        f"{name}[{edge_id}] must have shape {curve.shape}, "
                        f"not {array.shape}."
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"{name}[{edge_id}] contains non-finite values.")

            if self.edge_normals_xyz is not None:
                magnitudes = np.linalg.norm(self.edge_normals_xyz[edge_id], axis=1)
                if np.any(magnitudes <= 1.0e-12):
                    raise ValueError(f"edge_normals_xyz[{edge_id}] contains zero normals.")

            if self.edge_curves_uv is not None:
                uv = self.edge_curves_uv[edge_id]
                if uv.shape != (curve.shape[0], 2):
                    raise ValueError(
                        f"edge_curves_uv[{edge_id}] must have shape "
                        f"({curve.shape[0]}, 2), not {uv.shape}."
                    )
                if not np.isfinite(uv).all():
                    raise ValueError(f"edge_curves_uv[{edge_id}] contains non-finite values.")

            for name, values in (
                ("edge_pass_curves_xyz", self.edge_pass_curves_xyz),
                ("edge_pass_normals_xyz", self.edge_pass_normals_xyz),
                ("edge_pass_curves_uv", self.edge_pass_curves_uv),
            ):
                if values is None:
                    continue
                if len(values[edge_id]) != self.fibres_per_edge:
                    raise ValueError(
                        f"{name}[{edge_id}] must contain {self.fibres_per_edge} "
                        "per-fibre pass arrays."
                    )
                for fibre_id, pass_array in enumerate(values[edge_id]):
                    expected_shape = curve.shape if name != "edge_pass_curves_uv" else (curve.shape[0], 2)
                    if pass_array.shape != expected_shape:
                        raise ValueError(
                            f"{name}[{edge_id}][{fibre_id}] must have shape "
                            f"{expected_shape}, not {pass_array.shape}."
                        )
                    if not np.isfinite(pass_array).all():
                        raise ValueError(
                            f"{name}[{edge_id}][{fibre_id}] contains non-finite values."
                        )
                if name == "edge_pass_normals_xyz":
                    for fibre_id, pass_array in enumerate(values[edge_id]):
                        magnitudes = np.linalg.norm(pass_array, axis=1)
                        if np.any(magnitudes <= 1.0e-12):
                            raise ValueError(
                                f"{name}[{edge_id}][{fibre_id}] contains zero normals."
                            )
            if self.edge_pass_offsets is not None:
                offsets = self.edge_pass_offsets[edge_id]
                if offsets.shape != (self.fibres_per_edge,):
                    raise ValueError(
                        f"edge_pass_offsets[{edge_id}] must contain {self.fibres_per_edge} values."
                    )
                if not np.isfinite(offsets).all():
                    raise ValueError(f"edge_pass_offsets[{edge_id}] contains non-finite values.")

    # =========================================================
    # Graph construction
    # =========================================================

    def _measure_internal_turns(self, tolerance=1.0e-12):
        """Measure bends between successive nonzero segments of each curve."""
        result = []
        for edge_id, curve in enumerate(self.edge_curves_xyz):
            # Duplicate consecutive points have no tangent; skip them while
            # preserving indices in the original input polyline.
            kept = [0]
            for index in range(1, len(curve)):
                if np.linalg.norm(curve[index] - curve[kept[-1]]) > tolerance:
                    kept.append(index)
            for position in range(1, len(kept) - 1):
                previous, middle, following = kept[position - 1:position + 2]
                incoming = curve[middle] - curve[previous]
                outgoing = curve[following] - curve[middle]
                cosine = np.clip(
                    np.dot(incoming, outgoing)
                    / (np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
                    -1.0, 1.0,
                )
                result.append({
                    "edge_id": edge_id,
                    "point_index": middle,
                    "angle": float(np.degrees(np.arccos(cosine))),
                })
        return result

    def internal_turn_diagnostics(self, top_n=10):
        """Report the sharpest bends in the input curves, once per edge."""
        if top_n < 0:
            raise ValueError("top_n must be nonnegative.")
        worst = sorted(self._internal_turns, key=lambda item: item["angle"], reverse=True)
        print(f"Internal bends per physical graph: {len(worst)}")
        print(f"Maximum internal bend: {worst[0]['angle']:.2f} degrees" if worst
              else "Maximum internal bend: none")
        for item in worst[:top_n]:
            print(f"  Edge {item['edge_id']}, point {item['point_index']}: "
                  f"{item['angle']:.2f} degrees")
        return worst

    def build_multigraph(
        self,
        instance_order=None,
    ):
        """
        Build the Euler multigraph.

        Every item in instance_order is a pair:

            (physical_edge_id, fibre_id)

        Randomising instance_order changes the insertion order of all
        individual fibre-edge instances, rather than only changing the
        order of physical edges.
        """

        graph = nx.MultiGraph()

        physical_nodes = np.unique(
            self.edge_index
        )

        graph.add_nodes_from(
            int(node)
            for node in physical_nodes
        )

        # Create every individual fibre-edge instance.
        all_instances = [
            (edge_id, fibre_id)
            for edge_id in range(
                len(self.edge_index)
            )
            for fibre_id in range(
                self.fibres_per_edge
            )
        ]

        if instance_order is None:
            instance_order = all_instances

        else:
            instance_order = [
                (
                    int(edge_id),
                    int(fibre_id),
                )
                for edge_id, fibre_id
                in instance_order
            ]

            if (
                len(instance_order)
                != len(all_instances)
            ):
                raise ValueError(
                    "instance_order does not contain "
                    "the correct number of fibre-edge instances."
                )

            if sorted(instance_order) != sorted(all_instances):
                raise ValueError(
                    "instance_order must contain every "
                    "(physical_edge_id, fibre_id) pair exactly once."
                )

        for edge_id, fibre_id in instance_order:

            u, v = self.edge_index[
                edge_id
            ]

            graph.add_edge(
                int(u),
                int(v),
                physical_edge_id=int(
                    edge_id
                ),
                fibre_id=int(
                    fibre_id
                ),
            )

        self.G = graph

        # Routes and path coordinates from an earlier graph are no longer valid.
        self.euler_edges = None
        self.baseline_euler_edges = None
        self.toolpath_xyz = None

        return graph

    def validate_graph(
        self,
        graph=None,
        verbose=True,
    ):
        """
        Check graph connectivity and Eulerian status.
        """

        if graph is None:
            graph = self.G

        if graph is None:
            raise RuntimeError(
                "Run build_multigraph() first."
            )

        if graph.number_of_nodes() == 0:
            raise RuntimeError(
                "The graph contains no nodes."
            )

        connected = nx.is_connected(
            graph
        )

        odd_nodes = [
            int(node)
            for node, degree
            in graph.degree()
            if degree % 2 != 0
        ]

        eulerian = nx.is_eulerian(
            graph
        )

        if verbose:
            print(
                f"Physical edges       : "
                f"{len(self.edge_index)}"
            )
            print(
                f"Multigraph edges     : "
                f"{graph.number_of_edges()}"
            )
            print(
                f"Graph nodes          : "
                f"{graph.number_of_nodes()}"
            )
            print(
                f"Connected            : "
                f"{connected}"
            )
            print(
                f"Odd-degree nodes     : "
                f"{len(odd_nodes)}"
            )
            print(
                f"Eulerian             : "
                f"{eulerian}"
            )

        if not connected:
            components = sorted(
                (
                    len(component)
                    for component
                    in nx.connected_components(
                        graph
                    )
                ),
                reverse=True,
            )

            raise RuntimeError(
                "The graph is disconnected. "
                f"Component sizes: {components}"
            )

        if not eulerian:
            raise RuntimeError(
                "The graph is not Eulerian. "
                f"Odd-degree nodes: {odd_nodes}"
            )

        return {
            "connected": connected,
            "eulerian": eulerian,
            "odd_nodes": odd_nodes,
        }

    # =========================================================
    # Euler route generation
    # =========================================================

    @staticmethod
    def _networkx_route(
        graph,
        source=None,
    ):
        """
        Convert a NetworkX Euler circuit into our route format.
        """

        route = []

        circuit = nx.eulerian_circuit(
            graph,
            source=source,
            keys=True,
        )

        for u, v, key in circuit:

            data = graph.get_edge_data(
                u,
                v,
                key,
            )

            route.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "key": key,
                    "edge_id": int(
                        data[
                            "physical_edge_id"
                        ]
                    ),
                    "fibre_id": int(
                        data["fibre_id"]
                    ),
                }
            )

        return route

    def compute_euler_circuit(
        self,
        source=None,
    ):
        """
        Generate the baseline NetworkX Euler circuit.
        """

        if self.G is None:
            self.build_multigraph()

        self.validate_graph(
            verbose=False
        )

        route = self._networkx_route(
            self.G,
            source=source,
        )

        self.validate_route(
            route,
            raise_on_error=True,
            verbose=False,
        )

        self.euler_edges = route

        self.baseline_euler_edges = [
            dict(item)
            for item in route
        ]

        return route

    # =========================================================
    # Curve orientation and tangent calculations
    # =========================================================

    def get_oriented_curve(
        self,
        item,
    ):
        """
        Return one physical edge curve in the actual
        traversal direction.
        """

        edge_id = int(
            item["edge_id"]
        )

        curve = self.edge_curves_xyz[
            edge_id
        ]

        original_u = int(
            self.edge_index[
                edge_id, 0
            ]
        )

        original_v = int(
            self.edge_index[
                edge_id, 1
            ]
        )

        traversal_u = int(
            item["u"]
        )

        traversal_v = int(
            item["v"]
        )

        if (
            traversal_u == original_u
            and traversal_v == original_v
        ):
            return curve

        if (
            traversal_u == original_v
            and traversal_v == original_u
        ):
            return curve[::-1]

        raise RuntimeError(
            f"Traversal ({traversal_u}, "
            f"{traversal_v}) does not match "
            f"physical edge {edge_id}: "
            f"({original_u}, {original_v})."
        )

    def _is_forward_traversal(self, item):
        edge_id = int(item["edge_id"])
        original_u = int(self.edge_index[edge_id, 0])
        original_v = int(self.edge_index[edge_id, 1])
        traversal_u = int(item["u"])
        traversal_v = int(item["v"])
        if traversal_u == original_u and traversal_v == original_v:
            return True
        if traversal_u == original_v and traversal_v == original_u:
            return False
        raise RuntimeError(
            f"Traversal ({traversal_u}, {traversal_v}) does not match "
            f"physical edge {edge_id}: ({original_u}, {original_v})."
        )

    def _directed_half_edge_key(self, item):
        return (int(item["edge_id"]), 1 if self._is_forward_traversal(item) else -1)

    def _item_from_directed_half_edge_key(self, key):
        edge_id, direction = key
        u, v = self.edge_index[int(edge_id)]
        if int(direction) >= 0:
            return {"u": int(u), "v": int(v), "edge_id": int(edge_id), "fibre_id": 0, "key": None}
        return {"u": int(v), "v": int(u), "edge_id": int(edge_id), "fibre_id": 0, "key": None}

    def _build_transition_table(self, threshold_degrees=60.0):
        threshold = float(threshold_degrees)
        table = {}
        keys = []
        for edge_id in range(len(self.edge_index)):
            keys.append((edge_id, 1))
            keys.append((edge_id, -1))
        for first_key in keys:
            first = self._item_from_directed_half_edge_key(first_key)
            for second_key in keys:
                second = self._item_from_directed_half_edge_key(second_key)
                angle = self._transition_angle_uncached(first, second)
                huang_cost = max(angle - threshold, 0.0) ** 2
                table[(first_key, second_key)] = {
                    "angle": float(angle),
                    "huang_cost": float(huang_cost),
                    "above_60": bool(angle > threshold),
                    "above_90": bool(angle > 90.0),
                    "above_120": bool(angle > 120.0),
                    "above_150": bool(angle > 150.0),
                    "immediate_backtrack": bool(first_key[0] == second_key[0]),
                }
        self._transition_table = {
            "threshold_degrees": threshold,
            "table": table,
        }
        return self._transition_table

    def _transition_record(self, first_item, second_item, threshold_degrees=60.0):
        if (
            self._transition_table is None
            or self._transition_table["threshold_degrees"] != float(threshold_degrees)
        ):
            self._build_transition_table(threshold_degrees=threshold_degrees)
        key = (
            self._directed_half_edge_key(first_item),
            self._directed_half_edge_key(second_item),
        )
        return self._transition_table["table"][key]

    def _get_oriented_edge_array(self, item, arrays, name):
        if arrays is None:
            return None
        edge_id = int(item["edge_id"])
        values = arrays[edge_id]
        if self._is_forward_traversal(item):
            return values
        return values[::-1]

    def get_oriented_normals(self, item):
        """Return supplied CAD normals in traversal order, or None."""
        normals = self._get_oriented_edge_array(
            item, self.edge_normals_xyz, "edge_normals_xyz"
        )
        if normals is None:
            return None
        magnitudes = np.linalg.norm(normals, axis=1)
        return normals / magnitudes[:, None]

    def get_oriented_pass_curve(self, item):
        """Return per-fibre pass geometry when supplied, otherwise centreline."""
        if self.edge_pass_curves_xyz is None:
            return self.get_oriented_curve(item)
        edge_id = int(item["edge_id"])
        lane = int(item.get("offset_lane", item["fibre_id"]))
        curve = self.edge_pass_curves_xyz[edge_id][lane]
        if self._is_forward_traversal(item):
            return curve
        return curve[::-1]

    def get_oriented_pass_normals(self, item):
        """Return per-fibre pass normals when supplied, otherwise CAD normals."""
        if self.edge_pass_normals_xyz is None:
            return self.get_oriented_normals(item)
        edge_id = int(item["edge_id"])
        lane = int(item.get("offset_lane", item["fibre_id"]))
        normals = self.edge_pass_normals_xyz[edge_id][lane]
        if not self._is_forward_traversal(item):
            normals = normals[::-1]
        magnitudes = np.linalg.norm(normals, axis=1)
        return normals / magnitudes[:, None]

    def get_oriented_pass_uv(self, item):
        """Return per-lane pass UV samples in traversal order when available."""
        if self.edge_pass_curves_uv is None:
            return None
        edge_id = int(item["edge_id"])
        lane = int(item.get("offset_lane", item["fibre_id"]))
        uv = self.edge_pass_curves_uv[edge_id][lane]
        if self._is_forward_traversal(item):
            return uv
        return uv[::-1]

    def get_oriented_surface_positions(self, item):
        """Return supplied CAD-surface positions in traversal order, or None."""
        return self._get_oriented_edge_array(
            item, self.edge_surface_positions, "edge_surface_positions"
        )

    def get_oriented_stress_directions(self, item):
        """Return supplied stress directions in traversal order, or None."""
        values = self._get_oriented_edge_array(
            item, self.edge_stress_directions_xyz, "edge_stress_directions_xyz"
        )
        if values is None:
            return None
        magnitudes = np.linalg.norm(values, axis=1)
        valid = magnitudes > 1.0e-12
        unit = np.zeros_like(values)
        unit[valid] = values[valid] / magnitudes[valid, None]
        return unit

    @staticmethod
    def _curve_tangents(points, tolerance=1.0e-12):
        """Compute per-point travel tangents from an ordered polyline."""
        points = np.asarray(points, dtype=np.float64)
        tangents = np.zeros_like(points)
        if len(points) < 2:
            return tangents
        for index in range(len(points)):
            if index == 0:
                candidates = range(1, len(points))
                origin = points[0]
                sign = 1.0
            elif index == len(points) - 1:
                candidates = range(len(points) - 2, -1, -1)
                origin = points[-1]
                sign = -1.0
            else:
                before = None
                after = None
                for left in range(index - 1, -1, -1):
                    vector = points[index] - points[left]
                    if np.linalg.norm(vector) > tolerance:
                        before = vector
                        break
                for right in range(index + 1, len(points)):
                    vector = points[right] - points[index]
                    if np.linalg.norm(vector) > tolerance:
                        after = vector
                        break
                vectors = [value for value in (before, after) if value is not None]
                if vectors:
                    tangent = np.sum(
                        [value / np.linalg.norm(value) for value in vectors],
                        axis=0,
                    )
                    norm = np.linalg.norm(tangent)
                    if norm > tolerance:
                        tangents[index] = tangent / norm
                        continue
                candidates = ()
                origin = points[index]
                sign = 1.0
            for candidate in candidates:
                vector = sign * (points[candidate] - origin)
                norm = np.linalg.norm(vector)
                if norm > tolerance:
                    tangents[index] = vector / norm
                    break
        return tangents

    @staticmethod
    def _safe_unit(
        vector,
        tolerance=1.0e-12,
    ):
        vector = np.asarray(
            vector,
            dtype=np.float64,
        )

        magnitude = np.linalg.norm(
            vector
        )

        if magnitude <= tolerance:
            return None

        return vector / magnitude

    def start_tangent(
        self,
        item,
    ):
        """
        Unit tangent leaving the traversal's start node.
        """

        curve = self.get_oriented_curve(
            item
        )

        for point_index in range(
            1,
            len(curve),
        ):
            tangent = self._safe_unit(
                curve[point_index]
                - curve[0]
            )

            if tangent is not None:
                return tangent

        return None

    def end_tangent(
        self,
        item,
    ):
        """
        Unit tangent arriving at the traversal's end node.
        """

        curve = self.get_oriented_curve(
            item
        )

        for point_index in range(
            len(curve) - 2,
            -1,
            -1,
        ):
            tangent = self._safe_unit(
                curve[-1]
                - curve[point_index]
            )

            if tangent is not None:
                return tangent

        return None

    def _transition_angle_uncached(
        self,
        first_item,
        second_item,
    ):
        """
        Turning angle between consecutive traversals.

        0 degrees:
            Straight continuation.

        180 degrees:
            Complete U-turn.
        """

        tangent_in = self.end_tangent(
            first_item
        )

        tangent_out = self.start_tangent(
            second_item
        )

        if (
            tangent_in is None
            or tangent_out is None
        ):
            return 180.0

        cosine = np.clip(
            np.dot(
                tangent_in,
                tangent_out,
            ),
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(cosine)
            )
        )

    def transition_angle(
        self,
        first_item,
        second_item,
        threshold_degrees=60.0,
    ):
        """
        Turning angle between consecutive directed half-edges.

        Values are served from the precomputed directed half-edge transition
        table so construction, refinement, diagnostics, and scoring use the
        same incoming-to-outgoing geometry.
        """
        return self._transition_record(
            first_item, second_item, threshold_degrees=threshold_degrees
        )["angle"]

    @staticmethod
    def _angle_cost(angle, angle_weight, turn_120_weight, turn_150_weight):
        """Use the same angle penalty for construction and final scoring."""
        return (
            angle_weight * angle
            + turn_120_weight * max(angle - 120.0, 0.0) ** 2
            + turn_150_weight * max(angle - 150.0, 0.0) ** 2
        )

    @staticmethod
    def _huang_cost(angle, threshold_degrees=60.0):
        return max(float(angle) - float(threshold_degrees), 0.0) ** 2

    @staticmethod
    def _normalise_turn_objective_name(turn_objective):
        if turn_objective is None:
            return "legacy"
        if turn_objective == "thresholded_turn_energy":
            warnings.warn(
                "turn_objective='huang_2023' is deprecated; use "
                "'thresholded_turn_energy'. The objective is article-inspired "
                "and does not reproduce the article's routing solver.",
                DeprecationWarning,
                stacklevel=3,
            )
            return "thresholded_turn_energy"
        return str(turn_objective)

    # =========================================================
    # Route validation
    # =========================================================

    def validate_route(
        self,
        route=None,
        geometric_tolerance=1.0e-5,
        raise_on_error=False,
        verbose=True,
    ):
        """
        Validate:
        1. traversal count,
        2. topological continuity,
        3. closed-loop continuity,
        4. fibre-instance uniqueness,
        5. physical edge multiplicity,
        6. geometric endpoint continuity.
        """

        if route is None:
            route = self.euler_edges

        if route is None:
            raise RuntimeError(
                "No Euler route is available."
            )

        if len(route) == 0:
            raise ValueError("An Euler route must contain at least one traversal.")

        if geometric_tolerance < 0:
            raise ValueError("geometric_tolerance must be nonnegative.")

        expected_traversals = (
            len(self.edge_index)
            * self.fibres_per_edge
        )

        traversal_count_valid = (
            len(route)
            == expected_traversals
        )

        topology_errors = []

        geometric_gaps = []

        for index in range(
            len(route)
        ):
            current_item = route[
                index
            ]

            next_item = route[
                (index + 1) % len(route)
            ]

            if (
                int(current_item["v"])
                != int(next_item["u"])
            ):
                topology_errors.append(
                    index
                )

            current_curve = (
                self.get_oriented_curve(
                    current_item
                )
            )

            next_curve = (
                self.get_oriented_curve(
                    next_item
                )
            )

            gap = np.linalg.norm(
                current_curve[-1]
                - next_curve[0]
            )

            geometric_gaps.append(
                float(gap)
            )

        instance_pairs = [
            (
                int(item["edge_id"]),
                int(item["fibre_id"]),
            )
            for item in route
        ]

        unique_instance_pairs = set(
            instance_pairs
        )

        expected_pairs = {
            (edge_id, fibre_id)
            for edge_id in range(len(self.edge_index))
            for fibre_id in range(self.fibres_per_edge)
        }
        unique_instances_valid = (
            len(instance_pairs) == len(unique_instance_pairs)
            and unique_instance_pairs == expected_pairs
        )

        visit_counts = np.zeros(
            len(self.edge_index),
            dtype=np.int64,
        )

        for item in route:
            visit_counts[
                int(item["edge_id"])
            ] += 1

        multiplicity_valid = np.all(
            visit_counts
            == self.fibres_per_edge
        )

        geometric_gaps = np.asarray(
            geometric_gaps,
            dtype=np.float64,
        )

        geometry_valid = np.all(
            geometric_gaps
            <= geometric_tolerance
        )

        topology_valid = (
            len(topology_errors) == 0
        )

        valid = all(
            [
                traversal_count_valid,
                topology_valid,
                unique_instances_valid,
                multiplicity_valid,
                geometry_valid,
            ]
        )

        if verbose:
            print(
                f"Route traversals       : "
                f"{len(route)} / "
                f"{expected_traversals}"
            )
            print(
                f"Topology errors        : "
                f"{len(topology_errors)}"
            )
            print(
                f"Unique fibre instances : "
                f"{len(unique_instance_pairs)} / "
                f"{expected_traversals}"
            )
            print(
                f"Multiplicity valid     : "
                f"{multiplicity_valid}"
            )
            print(
                f"Maximum endpoint gap   : "
                f"{geometric_gaps.max():.8f}"
            )
            print(
                f"Route valid            : "
                f"{valid}"
            )

        if raise_on_error and not valid:
            reasons = []
            if not traversal_count_valid:
                reasons.append(
                    f"traversal count {len(route)} != {expected_traversals}"
                )
            if not topology_valid:
                reasons.append(
                    f"{len(topology_errors)} topological discontinuities "
                    f"(first at traversal {topology_errors[0]})"
                )
            if not unique_instances_valid:
                reasons.append("missing or repeated fibre instances")
            if not multiplicity_valid:
                reasons.append("incorrect physical-edge multiplicity")
            if not geometry_valid:
                worst = int(np.argmax(geometric_gaps))
                current = route[worst]
                following = route[(worst + 1) % len(route)]
                reasons.append(
                    f"endpoint gap {geometric_gaps[worst]:.8g} exceeds "
                    f"tolerance {geometric_tolerance:.8g} at traversal "
                    f"{worst}: edge {current['edge_id']} -> "
                    f"edge {following['edge_id']} "
                    f"(node {current['v']})"
                )
            raise RuntimeError(
                "The candidate route is not a valid "
                "closed Euler circuit: " + "; ".join(reasons)
            )

        return {
            "valid": valid,
            "traversal_count_valid":
                traversal_count_valid,
            "topology_valid":
                topology_valid,
            "topology_errors":
                topology_errors,
            "unique_instances_valid":
                unique_instances_valid,
            "multiplicity_valid":
                multiplicity_valid,
            "visit_counts":
                visit_counts,
            "geometry_valid":
                geometry_valid,
            "geometric_gaps":
                geometric_gaps,
        }

    # =========================================================
    # Route diagnostics
    # =========================================================

    def compute_turn_angles(
        self,
        route=None,
        include_closure=True,
        verbose=True,
    ):
        """
        Calculate orientation-aware turn angles.
        """

        if route is None:
            route = self.euler_edges

        if route is None:
            raise RuntimeError(
                "No Euler route is available."
            )

        if len(route) < 2:
            return np.empty(
                0,
                dtype=np.float64,
            )

        angles = []

        for index in range(
            len(route) - 1
        ):
            angles.append(
                self.transition_angle(
                    route[index],
                    route[index + 1],
                )
            )

        if include_closure:
            angles.append(
                self.transition_angle(
                    route[-1],
                    route[0],
                )
            )

        angles = np.asarray(
            angles,
            dtype=np.float64,
        )

        if verbose:
            print(
                f"Number of transitions : "
                f"{len(angles)}"
            )
            print(
                f"Mean turn angle       : "
                f"{angles.mean():.2f} degrees"
            )
            print(
                f"Median turn angle     : "
                f"{np.median(angles):.2f} degrees"
            )
            print(
                f"Maximum turn angle    : "
                f"{angles.max():.2f} degrees"
            )
            print(
                f"Turns above 90 deg    : "
                f"{np.sum(angles > 90.0)}"
            )
            print(
                f"Turns above 120 deg   : "
                f"{np.sum(angles > 120.0)}"
            )
            print(
                f"Turns above 150 deg   : "
                f"{np.sum(angles > 150.0)}"
            )

        return angles

    @staticmethod
    def _same_physical_edge(
        first_item,
        second_item,
    ):
        return (
            int(first_item["edge_id"])
            == int(second_item["edge_id"])
        )

    def count_immediate_backtracks(
        self,
        route=None,
        include_closure=True,
        verbose=True,
    ):
        """
        Count consecutive traversals of the same physical edge.
        """

        if route is None:
            route = self.euler_edges

        count = 0
        indices = []

        pair_count = (
            len(route)
            if include_closure
            else len(route) - 1
        )

        for index in range(
            pair_count
        ):
            first_item = route[index]

            second_item = route[
                (index + 1) % len(route)
            ]

            if self._same_physical_edge(
                first_item,
                second_item,
            ):
                count += 1
                indices.append(index)

        if verbose:
            print(
                "Immediate same-edge "
                f"backtracks: {count}"
            )

        return count, indices

    def assign_route_lanes(self, route=None):
        """
        Deterministically map each traversal to one unique offset lane.

        For every physical edge, the Euler route contains ``fibres_per_edge``
        occurrences. This assigns those occurrences bijectively to the
        available surface-offset lanes using actual lane endpoint positions.
        """
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        occurrences = {}
        for traversal_id, item in enumerate(route):
            occurrences.setdefault(int(item["edge_id"]), []).append(int(traversal_id))
        for edge_id in range(len(self.edge_index)):
            if len(occurrences.get(edge_id, [])) != self.fibres_per_edge:
                raise ManufacturingGeometryError(
                    f"Physical edge {edge_id} occurs {len(occurrences.get(edge_id, []))} times; "
                    f"expected {self.fibres_per_edge}."
                )

        lane_by_traversal = {}
        # Deterministic initial assignment by route occurrence order.
        for edge_id, traversal_ids in occurrences.items():
            for lane, traversal_id in enumerate(sorted(traversal_ids)):
                lane_by_traversal[int(traversal_id)] = int(lane)

        def oriented_lane_curve(item, lane):
            if self.edge_pass_curves_xyz is None:
                return self.get_oriented_curve(item)
            edge_id = int(item["edge_id"])
            curve = self.edge_pass_curves_xyz[edge_id][int(lane)]
            return curve if self._is_forward_traversal(item) else curve[::-1]

        def local_assignment_cost(traversal_id, lane):
            item = route[traversal_id]
            curve = oriented_lane_curve(item, lane)
            prev_id = (traversal_id - 1) % len(route)
            next_id = (traversal_id + 1) % len(route)
            prev_lane = lane_by_traversal.get(prev_id, 0)
            next_lane = lane_by_traversal.get(next_id, 0)
            prev_curve = oriented_lane_curve(route[prev_id], prev_lane)
            next_curve = oriented_lane_curve(route[next_id], next_lane)
            incoming_tangent = self._safe_unit(curve[1] - curve[0]) if len(curve) > 1 else None
            outgoing_tangent = self._safe_unit(curve[-1] - curve[-2]) if len(curve) > 1 else None
            prev_tangent = self._safe_unit(prev_curve[-1] - prev_curve[-2]) if len(prev_curve) > 1 else None
            next_tangent = self._safe_unit(next_curve[1] - next_curve[0]) if len(next_curve) > 1 else None
            gap_in = float(np.linalg.norm(prev_curve[-1] - curve[0]))
            gap_out = float(np.linalg.norm(curve[-1] - next_curve[0]))
            angles = []
            for a, b in ((prev_tangent, incoming_tangent), (outgoing_tangent, next_tangent)):
                if a is not None and b is not None:
                    angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))))
            max_turn = max(angles) if angles else 180.0
            hard_violations = sum(angle > 120.0 for angle in angles)
            immediate_reversal = int(any(angle > 175.0 for angle in angles))
            min_radius = np.inf
            for a, b, c in ((prev_curve[-1], curve[0], curve[1]), (curve[-2], curve[-1], next_curve[0])):
                min_radius = min(min_radius, self._circumradius(a, b, c))
            radius_penalty = 0.0 if np.isinf(min_radius) else 1.0 / max(min_radius, 1.0e-12)
            return (
                immediate_reversal,
                hard_violations,
                max_turn,
                gap_in + gap_out,
                radius_penalty,
            )

        def assignment_total_cost(traversal_ids=None):
            ids = range(len(route)) if traversal_ids is None else traversal_ids
            total = np.zeros(5, dtype=np.float64)
            for tid in ids:
                total += np.asarray(local_assignment_cost(tid, lane_by_traversal[tid]), dtype=np.float64)
            return tuple(total.tolist())

        for _ in range(2):
            for edge_id, traversal_ids in sorted(occurrences.items()):
                lanes = list(range(self.fibres_per_edge))
                best_perm = None
                best_cost = (np.inf, np.inf, np.inf, np.inf, np.inf)
                if self.fibres_per_edge <= 8:
                    permutations = itertools.permutations(lanes)
                else:
                    permutations = [tuple(lanes)]
                for perm in permutations:
                    old = {tid: lane_by_traversal[tid] for tid in traversal_ids}
                    for tid, lane in zip(traversal_ids, perm):
                        lane_by_traversal[tid] = int(lane)
                    cost = assignment_total_cost(traversal_ids)
                    if cost < best_cost:
                        best_cost = cost
                        best_perm = tuple(perm)
                    for tid, lane in old.items():
                        lane_by_traversal[tid] = lane
                for tid, lane in zip(traversal_ids, best_perm):
                    lane_by_traversal[tid] = int(lane)
                if self.fibres_per_edge > 8:
                    improved = True
                    while improved:
                        improved = False
                        for a, b in itertools.combinations(traversal_ids, 2):
                            old_cost = assignment_total_cost()
                            lane_by_traversal[a], lane_by_traversal[b] = lane_by_traversal[b], lane_by_traversal[a]
                            new_cost = assignment_total_cost()
                            if new_cost < old_cost:
                                improved = True
                            else:
                                lane_by_traversal[a], lane_by_traversal[b] = lane_by_traversal[b], lane_by_traversal[a]

        mapping = []
        edge_visit_counts = {}
        for traversal_id, item in enumerate(route):
            edge_id = int(item["edge_id"])
            direction = 1 if self._is_forward_traversal(item) else -1
            edge_visit_counts[edge_id] = edge_visit_counts.get(edge_id, 0) + 1
            visit_number = edge_visit_counts[edge_id]
            offset_lane = int(lane_by_traversal[traversal_id])
            signed_offset = None
            if self.edge_pass_offsets is not None:
                signed_offset = float(self.edge_pass_offsets[edge_id][offset_lane])
            entry = {
                "traversal_id": int(traversal_id),
                "physical_edge_id": edge_id,
                "edge_id": edge_id,
                "fibre_id": int(item.get("fibre_id", offset_lane)),
                "fibre_instance_id": int(item.get("fibre_id", offset_lane)),
                "direction": "forward" if direction > 0 else "reverse",
                "visit_number": int(visit_number),
                "offset_lane": int(offset_lane),
                "lane_id": int(offset_lane),
                "signed_offset": signed_offset,
            }
            mapping.append(entry)
        for edge_id, traversal_ids in occurrences.items():
            lanes = sorted(mapping[tid]["offset_lane"] for tid in traversal_ids)
            if lanes != list(range(self.fibres_per_edge)):
                raise ManufacturingGeometryError(f"Lane assignment for edge {edge_id} is not bijective.")
        return mapping

    def route_with_lane_assignment(self, route=None):
        if route is None:
            route = self.euler_edges
        mapping = self.assign_route_lanes(route)
        result = []
        for item, lane in zip(route, mapping):
            enriched = dict(item)
            enriched.update({
                "traversal_id": lane["traversal_id"],
                "offset_lane": lane["offset_lane"],
                "lane_id": lane["lane_id"],
                "signed_offset": lane["signed_offset"],
                "direction": lane["direction"],
                "visit_number": lane["visit_number"],
                "fibre_instance_id": lane["fibre_instance_id"],
            })
            result.append(enriched)
        return result

    def _transition_normals_from_mesh_or_linear(self, points, start_normal, end_normal, surface_vertices=None, surface_faces=None):
        if surface_vertices is not None and surface_faces is not None:
            return _nearest_mesh_normals(points, surface_vertices, surface_faces)
        alpha = np.linspace(0.0, 1.0, len(points))[:, None]
        normals = (1.0 - alpha) * start_normal + alpha * end_normal
        return _normalise_rows(normals, name="junction transition normals")

    @staticmethod
    def _polyline_segment_lengths(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 2:
            return np.asarray([], dtype=np.float64)
        return np.linalg.norm(np.diff(points, axis=0), axis=1)

    @staticmethod
    def _polyline_arc_length(points):
        return float(np.sum(EulerCCFToolpath._polyline_segment_lengths(points)))

    @staticmethod
    def _dedupe_consecutive_points(points, tolerance=1.0e-9):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return points
        kept = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - kept[-1]) > tolerance:
                kept.append(point)
        return np.asarray(kept, dtype=np.float64)

    @staticmethod
    def _resample_polyline_by_arclength(points, target_spacing=None, sample_count=None, tolerance=1.0e-12):
        points = EulerCCFToolpath._dedupe_consecutive_points(points, tolerance=tolerance)
        if len(points) <= 1:
            return points
        lengths = EulerCCFToolpath._polyline_segment_lengths(points)
        total = float(np.sum(lengths))
        if total <= tolerance:
            return points[:1].copy()
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        if sample_count is None:
            if target_spacing is None or target_spacing <= tolerance:
                sample_count = max(2, int(np.ceil(total / max(tolerance, total / 32.0))) + 1)
            else:
                sample_count = max(2, int(np.ceil(total / float(target_spacing))) + 1)
        targets = np.linspace(0.0, total, int(max(2, sample_count)))
        result = []
        seg_id = 0
        for target in targets:
            while seg_id < len(lengths) - 1 and cumulative[seg_id + 1] < target:
                seg_id += 1
            denom = max(lengths[seg_id], tolerance)
            alpha = (target - cumulative[seg_id]) / denom
            result.append((1.0 - alpha) * points[seg_id] + alpha * points[seg_id + 1])
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _resample_polyline_with_attributes_by_arclength(points, attributes=None, sample_count=None, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        attrs = [] if attributes is None else [np.asarray(values, dtype=np.float64) for values in attributes]
        if len(points) <= 1:
            return points.copy(), [values.copy() for values in attrs]
        lengths = EulerCCFToolpath._polyline_segment_lengths(points)
        total = float(np.sum(lengths))
        if total <= tolerance:
            return points[:1].copy(), [values[:1].copy() for values in attrs]
        sample_count = int(max(2, sample_count if sample_count is not None else len(points)))
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        targets = np.linspace(0.0, total, sample_count)
        resampled_points = []
        resampled_attrs = [[] for _ in attrs]
        seg_id = 0
        for target in targets:
            while seg_id < len(lengths) - 1 and cumulative[seg_id + 1] < target:
                seg_id += 1
            denom = max(lengths[seg_id], tolerance)
            alpha = (target - cumulative[seg_id]) / denom
            resampled_points.append((1.0 - alpha) * points[seg_id] + alpha * points[seg_id + 1])
            for attr_id, values in enumerate(attrs):
                resampled_attrs[attr_id].append((1.0 - alpha) * values[seg_id] + alpha * values[seg_id + 1])
        return (
            np.asarray(resampled_points, dtype=np.float64),
            [np.asarray(values, dtype=np.float64) for values in resampled_attrs],
        )

    @staticmethod
    def _dedupe_corresponding_xyz_uv(xyz, uv, tolerance=1.0e-9):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if len(xyz) != len(uv):
            raise ManufacturingGeometryError("XYZ/UV arrays must have matching lengths.")
        if len(xyz) == 0:
            return xyz, uv
        kept_xyz = [xyz[0]]
        kept_uv = [uv[0]]
        for point, uv_point in zip(xyz[1:], uv[1:]):
            if np.linalg.norm(point - kept_xyz[-1]) > tolerance:
                kept_xyz.append(point)
                kept_uv.append(uv_point)
        return np.asarray(kept_xyz, dtype=np.float64), np.asarray(kept_uv, dtype=np.float64)

    @staticmethod
    def _trim_polyline_end(points, trim_distance, from_end=True, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            points = points.reshape(-1, points.shape[-1] if points.size else 1)
        if len(points) < 2 or trim_distance <= tolerance:
            return points.copy()
        if not from_end:
            return EulerCCFToolpath._trim_polyline_end(points[::-1], trim_distance, from_end=True, tolerance=tolerance)[::-1]
        remaining = float(trim_distance)
        keep = points.copy()
        while len(keep) >= 2:
            segment = keep[-1] - keep[-2]
            length = float(np.linalg.norm(segment))
            if length <= tolerance:
                keep = keep[:-1]
                continue
            if remaining < length:
                keep[-1] = keep[-1] - (remaining / length) * segment
                return keep
            remaining -= length
            keep = keep[:-1]
            if remaining <= tolerance:
                return keep
        return points[:1].copy()

    @staticmethod
    def _trim_location_from_start(points, trim_distance, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or len(points) == 0:
            raise ManufacturingGeometryError("Cannot trim an empty polyline.")
        if len(points) == 1 or trim_distance <= tolerance:
            return 0, 0.0, points[0].copy()
        remaining = float(trim_distance)
        for index in range(len(points) - 1):
            segment = points[index + 1] - points[index]
            length = float(np.linalg.norm(segment))
            if length <= tolerance:
                continue
            if remaining <= length:
                alpha = remaining / length
                return index, float(alpha), (1.0 - alpha) * points[index] + alpha * points[index + 1]
            remaining -= length
        return len(points) - 2, 1.0, points[-1].copy()

    @staticmethod
    def _trim_polyline_between(points, start_distance=0.0, end_distance=0.0, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or len(points) < 2:
            raise ManufacturingGeometryError("Cannot trim a degenerate lane curve.")
        total = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if float(start_distance) + float(end_distance) >= total - tolerance:
            raise ManufacturingGeometryError(
                "Combined junction trim distances consume or collapse a manufactured lane."
            )
        start_index, start_alpha, start_point = EulerCCFToolpath._trim_location_from_start(points, start_distance, tolerance)
        end_from_start = total - float(end_distance)
        end_index, end_alpha, end_point = EulerCCFToolpath._trim_location_from_start(points, end_from_start, tolerance)
        trimmed = [start_point]
        first_full = start_index + 1
        last_full = end_index
        for index in range(first_full, last_full + 1):
            if 0 <= index < len(points):
                trimmed.append(points[index])
        trimmed.append(end_point)
        return EulerCCFToolpath._dedupe_consecutive_points(np.asarray(trimmed, dtype=np.float64), tolerance=tolerance)

    @staticmethod
    def trim_corresponding_xyz_uv(curve_xyz, curve_uv, start_distance=0.0, end_distance=0.0, tolerance=1.0e-12):
        xyz = np.asarray(curve_xyz, dtype=np.float64).reshape(-1, 3)
        uv = np.asarray(curve_uv, dtype=np.float64).reshape(-1, 2)
        if len(xyz) != len(uv):
            raise ManufacturingGeometryError("curve_xyz and curve_uv must have matching sample counts.")
        total = EulerCCFToolpath._polyline_arc_length(xyz)
        if float(start_distance) + float(end_distance) >= total - tolerance:
            raise ManufacturingGeometryError("Combined XYZ/UV trim distances collapse the curve.")
        start_index, start_alpha, start_xyz = EulerCCFToolpath._trim_location_from_start(xyz, start_distance, tolerance)
        end_index, end_alpha, end_xyz = EulerCCFToolpath._trim_location_from_start(xyz, total - float(end_distance), tolerance)
        start_uv = (1.0 - start_alpha) * uv[start_index] + start_alpha * uv[start_index + 1]
        end_uv = (1.0 - end_alpha) * uv[end_index] + end_alpha * uv[end_index + 1]
        xyz_points = [start_xyz]
        uv_points = [start_uv]
        for index in range(start_index + 1, end_index + 1):
            xyz_points.append(xyz[index])
            uv_points.append(uv[index])
        xyz_points.append(end_xyz)
        uv_points.append(end_uv)
        return EulerCCFToolpath._dedupe_corresponding_xyz_uv(
            np.asarray(xyz_points, dtype=np.float64),
            np.asarray(uv_points, dtype=np.float64),
            tolerance=tolerance,
        )

    @staticmethod
    def _trimmed_end_tangent(points, at_end=True, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 2:
            return None
        segment = points[-1] - points[-2] if at_end else points[1] - points[0]
        norm = float(np.linalg.norm(segment))
        return None if norm <= tolerance else segment / norm

    def _transition_geometry_metrics(
        self,
        *,
        unprojected_points,
        projected_points,
        tangent_in,
        tangent_out,
        projection_distances=None,
        transition_tolerance=1.0e-6,
        metric_sample_count=65,
    ):
        unprojected_points = np.asarray(unprojected_points, dtype=np.float64).reshape(-1, 3)
        raw_projected_points = np.asarray(projected_points, dtype=np.float64).reshape(-1, 3)
        unprojected_lengths = self._polyline_segment_lengths(unprojected_points)
        raw_projected_lengths = self._polyline_segment_lengths(raw_projected_points)
        repeated_consecutive = bool(raw_projected_lengths.size and np.any(raw_projected_lengths <= max(transition_tolerance, 1.0e-12)))
        projected_points = self._dedupe_consecutive_points(raw_projected_points, tolerance=transition_tolerance * 0.1)
        projected_lengths = self._polyline_segment_lengths(projected_points)
        total_unprojected = float(np.sum(unprojected_lengths))
        total_projected = float(np.sum(projected_lengths))
        min_projected = float(np.min(projected_lengths)) if projected_lengths.size else 0.0
        max_projected = float(np.max(projected_lengths)) if projected_lengths.size else 0.0
        positive = projected_lengths[projected_lengths > max(transition_tolerance, 1.0e-12)]
        min_positive = float(np.min(positive)) if positive.size else 0.0
        ratio = float(max_projected / min_positive) if min_positive > 0.0 else np.inf
        adjacent_ratios = []
        for first, second in zip(projected_lengths[:-1], projected_lengths[1:]):
            lo = min(float(first), float(second))
            hi = max(float(first), float(second))
            adjacent_ratios.append(np.inf if lo <= transition_tolerance else hi / lo)
        max_adjacent_ratio = float(np.max(adjacent_ratios)) if adjacent_ratios else 1.0
        max_projection = float(np.max(projection_distances)) if projection_distances is not None and len(projection_distances) else 0.0

        resampled = self._resample_polyline_by_arclength(
            projected_points,
            sample_count=max(int(metric_sample_count), 9),
            tolerance=transition_tolerance * 0.1,
        )
        turns = []
        radii = []
        for idx in range(1, len(resampled) - 1):
            v0 = self._safe_unit(resampled[idx] - resampled[idx - 1])
            v1 = self._safe_unit(resampled[idx + 1] - resampled[idx])
            if v0 is not None and v1 is not None:
                turns.append(float(np.degrees(np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0)))))
            radius = self._circumradius(resampled[idx - 1], resampled[idx], resampled[idx + 1])
            if np.isfinite(radius):
                radii.append(float(radius))
        min_radius = float(np.min(radii)) if radii else None
        max_curvature = None if min_radius is None or min_radius <= 0.0 else float(1.0 / min_radius)
        transition_tangents = self._curve_tangents(projected_points) if len(projected_points) >= 2 else np.empty((0, 3))
        start_error = None
        end_error = None
        if len(transition_tangents):
            start_error = float(np.degrees(np.arccos(np.clip(np.dot(tangent_in, transition_tangents[0]), -1.0, 1.0))))
            end_error = float(np.degrees(np.arccos(np.clip(np.dot(transition_tangents[-1], tangent_out), -1.0, 1.0))))
        projection_reversed = False
        if len(unprojected_points) == len(projected_points) and len(projected_points) >= 2:
            unproj_segments = np.diff(unprojected_points, axis=0)
            proj_segments = np.diff(projected_points, axis=0)
            for a_seg, b_seg in zip(unproj_segments, proj_segments):
                a_unit = self._safe_unit(a_seg)
                b_unit = self._safe_unit(b_seg)
                if a_unit is not None and b_unit is not None and float(np.dot(a_unit, b_unit)) < -0.1:
                    projection_reversed = True
                    break
        collapsed = bool(repeated_consecutive or min_projected <= max(transition_tolerance, 1.0e-12))
        extreme_ratio = bool(ratio > 100.0 or max_adjacent_ratio > 50.0)
        tangent_destroyed = bool(
            (start_error is not None and start_error > 75.0)
            or (end_error is not None and end_error > 75.0)
        )
        degeneration = collapsed or extreme_ratio or projection_reversed or tangent_destroyed
        return {
            "points": projected_points,
            "metric_points": resampled,
            "unprojected_segment_lengths": unprojected_lengths,
            "projected_segment_lengths": projected_lengths,
            "transition_arc_length": total_projected,
            "unprojected_arc_length": total_unprojected,
            "minimum_segment_length": min_projected,
            "maximum_segment_length": max_projected,
            "segment_length_ratio": ratio,
            "maximum_adjacent_segment_length_ratio": max_adjacent_ratio,
            "maximum_projection_correction": max_projection,
            "endpoint_tangent_error_degrees": {
                "start": start_error,
                "end": end_error,
                "maximum": max(v for v in (start_error, end_error) if v is not None) if (start_error is not None or end_error is not None) else None,
            },
            "maximum_local_turn": float(np.max(turns)) if turns else 0.0,
            "minimum_local_radius": min_radius,
            "maximum_curvature": max_curvature,
            "projection_reversed": bool(projection_reversed),
            "repeated_consecutive_points": bool(repeated_consecutive),
            "collapsed_projected_segments": bool(collapsed),
            "extreme_segment_length_ratio": bool(extreme_ratio),
            "endpoint_tangency_destroyed": bool(tangent_destroyed),
            "projection_degenerate": bool(degeneration),
        }

    def _manufactured_lane_curve_for_transition(self, route_lanes, transitions, traversal_id, tolerance=1.0e-9):
        transition = transitions[int(traversal_id)]
        if "manufactured_lane_xyz" in transition:
            return np.asarray(transition["manufactured_lane_xyz"], dtype=np.float64)
        item = route_lanes[int(traversal_id)]
        curve = self.get_oriented_pass_curve(item)
        previous_transition = transitions[(int(traversal_id) - 1) % len(transitions)]
        current_transition = transitions[int(traversal_id)]
        start_trim = float(previous_transition.get("trim_distance_from_next_lane", 0.0))
        end_trim = float(current_transition.get("trim_distance_from_previous_lane", 0.0))
        return self._trim_polyline_between(curve, start_trim, end_trim, tolerance=tolerance)

    def _apply_canonical_manufactured_lanes(self, route_lanes, transitions, surface_evaluator=None, tolerance=1.0e-9):
        for traversal_id, item in enumerate(route_lanes):
            if not transitions[traversal_id].get("accepted", True):
                continue
            previous_transition = transitions[(traversal_id - 1) % len(transitions)]
            current_transition = transitions[traversal_id]
            start_trim = float(previous_transition.get("trim_distance_from_next_lane", 0.0))
            end_trim = float(current_transition.get("trim_distance_from_previous_lane", 0.0))
            curve = self.get_oriented_pass_curve(item)
            uv = self.get_oriented_pass_uv(item)
            normals = self.get_oriented_pass_normals(item)
            lane_uv = None
            if surface_evaluator is not None and uv is not None:
                lane_xyz, lane_uv = self.trim_corresponding_xyz_uv(
                    curve,
                    uv,
                    start_distance=start_trim,
                    end_distance=end_trim,
                    tolerance=tolerance,
                )
                evaluated = surface_evaluator(lane_uv)
                if not isinstance(evaluated, dict) or "xyz" not in evaluated:
                    raise ManufacturingGeometryError("surface_evaluator must return {'xyz': ...} for manufactured lane evaluation.")
                lane_xyz = _to_numpy_array(evaluated["xyz"])
                if "normal" in evaluated:
                    lane_normals = _normalise_rows(evaluated["normal"], name="manufactured lane CAD normals")
                elif "Xu" in evaluated and "Xv" in evaluated:
                    lane_normals = _normalise_rows(
                        np.cross(_to_numpy_array(evaluated["Xu"]), _to_numpy_array(evaluated["Xv"])),
                        name="manufactured lane CAD normals",
                    )
                else:
                    lane_normals = self._transition_normals_from_mesh_or_linear(lane_xyz, normals[0], normals[-1], None, None)
            else:
                lane_xyz = self._trim_polyline_between(curve, start_trim, end_trim, tolerance=tolerance)
                lane_normals = self._transition_normals_from_mesh_or_linear(lane_xyz, normals[0], normals[-1], None, None)
            transitions[traversal_id]["manufactured_lane_xyz"] = lane_xyz
            transitions[traversal_id]["manufactured_lane_uv"] = lane_uv
            transitions[traversal_id]["manufactured_lane_normals"] = lane_normals
            if transitions[traversal_id]["points"].size:
                transitions[traversal_id]["points"][0] = lane_xyz[-1]
                transitions[traversal_id]["metric_points"][0] = lane_xyz[-1]
            next_transition = transitions[(traversal_id - 1) % len(transitions)]
            if next_transition.get("accepted", True) and next_transition["points"].size:
                next_transition["points"][-1] = lane_xyz[0]
                next_transition["metric_points"][-1] = lane_xyz[0]

    def build_junction_transitions(
        self,
        route=None,
        surface_evaluator=None,
        surface_vertices=None,
        surface_faces=None,
        transition_samples=5,
        transition_tolerance=1.0e-6,
        min_bend_radius=None,
        max_joined_turn_degrees=None,
        mesh_projection_tolerance=None,
        junction_trim_distance=None,
        transition_metric_samples=65,
    ):
        """
        Build explicit surface-constrained transition samples between offset lanes.

        The construction is a local linear/Bezier-style connection projected to
        the supplied CAD evaluator or triangle mesh when available. It is a
        manufacturing post-processing layer; structural centrelines are not
        modified.
        """
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        if self.edge_pass_curves_xyz is None:
            raise ManufacturingGeometryError("Offset pass curves are required before junction transitions can be built.")
        route_lanes = self.route_with_lane_assignment(route)
        transition_samples = max(int(transition_samples), 2)
        mesh_vertices = mesh_faces = None
        mesh_projector = None
        if surface_vertices is not None or surface_faces is not None:
            mesh_vertices, mesh_faces = coerce_surface_mesh(surface_vertices, surface_faces)
            mesh_projector = MeshSurfaceProjector(mesh_vertices, mesh_faces)
        mesh_projection_limit = None
        if mesh_projection_tolerance is not None:
            mesh_projection_limit = float(mesh_projection_tolerance)
        elif mesh_projector is not None:
            mesh_projection_limit = max(float(mesh_projector.max_triangle_radius), transition_tolerance)
        transitions = []
        violations = []
        for transition_id, item in enumerate(route_lanes):
            following = route_lanes[(transition_id + 1) % len(route_lanes)]
            if int(item["v"]) != int(following["u"]):
                raise ManufacturingGeometryError(
                    "Junction transition requires consecutive routed half-edges: "
                    f"item['v']={item['v']} must equal following['u']={following['u']}."
                )
            junction_node_id = int(item["v"])
            curve_a = self.get_oriented_pass_curve(item)
            curve_b = self.get_oriented_pass_curve(following)
            normals_a = self.get_oriented_pass_normals(item)
            normals_b = self.get_oriented_pass_normals(following)
            lane_a_length = self._polyline_arc_length(curve_a)
            lane_b_length = self._polyline_arc_length(curve_b)
            untrimmed_start = curve_a[-1]
            untrimmed_end = curve_b[0]
            untrimmed_gap = float(np.linalg.norm(untrimmed_end - untrimmed_start))
            max_trim = max(0.0, min(0.25 * lane_a_length, 0.25 * lane_b_length))
            uv_a = self.get_oriented_pass_uv(item)
            uv_b = self.get_oriented_pass_uv(following)
            requested_trim = min(
                max_trim,
                float(junction_trim_distance) if junction_trim_distance is not None else max(
                    2.0 * transition_tolerance,
                    0.5 * untrimmed_gap,
                    0.25 * float(min_bend_radius) if min_bend_radius is not None else 0.0,
                ),
            )
            trim_candidates = sorted({
                float(np.clip(requested_trim * factor, 0.0, max_trim))
                for factor in (0.5, 1.0, 1.5, 2.0)
            })
            if not trim_candidates:
                trim_candidates = [0.0]
            handle_factors = (0.5, 1.0, 1.5, 2.0)
            candidate_stats = {
                "attempted": 0,
                "rejected_projection_degenerate": 0,
                "rejected_joined_turn": 0,
                "rejected_radius": 0,
            }
            valid_candidates = []
            rejected_records = []
            highres_samples = 129
            for trim_distance in trim_candidates:
                try:
                    trimmed_curve_a = self._trim_polyline_between(curve_a, 0.0, trim_distance, tolerance=transition_tolerance * 0.1)
                    trimmed_curve_b = self._trim_polyline_between(curve_b, trim_distance, 0.0, tolerance=transition_tolerance * 0.1)
                except ManufacturingGeometryError as exc:
                    rejected_records.append({"reason": "lane_trim_collapsed", "trim_distance": float(trim_distance), "message": str(exc)})
                    continue
                trimmed_uv_a = None
                trimmed_uv_b = None
                if surface_evaluator is not None and uv_a is not None and uv_b is not None:
                    try:
                        trimmed_curve_a, trimmed_uv_a = self.trim_corresponding_xyz_uv(curve_a, uv_a, 0.0, trim_distance, tolerance=transition_tolerance * 0.1)
                        trimmed_curve_b, trimmed_uv_b = self.trim_corresponding_xyz_uv(curve_b, uv_b, trim_distance, 0.0, tolerance=transition_tolerance * 0.1)
                        eval_a = surface_evaluator(trimmed_uv_a)
                        eval_b = surface_evaluator(trimmed_uv_b)
                        if not isinstance(eval_a, dict) or "xyz" not in eval_a or not isinstance(eval_b, dict) or "xyz" not in eval_b:
                            raise ManufacturingGeometryError("surface_evaluator must return {'xyz': ...} for trimmed lane evaluation.")
                        trimmed_curve_a = _to_numpy_array(eval_a["xyz"])
                        trimmed_curve_b = _to_numpy_array(eval_b["xyz"])
                    except ManufacturingGeometryError as exc:
                        rejected_records.append({"reason": "cad_trim_evaluation_failed", "trim_distance": float(trim_distance), "message": str(exc)})
                        continue
                start = trimmed_curve_a[-1]
                end = trimmed_curve_b[0]
                gap = float(np.linalg.norm(end - start))
                tangent_in = self._trimmed_end_tangent(trimmed_curve_a, at_end=True, tolerance=transition_tolerance * 0.1)
                tangent_out = self._trimmed_end_tangent(trimmed_curve_b, at_end=False, tolerance=transition_tolerance * 0.1)
                if tangent_in is None:
                    tangent_in = self._safe_unit(end - start)
                if tangent_out is None:
                    tangent_out = self._safe_unit(end - start)
                if tangent_in is None:
                    tangent_in = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                if tangent_out is None:
                    tangent_out = tangent_in.copy()
                s = np.linspace(0.0, 1.0, highres_samples)[:, None]
                h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
                h10 = s**3 - 2.0 * s**2 + s
                h01 = -2.0 * s**3 + 3.0 * s**2
                h11 = s**3 - s**2
                for handle_factor in handle_factors:
                    candidate_stats["attempted"] += 1
                    handle = max(gap / 3.0, transition_tolerance) * float(handle_factor)
                    unprojected_points = (
                        h00 * start
                        + h10 * (handle * tangent_in)
                        + h01 * end
                        + h11 * (handle * tangent_out)
                    )
                    points = unprojected_points.copy()
                    uv_points = None
                    method = "cubic_hermite_xyz"
                    projection_distances = np.zeros((len(points),), dtype=np.float64)
                    normals = self._transition_normals_from_mesh_or_linear(points, normals_a[-1], normals_b[0], None, None)
                    domain_valid = True
                    if surface_evaluator is not None and uv_a is not None and uv_b is not None:
                        uv_start = trimmed_uv_a[-1]
                        uv_end = trimmed_uv_b[0]
                        endpoint_eval = surface_evaluator(np.vstack((uv_start, uv_end)))
                        if not isinstance(endpoint_eval, dict) or "Xu" not in endpoint_eval or "Xv" not in endpoint_eval:
                            raise ManufacturingGeometryError("surface_evaluator must return Xu and Xv for tangent-controlled CAD transitions.")
                        xu_endpoint = _to_numpy_array(endpoint_eval["Xu"])
                        xv_endpoint = _to_numpy_array(endpoint_eval["Xv"])
                        j_start = np.column_stack((xu_endpoint[0], xv_endpoint[0]))
                        j_end = np.column_stack((xu_endpoint[1], xv_endpoint[1]))
                        uv_tangent_in = np.linalg.pinv(j_start) @ (handle * tangent_in)
                        uv_tangent_out = np.linalg.pinv(j_end) @ (handle * tangent_out)
                        uv_points = (
                            h00[:, :1] * uv_start
                            + h10[:, :1] * uv_tangent_in
                            + h01[:, :1] * uv_end
                            + h11[:, :1] * uv_tangent_out
                        )
                        evaluated = surface_evaluator(uv_points)
                        if not isinstance(evaluated, dict) or "xyz" not in evaluated:
                            raise ManufacturingGeometryError("surface_evaluator must return {'xyz': ...} for transition projection.")
                        points = _to_numpy_array(evaluated["xyz"])
                        saw_inside_mask = False
                        for key in ("inside", "inside_mask", "is_inside", "trimmed_inside"):
                            if key in evaluated:
                                saw_inside_mask = True
                                domain_valid = bool(np.all(_to_numpy_array(evaluated[key], dtype=bool)))
                                break
                        projection_distances = np.linalg.norm(points - unprojected_points, axis=1)
                        if mesh_projector is not None:
                            projected_points, projected_normals, mesh_distances = mesh_projector.project(points)
                            projection_distances = np.maximum(projection_distances, mesh_distances)
                            if not saw_inside_mask:
                                points = projected_points
                                normals = projected_normals
                                domain_valid = bool(np.max(mesh_distances) <= mesh_projection_limit)
                        elif not saw_inside_mask:
                            domain_valid = False
                        if not (mesh_projector is not None and not saw_inside_mask):
                            if "normal" in evaluated:
                                normals = _normalise_rows(evaluated["normal"], name="junction CAD normals")
                            elif "Xu" in evaluated and "Xv" in evaluated:
                                normals = _normalise_rows(np.cross(_to_numpy_array(evaluated["Xu"]), _to_numpy_array(evaluated["Xv"])), name="junction CAD normals")
                        method = "cad_uv_transition"
                    elif mesh_vertices is not None:
                        points, normals, distances = mesh_projector.project(points)
                        projection_distances = distances
                        method = "mesh_projected_cubic_hermite"
                    metrics = self._transition_geometry_metrics(
                        unprojected_points=unprojected_points,
                        projected_points=points,
                        tangent_in=tangent_in,
                        tangent_out=tangent_out,
                        projection_distances=projection_distances,
                        transition_tolerance=transition_tolerance,
                        metric_sample_count=transition_metric_samples,
                    )
                    metric_points = metrics["metric_points"]
                    joined = np.vstack((trimmed_curve_a[-2:], metric_points, trimmed_curve_b[:2]))
                    joined_turns = []
                    joined_radii = []
                    for idx in range(1, len(joined) - 1):
                        v0 = self._safe_unit(joined[idx] - joined[idx - 1])
                        v1 = self._safe_unit(joined[idx + 1] - joined[idx])
                        if v0 is not None and v1 is not None:
                            joined_turns.append(float(np.degrees(np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0)))))
                        radius = self._circumradius(joined[idx - 1], joined[idx], joined[idx + 1])
                        if np.isfinite(radius):
                            joined_radii.append(float(radius))
                    max_joined_turn = float(np.max(joined_turns)) if joined_turns else 0.0
                    min_joined_radius = float(np.min(joined_radii)) if joined_radii else None
                    if (not domain_valid) or metrics["projection_degenerate"]:
                        candidate_stats["rejected_projection_degenerate"] += 1
                        rejected_records.append({"reason": "projection_degenerate", "trim_distance": float(trim_distance), "handle_factor": float(handle_factor)})
                        continue
                    turn_limit = 179.0 if max_joined_turn_degrees is None else min(179.0, float(max_joined_turn_degrees))
                    if max_joined_turn > turn_limit + transition_tolerance:
                        candidate_stats["rejected_joined_turn"] += 1
                        rejected_records.append({"reason": "joined_turn", "maximum_joined_turn_angle": max_joined_turn})
                        continue
                    if min_bend_radius is not None and min_joined_radius is not None and min_joined_radius < float(min_bend_radius) - transition_tolerance:
                        candidate_stats["rejected_radius"] += 1
                        rejected_records.append({"reason": "radius", "minimum_joined_radius": min_joined_radius})
                        continue
                    attr_inputs = []
                    if uv_points is not None and len(uv_points) == len(metrics["points"]):
                        attr_inputs.append(uv_points)
                    else:
                        uv_points = None
                    if normals is not None and len(normals) == len(metrics["points"]):
                        attr_inputs.append(normals)
                    export_points, export_attrs = self._resample_polyline_with_attributes_by_arclength(
                        metrics["points"],
                        attr_inputs,
                        sample_count=transition_samples,
                        tolerance=transition_tolerance * 0.1,
                    )
                    attr_cursor = 0
                    export_uv = None
                    if uv_points is not None:
                        export_uv = export_attrs[attr_cursor]
                        attr_cursor += 1
                    if normals is not None and len(normals) == len(metrics["points"]):
                        export_normals = _normalise_rows(export_attrs[attr_cursor], name="resampled transition normals")
                    else:
                        export_normals = self._transition_normals_from_mesh_or_linear(export_points, normals_a[-1], normals_b[0], mesh_vertices, mesh_faces)
                    tangent_error = metrics["endpoint_tangent_error_degrees"]["maximum"] or 0.0
                    radius_rank = -min_joined_radius if min_joined_radius is not None else np.inf
                    valid_candidates.append({
                        "rank": (max_joined_turn, radius_rank, tangent_error, metrics["transition_arc_length"], metrics["maximum_projection_correction"]),
                        "trim_distance": float(trim_distance),
                        "handle_factor": float(handle_factor),
                        "gap": gap,
                        "points": export_points,
                        "normals": export_normals,
                        "uv": export_uv,
                        "method": method,
                        "metrics": metrics,
                        "joined_turns": np.asarray(joined_turns, dtype=np.float64),
                        "max_joined_turn": max_joined_turn,
                        "min_joined_radius": min_joined_radius,
                        "trimmed_curve_a": trimmed_curve_a,
                        "trimmed_curve_b": trimmed_curve_b,
                    })
            if not valid_candidates:
                failed = {
                    "transition_id": int(transition_id),
                    "from_traversal_id": int(transition_id),
                    "to_traversal_id": int((transition_id + 1) % len(route_lanes)),
                    "edge_ids": [int(item["edge_id"]), int(following["edge_id"])],
                    "junction_node_id": junction_node_id,
                    "points": np.empty((0, 3), dtype=np.float64),
                    "normals": np.empty((0, 3), dtype=np.float64),
                    "uv": None,
                    "gap": untrimmed_gap,
                    "untrimmed_gap": untrimmed_gap,
                    "method": "failed_no_valid_transition",
                    "accepted": False,
                    "failure_reason": "no_valid_transition",
                    "candidate_statistics": candidate_stats,
                    "rejected_candidates": rejected_records,
                    "closing": bool(transition_id == len(route_lanes) - 1),
                    "trim_distance_from_previous_lane": 0.0,
                    "trim_distance_from_next_lane": 0.0,
                }
                transitions.append(failed)
                violations.append({
                    "kind": "junction_transition_no_valid_candidate",
                    "transition_id": int(transition_id),
                    "edge_ids": failed["edge_ids"],
                    "measured_value": 0.0,
                    "limit": 1.0,
                    "action": "offset_geometry",
                    "candidate_statistics": candidate_stats,
                })
                continue
            selected = sorted(valid_candidates, key=lambda item: item["rank"])[0]
            metrics = selected["metrics"]
            transitions.append({
                "transition_id": int(transition_id),
                "from_traversal_id": int(transition_id),
                "to_traversal_id": int((transition_id + 1) % len(route_lanes)),
                "edge_ids": [int(item["edge_id"]), int(following["edge_id"])],
                "junction_node_id": junction_node_id,
                "points": selected["points"],
                "normals": selected["normals"],
                "uv": selected["uv"],
                "gap": selected["gap"],
                "untrimmed_gap": untrimmed_gap,
                "method": selected["method"],
                "accepted": True,
                "closing": bool(transition_id == len(route_lanes) - 1),
                "start_tangent_angle": metrics["endpoint_tangent_error_degrees"]["start"],
                "end_tangent_angle": metrics["endpoint_tangent_error_degrees"]["end"],
                "joined_turn_angles": selected["joined_turns"],
                "maximum_joined_turn_angle": selected["max_joined_turn"],
                "minimum_realised_bend_radius": selected["min_joined_radius"],
                "trim_distance_from_previous_lane": selected["trim_distance"],
                "trim_distance_from_next_lane": selected["trim_distance"],
                "trimmed_previous_curve": selected["trimmed_curve_a"],
                "trimmed_next_curve": selected["trimmed_curve_b"],
                "transition_arc_length": metrics["transition_arc_length"],
                "minimum_segment_length": metrics["minimum_segment_length"],
                "maximum_segment_length": metrics["maximum_segment_length"],
                "segment_length_ratio": metrics["segment_length_ratio"],
                "endpoint_tangent_error_degrees": metrics["endpoint_tangent_error_degrees"],
                "maximum_local_turn": metrics["maximum_local_turn"],
                "minimum_local_radius": metrics["minimum_local_radius"],
                "maximum_curvature": metrics["maximum_curvature"],
                "maximum_projection_correction": metrics["maximum_projection_correction"],
                "projection_degenerate": metrics["projection_degenerate"],
                "metric_points": metrics["metric_points"],
                "candidate_statistics": candidate_stats,
                "selected_handle_factor": selected["handle_factor"],
            })
        self._apply_canonical_manufactured_lanes(
            route_lanes,
            transitions,
            surface_evaluator=surface_evaluator,
            tolerance=transition_tolerance * 0.1,
        )
        self.junction_transitions = transitions
        aggregate_stats = {
            "attempted_transition_candidates": int(sum(t.get("candidate_statistics", {}).get("attempted", 0) for t in transitions)),
            "rejected_projection_degenerate_candidates": int(sum(t.get("candidate_statistics", {}).get("rejected_projection_degenerate", 0) for t in transitions)),
            "rejected_joined_turn_candidates": int(sum(t.get("candidate_statistics", {}).get("rejected_joined_turn", 0) for t in transitions)),
            "rejected_radius_candidates": int(sum(t.get("candidate_statistics", {}).get("rejected_radius", 0) for t in transitions)),
            "junctions_with_no_valid_transition": int(sum(1 for t in transitions if not t.get("accepted", True))),
        }
        return {
            "transitions": transitions,
            "violations": violations,
            "transition_count": len(transitions),
            "candidate_statistics": aggregate_stats,
            "method_note": "Exact even-multiplicity Euler route with explicit local surface-constrained junction transitions.",
        }

    def edge_revisit_distances(
        self,
        route=None,
        cyclic=True,
        verbose=True,
    ):
        """
        Return spacing between consecutive fibre passes
        of each physical edge.

        For two fibres per edge, one distance is returned
        per physical edge.
        """

        if route is None:
            route = self.euler_edges

        route_length = len(route)

        occurrences = {}

        for index, item in enumerate(
            route
        ):
            edge_id = int(
                item["edge_id"]
            )

            occurrences.setdefault(
                edge_id,
                [],
            ).append(index)

        distances = []

        for edge_id in range(
            len(self.edge_index)
        ):
            positions = sorted(
                occurrences.get(
                    edge_id,
                    [],
                )
            )

            if len(positions) != (
                self.fibres_per_edge
            ):
                continue

            for occurrence_index in range(
                len(positions) - 1
            ):
                distances.append(
                    positions[
                        occurrence_index + 1
                    ]
                    - positions[
                        occurrence_index
                    ]
                )

            if (
                cyclic
                and len(positions) > 1
            ):
                wrap_distance = (
                    route_length
                    - positions[-1]
                    + positions[0]
                )

                # For two copies, retain the shorter circular
                # separation rather than reporting both arcs.
                if len(positions) == 2:
                    forward_distance = (
                        positions[1]
                        - positions[0]
                    )

                    distances[-1] = min(
                        forward_distance,
                        wrap_distance,
                    )
                else:
                    distances.append(
                        wrap_distance
                    )

        distances = np.asarray(
            distances,
            dtype=np.int64,
        )

        if verbose and len(distances):
            print(
                f"Mean revisit distance : "
                f"{distances.mean():.2f}"
            )
            print(
                f"Minimum revisit distance: "
                f"{distances.min()}"
            )
            print(
                f"Maximum revisit distance: "
                f"{distances.max()}"
            )
            print(
                f"Revisit distance = 1 : "
                f"{np.sum(distances == 1)}"
            )
            print(
                f"Revisit distance <= 5: "
                f"{np.sum(distances <= 5)}"
            )

        return distances

    def route_metrics(
        self,
        route=None,
        huang_angle_threshold_degrees=60.0,
        verbose=True,
    ):
        """
        Return manufacturing metrics for one Euler circuit.
        """

        if route is None:
            route = self.euler_edges

        angles = self.compute_turn_angles(
            route=route,
            include_closure=True,
            verbose=False,
        )
        thresholded_excess = np.maximum(angles - float(huang_angle_threshold_degrees), 0.0)
        thresholded_turn_energy = float(np.mean(thresholded_excess ** 2)) if angles.size else 0.0

        # Every valid Euler route traverses each physical curve the same
        # number of times. Reversing an edge preserves its internal angles.
        internal_angles = self._printed_internal_angles

        backtrack_count, _ = (
            self.count_immediate_backtracks(
                route=route,
                include_closure=True,
                verbose=False,
            )
        )

        revisit_distances = (
            self.edge_revisit_distances(
                route=route,
                cyclic=True,
                verbose=False,
            )
        )

        metrics = {
            "mean_angle":
                float(angles.mean()),
            "median_angle":
                float(np.median(angles)),
            "max_angle":
                float(angles.max()),
            "turns_above_90":
                int(np.sum(angles > 90.0)),
            "turns_above_60":
                int(np.sum(angles > float(huang_angle_threshold_degrees))),
            "turns_above_120":
                int(np.sum(angles > 120.0)),
            "turns_above_150":
                int(np.sum(angles > 150.0)),
            "internal_max_angle":
                self._internal_max,
            "internal_turns_above_90":
                self._internal_counts[90],
            "internal_turns_above_120":
                self._internal_counts[120],
            "internal_turns_above_150":
                self._internal_counts[150],
            "overall_max_angle":
                max(float(angles.max()) if angles.size else 0.0,
                    self._internal_max),
            "overall_turns_above_90":
                int(np.sum(angles > 90.0)) + self._internal_counts[90],
            "thresholded_turn_energy":
                thresholded_turn_energy,
            "huang_energy":
                thresholded_turn_energy,
            "huang_angle_threshold_degrees":
                float(huang_angle_threshold_degrees),
            "overall_turns_above_120":
                int(np.sum(angles > 120.0)) + self._internal_counts[120],
            "overall_turns_above_150":
                int(np.sum(angles > 150.0)) + self._internal_counts[150],
            "immediate_backtracks":
                int(backtrack_count),
            "mean_revisit_distance":
                float(
                    revisit_distances.mean()
                ),
            "minimum_revisit_distance":
                int(
                    revisit_distances.min()
                ),
            "revisit_distance_one":
                int(
                    np.sum(
                        revisit_distances == 1
                    )
                ),
            "angles":
                angles,
            "internal_angles":
                internal_angles,
            "revisit_distances":
                revisit_distances,
        }

        if verbose:
            self.print_metrics(
                metrics
            )

        return metrics

    @staticmethod
    def print_metrics(
        metrics,
        title="Route metrics",
    ):
        print(title)
        print("-" * len(title))
        print(
            f"Mean junction angle    : "
            f"{metrics['mean_angle']:.2f}"
        )
        print(
            f"Median junction angle  : "
            f"{metrics['median_angle']:.2f}"
        )
        print(
            f"Maximum junction angle : "
            f"{metrics['max_angle']:.2f}"
        )
        print(
            f"Junctions above 90 deg : "
            f"{metrics['turns_above_90']}"
        )
        print(
            f"Junctions above 120 deg: "
            f"{metrics['turns_above_120']}"
        )
        print(
            f"Junctions above 150 deg: "
            f"{metrics['turns_above_150']}"
        )
        print(
            f"Immediate backtracks   : "
            f"{metrics['immediate_backtracks']}"
        )
        print(f"Maximum internal angle : {metrics['internal_max_angle']:.2f}")
        print(f"Internal bends >120 deg: {metrics['internal_turns_above_120']}")
        print(f"Internal bends >150 deg: {metrics['internal_turns_above_150']}")
        print(f"Maximum full-path angle: {metrics['overall_max_angle']:.2f}")
        print(f"Full-path turns >120 deg: {metrics['overall_turns_above_120']}")
        print(f"Full-path turns >150 deg: {metrics['overall_turns_above_150']}")
        print(
            f"Mean revisit distance  : "
            f"{metrics['mean_revisit_distance']:.2f}"
        )
        print(
            f"Minimum revisit distance: "
            f"{metrics['minimum_revisit_distance']}"
        )
        print(
            f"Revisit distance = 1   : "
            f"{metrics['revisit_distance_one']}"
        )

    def validate_print_turns(
        self, route=None, max_allowed_angle=120.0, raise_on_error=False,
        verbose=True,
    ):
        """Check both curve-internal bends and closed-route junctions."""
        if not 0.0 <= max_allowed_angle <= 180.0:
            raise ValueError("max_allowed_angle must be between 0 and 180 degrees.")
        if route is None:
            route = self.euler_edges
        self.validate_route(route, raise_on_error=True, verbose=False)
        junctions = self.compute_turn_angles(route, verbose=False)
        bad_junctions = [
            (int(index), float(junctions[index]))
            for index in np.flatnonzero(junctions > max_allowed_angle + 1e-9)
        ]
        bad_internal = [
            dict(item) for item in self._internal_turns
            if item["angle"] > max_allowed_angle + 1e-9
        ]
        valid = not bad_junctions and not bad_internal
        if verbose:
            print(f"Maximum permitted turn : {max_allowed_angle:.2f} degrees")
            print(f"Exceeding junctions    : {len(bad_junctions)}")
            print(f"Exceeding edge bends   : {len(bad_internal)} physical locations "
                  f"({len(bad_internal) * self.fibres_per_edge} printed passes)")
            print(f"Print turns valid      : {valid}")
        if raise_on_error and not valid:
            raise RuntimeError("The toolpath exceeds the permitted turn angle.")
        return {
            "valid": valid,
            "junctions": bad_junctions,
            "internal_bends": bad_internal,
        }

    @staticmethod
    def default_manufacturing_settings(**overrides):
        """Return configurable print-feasibility limits and score weights."""
        settings = {
            "endpoint_tolerance": 1.0e-5,
            "max_junction_turn_degrees": None,
            "max_internal_bend_degrees": None,
            "min_bend_radius": None,
            "junction_fillet_radius": None,
            "min_revisit_distance": None,
            "min_pass_spacing": None,
            "pass_spacing_samples": 9,
            "max_normal_change_degrees": None,
            "min_stress_alignment_cosine": None,
            "nozzle_clearance": None,
            "nozzle_approach": "+normal",
            "turn_objective": "legacy",
            "huang_angle_threshold_degrees": 60.0,
            "infeasible_policy": "return_best",
            "score_weights": {
                "backtrack": 100000.0,
                "severe_turn": 500.0,
                "large_turn": 100.0,
                "angle": 3.0,
                "revisit_shortfall": 100.0,
            },
        }
        settings.update(overrides)
        return settings

    def _junction_bend_radius_report(self, route, settings):
        min_radius = settings.get("min_bend_radius")
        if min_radius is None:
            return [], []
        fillet_radius = settings.get("junction_fillet_radius")
        reports = []
        violations = []
        for index, item in enumerate(route):
            following = route[(index + 1) % len(route)]
            angle = self.transition_angle(item, following)
            location = {
                "transition_index": int(index),
                "node": int(item["v"]),
                "model": "circular_fillet_radius",
                "closing_junction": bool(index == len(route) - 1),
            }
            if fillet_radius is None:
                reports.append({
                    "kind": "junction_bend_radius_unknown",
                    "location": location,
                    "edge_ids": [int(item["edge_id"]), int(following["edge_id"])],
                    "measured_value": None,
                    "limit": float(min_radius),
                    "message": (
                        "No junction transition geometry or fillet radius was supplied. "
                        "The junction angle is known, but physical bend radius cannot be "
                        "verified from centreline edge ordering alone."
                    ),
                    "action": "machine_data",
                    "junction_angle_degrees": float(angle),
                })
            else:
                measured = float(fillet_radius)
                entry = self._as_violation(
                    "junction_bend_radius",
                    location,
                    [item["edge_id"], following["edge_id"]],
                    measured,
                    min_radius,
                    "Configured circular fillet radius at this junction is below the minimum bend radius.",
                    "machine_data" if measured < float(min_radius) else "route_ordering",
                )
                entry["junction_angle_degrees"] = float(angle)
                reports.append(entry)
                if measured < float(min_radius) - 1.0e-9:
                    violations.append(entry)
        return reports, violations

    @staticmethod
    def _circumradius(a, b, c, tolerance=1.0e-12):
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(c - a)
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        if min(ab, bc, ca) <= tolerance:
            return np.inf
        if area2 <= tolerance:
            return np.inf
        return float(ab * bc * ca / (2.0 * area2))

    @staticmethod
    def _as_violation(
        kind,
        location,
        edge_ids,
        measured_value,
        limit,
        message,
        action="route_ordering",
    ):
        return {
            "kind": kind,
            "location": location,
            "edge_ids": [int(edge_id) for edge_id in edge_ids],
            "measured_value": float(measured_value),
            "limit": None if limit is None else float(limit),
            "message": message,
            "action": action,
        }

    @staticmethod
    def _segment_intersection_3d(a, b, c, d, tolerance=1.0e-9, angular_tolerance=1.0e-8):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        u = b - a
        v = d - c
        length_u = float(np.linalg.norm(u))
        length_v = float(np.linalg.norm(v))

        def closest_point_result(p_first, p_second, s, t):
            distance = float(np.linalg.norm(p_first - p_second))
            return {
                "type": "point" if distance <= tolerance else "none",
                "point_first": p_first,
                "point_second": p_second,
                "distance": distance,
                "parameter_first": float(s),
                "parameter_second": float(t),
                "overlap_length": 0.0,
            }

        def closest_segment_points():
            uu = float(np.dot(u, u))
            vv = float(np.dot(v, v))
            if uu <= tolerance * tolerance and vv <= tolerance * tolerance:
                return a, c, 0.0, 0.0
            if uu <= tolerance * tolerance:
                t0 = float(np.clip(np.dot(a - c, v) / vv, 0.0, 1.0))
                return a, c + t0 * v, 0.0, t0
            if vv <= tolerance * tolerance:
                s0 = float(np.clip(np.dot(c - a, u) / uu, 0.0, 1.0))
                return a + s0 * u, c, s0, 0.0
            w0 = a - c
            a_dot = uu
            b_dot = float(np.dot(u, v))
            c_dot = vv
            d_dot = float(np.dot(u, w0))
            e_dot = float(np.dot(v, w0))
            denom = a_dot * c_dot - b_dot * b_dot
            small = 1.0e-30
            if denom < small:
                s_numer = 0.0
                s_denom = 1.0
                t_numer = e_dot
                t_denom = c_dot
            else:
                s_numer = b_dot * e_dot - c_dot * d_dot
                t_numer = a_dot * e_dot - b_dot * d_dot
                s_denom = denom
                t_denom = denom
                if s_numer < 0.0:
                    s_numer = 0.0
                    t_numer = e_dot
                    t_denom = c_dot
                elif s_numer > s_denom:
                    s_numer = s_denom
                    t_numer = e_dot + b_dot
                    t_denom = c_dot
            if t_numer < 0.0:
                t_numer = 0.0
                if -d_dot < 0.0:
                    s_numer = 0.0
                    s_denom = 1.0
                elif -d_dot > a_dot:
                    s_numer = 1.0
                    s_denom = 1.0
                else:
                    s_numer = -d_dot
                    s_denom = a_dot
            elif t_numer > t_denom:
                t_numer = t_denom
                if (-d_dot + b_dot) < 0.0:
                    s_numer = 0.0
                    s_denom = 1.0
                elif (-d_dot + b_dot) > a_dot:
                    s_numer = 1.0
                    s_denom = 1.0
                else:
                    s_numer = -d_dot + b_dot
                    s_denom = a_dot
            s = 0.0 if abs(s_numer) <= small else s_numer / s_denom
            t = 0.0 if abs(t_numer) <= small else t_numer / t_denom
            s = float(np.clip(s, 0.0, 1.0))
            t = float(np.clip(t, 0.0, 1.0))
            return a + s * u, c + t * v, s, t

        if length_u <= tolerance or length_v <= tolerance:
            p_first, p_second, s, t = closest_segment_points()
            return closest_point_result(p_first, p_second, s, t)

        uu = float(np.dot(u, u))
        vv = float(np.dot(v, v))
        cross = np.cross(u, v)
        sin_angle = float(np.linalg.norm(cross) / (length_u * length_v))
        parallel = sin_angle <= float(angular_tolerance)
        if parallel:
            line_distance = float(np.linalg.norm(np.cross(c - a, u)) / length_u)
            if line_distance <= tolerance:
                t0 = float(np.dot(c - a, u) / uu)
                t1 = float(np.dot(d - a, u) / uu)
                lo = max(0.0, min(t0, t1))
                hi = min(1.0, max(t0, t1))
                overlap_parameter_length = max(0.0, hi - lo)
                overlap_length = overlap_parameter_length * length_u
                if overlap_length > tolerance:
                    s_mid = 0.5 * (lo + hi)
                    p_first = a + s_mid * u
                    t_second = float(np.dot(p_first - c, v) / vv)
                    t_second_clipped = float(np.clip(t_second, 0.0, 1.0))
                    p_second = c + t_second_clipped * v
                    distance = float(np.linalg.norm(p_first - p_second))
                    if distance <= tolerance:
                        second_parameters = sorted(
                            float(np.clip(np.dot((a + s * u) - c, v) / vv, 0.0, 1.0))
                            for s in (lo, hi)
                        )
                        return {
                            "type": "overlap",
                            "point_first": p_first,
                            "point_second": p_second,
                            "distance": distance,
                            "parameter_first": float(s_mid),
                            "parameter_second": t_second_clipped,
                            "overlap_length": float(overlap_length),
                            "overlap_interval_first": (float(lo), float(hi)),
                            "overlap_interval_second": (second_parameters[0], second_parameters[1]),
                        }
                if hi >= lo:
                    s_point = 0.5 * (lo + hi)
                    p_first = a + s_point * u
                    t_second = float(np.dot(p_first - c, v) / vv)
                    p_second = c + np.clip(t_second, 0.0, 1.0) * v
                    result = closest_point_result(
                        p_first,
                        p_second,
                        float(np.clip(s_point, 0.0, 1.0)),
                        float(np.clip(t_second, 0.0, 1.0)),
                    )
                    result["overlap_length"] = float(overlap_length)
                    result["overlap_interval_first"] = (float(lo), float(hi))
                    return result

        p_first, p_second, s, t = closest_segment_points()
        return closest_point_result(p_first, p_second, s, t)

    @staticmethod
    def _segment_intersection_xy(a, b, c, d, tolerance=1.0e-9):
        """Deprecated compatibility wrapper for the 3D segment intersection test."""
        hit = EulerCCFToolpath._segment_intersection_3d(a, b, c, d, tolerance=tolerance)
        if hit["type"] == "none":
            return None
        return {
            "type": hit["type"],
            "point": hit["point_first"],
            "parameters": (hit["parameter_first"], hit["parameter_second"]),
        }

    def _manufactured_segment_records(self, route_with_lanes, transitions):
        records = []
        order = 0
        for traversal_id, item in enumerate(route_with_lanes):
            transition = transitions[traversal_id]
            if not transition.get("accepted", True):
                raise ManufacturingGeometryError(
                    f"Cannot assemble failed transition {traversal_id}; no accepted connector geometry exists."
                )
            curve = self._manufactured_lane_curve_for_transition(route_with_lanes, transitions, traversal_id)
            edge_nodes = set(map(int, self.edge_index[int(item["edge_id"])]))
            for i in range(len(curve) - 1):
                records.append({
                    "order": order,
                    "kind": "fibre_lane",
                    "edge_id": int(item["edge_id"]),
                    "edge_ids": [int(item["edge_id"])],
                    "node_ids": set(edge_nodes),
                    "lane_id": int(item.get("offset_lane", item["fibre_id"])),
                    "traversal_id": int(traversal_id),
                    "segment_index": int(i),
                    "a": curve[i],
                    "b": curve[i + 1],
                })
                order += 1
            points = transition["points"]
            transition_edge_ids = [int(edge_id) for edge_id in transition.get("edge_ids", [item["edge_id"]])]
            if "junction_node_id" in transition:
                transition_node_ids = {int(transition["junction_node_id"])}
            else:
                transition_node_ids = set()
                for edge_id in transition_edge_ids:
                    transition_node_ids.update(map(int, self.edge_index[edge_id]))
            for i in range(len(points) - 1):
                records.append({
                    "order": order,
                    "kind": "closure_transition" if transition.get("closing") else "junction_transition",
                    "edge_id": int(item["edge_id"]),
                    "edge_ids": transition_edge_ids,
                    "node_ids": set(transition_node_ids),
                    "lane_id": int(item.get("offset_lane", item["fibre_id"])),
                    "traversal_id": int(traversal_id),
                    "segment_index": int(i),
                    "a": points[i],
                    "b": points[i + 1],
                })
                order += 1
        return records

    def _junction_points_xyz(self):
        node_points = {}
        for edge_id, (u, v) in enumerate(self.edge_index):
            curve = self.edge_curves_xyz[edge_id]
            node_points.setdefault(int(u), []).append(curve[0])
            node_points.setdefault(int(v), []).append(curve[-1])
        return {
            node: np.mean(np.asarray(points, dtype=np.float64), axis=0)
            for node, points in node_points.items()
        }

    def _junction_radii(self, fibre_line_width=None, strut_thickness=None, user_supplied_junction_radius=None):
        base = float(user_supplied_junction_radius) if user_supplied_junction_radius is not None else 0.0
        if fibre_line_width is not None:
            max_abs_lane_offset = 0.0
            if self.edge_pass_offsets is not None:
                max_abs_lane_offset = max(
                    float(np.max(np.abs(offsets))) for offsets in self.edge_pass_offsets
                )
            base = max(base, max_abs_lane_offset + 0.5 * float(fibre_line_width))
        thicknesses = None
        if strut_thickness is not None:
            thicknesses = np.asarray(strut_thickness, dtype=np.float64)
            if thicknesses.ndim == 0:
                thicknesses = np.full((len(self.edge_index),), float(thicknesses), dtype=np.float64)
        radii = {}
        for node_id, node_point in self._junction_points_xyz().items():
            radius = base
            if thicknesses is not None and thicknesses.shape == (len(self.edge_index),):
                incident = [
                    edge_id for edge_id, (u, v) in enumerate(self.edge_index)
                    if int(u) == int(node_id) or int(v) == int(node_id)
                ]
                if incident:
                    radius = max(radius, 0.5 * float(np.max(thicknesses[incident])))
            radii[int(node_id)] = max(radius, 0.0)
        return radii

    def _accepted_junction_node(self, point, first, second, junction_radii):
        shared_nodes = set(first.get("node_ids", set())) & set(second.get("node_ids", set()))
        if not shared_nodes:
            return None
        node_points = self._junction_points_xyz()
        point = np.asarray(point, dtype=np.float64)
        for node_id in shared_nodes:
            radius = float(junction_radii.get(int(node_id), 0.0))
            if radius > 0.0 and np.linalg.norm(point - node_points[int(node_id)]) <= radius:
                return int(node_id)
        return None

    @staticmethod
    def _segments_share_endpoint(first, second, tolerance):
        endpoints_a = (first["a"], first["b"])
        endpoints_b = (second["a"], second["b"])
        return any(
            np.linalg.norm(np.asarray(a) - np.asarray(b)) <= tolerance
            for a in endpoints_a
            for b in endpoints_b
        )

    @staticmethod
    def _aabb_candidate_pairs(records, tolerance, search_clearance=None):
        if not records:
            return []
        clearance = float(max(tolerance, 0.0 if search_clearance is None else search_clearance))
        mins = []
        maxs = []
        centers = []
        radii = []
        for rec in records:
            a = np.asarray(rec["a"], dtype=np.float64)
            b = np.asarray(rec["b"], dtype=np.float64)
            mn = np.minimum(a, b) - clearance
            mx = np.maximum(a, b) + clearance
            mins.append(mn)
            maxs.append(mx)
            center = 0.5 * (mn + mx)
            centers.append(center)
            radii.append(float(np.linalg.norm(mx - center)))
        mins = np.asarray(mins)
        maxs = np.asarray(maxs)
        centers = np.asarray(centers)
        radii = np.asarray(radii)
        pairs = set()
        if cKDTree is not None:
            tree = cKDTree(centers)
            max_radius = float(radii.max()) if radii.size else 0.0
            for i, center in enumerate(centers):
                for j in tree.query_ball_point(center, r=float(radii[i] + max_radius)):
                    if j <= i:
                        continue
                    if np.all(maxs[i] >= mins[j]) and np.all(maxs[j] >= mins[i]):
                        pairs.add((int(i), int(j)))
        else:
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    if np.all(maxs[i] >= mins[j]) and np.all(maxs[j] >= mins[i]):
                        pairs.add((i, j))
        return sorted(pairs)

    def _validate_manufactured_intersections(
        self,
        route_with_lanes,
        transitions,
        tolerance=1.0e-6,
        junction_region_radius=None,
        fibre_line_width=None,
        strut_thickness=None,
        min_unrelated_segment_spacing=None,
    ):
        records = self._manufactured_segment_records(route_with_lanes, transitions)
        if not records:
            return {
                "self_intersections": [],
                "lane_crossovers": [],
                "unrelated_segment_intersections": [],
                "transition_intersections": [],
                "accepted_junction_intersections": [],
                "accepted_junction_intersection_count": 0,
                "off_junction_intersection_count": 0,
                "finite_overlap_count": 0,
                "minimum_distance_between_unrelated_segments": None,
                "violations": [],
            }
        junction_radii = self._junction_radii(
            fibre_line_width=fibre_line_width,
            strut_thickness=strut_thickness,
            user_supplied_junction_radius=junction_region_radius,
        )
        violations = []
        lane_crossovers = []
        unrelated = []
        transition_hits = []
        self_hits = []
        accepted = []
        minimum_unrelated_spacing = np.inf
        spacing_evaluated = False
        search_clearance = max(float(tolerance), float(min_unrelated_segment_spacing or 0.0))
        for i, j in self._aabb_candidate_pairs(records, tolerance, search_clearance=search_clearance):
            first = records[i]
            second = records[j]
            adjacent_in_order = abs(first["order"] - second["order"]) <= 1 or (i == 0 and j == len(records) - 1)
            adjacent = adjacent_in_order and self._segments_share_endpoint(first, second, tolerance)
            hit = self._segment_intersection_3d(first["a"], first["b"], second["a"], second["b"], tolerance=tolerance)
            point = 0.5 * (hit["point_first"] + hit["point_second"])
            accepted_node = None
            if hit["type"] == "point":
                accepted_node = self._accepted_junction_node(point, first, second, junction_radii)
            spacing_exempt = self._accepted_junction_node(point, first, second, junction_radii) is not None
            unrelated_pair = not adjacent and set(first.get("edge_ids", [])) != set(second.get("edge_ids", []))
            if unrelated_pair and not spacing_exempt:
                spacing_evaluated = True
                minimum_unrelated_spacing = min(minimum_unrelated_spacing, float(hit["distance"]))
            if hit["type"] == "none":
                continue
            if adjacent and hit["type"] != "overlap":
                continue
            if accepted_node is not None:
                accepted.append({
                    "segments": [first, second],
                    "point": point,
                    "node_id": int(accepted_node),
                    "distance": float(hit["distance"]),
                })
                continue
            edge_ids = sorted(set(first.get("edge_ids", [first["edge_id"]])) | set(second.get("edge_ids", [second["edge_id"]])))
            kind = "finite_segment_overlap" if hit["type"] == "overlap" else "off_junction_intersection"
            if hit["type"] == "point" and first["edge_id"] == second["edge_id"] and first["lane_id"] != second["lane_id"]:
                kind = "lane_crossover"
            hit_record = {"segments": [first, second], "point": point, "type": hit["type"], "distance": hit["distance"]}
            if "transition" in first["kind"] or "transition" in second["kind"]:
                transition_hits.append(hit_record)
            elif first["edge_id"] != second["edge_id"]:
                unrelated.append(hit_record)
            elif first["lane_id"] != second["lane_id"]:
                lane_crossovers.append(hit_record)
            else:
                self_hits.append(hit_record)
            violations.append({
                "kind": kind,
                "location": {
                    "segment_orders": [int(first["order"]), int(second["order"])],
                    "point_type": hit["type"],
                },
                "edge_ids": edge_ids,
                "measured_value": float(hit["distance"]),
                "limit": float(tolerance),
                "message": "Manufactured path segments intersect outside an accepted graph-junction point or overlap over finite length.",
                "action": "offset_geometry",
            })
        if min_unrelated_segment_spacing is not None and np.isfinite(minimum_unrelated_spacing):
            if minimum_unrelated_spacing < float(min_unrelated_segment_spacing) - float(tolerance):
                violations.append({
                    "kind": "unrelated_segment_spacing",
                    "measured_value": float(minimum_unrelated_spacing),
                    "limit": float(min_unrelated_segment_spacing),
                    "action": "offset_geometry",
                })
        return {
            "self_intersections": self_hits,
            "lane_crossovers": lane_crossovers,
            "unrelated_segment_intersections": unrelated,
            "transition_intersections": transition_hits,
            "accepted_junction_intersections": accepted,
            "accepted_junction_intersection_count": len(accepted),
            "off_junction_intersection_count": sum(1 for item in violations if item["kind"] == "off_junction_intersection"),
            "finite_overlap_count": sum(1 for item in violations if item["kind"] == "finite_segment_overlap"),
            "minimum_distance_between_unrelated_segments": (
                float(minimum_unrelated_spacing) if spacing_evaluated and np.isfinite(minimum_unrelated_spacing) else None
            ),
            "unrelated_segment_spacing_evaluated": bool(spacing_evaluated),
            "violations": violations,
        }

    def _assembled_manufactured_points(self, route_with_lanes, transitions, tolerance=1.0e-6):
        points = []
        for traversal_id, item in enumerate(route_with_lanes):
            transition = transitions[traversal_id]
            if not transition.get("accepted", True):
                raise ManufacturingGeometryError(
                    f"Cannot assemble failed transition {traversal_id}; no accepted connector geometry exists."
                )
            for segment in (self._manufactured_lane_curve_for_transition(route_with_lanes, transitions, traversal_id), transition["points"]):
                for point in np.asarray(segment, dtype=np.float64):
                    if points and np.linalg.norm(point - points[-1]) <= tolerance:
                        continue
                    points.append(point)
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)

    def _assembled_manufactured_metric_points(self, route_with_lanes, transitions, tolerance=1.0e-6):
        points = []
        for traversal_id, item in enumerate(route_with_lanes):
            transition = transitions[traversal_id]
            if not transition.get("accepted", True):
                raise ManufacturingGeometryError(
                    f"Cannot assemble failed transition {traversal_id}; no accepted connector geometry exists."
                )
            lane_curve = self._manufactured_lane_curve_for_transition(route_with_lanes, transitions, traversal_id, tolerance=tolerance * 0.1)
            metric_transition = np.asarray(
                transition.get("metric_points", transition["points"]),
                dtype=np.float64,
            )
            for segment in (lane_curve, metric_transition):
                for point in np.asarray(segment, dtype=np.float64):
                    if points and np.linalg.norm(point - points[-1]) <= tolerance:
                        continue
                    points.append(point)
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)

    def _manufactured_path_metrics(self, points, hard_limit=None, tolerance=1.0e-12):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 3:
            empty = np.asarray([], dtype=np.float64)
            return {
                "manufactured_mean_turn_angle": 0.0,
                "manufactured_median_turn_angle": 0.0,
                "manufactured_maximum_turn_angle": 0.0,
                "manufactured_turns_above_90": 0,
                "manufactured_turns_above_120": 0,
                "manufactured_turns_above_hard_limit": 0,
                "manufactured_minimum_bend_radius": None,
                "manufactured_turn_angles": empty,
            }
        angles = []
        radii = []
        closed = np.linalg.norm(points[0] - points[-1]) <= max(tolerance, 1.0e-9)
        if closed:
            points = points[:-1]
        count = len(points)
        if count < 3:
            empty = np.asarray([], dtype=np.float64)
            return {
                "manufactured_mean_turn_angle": 0.0,
                "manufactured_median_turn_angle": 0.0,
                "manufactured_maximum_turn_angle": 0.0,
                "manufactured_turns_above_90": 0,
                "manufactured_turns_above_120": 0,
                "manufactured_turns_above_hard_limit": 0,
                "manufactured_minimum_bend_radius": None,
                "manufactured_turn_angles": empty,
            }
        for idx in range(count):
            if not closed and (idx == 0 or idx == count - 1):
                continue
            prev_point = points[(idx - 1) % count]
            point = points[idx]
            next_point = points[(idx + 1) % count]
            v0 = self._safe_unit(point - prev_point)
            v1 = self._safe_unit(next_point - point)
            if v0 is None or v1 is None:
                continue
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0)))))
            radius = self._circumradius(prev_point, point, next_point)
            if np.isfinite(radius):
                radii.append(radius)
        angles = np.asarray(angles, dtype=np.float64)
        hard = float(hard_limit) if hard_limit is not None else np.inf
        return {
            "manufactured_mean_turn_angle": float(np.mean(angles)) if angles.size else 0.0,
            "manufactured_median_turn_angle": float(np.median(angles)) if angles.size else 0.0,
            "manufactured_maximum_turn_angle": float(np.max(angles)) if angles.size else 0.0,
            "manufactured_turns_above_90": int(np.sum(angles > 90.0)) if angles.size else 0,
            "manufactured_turns_above_120": int(np.sum(angles > 120.0)) if angles.size else 0,
            "manufactured_turns_above_hard_limit": int(np.sum(angles > hard)) if np.isfinite(hard) and angles.size else 0,
            "manufactured_minimum_bend_radius": float(np.min(radii)) if radii else None,
            "manufactured_turn_angles": angles,
        }

    def _endpoint_continuity(self, route, tolerance):
        gaps = []
        violations = []
        for index, item in enumerate(route):
            following = route[(index + 1) % len(route)]
            curve = self.get_oriented_curve(item)
            next_curve = self.get_oriented_curve(following)
            gap = float(np.linalg.norm(curve[-1] - next_curve[0]))
            gaps.append(gap)
            if gap > tolerance:
                violations.append(self._as_violation(
                    "endpoint_gap",
                    {
                        "transition_index": int(index),
                        "traversal_ids": [int(index), int((index + 1) % len(route))],
                        "node": int(item["v"]),
                    },
                    [item["edge_id"], following["edge_id"]],
                    gap,
                    tolerance,
                    "Endpoint gap exceeds the configured continuity tolerance.",
                    "new_geometry",
                ))
        return np.asarray(gaps, dtype=np.float64), violations

    def _internal_bend_radius_violations(self, settings):
        max_bend = settings.get("max_internal_bend_degrees")
        min_radius = settings.get("min_bend_radius")
        violations = []
        radii = []
        for edge_id, curve in enumerate(self.edge_curves_xyz):
            for point_index in range(1, len(curve) - 1):
                radius = self._circumradius(
                    curve[point_index - 1], curve[point_index], curve[point_index + 1]
                )
                radii.append(radius)
                incoming = curve[point_index] - curve[point_index - 1]
                outgoing = curve[point_index + 1] - curve[point_index]
                if np.linalg.norm(incoming) <= 1.0e-12 or np.linalg.norm(outgoing) <= 1.0e-12:
                    continue
                cosine = np.clip(
                    np.dot(incoming, outgoing)
                    / (np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
                    -1.0, 1.0,
                )
                angle = float(np.degrees(np.arccos(cosine)))
                if max_bend is not None and angle > float(max_bend) + 1.0e-9:
                    violations.append(self._as_violation(
                        "internal_bend_angle",
                        {"edge_id": int(edge_id), "point_index": int(point_index)},
                        [edge_id],
                        angle,
                        max_bend,
                        "Internal curve bend exceeds the configured limit; route ordering cannot fix this without changing edge geometry.",
                        "new_geometry",
                    ))
                if min_radius is not None and radius < float(min_radius) - 1.0e-9:
                    violations.append(self._as_violation(
                        "bend_radius",
                        {"edge_id": int(edge_id), "point_index": int(point_index)},
                        [edge_id],
                        radius,
                        min_radius,
                        "Internal curve bend radius is below the configured physical XYZ limit; route ordering cannot fix this without changing edge geometry.",
                        "new_geometry",
                    ))
        return np.asarray(radii, dtype=np.float64), violations

    def _static_settings_key(self, settings):
        return (
            settings.get("max_internal_bend_degrees"),
            settings.get("min_bend_radius"),
            settings.get("min_pass_spacing"),
            int(settings.get("pass_spacing_samples", 9)),
            self.edge_pass_curves_xyz is not None,
        )

    def _route_independent_manufacturing_checks(self, settings):
        key = self._static_settings_key(settings)
        cached = self._manufacturing_static_cache.get(key)
        if cached is not None:
            return cached
        start = perf_counter()
        radii, internal_violations = self._internal_bend_radius_violations(settings)
        coincident_violations = self._coincident_pass_spacing_violations(settings)
        result = {
            "internal_bend_radii": radii,
            "internal_violations": internal_violations,
            "coincident_pass_violations": coincident_violations,
            "elapsed_seconds": perf_counter() - start,
        }
        self._manufacturing_static_cache[key] = result
        return result

    def _coincident_pass_spacing_violations(self, settings):
        min_spacing = settings.get("min_pass_spacing")
        if min_spacing is None:
            return []
        violations = []
        sample_count = int(settings.get("pass_spacing_samples", 9))
        for edge_id, centreline in enumerate(self.edge_curves_xyz):
            if self.fibres_per_edge < 2:
                continue
            if self.edge_pass_curves_xyz is None:
                violations.append(self._as_violation(
                    "coincident_fibre_pass_geometry",
                    {"edge_id": int(edge_id), "fibre_ids": list(range(self.fibres_per_edge))},
                    [edge_id],
                    0.0,
                    min_spacing,
                    "This physical edge has multiple fibre instances but only one centreline. Positive pass spacing is impossible until distinct offset pass geometries are supplied.",
                    "new_geometry",
                ))
                continue
            samples = [
                self._sample_curve_interior(
                    self.edge_pass_curves_xyz[edge_id][fibre_id], sample_count
                )
                for fibre_id in range(self.fibres_per_edge)
            ]
            for first in range(self.fibres_per_edge):
                for second in range(first + 1, self.fibres_per_edge):
                    if len(samples[first]) == 0 or len(samples[second]) == 0:
                        continue
                    distances = np.linalg.norm(
                        samples[first][:, None, :] - samples[second][None, :, :],
                        axis=2,
                    )
                    minimum = float(np.min(distances))
                    if minimum < float(min_spacing) - 1.0e-9:
                        violations.append(self._as_violation(
                            "coincident_fibre_pass_geometry",
                            {"edge_id": int(edge_id), "fibre_ids": [int(first), int(second)]},
                            [edge_id],
                            minimum,
                            min_spacing,
                            "Per-fibre pass geometries are closer than the configured spacing.",
                            "new_geometry",
                        ))
        return violations

    @staticmethod
    def _sample_curve_interior(curve, sample_count):
        sample_count = int(sample_count)
        if sample_count <= 0:
            return np.empty((0, 3), dtype=np.float64)
        if len(curve) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if len(curve) == 1:
            return curve.copy()
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        total = float(lengths[-1])
        if total <= 1.0e-12:
            return curve[:1].copy()
        distances = np.linspace(0.05 * total, 0.95 * total, sample_count)
        samples = []
        for distance in distances:
            index = min(int(np.searchsorted(lengths, distance, side="right")), len(curve) - 1)
            left = lengths[index - 1]
            right = lengths[index]
            t = (distance - left) / (right - left) if right > left else 0.0
            samples.append(curve[index - 1] + t * (curve[index] - curve[index - 1]))
        return np.asarray(samples, dtype=np.float64)

    def _pass_spacing_violations(self, route, settings):
        min_spacing = settings.get("min_pass_spacing")
        if min_spacing is None:
            return None, []
        sample_count = int(settings.get("pass_spacing_samples", 9))
        samples = [
            self._sample_curve_interior(self.get_oriented_pass_curve(item), sample_count)
            for item in route
        ]
        best_distance = np.inf
        violations = []
        for first in range(len(route)):
            if len(samples[first]) == 0:
                continue
            for second in range(first + 1, len(route)):
                if int(route[first]["edge_id"]) == int(route[second]["edge_id"]):
                    continue
                if len(samples[second]) == 0:
                    continue
                delta = samples[first][:, None, :] - samples[second][None, :, :]
                distances = np.linalg.norm(delta, axis=2)
                pair_min = float(np.min(distances))
                best_distance = min(best_distance, pair_min)
                if pair_min < float(min_spacing) - 1.0e-9:
                    violations.append(self._as_violation(
                        "pass_spacing",
                        {
                            "traversal_ids": [int(first), int(second)],
                            "fibre_ids": [
                                int(route[first]["fibre_id"]),
                                int(route[second]["fibre_id"]),
                            ],
                        },
                        [route[first]["edge_id"], route[second]["edge_id"]],
                        pair_min,
                        min_spacing,
                        "Distinct deposited fibre passes are closer than the configured spacing. Coincident repeated centrelines require graph or edge-offset geometry changes.",
                        "new_geometry",
                    ))
        if not np.isfinite(best_distance):
            best_distance = None
        return best_distance, violations

    def _optional_data_violations(self, route, settings):
        violations = []
        max_normal_change = settings.get("max_normal_change_degrees")
        min_stress_alignment = settings.get("min_stress_alignment_cosine")
        normal_angles = []
        stress_alignment = []

        if max_normal_change is not None:
            if self.edge_normals_xyz is None:
                violations.append({
                    "kind": "missing_normal_data",
                    "location": "route",
                    "edge_ids": [],
                    "measured_value": None,
                    "limit": float(max_normal_change),
                    "message": "Normal-change checking requires CAD normals for every edge.",
                    "action": "machine_data",
                })
            else:
                for index, item in enumerate(route):
                    following = route[(index + 1) % len(route)]
                    n0 = self.get_oriented_normals(item)[-1]
                    n1 = self.get_oriented_normals(following)[0]
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))))
                    normal_angles.append(angle)
                    if angle > float(max_normal_change) + 1.0e-9:
                        violations.append(self._as_violation(
                            "surface_normal_change",
                            {"transition_index": int(index)},
                            [item["edge_id"], following["edge_id"]],
                            angle,
                            max_normal_change,
                            "Surface-normal change exceeds the configured limit.",
                        ))

        if min_stress_alignment is not None:
            if self.edge_stress_directions_xyz is None:
                violations.append({
                    "kind": "missing_stress_direction_data",
                    "location": "route",
                    "edge_ids": [],
                    "measured_value": None,
                    "limit": float(min_stress_alignment),
                    "message": "Stress-direction alignment requires per-point stress directions.",
                    "action": "machine_data",
                })
            else:
                for traversal_id, item in enumerate(route):
                    curve = self.get_oriented_curve(item)
                    tangents = self._curve_tangents(curve)
                    stress = self.get_oriented_stress_directions(item)
                    dots = np.abs(np.sum(tangents * stress, axis=1))
                    if dots.size:
                        value = float(np.min(dots))
                        stress_alignment.append(value)
                        if value < float(min_stress_alignment) - 1.0e-9:
                            violations.append(self._as_violation(
                                "stress_direction_alignment",
                                {"traversal_id": int(traversal_id)},
                                [item["edge_id"]],
                                value,
                                min_stress_alignment,
                                "Fibre tangent is insufficiently aligned with supplied stress direction.",
                            ))
        if settings.get("nozzle_clearance") is not None:
            violations.append({
                "kind": "missing_nozzle_clearance_data",
                "location": "route",
                "edge_ids": [],
                "measured_value": None,
                "limit": float(settings["nozzle_clearance"]),
                "message": (
                    "Nozzle-clearance checking requires machine head geometry, "
                    "tool approach convention, coordinate frame, and obstacle/surface "
                    "clearance data. It is not inferred by the Euler solver."
                ),
                "action": "machine_data",
            })
        return {
            "normal_change_angles": np.asarray(normal_angles, dtype=np.float64),
            "stress_alignment": np.asarray(stress_alignment, dtype=np.float64),
            "violations": violations,
        }

    def check_manufacturing_constraints(self, route=None, settings=None, verbose=True):
        """Evaluate hard manufacturing limits without changing the graph."""
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        settings = self.default_manufacturing_settings(**(settings or {}))
        self.validate_route(
            route,
            geometric_tolerance=float(settings["endpoint_tolerance"]),
            raise_on_error=False,
            verbose=False,
        )

        endpoint_gaps, endpoint_violations = self._endpoint_continuity(
            route, float(settings["endpoint_tolerance"])
        )
        junction_angles = self.compute_turn_angles(route, include_closure=True, verbose=False)
        backtrack_count, backtrack_indices = self.count_immediate_backtracks(
            route, include_closure=True, verbose=False
        )
        revisit_distances = self.edge_revisit_distances(route, cyclic=True, verbose=False)
        static = self._route_independent_manufacturing_checks(settings)
        radii = static["internal_bend_radii"]
        internal_violations = static["internal_violations"]
        coincident_violations = static["coincident_pass_violations"]
        junction_radius_reports, junction_radius_violations = (
            self._junction_bend_radius_report(route, settings)
        )
        spacing_min, spacing_violations = self._pass_spacing_violations(route, settings)
        optional = self._optional_data_violations(route, settings)

        violations = []
        violations.extend(endpoint_violations)
        max_junction_turn = settings.get("max_junction_turn_degrees")
        if max_junction_turn is not None:
            for index in np.flatnonzero(junction_angles > float(max_junction_turn) + 1.0e-9):
                item = route[int(index)]
                following = route[(int(index) + 1) % len(route)]
                violations.append(self._as_violation(
                    "junction_turn",
                    {"transition_index": int(index), "node": int(item["v"])},
                    [item["edge_id"], following["edge_id"]],
                    float(junction_angles[int(index)]),
                    max_junction_turn,
                    "Junction turn exceeds the configured limit.",
                    "route_ordering",
                ))
        min_revisit = settings.get("min_revisit_distance")
        if min_revisit is not None:
            for index in np.flatnonzero(revisit_distances < int(min_revisit)):
                violations.append(self._as_violation(
                    "short_revisit_distance",
                    {"revisit_index": int(index)},
                    [],
                    float(revisit_distances[int(index)]),
                    min_revisit,
                    "A physical edge is revisited sooner than the configured traversal-distance limit.",
                    "route_ordering",
                ))
        violations.extend(internal_violations)
        violations.extend(junction_radius_violations)
        violations.extend(coincident_violations)
        violations.extend(spacing_violations)
        violations.extend(optional["violations"])

        awaiting_data = [
            violation for violation in (violations + junction_radius_reports)
            if violation.get("action") == "machine_data"
            and violation.get("kind", "").endswith("_unknown")
        ]
        route_ordering_violations = [
            violation for violation in violations
            if violation.get("action") == "route_ordering"
        ]
        graph_or_geometry_changes = [
            violation for violation in violations
            if violation.get("action") == "new_geometry"
        ]
        machine_data_needed = [
            violation for violation in (violations + junction_radius_reports)
            if violation.get("action") == "machine_data"
        ]
        feasible = len(violations) == 0 and len(awaiting_data) == 0
        report = {
            "feasible": feasible,
            "manufacturing_ready": feasible,
            "route_ordering_feasible": len(route_ordering_violations) == 0,
            "settings": settings,
            "violations": violations,
            "route_ordering_violations": route_ordering_violations,
            "requires_graph_or_geometry_change": graph_or_geometry_changes,
            "requires_machine_data": machine_data_needed,
            "checks_awaiting_geometry_or_machine_data": awaiting_data,
            "endpoint_gaps": endpoint_gaps,
            "max_endpoint_gap": float(endpoint_gaps.max()) if endpoint_gaps.size else 0.0,
            "junction_angles": junction_angles,
            "junction_bend_radius_reports": junction_radius_reports,
            "immediate_backtracks": int(backtrack_count),
            "backtrack_indices": backtrack_indices,
            "revisit_distances": revisit_distances,
            "minimum_revisit_distance": (
                int(revisit_distances.min()) if revisit_distances.size else None
            ),
            "internal_bend_radii": radii,
            "minimum_bend_radius": (
                float(np.min(radii[np.isfinite(radii)]))
                if np.any(np.isfinite(radii)) else None
            ),
            "minimum_pass_spacing": spacing_min,
            "normal_change_angles": optional["normal_change_angles"],
            "stress_alignment": optional["stress_alignment"],
            "static_check_elapsed_seconds": float(static["elapsed_seconds"]),
            "nozzle_approach": settings.get("nozzle_approach", "+normal"),
            "notes": [
                "Nozzle clearance needs machine/head geometry and obstacle data; the Euler solver only reports it when supplied data is sufficient.",
                "G-code or robot commands are intentionally not generated here; use a machine-specific postprocessor with explicit calibration, frames, speeds, tool orientation, and collision constraints.",
            ],
        }
        if verbose:
            print(f"Manufacturing feasible : {feasible}")
            print(f"Violations             : {len(violations)}")
            for violation in violations[:20]:
                print(
                    f"  {violation['kind']}: value={violation['measured_value']} "
                    f"limit={violation['limit']} edge_ids={violation['edge_ids']} "
                    f"location={violation['location']} action={violation.get('action')}"
                )
            if len(violations) > 20:
                print(f"  ... {len(violations) - 20} more")
        return report

    def validate_manufactured_paths(
        self,
        route=None,
        surface_evaluator=None,
        surface_vertices=None,
        surface_faces=None,
        fibre_line_width=None,
        strut_thickness=None,
        boundary_margin=0.0,
        min_pass_spacing=None,
        min_bend_radius=None,
        max_manufactured_turn_degrees=None,
        mesh_projection_tolerance=None,
        transition_samples=5,
        require_surface_check=True,
        require_unrelated_segment_spacing=True,
        require_self_intersection_check=True,
        require_lane_crossover_check=True,
        require_junction_bend_radius=True,
        allow_unchecked_intersections=False,
        junction_region_radius=None,
        min_unrelated_segment_spacing=None,
        tolerance=1.0e-6,
        verbose=True,
        junction_trim_distance=None,
    ):
        """
        Validate offset deposition paths separately from Euler route ordering.

        This routine does not mutate structural centrelines. It reports which
        checks are complete and which require richer CAD/machine data.
        """
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        route_validation = self.validate_route(route, raise_on_error=False, verbose=False)
        route_ordering_ready = bool(route_validation["valid"])
        if self.edge_pass_curves_xyz is None:
            return {
                "route_ordering_ready": route_ordering_ready,
                "offset_geometry_ready": False,
                "machine_ready": False,
                "manufacturing_ready": False,
                "failure_category": "missing_machine_parameters",
                "missing_data": ["edge_pass_curves_xyz"],
                "violations": [{
                    "kind": "missing_offset_pass_geometry",
                    "message": "Offset-path validation requires generated edge_pass_curves_xyz.",
                    "action": "offset_geometry",
                }],
            }

        mesh_vertices = mesh_faces = None
        if surface_vertices is not None or surface_faces is not None:
            mesh_vertices, mesh_faces = coerce_surface_mesh(surface_vertices, surface_faces)
        projector = MeshSurfaceProjector(mesh_vertices, mesh_faces) if mesh_vertices is not None else None
        route_with_lanes = self.route_with_lane_assignment(route)
        transition_report = self.build_junction_transitions(
            route,
            surface_evaluator=surface_evaluator,
            surface_vertices=mesh_vertices,
            surface_faces=mesh_faces,
            transition_samples=transition_samples,
            transition_tolerance=tolerance,
            min_bend_radius=min_bend_radius,
            max_joined_turn_degrees=max_manufactured_turn_degrees,
            mesh_projection_tolerance=mesh_projection_tolerance,
            junction_trim_distance=junction_trim_distance,
        )
        transitions = transition_report["transitions"]
        has_failed_transitions = any(not transition.get("accepted", True) for transition in transitions)
        failed_transition_violations = []
        maximum_unbridged_discontinuity = 0.0
        maximum_transition_span = 0.0
        discontinuity_violations = []
        for index, item in enumerate(route_with_lanes):
            following = route_with_lanes[(index + 1) % len(route_with_lanes)]
            transition = transitions[index]
            if not transition.get("accepted", True):
                continue
            previous_curve = self._manufactured_lane_curve_for_transition(route_with_lanes, transitions, index, tolerance=tolerance * 0.1)
            next_curve = self._manufactured_lane_curve_for_transition(route_with_lanes, transitions, (index + 1) % len(route_with_lanes), tolerance=tolerance * 0.1)
            previous_end = previous_curve[-1]
            transition_start = transition["points"][0]
            transition_end = transition["points"][-1]
            next_start = next_curve[0]
            gap = max(
                float(np.linalg.norm(previous_end - transition_start)),
                float(np.linalg.norm(transition_end - next_start)),
            )
            maximum_unbridged_discontinuity = max(maximum_unbridged_discontinuity, gap)
            maximum_transition_span = max(maximum_transition_span, float(transition["gap"]))
            if gap > float(tolerance):
                discontinuity_violations.append({
                    "kind": "offset_path_discontinuity",
                    "location": {"transition_index": int(index), "closing": bool(index == len(route_with_lanes) - 1)},
                    "edge_ids": [int(item["edge_id"]), int(following["edge_id"])],
                    "measured_value": gap,
                    "limit": float(tolerance),
                    "action": "offset_geometry",
                })

        min_same_edge_spacing = np.inf
        minimum_unrelated_spacing = None
        max_centreline_deviation = 0.0
        inside_strut = True
        strut_limits = None
        if strut_thickness is not None and fibre_line_width is not None:
            thicknesses = np.asarray(strut_thickness, dtype=np.float64)
            if thicknesses.ndim == 0:
                thicknesses = np.full((len(self.edge_index),), float(thicknesses), dtype=np.float64)
            elif thicknesses.shape != (len(self.edge_index),):
                raise ManufacturingGeometryError(
                    "strut_thickness must be scalar or contain one value per physical edge."
                )
            strut_limits = 0.5 * thicknesses - 0.5 * float(fibre_line_width) - float(boundary_margin)
        for edge_id, centreline in enumerate(self.edge_curves_xyz):
            edge_passes = self.edge_pass_curves_xyz[edge_id]
            for lane_a in range(len(edge_passes)):
                for lane_b in range(lane_a + 1, len(edge_passes)):
                    sample_count = min(len(edge_passes[lane_a]), len(edge_passes[lane_b]))
                    distances = np.linalg.norm(
                        edge_passes[lane_a][:sample_count]
                        - edge_passes[lane_b][:sample_count],
                        axis=1,
                    )
                    if distances.size:
                        min_same_edge_spacing = min(min_same_edge_spacing, float(distances.min()))
            for lane_curve in self.edge_pass_curves_xyz[edge_id]:
                sample_count = min(len(centreline), len(lane_curve))
                deviation = np.linalg.norm(lane_curve[:sample_count] - centreline[:sample_count], axis=1)
                if deviation.size:
                    max_centreline_deviation = max(max_centreline_deviation, float(deviation.max()))
                    if strut_limits is not None and np.any(deviation > strut_limits[edge_id] + float(tolerance)):
                        inside_strut = False
        if not np.isfinite(min_same_edge_spacing):
            min_same_edge_spacing = None

        max_surface_distance = None
        missing_data = []
        if mesh_vertices is not None and mesh_faces is not None:
            all_points = np.vstack([
                curve for edge_passes in self.edge_pass_curves_xyz for curve in edge_passes
            ])
            all_transition_points = np.vstack([item["points"] for item in transitions]) if transitions else np.empty((0, 3))
            if all_transition_points.size:
                all_points = np.vstack((all_points, all_transition_points))
            _, _, distances = projector.project(all_points)
            max_surface_distance = float(distances.max()) if distances.size else 0.0
            if surface_evaluator is None and require_surface_check and max_surface_distance > float(tolerance):
                discontinuity_violations.append({
                    "kind": "surface_distance",
                    "measured_value": max_surface_distance,
                    "limit": float(tolerance),
                    "action": "offset_geometry",
                })
        elif require_surface_check:
            missing_data.append("surface_vertices/surface_faces")

        normal_consistency = True
        if self.edge_pass_normals_xyz is not None:
            for edge_passes in self.edge_pass_normals_xyz:
                for normals in edge_passes:
                    unit = _normalise_rows(normals, name="edge_pass_normals_xyz")
                    dots = np.sum(unit[1:] * unit[:-1], axis=1)
                    if np.any(dots < -0.05):
                        normal_consistency = False
        else:
            missing_data.append("edge_pass_normals_xyz")

        coverage_ok = True
        coverage_missing = []
        for edge_id, passes in enumerate(self.edge_pass_curves_xyz):
            if len(passes) != self.fibres_per_edge:
                coverage_ok = False
                coverage_missing.append(edge_id)
        covered_pairs = sorted((int(item["edge_id"]), int(item.get("offset_lane", item["fibre_id"]))) for item in route_with_lanes)
        expected_pairs = sorted((edge_id, lane) for edge_id in range(len(self.edge_index)) for lane in range(self.fibres_per_edge))
        if covered_pairs != expected_pairs:
            coverage_ok = False

        intersection_report = None
        if (require_unrelated_segment_spacing or require_self_intersection_check or require_lane_crossover_check) and not has_failed_transitions:
            if allow_unchecked_intersections:
                if require_unrelated_segment_spacing:
                    missing_data.append("minimum_distance_between_unrelated_segments")
                if require_self_intersection_check:
                    missing_data.append("self_intersections")
                if require_lane_crossover_check:
                    missing_data.append("lane_crossovers")
            else:
                intersection_report = self._validate_manufactured_intersections(
                    route_with_lanes,
                    transitions,
                    tolerance=tolerance,
                    junction_region_radius=junction_region_radius,
                    fibre_line_width=fibre_line_width,
                    strut_thickness=strut_thickness,
                    min_unrelated_segment_spacing=min_unrelated_segment_spacing,
                )
        if require_junction_bend_radius and min_bend_radius is None:
            missing_data.append("minimum_realised_bend_radius")

        violations = list(discontinuity_violations)
        violations.extend(failed_transition_violations)
        violations.extend(transition_report["violations"])
        if intersection_report is not None:
            violations.extend(intersection_report["violations"])
            minimum_unrelated_spacing = intersection_report["minimum_distance_between_unrelated_segments"]
            if (
                require_unrelated_segment_spacing
                and not intersection_report.get("unrelated_segment_spacing_evaluated", False)
            ):
                missing_data.append("minimum_distance_between_unrelated_segments")
        if min_pass_spacing is not None and min_same_edge_spacing is not None and min_same_edge_spacing < float(min_pass_spacing) - float(tolerance):
            violations.append({
                "kind": "minimum_pass_spacing",
                "measured_value": min_same_edge_spacing,
                "limit": float(min_pass_spacing),
                "action": "offset_geometry",
            })
        if strut_limits is not None and not inside_strut:
            violations.append({
                "kind": "deposited_half_width_outside_strut",
                "measured_value": max_centreline_deviation + 0.5 * float(fibre_line_width),
                "limit": float(np.min(0.5 * np.asarray(strut_limits) + 0.5 * float(fibre_line_width))),
                "action": "insufficient_strut_width",
            })
        if not normal_consistency:
            violations.append({
                "kind": "normal_orientation_inconsistency",
                "action": "offset_geometry",
            })
        if not coverage_ok:
            violations.append({
                "kind": "incomplete_fibre_instance_coverage",
                "edge_ids": coverage_missing,
                "action": "offset_geometry",
            })

        if has_failed_transitions:
            manufactured_points = np.empty((0, 3), dtype=np.float64)
            manufactured_metrics = self._manufactured_path_metrics(
                manufactured_points,
                hard_limit=max_manufactured_turn_degrees,
                tolerance=tolerance,
            )
        else:
            manufactured_points = self._assembled_manufactured_metric_points(route_with_lanes, transitions, tolerance=tolerance)
            manufactured_metrics = self._manufactured_path_metrics(
                manufactured_points,
                hard_limit=max_manufactured_turn_degrees,
                tolerance=tolerance,
            )
        if max_manufactured_turn_degrees is not None:
            manufactured_max = manufactured_metrics["manufactured_maximum_turn_angle"]
            if manufactured_max > float(max_manufactured_turn_degrees) + float(tolerance):
                violations.append({
                    "kind": "manufactured_turn_angle",
                    "measured_value": manufactured_max,
                    "limit": float(max_manufactured_turn_degrees),
                    "action": "offset_geometry",
                })
        offset_geometry_ready = len(violations) == 0
        machine_ready = len(missing_data) == 0
        ready = route_ordering_ready and offset_geometry_ready and machine_ready
        report = {
            "route_ordering_ready": route_ordering_ready,
            "offset_geometry_ready": offset_geometry_ready,
            "machine_ready": machine_ready,
            "manufacturing_ready": ready,
            "failure_category": (
                None if ready else
                "missing_cad_or_surface_data" if missing_data else
                "offset_geometry_failure" if not offset_geometry_ready else
                "route_ordering_failure"
            ),
            "violations": violations,
            "missing_data": missing_data,
            "max_distance_from_surface": max_surface_distance,
            "max_path_discontinuity": float(maximum_unbridged_discontinuity),
            "maximum_unbridged_discontinuity": float(maximum_unbridged_discontinuity),
            "maximum_transition_span": float(maximum_transition_span),
            "minimum_distance_between_two_passes_per_edge": min_same_edge_spacing,
            "minimum_distance_between_unrelated_segments": minimum_unrelated_spacing,
            "maximum_deviation_from_structural_centreline": float(max_centreline_deviation),
            "inside_strut_with_half_width": bool(inside_strut),
            "self_intersections": None if intersection_report is None else intersection_report["self_intersections"],
            "lane_crossovers": None if intersection_report is None else intersection_report["lane_crossovers"],
            "unrelated_segment_intersections": None if intersection_report is None else intersection_report["unrelated_segment_intersections"],
            "transition_intersections": None if intersection_report is None else intersection_report["transition_intersections"],
            "minimum_realised_bend_radius": manufactured_metrics["manufactured_minimum_bend_radius"],
            "maximum_turn_angle": float(self.compute_turn_angles(route, verbose=False).max()),
            "manufactured_mean_turn_angle": manufactured_metrics["manufactured_mean_turn_angle"],
            "manufactured_median_turn_angle": manufactured_metrics["manufactured_median_turn_angle"],
            "manufactured_maximum_turn_angle": manufactured_metrics["manufactured_maximum_turn_angle"],
            "manufactured_turns_above_90": manufactured_metrics["manufactured_turns_above_90"],
            "manufactured_turns_above_120": manufactured_metrics["manufactured_turns_above_120"],
            "manufactured_turns_above_hard_limit": manufactured_metrics["manufactured_turns_above_hard_limit"],
            "manufactured_minimum_bend_radius": manufactured_metrics["manufactured_minimum_bend_radius"],
            "manufactured_turn_angles": manufactured_metrics["manufactured_turn_angles"],
            "junction_transitions": transitions,
            "transition_candidate_statistics": transition_report.get("candidate_statistics", {}),
            "attempted_transition_candidates": transition_report.get("candidate_statistics", {}).get("attempted_transition_candidates", 0),
            "rejected_projection_degenerate_candidates": transition_report.get("candidate_statistics", {}).get("rejected_projection_degenerate_candidates", 0),
            "rejected_joined_turn_candidates": transition_report.get("candidate_statistics", {}).get("rejected_joined_turn_candidates", 0),
            "rejected_radius_candidates": transition_report.get("candidate_statistics", {}).get("rejected_radius_candidates", 0),
            "junctions_with_no_valid_transition": transition_report.get("candidate_statistics", {}).get("junctions_with_no_valid_transition", 0),
            "normal_orientation_consistent": bool(normal_consistency),
            "complete_coverage": bool(coverage_ok),
            "original_physical_edge_ids": self.original_physical_edge_ids.copy(),
            "physical_edge_metadata": dict(self.physical_edge_metadata),
            "accepted_junction_intersection_count": 0 if intersection_report is None else intersection_report["accepted_junction_intersection_count"],
            "off_junction_intersection_count": 0 if intersection_report is None else intersection_report["off_junction_intersection_count"],
            "finite_overlap_count": 0 if intersection_report is None else intersection_report["finite_overlap_count"],
            "accepted_junction_intersections": [] if intersection_report is None else intersection_report["accepted_junction_intersections"],
            "notes": [
                "Self-intersection, lane-crossover, and unrelated-segment checks use local segment intersection validation; machine collision checks require machine data.",
                "A route is manufacturing-ready only when route validation and this offset-path validation both pass.",
            ],
        }
        if verbose:
            print(f"Offset geometry ready : {report['offset_geometry_ready']}")
            print(f"Manufacturing ready   : {report['manufacturing_ready']}")
            print(f"Offset violations     : {len(violations)}")
        return report

    def _geometry_ready_for_export(self, validation, require_machine_ready=False):
        if require_machine_ready:
            return bool(validation["manufacturing_ready"])
        return bool(validation["route_ordering_ready"] and validation["offset_geometry_ready"])

    # =========================================================
    # Route scoring
    # =========================================================

    def route_score(
        self,
        route,
        angle_weight=1.0,
        turn_120_weight=20.0,
        turn_150_weight=100.0,
        backtrack_weight=10000.0,
        revisit_weight=100.0,
        target_revisit_distance=10,
        manufacturing_settings=None,
        turn_objective=None,
        huang_angle_threshold_degrees=60.0,
        return_breakdown=False,
    ):
        """
        Lower score is better.

        The score optimizes junctions between edges. Internal bends are
        reported and checked separately: each valid route prints every edge
        the same number of times, so those bends add the same constant to
        the score regardless of the traversal order.

        Important order of preference:
        1. Avoid immediate U-turns.
        2. Avoid turns above 150 degrees.
        3. Avoid turns above 120 degrees.
        4. Encourage separation of duplicate fibre passes.
        5. Reduce overall turning angle.
        """

        breakdown = self.route_score_breakdown(
            route=route,
            angle_weight=angle_weight,
            turn_120_weight=turn_120_weight,
            turn_150_weight=turn_150_weight,
            backtrack_weight=backtrack_weight,
            revisit_weight=revisit_weight,
            target_revisit_distance=target_revisit_distance,
            manufacturing_settings=manufacturing_settings,
            turn_objective=turn_objective,
            huang_angle_threshold_degrees=huang_angle_threshold_degrees,
        )
        if return_breakdown:
            return breakdown
        return float(breakdown["total_score"])

    def route_score_breakdown(
        self,
        route,
        angle_weight=1.0,
        turn_120_weight=20.0,
        turn_150_weight=100.0,
        backtrack_weight=10000.0,
        revisit_weight=100.0,
        target_revisit_distance=10,
        manufacturing_settings=None,
        turn_objective=None,
        huang_angle_threshold_degrees=60.0,
    ):
        """Return score terms explaining route selection."""
        if manufacturing_settings is not None:
            turn_objective = manufacturing_settings.get("turn_objective", turn_objective)
            huang_angle_threshold_degrees = manufacturing_settings.get(
                "huang_angle_threshold_degrees", huang_angle_threshold_degrees
            )
            setting_weights = manufacturing_settings.get("score_weights", {})
            angle_weight = setting_weights.get("angle", angle_weight)
            turn_120_weight = setting_weights.get("large_turn", turn_120_weight)
            turn_150_weight = setting_weights.get("severe_turn", turn_150_weight)
            backtrack_weight = setting_weights.get("backtrack", backtrack_weight)
            revisit_weight = setting_weights.get(
                "revisit_shortfall", revisit_weight
            )

        turn_objective = self._normalise_turn_objective_name(turn_objective)
        if turn_objective not in ("legacy", "thresholded_turn_energy"):
            raise ValueError("turn_objective must be 'legacy' or 'thresholded_turn_energy'.")

        metrics = self.route_metrics(
            route=route,
            huang_angle_threshold_degrees=huang_angle_threshold_degrees,
            verbose=False,
        )
        angles = metrics["angles"]

        revisit_distances = metrics[
            "revisit_distances"
        ]

        turn_120_excess = np.maximum(
            angles - 120.0,
            0.0,
        )

        turn_150_excess = np.maximum(
            angles - 150.0,
            0.0,
        )

        revisit_shortfall = np.maximum(
            target_revisit_distance
            - revisit_distances,
            0,
        )

        if turn_objective == "thresholded_turn_energy":
            huang_excess = np.maximum(
                angles - float(huang_angle_threshold_degrees), 0.0
            )
            terms = {
                "thresholded_turn_energy": float(np.mean(huang_excess ** 2)) if angles.size else 0.0,
                "turns_above_90": float(metrics["turns_above_90"]),
                "turns_above_120": float(metrics["turns_above_120"]),
                "turns_above_150": float(metrics["turns_above_150"]),
                "immediate_backtracks": float(metrics["immediate_backtracks"]),
                "revisit_shortfall": float(np.sum(revisit_shortfall ** 2)),
            }
        else:
            terms = {
                "angle_sum": float(angle_weight * np.sum(angles)),
                "turns_above_120": float(turn_120_weight * np.sum(turn_120_excess ** 2)),
                "turns_above_150": float(turn_150_weight * np.sum(turn_150_excess ** 2)),
                "immediate_backtracks": float(
                    backtrack_weight * metrics["immediate_backtracks"]
                ),
                "revisit_shortfall": float(
                    revisit_weight * np.sum(revisit_shortfall ** 2)
                ),
            }
        manufacturing = None
        feasible = True
        hard_violation_count = 0
        manufacturing_ready = True
        if manufacturing_settings is not None:
            manufacturing = self.check_manufacturing_constraints(
                route=route, settings=manufacturing_settings, verbose=False
            )
            feasible = bool(manufacturing["route_ordering_feasible"])
            manufacturing_ready = bool(manufacturing["manufacturing_ready"])
            hard_violation_count = len(manufacturing["violations"])
        total = float(sum(terms.values()))
        breakdown = {
            "total_score": total,
            "feasible": feasible,
            "manufacturing_ready": manufacturing_ready,
            "hard_violation_count": int(hard_violation_count),
            "terms": terms,
            "metrics": metrics,
            "manufacturing": manufacturing,
            "turn_objective": turn_objective,
            "huang_angle_threshold_degrees": float(huang_angle_threshold_degrees),
            "priority_order": [
                "valid closed Euler circuit",
                "zero immediate backtracks",
                "configured hard maximum angles",
                "minimum turns above 120 degrees",
                "minimum turns above 90 degrees",
                "minimum thresholded turn energy" if turn_objective == "thresholded_turn_energy" else "lower legacy weighted turn score",
                "better duplicate-pass revisit separation",
            ],
            "global_optimality_claim": "No global optimality is claimed; this is the best validated route found by the configured construction and refinement search.",
            "algorithm_note": (
                "The thresholded angular cost is article-inspired. This implementation "
                "retains the exact even-multiplicity Euler formulation: every "
                "fibre-edge instance is visited exactly once in a closed Euler circuit."
            ),
        }
        breakdown["rank_tuple"] = self._route_rank_tuple(breakdown)
        return breakdown

    @staticmethod
    def _route_rank_tuple(breakdown):
        metrics = breakdown.get("metrics", {})
        if breakdown.get("turn_objective") == "thresholded_turn_energy":
            manufacturing = breakdown.get("manufacturing") or {}
            hard_angle_violations = [
                item for item in manufacturing.get("route_ordering_violations", [])
                if item.get("kind") == "junction_turn"
            ]
            readiness_rank = 0
            if manufacturing:
                readiness_rank = 0 if manufacturing.get("manufacturing_ready", False) else 1
            return (
                readiness_rank,
                0 if int(metrics.get("immediate_backtracks", 0)) == 0 else 1,
                len(hard_angle_violations),
                int(metrics.get("turns_above_120", 0)),
                int(metrics.get("turns_above_90", 0)),
                float(metrics.get("thresholded_turn_energy", metrics.get("huang_energy", 0.0))),
                -float(metrics.get("minimum_revisit_distance", 0) or 0),
                float(breakdown["total_score"]),
            )
        manufacturing = breakdown.get("manufacturing")
        if manufacturing is None:
            return (0, 0, 0, float(breakdown["total_score"]))
        route_violations = len(manufacturing.get("route_ordering_violations", []))
        geometry_violations = len(manufacturing.get("requires_graph_or_geometry_change", []))
        machine_data = len(manufacturing.get("requires_machine_data", []))
        unknown = len(manufacturing.get("checks_awaiting_geometry_or_machine_data", []))
        readiness_rank = 0 if manufacturing.get("manufacturing_ready", False) else 1
        return (
            readiness_rank,
            int(route_violations),
            int(geometry_violations),
            int(machine_data + unknown),
            float(breakdown["total_score"]),
        )

    def _route_is_better(self, candidate_breakdown, best_breakdown):
        return tuple(candidate_breakdown["rank_tuple"]) < tuple(best_breakdown["rank_tuple"])

    def _halfedge_item(self, edge_id, direction, visit_number=1):
        u0, v0 = self.edge_index[int(edge_id)]
        if int(direction) >= 0:
            u, v = int(u0), int(v0)
        else:
            u, v = int(v0), int(u0)
        return {
            "u": u,
            "v": v,
            "key": (int(edge_id), int(direction), int(visit_number)),
            "edge_id": int(edge_id),
            "physical_edge_id": int(edge_id),
            "halfedge_direction": "forward" if int(direction) >= 0 else "reverse",
            "direction_sign": int(direction),
            "fibre_id": max(int(visit_number) - 1, 0),
            "visit_number": int(visit_number),
        }

    def _project_to_tangent_plane(self, vector, normal):
        normal = self._safe_unit(normal)
        if normal is None:
            return self._safe_unit(vector)
        projected = np.asarray(vector, dtype=np.float64) - np.dot(vector, normal) * normal
        return self._safe_unit(projected)

    def _node_normal(self, node_id):
        normals = []
        if self.edge_normals_xyz is None:
            return None
        for edge_id, (u, v) in enumerate(self.edge_index):
            if int(u) == int(node_id):
                normals.append(self.edge_normals_xyz[edge_id][0])
            if int(v) == int(node_id):
                normals.append(self.edge_normals_xyz[edge_id][-1])
        if not normals:
            return None
        return self._safe_unit(np.sum(normals, axis=0))

    def projected_halfedge_transition_angle(self, first_item, second_item):
        if int(first_item["v"]) != int(second_item["u"]):
            return None
        node_normal = self._node_normal(int(first_item["v"]))
        tangent_in = self.end_tangent(first_item)
        tangent_out = self.start_tangent(second_item)
        tangent_in = self._project_to_tangent_plane(tangent_in, node_normal)
        tangent_out = self._project_to_tangent_plane(tangent_out, node_normal)
        if tangent_in is None or tangent_out is None:
            return 180.0
        return float(np.degrees(np.arccos(np.clip(np.dot(tangent_in, tangent_out), -1.0, 1.0))))

    def _deprecated_huang_dual_graph_route(self, *args, **kwargs):
        raise NotImplementedError(
            "huang_dual_graph_route is out of scope for this implementation. "
            "The articles are used as references for surface offsets, junction "
            "intersections, sharp-turn reduction, and path representation; routing "
            "remains an exact even-multiplicity Euler circuit."
        )

    def compute_route(self, route_solver="euler_exact_even", **kwargs):
        if route_solver in ("euler_exact_even", "euler_exact_two", "euler"):
            return self.compute_euler_circuit(**kwargs)
        raise ValueError(
            "route_solver must be 'euler_exact_even'. Article dual-graph/variable-"
            "multiplicity solvers are intentionally not implemented here."
        )

    def compare_manufacturing_routes(self, *, local_offset_report=None):
        """Compare structural Euler route and local manufactured-offset status."""
        if self.G is None:
            self.build_multigraph()
        euler_route = self.compute_euler_circuit()
        euler_metrics = self.route_metrics(euler_route, verbose=False)
        return {
            "euler_exact_even_multiplicity_route": {
                "route_solver": "euler_exact_two" if self.fibres_per_edge == 2 else "euler_exact_even",
                "traversal_count": len(euler_route),
                "mean_angle": euler_metrics["mean_angle"],
                "max_angle": euler_metrics["max_angle"],
                "thresholded_turn_energy": euler_metrics.get("thresholded_turn_energy"),
            },
            "centreline_route": {
                "geometry": "structural_centreline",
                "manufacturing_offset": False,
            },
            "surface_offset_route": {
                "offset_method": "local_surface_projection",
                "label": "approximate_surface_constrained_offset",
                "report": local_offset_report,
            },
            "scope_note": (
                "The articles are used as references for surface offsetting, acceptable "
                "junction intersections, sharp-turn reduction and path representation. "
                "The implemented route remains an exact even-multiplicity Euler circuit "
                "and does not reproduce the complete algorithms proposed in those articles."
            ),
        }

    def _candidate_is_bridge(
        self,
        remaining_graph,
        u,
        v,
        key,
    ):
        """
        Return True if removing this specific multigraph edge
        disconnects the remaining non-isolated graph.

        Parallel edge copies are handled correctly because one
        specific MultiGraph key is removed at a time.
        """

        # If another parallel edge exists between the same nodes,
        # this edge instance cannot be a bridge.
        parallel_keys = list(
            remaining_graph[u][v].keys()
        )

        if len(parallel_keys) > 1:
            return False

        graph_copy = remaining_graph.copy()

        graph_copy.remove_edge(
            u,
            v,
            key=key,
        )

        # A leaf endpoint becoming isolated is precisely a bridge: checking
        # connectivity only among non-isolated nodes would miss that case.
        return not nx.has_path(graph_copy, u, v)

    def _make_traversal_item(
        self,
        graph,
        u,
        v,
        key,
    ):
        """
        Build one directed traversal dictionary from one
        undirected multigraph edge instance.
        """

        data = graph.get_edge_data(
            u,
            v,
            key,
        )

        return {
            "u": int(u),
            "v": int(v),
            "key": key,
            "edge_id": int(
                data["physical_edge_id"]
            ),
            "fibre_id": int(
                data["fibre_id"]
            ),
        }

    def _construct_turn_aware_euler_route(
        self,
        start_node,
        rng,
        angle_weight=1.0,
        turn_120_weight=20.0,
        turn_150_weight=100.0,
        same_edge_penalty=100000.0,
        recent_edge_weight=500.0,
        random_jitter=1.0,
        avoid_same_edge_when_possible=True,
        turn_objective="legacy",
        huang_angle_threshold_degrees=60.0,
    ):
        """
        Construct one complete Euler circuit using a
        turn-aware Fleury-style strategy.

        Selection priorities:
        1. Avoid bridges unless no alternative exists.
        2. Avoid immediately reusing the same physical edge.
        3. Minimise the local turning angle.
        4. Prefer physical edges not used recently.

        Returns
        -------
        route : list[dict]
            Directed Euler traversals.

        Important
        ---------
        This method removes one multigraph edge instance at each
        step and validates that every fibre-edge instance is used
        exactly once.
        """

        if self.G is None:
            raise RuntimeError(
                "Run build_multigraph() first."
            )

        remaining_graph = self.G.copy()

        total_instances = (
            remaining_graph.number_of_edges()
        )

        current_node = int(
            start_node
        )

        route = []

        last_used_step = {}

        for step in range(
            total_instances
        ):

            incident_edges = list(
                remaining_graph.edges(
                    current_node,
                    keys=True,
                )
            )

            if len(incident_edges) == 0:
                raise RuntimeError(
                    "The turn-aware traversal became stuck "
                    f"at node {current_node} after {step} "
                    f"of {total_instances} traversals."
                )

            candidates = []

            for u_raw, v_raw, key in incident_edges:

                # MultiGraph.edges(node) normally returns current_node
                # first, but handle both orientations safely.
                if int(u_raw) == current_node:
                    next_node = int(v_raw)
                else:
                    next_node = int(u_raw)

                item = self._make_traversal_item(
                    remaining_graph,
                    current_node,
                    next_node,
                    key,
                )

                is_bridge = self._candidate_is_bridge(
                    remaining_graph,
                    current_node,
                    next_node,
                    key,
                )

                if len(route) == 0:
                    turn_angle = 0.0
                    same_edge = False
                else:
                    record = self._transition_record(
                        route[-1], item,
                        threshold_degrees=huang_angle_threshold_degrees,
                    )
                    turn_angle = record["angle"]

                    same_edge = (
                        int(route[-1]["edge_id"])
                        == int(item["edge_id"])
                    )

                # The last edge also creates a junction with the first edge.
                closure_cost = 0.0
                closure_backtrack = False
                if step == total_instances - 1 and route:
                    closure_record = self._transition_record(
                        item, route[0],
                        threshold_degrees=huang_angle_threshold_degrees,
                    )
                    closure_angle = closure_record["angle"]
                    if turn_objective == "thresholded_turn_energy":
                        closure_cost = closure_record["huang_cost"]
                    else:
                        closure_cost = self._angle_cost(
                            closure_angle, angle_weight,
                            turn_120_weight, turn_150_weight,
                        )
                    closure_backtrack = int(item["edge_id"]) == int(route[0]["edge_id"])

                edge_id = int(
                    item["edge_id"]
                )

                previous_use = last_used_step.get(
                    edge_id,
                    None,
                )

                if previous_use is None:
                    revisit_penalty = 0.0
                else:
                    revisit_distance = (
                        step - previous_use
                    )

                    revisit_penalty = (
                        recent_edge_weight
                        / max(
                            revisit_distance,
                            1,
                        )
                    )

                if turn_objective == "thresholded_turn_energy":
                    local_cost = self._huang_cost(
                        turn_angle, threshold_degrees=huang_angle_threshold_degrees
                    ) + closure_cost
                else:
                    local_cost = self._angle_cost(
                        turn_angle, angle_weight,
                        turn_120_weight, turn_150_weight,
                    ) + closure_cost

                if same_edge:
                    local_cost += (
                        same_edge_penalty
                    )

                if closure_backtrack:
                    local_cost += same_edge_penalty

                local_cost += (
                    revisit_penalty
                )

                local_cost += (
                    random_jitter
                    * float(
                        rng.random()
                    )
                )

                candidates.append(
                    {
                        "item": item,
                        "key": key,
                        "next_node": next_node,
                        "is_bridge": bool(
                            is_bridge
                        ),
                        "turn_angle": float(
                            turn_angle
                        ),
                        "same_edge": bool(
                            same_edge
                        ),
                        "cost": float(
                            local_cost
                        ),
                    }
                )

            # Fleury rule:
            # Do not select a bridge while a non-bridge alternative exists.
            non_bridge_candidates = [
                candidate
                for candidate in candidates
                if not candidate[
                    "is_bridge"
                ]
            ]

            if non_bridge_candidates:
                allowed_candidates = (
                    non_bridge_candidates
                )
            else:
                allowed_candidates = (
                    candidates
                )

            # Hard preference:
            # Exclude the second copy of the same physical edge
            # whenever another Euler-safe continuation exists.
            if route and avoid_same_edge_when_possible:

                different_edge_candidates = [
                    candidate
                    for candidate
                    in allowed_candidates
                    if not candidate[
                        "same_edge"
                    ]
                ]

                if different_edge_candidates:
                    allowed_candidates = (
                        different_edge_candidates
                    )

            selected = min(
                allowed_candidates,
                key=lambda candidate:
                    candidate["cost"],
            )

            selected_item = selected[
                "item"
            ]

            route.append(
                selected_item
            )

            selected_edge_id = int(
                selected_item["edge_id"]
            )

            last_used_step[
                selected_edge_id
            ] = step

            remaining_graph.remove_edge(
                current_node,
                selected[
                    "next_node"
                ],
                key=selected["key"],
            )

            current_node = selected[
                "next_node"
            ]

        if remaining_graph.number_of_edges() != 0:
            raise RuntimeError(
                "The turn-aware constructor did not use "
                "all fibre-edge instances."
            )

        if len(route) != total_instances:
            raise RuntimeError(
                "The generated route has an incorrect "
                "number of traversals."
            )

        if route[-1]["v"] != route[0]["u"]:
            raise RuntimeError(
                "The generated route is not closed."
            )

        validation = self.validate_route(
            route=route,
            raise_on_error=False,
            verbose=False,
        )

        if not validation["valid"]:
            raise RuntimeError(
                "The generated turn-aware route failed "
                "Euler-route validation."
            )

        return route

    def _refine_closed_subtrails(
        self, route, score_kwargs, rng, passes=2, candidates_per_pass=300,
    ):
        """Reverse closed subtrails when doing so improves the full route score.

        An undirected closed subtrail can be reversed without changing the
        traversed fibre instances or disconnecting the Euler circuit.
        """
        best = [dict(item) for item in route]
        best_breakdown = self.route_score(best, return_breakdown=True, **score_kwargs)
        improvements = 0
        for _ in range(passes):
            # Find segments whose starting and ending nodes coincide.
            pairs = [(i, j) for i in range(len(best) - 1)
                     for j in range(i + 1, len(best))
                     if best[i]["u"] == best[j]["v"]]
            if not pairs:
                break
            indices = rng.permutation(len(pairs))[:candidates_per_pass]
            improved = False
            for index in indices:
                i, j = pairs[int(index)]
                reversed_segment = [
                    {**item, "u": item["v"], "v": item["u"]}
                    for item in reversed(best[i:j + 1])
                ]
                candidate = best[:i] + reversed_segment + best[j + 1:]
                candidate_breakdown = self.route_score(
                    candidate, return_breakdown=True, **score_kwargs
                )
                if self._route_is_better(candidate_breakdown, best_breakdown):
                    self.validate_route(candidate, raise_on_error=True, verbose=False)
                    best, best_breakdown = candidate, candidate_breakdown
                    improvements += 1
                    improved = True
                    break  # Node positions may have changed; rebuild the pairs.
            if not improved:
                break
        return best, float(best_breakdown["total_score"]), improvements

    def optimize_turn_aware_euler_circuit(
        self,
        trials=200,
        random_seed=42,
        angle_weight=1.0,
        same_edge_penalty=100000.0,
        recent_edge_weight=500.0,
        random_jitter=2.0,
        avoid_same_edge_when_possible=True,
        score_angle_weight=1.0,
        score_turn_120_weight=20.0,
        score_turn_150_weight=100.0,
        score_backtrack_weight=100000.0,
        score_revisit_weight=100.0,
        target_revisit_distance=10,
        turn_objective=None,
        huang_angle_threshold_degrees=60.0,
        refinement_passes=2,
        refinement_candidates_per_pass=300,
        manufacturing_settings=None,
        infeasible_policy=None,
        verbose=True,
    ):
        """
        Generate turn-aware Euler circuits and retain the
        lowest-cost valid route.

        Different trials vary:
        - start node,
        - tie-breaking jitter.

        Every candidate is a complete, validated Euler circuit.
        """

        if self.G is None:
            self.build_multigraph()

        self.validate_graph(
            verbose=False
        )

        if self.euler_edges is None:
            self.compute_euler_circuit()

        if int(trials) < 0 or int(refinement_passes) < 0 or int(refinement_candidates_per_pass) < 0:
            raise ValueError("Trial and refinement counts must be nonnegative.")
        if manufacturing_settings is not None:
            manufacturing_settings = self.default_manufacturing_settings(
                **manufacturing_settings
            )
            if turn_objective is None:
                turn_objective = manufacturing_settings.get("turn_objective", "legacy")
            huang_angle_threshold_degrees = manufacturing_settings.get(
                "huang_angle_threshold_degrees", huang_angle_threshold_degrees
            )
            self._route_independent_manufacturing_checks(manufacturing_settings)
            if infeasible_policy is None:
                infeasible_policy = manufacturing_settings.get(
                    "infeasible_policy", "return_best"
                )
        if infeasible_policy is None:
            infeasible_policy = "return_best"
        if infeasible_policy not in ("return_best", "raise"):
            raise ValueError("infeasible_policy must be 'return_best' or 'raise'.")
        turn_objective = self._normalise_turn_objective_name(turn_objective)
        if turn_objective not in ("legacy", "thresholded_turn_energy"):
            raise ValueError("turn_objective must be 'legacy' or 'thresholded_turn_energy'.")

        rng = np.random.default_rng(
            random_seed
        )

        baseline_route = [
            dict(item)
            for item in self.euler_edges
        ]

        score_kwargs = dict(
            angle_weight=score_angle_weight,
            turn_120_weight=score_turn_120_weight,
            turn_150_weight=score_turn_150_weight,
            backtrack_weight=score_backtrack_weight,
            revisit_weight=score_revisit_weight,
            target_revisit_distance=target_revisit_distance,
            manufacturing_settings=manufacturing_settings,
            turn_objective=turn_objective,
            huang_angle_threshold_degrees=huang_angle_threshold_degrees,
        )
        baseline_score = self.route_score(baseline_route, **score_kwargs)
        baseline_breakdown = self.route_score(baseline_route, return_breakdown=True, **score_kwargs)

        best_route = baseline_route
        best_breakdown = baseline_breakdown
        best_score = float(best_breakdown["total_score"])

        valid_candidates = 0
        failed_candidates = 0
        best_route_updates = 0

        graph_nodes = np.asarray(
            list(self.G.nodes),
            dtype=np.int64,
        )

        for trial in range(
            int(trials)
        ):

            start_node = int(
                rng.choice(
                    graph_nodes
                )
            )

            try:
                candidate_route = (
                    self._construct_turn_aware_euler_route(
                        start_node=
                            start_node,
                        rng=rng,
                        angle_weight=
                            angle_weight,
                        turn_120_weight=score_turn_120_weight,
                        turn_150_weight=score_turn_150_weight,
                        same_edge_penalty=
                            same_edge_penalty,
                        recent_edge_weight=
                            recent_edge_weight,
                        random_jitter=
                            random_jitter,
                        avoid_same_edge_when_possible=avoid_same_edge_when_possible,
                        turn_objective=turn_objective,
                        huang_angle_threshold_degrees=huang_angle_threshold_degrees,
                    )
                )

            except RuntimeError:
                failed_candidates += 1
                continue

            validation = self.validate_route(
                route=candidate_route,
                raise_on_error=False,
                verbose=False,
            )

            if not validation["valid"]:
                failed_candidates += 1
                continue

            valid_candidates += 1

            candidate_breakdown = self.route_score(
                candidate_route, return_breakdown=True, **score_kwargs
            )

            if self._route_is_better(candidate_breakdown, best_breakdown):

                best_breakdown = candidate_breakdown
                best_score = float(best_breakdown["total_score"])

                best_route = [
                    dict(item)
                    for item
                    in candidate_route
                ]

                best_route_updates += 1

        best_route, best_score, refinement_updates = self._refine_closed_subtrails(
            best_route, score_kwargs, rng,
            passes=int(refinement_passes),
            candidates_per_pass=int(refinement_candidates_per_pass),
        )
        best_breakdown = self.route_score(best_route, return_breakdown=True, **score_kwargs)

        self.baseline_euler_edges = [
            dict(item)
            for item
            in baseline_route
        ]

        self.euler_edges = [
            dict(item)
            for item
            in best_route
        ]

        final_validation = self.validate_route(
            route=self.euler_edges,
            raise_on_error=True,
            verbose=False,
        )

        baseline_metrics = self.route_metrics(
            route=self.baseline_euler_edges,
            verbose=False,
        )

        optimized_metrics = self.route_metrics(
            route=self.euler_edges,
            verbose=False,
        )
        optimized_breakdown = self.route_score(
            self.euler_edges, return_breakdown=True, **score_kwargs
        )
        self.baseline_score_breakdown = baseline_breakdown
        self.score_breakdown = optimized_breakdown
        self.baseline_manufacturing_report = baseline_breakdown["manufacturing"]
        self.manufacturing_report = optimized_breakdown["manufacturing"]

        if verbose:

            print(
                f"Turn-aware trials      : "
                f"{trials}"
            )

            print(
                f"Valid candidates       : "
                f"{valid_candidates}"
            )

            print(
                f"Failed candidates      : "
                f"{failed_candidates}"
            )

            print(
                f"Best-route updates     : "
                f"{best_route_updates}"
            )

            print(f"Refinement updates     : {refinement_updates}")

            print(
                f"Baseline score         : "
                f"{baseline_score:.2f}"
            )

            print(
                f"Optimised score        : "
                f"{best_score:.2f}"
            )
            if manufacturing_settings is not None:
                print(
                    f"Baseline feasible      : "
                    f"{baseline_breakdown['manufacturing_ready']} "
                    f"({baseline_breakdown['hard_violation_count']} violations)"
                )
                print(
                    f"Optimised feasible     : "
                    f"{optimized_breakdown['manufacturing_ready']} "
                    f"({optimized_breakdown['hard_violation_count']} violations)"
                )
                if not optimized_breakdown["manufacturing_ready"]:
                    print(
                        "No manufacturing-ready route was found for the configured "
                        "limits. Returning the best Euler-valid route found with "
                        "labelled violations."
                    )

            print()

            self.print_metrics(
                baseline_metrics,
                title="Baseline route",
            )

            print()

            self.print_metrics(
                optimized_metrics,
                title="Turn-aware route",
            )

            print()

            print(
                f"Final route valid      : "
                f"{final_validation['valid']}"
            )

        if manufacturing_settings is not None and not optimized_breakdown["manufacturing_ready"]:
            if infeasible_policy == "raise":
                report = optimized_breakdown["manufacturing"]
                raise ManufacturingInfeasibilityError(
                    "No manufacturing-ready Euler route was found. "
                    f"Route-ordering violations: {len(report['route_ordering_violations'])}; "
                    f"geometry changes required: {len(report['requires_graph_or_geometry_change'])}; "
                    f"machine data required: {len(report['requires_machine_data'])}."
                )

        return (
            self.euler_edges,
            optimized_metrics,
        )

    def export_ordered_path(
        self,
        route=None,
        filename=None,
        format=None,
        remove_duplicate_junctions=True,
        nozzle_approach="+normal",
        frame_tolerance=1.0e-8,
        surface_evaluator=None,
        surface_vertices=None,
        surface_faces=None,
        fibre_line_width=None,
        strut_thickness=None,
        boundary_margin=0.0,
        min_pass_spacing=None,
        min_bend_radius=None,
        max_manufactured_turn_degrees=None,
        mesh_projection_tolerance=None,
        transition_samples=5,
        continuity_tolerance=1.0e-6,
        allow_incomplete_geometry=False,
        require_machine_ready=False,
        junction_trim_distance=None,
    ):
        """
        Export an ordered, machine-independent fibre path.

        Each point contains XYZ, supplied CAD-surface normal, recomputed travel
        tangent, edge/fibre/traversal IDs, and optional supplied surface
        position. The nozzle approach convention is recorded as ``+normal`` or
        ``-normal``; full printer-head pose generation belongs in a separate
        machine-specific postprocessor.
        """
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        if self.edge_normals_xyz is None and self.edge_pass_normals_xyz is None:
            raise RuntimeError(
                "export_ordered_path requires CAD-surface normals supplied as "
                "edge_normals_xyz or edge_pass_normals_xyz. Normals are not invented from a flat projection."
            )
        if nozzle_approach not in ("+normal", "-normal"):
            raise ValueError("nozzle_approach must be '+normal' or '-normal'.")
        approach_sign = 1.0 if nozzle_approach == "+normal" else -1.0
        self.validate_route(route, raise_on_error=True, verbose=False)
        if not allow_incomplete_geometry:
            validation = self.validate_manufactured_paths(
                route,
                surface_evaluator=surface_evaluator,
                surface_vertices=surface_vertices,
                surface_faces=surface_faces,
                fibre_line_width=fibre_line_width,
                strut_thickness=strut_thickness,
                boundary_margin=boundary_margin,
                min_pass_spacing=min_pass_spacing,
                min_bend_radius=min_bend_radius,
                max_manufactured_turn_degrees=max_manufactured_turn_degrees,
                mesh_projection_tolerance=mesh_projection_tolerance,
                junction_trim_distance=junction_trim_distance,
                transition_samples=transition_samples,
                tolerance=continuity_tolerance,
                verbose=False,
            )
            if not self._geometry_ready_for_export(validation, require_machine_ready=require_machine_ready):
                raise ManufacturingInfeasibilityError(
                    "Refusing to export incomplete manufacturing geometry. "
                    f"route_ordering_ready={validation['route_ordering_ready']}, "
                    f"offset_geometry_ready={validation['offset_geometry_ready']}, "
                    f"machine_ready={validation['machine_ready']}, "
                    f"missing_data={validation['missing_data']}, "
                    f"violations={len(validation['violations'])}."
                )
        route = self.route_with_lane_assignment(route)
        lane_mapping = self.assign_route_lanes(route)
        transition_report = self.build_junction_transitions(
            route,
            surface_evaluator=surface_evaluator,
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            transition_samples=transition_samples,
            transition_tolerance=continuity_tolerance,
            min_bend_radius=min_bend_radius,
            max_joined_turn_degrees=max_manufactured_turn_degrees,
            mesh_projection_tolerance=mesh_projection_tolerance,
            junction_trim_distance=junction_trim_distance,
        )
        transitions = transition_report["transitions"]

        rows = []
        violations = []
        previous_normal = None
        previous_point = None
        path_index = 0
        for traversal_id, item in enumerate(route):
            transition = transitions[traversal_id]
            if not transition.get("accepted", True):
                raise ManufacturingInfeasibilityError(
                    f"Refusing to export failed transition {traversal_id}; no accepted connector geometry exists."
                )
            curve = self._manufactured_lane_curve_for_transition(route, transitions, traversal_id)
            normals = transition.get("manufactured_lane_normals", self.get_oriented_pass_normals(item))
            if len(normals) != len(curve):
                normals = self._transition_normals_from_mesh_or_linear(
                    curve,
                    normals[0],
                    normals[-1],
                    surface_vertices,
                    surface_faces,
                )
            surface_positions = self.get_oriented_surface_positions(item)
            tangents = self._curve_tangents(curve)

            for local_index in range(len(curve)):
                point = curve[local_index]
                if (
                    remove_duplicate_junctions
                    and previous_point is not None
                    and np.linalg.norm(point - previous_point) <= continuity_tolerance
                ):
                    continue
                surface_normal = normals[local_index]
                normal = approach_sign * surface_normal
                tangent = tangents[local_index]
                tangent_norm = float(np.linalg.norm(tangent))
                normal_norm = float(np.linalg.norm(normal))
                frame_norm = float(np.linalg.norm(np.cross(tangent, normal)))
                if tangent_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_tangent",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index)},
                        [item["edge_id"]],
                        tangent_norm,
                        frame_tolerance,
                        "Travel tangent is not usable at this path point.",
                    ))
                if normal_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_normal",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index)},
                        [item["edge_id"]],
                        normal_norm,
                        frame_tolerance,
                        "CAD normal is not usable at this path point.",
                    ))
                if frame_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_frame",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index)},
                        [item["edge_id"]],
                        frame_norm,
                        frame_tolerance,
                        "Travel tangent and CAD normal do not define a usable local frame.",
                    ))
                if previous_normal is not None and np.dot(previous_normal, normal) < -0.05:
                    violations.append(self._as_violation(
                        "normal_orientation_flip",
                        {"path_index": int(path_index)},
                        [item["edge_id"]],
                        float(np.dot(previous_normal, normal)),
                        -0.05,
                        "CAD normals are not consistently oriented along the ordered path.",
                    ))
                row = {
                    "path_index": int(path_index),
                    "traversal_id": int(traversal_id),
                    "physical_edge_id": int(item["edge_id"]),
                    "original_physical_edge_id": int(self.original_physical_edge_ids[int(item["edge_id"])]),
                    "edge_id": int(item["edge_id"]),
                    "fibre_id": int(item["fibre_id"]),
                    "fibre_instance_id": int(item.get("fibre_instance_id", item["fibre_id"])),
                    "u": int(item["u"]),
                    "v": int(item["v"]),
                    "offset_lane": None if "offset_lane" not in item else int(item["offset_lane"]),
                    "lane_id": None if "lane_id" not in item else int(item["lane_id"]),
                    "signed_offset": item.get("signed_offset"),
                    "fibre_on": True,
                    "point_type": "fibre_lane",
                    "point_index_in_traversal": int(local_index),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "normal_x": float(normal[0]),
                    "normal_y": float(normal[1]),
                    "normal_z": float(normal[2]),
                    "tangent_x": float(tangent[0]),
                    "tangent_y": float(tangent[1]),
                    "tangent_z": float(tangent[2]),
                }
                if surface_positions is not None:
                    surface = surface_positions[local_index]
                    row.update({
                        "surface_x": float(surface[0]),
                        "surface_y": float(surface[1]),
                        "surface_z": float(surface[2]),
                    })
                rows.append(row)
                previous_normal = normal
                previous_point = point
                path_index += 1
            transition_point_type = "closure_transition" if transition.get("closing") else "junction_transition"
            transition_points = transition["points"]
            transition_normals = transition["normals"]
            transition_tangents = self._curve_tangents(transition_points)
            for local_index, point in enumerate(transition_points):
                if (
                    remove_duplicate_junctions
                    and previous_point is not None
                    and np.linalg.norm(point - previous_point) <= continuity_tolerance
                ):
                    continue
                surface_normal = transition_normals[local_index]
                normal = approach_sign * surface_normal
                tangent = transition_tangents[local_index]
                tangent_norm = float(np.linalg.norm(tangent))
                normal_norm = float(np.linalg.norm(normal))
                frame_norm = float(np.linalg.norm(np.cross(tangent, normal)))
                if tangent_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_tangent",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index), "point_type": transition_point_type},
                        transition["edge_ids"],
                        tangent_norm,
                        frame_tolerance,
                        "Travel tangent is not usable at this transition point.",
                    ))
                if normal_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_normal",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index), "point_type": transition_point_type},
                        transition["edge_ids"],
                        normal_norm,
                        frame_tolerance,
                        "CAD/surface normal is not usable at this transition point.",
                    ))
                if frame_norm <= frame_tolerance:
                    violations.append(self._as_violation(
                        "path_frame",
                        {"traversal_id": int(traversal_id), "point_index": int(local_index), "point_type": transition_point_type},
                        transition["edge_ids"],
                        frame_norm,
                        frame_tolerance,
                        "Travel tangent and surface normal do not define a usable local frame.",
                    ))
                if previous_normal is not None and np.dot(previous_normal, normal) < -0.05:
                    violations.append(self._as_violation(
                        "normal_orientation_flip",
                        {"path_index": int(path_index)},
                        transition["edge_ids"],
                        float(np.dot(previous_normal, normal)),
                        -0.05,
                        "Normals are not consistently oriented along the ordered path.",
                    ))
                rows.append({
                    "path_index": int(path_index),
                    "traversal_id": int(traversal_id),
                    "physical_edge_id": int(item["edge_id"]),
                    "original_physical_edge_id": int(self.original_physical_edge_ids[int(item["edge_id"])]),
                    "edge_id": int(item["edge_id"]),
                    "fibre_id": int(item["fibre_id"]),
                    "fibre_instance_id": int(item.get("fibre_instance_id", item["fibre_id"])),
                    "u": int(item["u"]),
                    "v": int(item["v"]),
                    "offset_lane": int(item.get("offset_lane", item["fibre_id"])),
                    "lane_id": int(item.get("lane_id", item.get("offset_lane", item["fibre_id"]))),
                    "signed_offset": item.get("signed_offset"),
                    "fibre_on": True,
                    "point_type": transition_point_type,
                    "point_index_in_traversal": int(local_index),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "normal_x": float(normal[0]),
                    "normal_y": float(normal[1]),
                    "normal_z": float(normal[2]),
                    "tangent_x": float(tangent[0]),
                    "tangent_y": float(tangent[1]),
                    "tangent_z": float(tangent[2]),
                })
                previous_normal = normal
                previous_point = point
                path_index += 1

        if rows:
            ordered_points = np.asarray([[row["x"], row["y"], row["z"]] for row in rows], dtype=np.float64)
            ordered_tangents = self._curve_tangents(ordered_points)
            for row, tangent in zip(rows, ordered_tangents):
                row["tangent_x"] = float(tangent[0])
                row["tangent_y"] = float(tangent[1])
                row["tangent_z"] = float(tangent[2])

        metadata = {
            "nozzle_approach": nozzle_approach,
            "point_count": len(rows),
            "traversal_count": len(route),
            "route_to_lane_mapping": lane_mapping,
            "original_physical_edge_ids": self.original_physical_edge_ids.tolist(),
            "physical_edge_metadata": dict(self.physical_edge_metadata),
            "frame_valid": len(violations) == 0,
            "violations": violations,
            "junction_transition_count": len(transitions),
            "postprocessor_note": (
                "This export is machine-independent. G-code or robot-command "
                "generation must be performed separately with explicit machine "
                "settings, coordinate frame, extrusion calibration, speeds, "
                "tool-orientation rules, and collision constraints."
            ),
        }
        if violations:
            first = violations[0]
            raise RuntimeError(
                "Ordered path local-frame validation failed: "
                f"{first['kind']} at {first['location']} "
                f"value={first['measured_value']} limit={first['limit']}"
            )

        if filename is not None:
            if format is None:
                suffix = str(filename).lower().rsplit(".", 1)[-1]
                format = "json" if suffix == "json" else "csv"
            if format == "json":
                with open(filename, "w", encoding="utf-8") as handle:
                    json.dump({"metadata": metadata, "points": rows}, handle, indent=2)
            elif format == "csv":
                if rows:
                    fieldnames = list(rows[0].keys())
                else:
                    fieldnames = []
                with open(filename, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                raise ValueError("format must be 'csv' or 'json'.")
        return {"metadata": metadata, "points": rows}

    def export_minimal_xyz_normal_path(
        self,
        route=None,
        filename=None,
        delimiter=",",
        **kwargs,
    ):
        """
        Export or return the minimal geometric deposition path as [N, 6].

        Columns are ``x, y, z, normal_x, normal_y, normal_z``. Validation and
        route-to-lane assignment are delegated to ``export_ordered_path``.
        """
        exported = self.export_ordered_path(route=route, **kwargs)
        points = exported["points"]
        data = np.asarray(
            [
                [
                    row["x"],
                    row["y"],
                    row["z"],
                    row["normal_x"],
                    row["normal_y"],
                    row["normal_z"],
                ]
                for row in points
            ],
            dtype=np.float64,
        )
        if filename is not None:
            np.savetxt(filename, data, delimiter=delimiter)
        return data

    def build_xyz_toolpath(
        self,
        route=None,
        remove_duplicate_junctions=True,
    ):
        """
        Concatenate oriented structural centreline curves into one XYZ polyline.

        This is not manufacturing deposition geometry. Use
        ``build_manufactured_xyz_toolpath`` after offset-pass and junction
        generation when exporting a physical fibre path.
        """

        if route is None:
            route = self.euler_edges

        if route is None:
            raise RuntimeError(
                "No Euler route is available."
            )

        path_segments = []

        for traversal_index, item in enumerate(
            route
        ):
            curve = (
                self.get_oriented_curve(
                    item
                ).copy()
            )

            if (
                remove_duplicate_junctions
                and traversal_index > 0
            ):
                curve = curve[1:]

            if len(curve) > 0:
                path_segments.append(
                    curve
                )

        if len(path_segments) == 0:
            toolpath = np.empty(
                (0, 3),
                dtype=np.float64,
            )
        else:
            toolpath = np.vstack(
                path_segments
            )

        self.toolpath_xyz = toolpath

        return toolpath

    def build_manufactured_xyz_toolpath(
        self,
        route=None,
        surface_evaluator=None,
        surface_vertices=None,
        surface_faces=None,
        transition_samples=5,
        continuity_tolerance=1.0e-6,
        validate_geometry=True,
        allow_incomplete_geometry=False,
        fibre_line_width=None,
        strut_thickness=None,
        boundary_margin=0.0,
        min_pass_spacing=None,
        min_bend_radius=None,
        max_manufactured_turn_degrees=None,
        mesh_projection_tolerance=None,
        require_machine_ready=False,
        junction_trim_distance=None,
    ):
        """Concatenate lane-assigned offset curves and junction transitions."""
        if route is None:
            route = self.euler_edges
        if route is None:
            raise RuntimeError("No Euler route is available.")
        if self.edge_pass_curves_xyz is None:
            raise ManufacturingGeometryError("build_manufactured_xyz_toolpath requires edge_pass_curves_xyz.")
        if validate_geometry and not allow_incomplete_geometry:
            validation = self.validate_manufactured_paths(
                route,
                surface_evaluator=surface_evaluator,
                surface_vertices=surface_vertices,
                surface_faces=surface_faces,
                fibre_line_width=fibre_line_width,
                strut_thickness=strut_thickness,
                boundary_margin=boundary_margin,
                min_pass_spacing=min_pass_spacing,
                min_bend_radius=min_bend_radius,
                max_manufactured_turn_degrees=max_manufactured_turn_degrees,
                mesh_projection_tolerance=mesh_projection_tolerance,
                junction_trim_distance=junction_trim_distance,
                transition_samples=transition_samples,
                tolerance=continuity_tolerance,
                verbose=False,
            )
            if not self._geometry_ready_for_export(validation, require_machine_ready=require_machine_ready):
                raise ManufacturingInfeasibilityError(
                    "Refusing to build manufactured XYZ from invalid manufacturing geometry. "
                    f"route_ordering_ready={validation['route_ordering_ready']}, "
                    f"offset_geometry_ready={validation['offset_geometry_ready']}, "
                    f"machine_ready={validation['machine_ready']}, "
                    f"missing_data={validation['missing_data']}, "
                    f"violations={len(validation['violations'])}."
                )
        route_lanes = self.route_with_lane_assignment(route)
        transition_report = self.build_junction_transitions(
            route_lanes,
            surface_evaluator=surface_evaluator,
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            transition_samples=transition_samples,
            transition_tolerance=continuity_tolerance,
            min_bend_radius=min_bend_radius,
            max_joined_turn_degrees=max_manufactured_turn_degrees,
            mesh_projection_tolerance=mesh_projection_tolerance,
            junction_trim_distance=junction_trim_distance,
        )
        segments = []
        previous = None
        for traversal_id, item in enumerate(route_lanes):
            transition = transition_report["transitions"][traversal_id]
            if not transition.get("accepted", True):
                raise ManufacturingInfeasibilityError(
                    f"Refusing to build manufactured XYZ with failed transition {traversal_id}."
                )
            for segment in (self._manufactured_lane_curve_for_transition(route_lanes, transition_report["transitions"], traversal_id), transition["points"]):
                segment = np.asarray(segment, dtype=np.float64)
                if previous is not None and len(segment) and np.linalg.norm(segment[0] - previous) <= continuity_tolerance:
                    segment = segment[1:]
                if len(segment):
                    segments.append(segment)
                    previous = segment[-1]
        if not segments:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(segments)

    # =========================================================
    # Before and after comparison
    # =========================================================

    def compare_routes(
        self,
    ):
        if self.baseline_euler_edges is None:
            raise RuntimeError(
                "No baseline route is stored."
            )

        if self.euler_edges is None:
            raise RuntimeError(
                "No current route is stored."
            )

        baseline = self.route_metrics(
            self.baseline_euler_edges,
            verbose=False,
        )

        optimized = self.route_metrics(
            self.euler_edges,
            verbose=False,
        )

        print("Baseline versus optimized")
        print("---------------------------")

        items = [
            (
                "Mean angle",
                baseline["mean_angle"],
                optimized["mean_angle"],
            ),
            (
                "Maximum angle",
                baseline["max_angle"],
                optimized["max_angle"],
            ),
            (
                "Full-path maximum angle",
                baseline["overall_max_angle"],
                optimized["overall_max_angle"],
            ),
            (
                "Turns above 120",
                baseline[
                    "turns_above_120"
                ],
                optimized[
                    "turns_above_120"
                ],
            ),
            (
                "Turns above 150",
                baseline[
                    "turns_above_150"
                ],
                optimized[
                    "turns_above_150"
                ],
            ),
            (
                "Immediate backtracks",
                baseline[
                    "immediate_backtracks"
                ],
                optimized[
                    "immediate_backtracks"
                ],
            ),
            (
                "Mean revisit distance",
                baseline[
                    "mean_revisit_distance"
                ],
                optimized[
                    "mean_revisit_distance"
                ],
            ),
        ]

        for name, before, after in items:
            print(
                f"{name:25s}: "
                f"{before:8.2f} -> "
                f"{after:8.2f}"
            )

        return {
            "baseline": baseline,
            "optimized": optimized,
        }

    def analyze_topological_uturn_limits(
    self,
        verbose=True,
    ):
        """
        Analyze physical-graph features that can force U-turns.

        A degree-1 physical node forces an immediate reversal when
        every physical edge is represented by parallel fibre passes.

        Bridges do not always force an immediate reversal, but they
        identify branches that the closed Euler circuit must enter
        and later leave through the same physical connection.
        """

        physical_graph = nx.MultiGraph()

        physical_graph.add_edges_from(
            (
                int(u),
                int(v),
            )
            for u, v in self.edge_index
        )

        degree_one_nodes = sorted(
            int(node)
            for node, degree
            in physical_graph.degree()
            if degree == 1
        )

        degree_two_nodes = sorted(
            int(node)
            for node, degree
            in physical_graph.degree()
            if degree == 2
        )

        branch_nodes = sorted(
            int(node)
            for node, degree
            in physical_graph.degree()
            if degree >= 3
        )

        bridges = sorted(
            (
                int(min(u, v)),
                int(max(u, v)),
            )
            for u, v
            in nx.bridges(nx.Graph(physical_graph))
        )

        terminal_edge_ids = []

        degree_one_set = set(
            degree_one_nodes
        )

        for edge_id, (u, v) in enumerate(
            self.edge_index
        ):
            if (
                int(u) in degree_one_set
                or int(v) in degree_one_set
            ):
                terminal_edge_ids.append(
                    int(edge_id)
                )

        # With two fibre copies, each degree-1 terminal edge forces
        # one immediate pass reversal at its terminal node.
        if self.fibres_per_edge == 2:
            minimum_forced_backtracks = len(
                terminal_edge_ids
            )
        else:
            # For 2k parallel copies at a terminal node, all copies
            # must be paired locally. Each pair creates a reversal.
            minimum_forced_backtracks = (
                len(terminal_edge_ids)
                * self.fibres_per_edge
                // 2
            )

        report = {
            "physical_node_count":
                physical_graph.number_of_nodes(),

            "physical_edge_count":
                physical_graph.number_of_edges(),

            "degree_one_nodes":
                degree_one_nodes,

            "degree_two_nodes":
                degree_two_nodes,

            "branch_nodes":
                branch_nodes,

            "bridges":
                bridges,

            "terminal_edge_ids":
                terminal_edge_ids,

            "minimum_forced_backtracks":
                int(minimum_forced_backtracks),
        }

        if verbose:

            print("Physical graph topology")
            print("-----------------------")

            print(
                f"Physical nodes                 : "
                f"{report['physical_node_count']}"
            )

            print(
                f"Physical edges                 : "
                f"{report['physical_edge_count']}"
            )

            print(
                f"Degree-1 terminal nodes        : "
                f"{len(degree_one_nodes)}"
            )

            print(
                f"Degree-2 nodes                 : "
                f"{len(degree_two_nodes)}"
            )

            print(
                f"Branch nodes, degree >= 3      : "
                f"{len(branch_nodes)}"
            )

            print(
                f"Bridge edges                   : "
                f"{len(bridges)}"
            )

            print(
                f"Terminal physical edges        : "
                f"{len(terminal_edge_ids)}"
            )

            print(
                f"Theoretical forced backtracks  : "
                f"{minimum_forced_backtracks}"
            )

            print(
                f"Current immediate backtracks   : "
                f"{self.count_immediate_backtracks(verbose=False)[0]}"
            )

            if degree_one_nodes:

                print()
                print(
                    "Degree-1 node IDs:"
                )

                print(degree_one_nodes)

                print()
                print(
                    "Terminal physical edge IDs:"
                )

                print(terminal_edge_ids)

        return report

    def export_euler_debug_video_cv2(
        self,
        filename="ccf_euler_debug.avi",
        fps=10,
        samples_per_edge=10,
        hold_frames=2,
        width=1400,
        height=900,
        margin=70,
        projection="pca",
        show_edge_ids=False,
        save_debug_frame=True,
    ):
        """Export the current closed Euler route as an MJPEG/AVI animation.

        Gray: unvisited; blue: visited once; green: visited twice or more;
        red: current traversal; yellow: nozzle. The projection is fixed
        across frames. Animation samples each curve by arc length.
        """
        from pathlib import Path

        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Video export requires opencv-python (cv2).") from exc

        if self.euler_edges is None:
            raise RuntimeError("Compute an Euler circuit before exporting the video.")
        self.validate_route(self.euler_edges, raise_on_error=True, verbose=False)
        if not isinstance(samples_per_edge, (int, np.integer)) or samples_per_edge < 2:
            raise ValueError("samples_per_edge must be an integer of at least 2.")
        if not isinstance(hold_frames, (int, np.integer)) or hold_frames < 1:
            raise ValueError("hold_frames must be a positive integer.")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite.")
        if width <= 0 or height <= 0 or margin < 0 or min(width, height) <= 2 * margin:
            raise ValueError("width and height must exceed twice the nonnegative margin.")
        if projection not in ("xy", "xz", "yz", "pca", "oblique"):
            raise ValueError("projection must be 'xy', 'xz', 'yz', 'pca', or 'oblique'.")

        output_path = Path(filename).with_suffix(".avi")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        curves = self.edge_curves_xyz
        all_points = np.vstack(curves)
        if projection == "pca":
            centre = all_points.mean(axis=0)
            _, _, vh = np.linalg.svd(all_points - centre, full_matrices=False)
            basis = vh[:2].T

            def project(points):
                return (points - centre) @ basis
        elif projection == "oblique":
            centre = all_points.mean(axis=0)
            view_x = np.asarray([1.0, -1.0, 0.0], dtype=np.float64)
            view_x /= np.linalg.norm(view_x)
            view_y = np.asarray([0.45, 0.45, 1.0], dtype=np.float64)
            view_y = view_y - np.dot(view_y, view_x) * view_x
            view_y /= np.linalg.norm(view_y)
            basis = np.column_stack((view_x, view_y))

            def project(points):
                return (points - centre) @ basis
        else:
            axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[projection]

            def project(points):
                return points[:, axes]

        curves_2d = [project(curve) for curve in curves]
        projected = np.vstack(curves_2d)
        low = projected.min(axis=0)
        high = projected.max(axis=0)
        span = high - low
        if np.max(span) <= 1e-12:
            raise ValueError("Projected geometry is a point; select another view.")
        # Degenerate projections (for example a straight line) are allowed.
        scale = min((width - 2 * margin) / max(span[0], 1e-12),
                    (height - 2 * margin) / max(span[1], 1e-12))
        offset = np.array([(width - span[0] * scale) / 2,
                           (height - span[1] * scale) / 2])

        def pixels(points):
            result = np.empty_like(points, dtype=np.float64)
            result[:, 0] = (points[:, 0] - low[0]) * scale + offset[0]
            result[:, 1] = (high[1] - points[:, 1]) * scale + offset[1]
            return np.rint(result).astype(np.int32)

        curves_px = [pixels(curve) for curve in curves_2d]
        white = (255, 255, 255)
        panel = (248, 248, 248)
        outline = (35, 35, 35)
        unvisited = (190, 190, 190)
        visited_once = (150, 80, 20)
        visited_twice = (55, 135, 55)
        current = (35, 35, 205)
        nozzle_colour = (30, 195, 225)

        def draw(frame, polyline, colour, thickness, bordered=False):
            if len(polyline) < 2:
                return
            polygon = np.asarray(polyline, dtype=np.int32).reshape(-1, 1, 2)
            if bordered:
                cv2.polylines(frame, [polygon], False, outline, thickness + 4, cv2.LINE_AA)
            cv2.polylines(frame, [polygon], False, colour, thickness, cv2.LINE_AA)

        def label(frame, value, x, y, size=0.65):
            cv2.putText(frame, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        size, (25, 25, 25), 1, cv2.LINE_AA)

        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 float(fps), (int(width), int(height)))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MJPEG video writer: {output_path}")

        counts = np.zeros(len(curves), dtype=np.int64)
        total = len(self.euler_edges)
        debug_path = output_path.with_name(output_path.stem + "_debug_frame.png")
        debug_saved = False
        try:
            for step, item in enumerate(self.euler_edges):
                edge_id = int(item["edge_id"])
                oriented = self.get_oriented_curve(item)
                samples = np.linspace(0, 1, samples_per_edge)
                lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(oriented, axis=0), axis=1))]
                total_length = lengths[-1]
                for frame_index, fraction in enumerate(samples):
                    frame = np.full((height, width, 3), white, dtype=np.uint8)
                    for eid, curve_px in enumerate(curves_px):
                        colour = (unvisited if counts[eid] == 0 else
                                  visited_once if counts[eid] == 1 else visited_twice)
                        draw(frame, curve_px, colour, 4)

                    distance = fraction * total_length
                    index = min(int(np.searchsorted(lengths, distance, side="right")),
                                len(oriented) - 1)
                    if index == 0:
                        partial = oriented[:1]
                    else:
                        left, right = lengths[index - 1], lengths[index]
                        t = (distance - left) / (right - left) if right > left else 0.0
                        point = oriented[index - 1] + t * (oriented[index] - oriented[index - 1])
                        partial = np.vstack((oriented[:index], point))
                    partial_px = pixels(project(partial))
                    draw(frame, partial_px, current, 7, bordered=True)
                    position = tuple(map(int, partial_px[-1]))
                    cv2.circle(frame, position, 14, outline, -1, cv2.LINE_AA)
                    cv2.circle(frame, position, 10, nozzle_colour, -1, cv2.LINE_AA)
                    if len(partial_px) > 1:
                        delta = partial_px[-1].astype(float) - partial_px[-2]
                        norm = np.linalg.norm(delta)
                        if norm > 1e-8:
                            arrow = tuple(np.rint(partial_px[-1] + 22 * delta / norm).astype(int))
                            cv2.arrowedLine(frame, position, arrow, outline, 3,
                                            cv2.LINE_AA, tipLength=0.35)

                    info = [f"Step: {step + 1} / {total}",
                            f"Edge: {edge_id}",
                            f"Fibre: {item['fibre_id']}",
                            f"Pass: {counts[edge_id] + 1} / {self.fibres_per_edge}",
                            f"Nodes: {item['u']} -> {item['v']}",
                            f"Progress: {100 * fraction:.0f}%"]
                    info_width = max(
                        cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                        for message in info
                    )
                    cv2.rectangle(frame, (10, 10), (30 + info_width, 155), panel, -1)
                    cv2.rectangle(frame, (10, 10), (30 + info_width, 155),
                                  (180, 180, 180), 1)
                    for row, message in enumerate(info):
                        label(frame, message, 20, 32 + 22 * row, 0.5)
                    legend_y = height - 96
                    cv2.rectangle(frame, (10, legend_y - 16), (240, height - 8), panel, -1)
                    cv2.rectangle(frame, (10, legend_y - 16), (240, height - 8),
                                  (180, 180, 180), 1)
                    legend = [("Unvisited", unvisited), ("1 pass", visited_once),
                              ("2+ passes", visited_twice),
                              ("Current pass", current)]
                    for row, (name, colour) in enumerate(legend):
                        y = legend_y + row * 24
                        cv2.line(frame, (20, y), (55, y), outline, 7, cv2.LINE_AA)
                        cv2.line(frame, (20, y), (55, y), colour, 4, cv2.LINE_AA)
                        label(frame, name, 68, y + 5, 0.48)
                    if show_edge_ids:
                        for eid, curve_px in enumerate(curves_px):
                            label(frame, eid, *map(int, curve_px[len(curve_px) // 2]), 0.4)
                    writer.write(np.ascontiguousarray(frame))
                    if save_debug_frame and not debug_saved and step == min(10, total - 1) and frame_index == samples_per_edge // 2:
                        cv2.imwrite(str(debug_path), frame)
                        debug_saved = True
                counts[edge_id] += 1
                for _ in range(hold_frames - 1):
                    writer.write(np.ascontiguousarray(frame))
        finally:
            writer.release()

        self.edge_visit_counts = counts
        if np.any(counts != self.fibres_per_edge):
            raise RuntimeError("Video export ended with incorrect edge visit counts.")
        print(f"Saved video: {output_path}")
        if debug_saved:
            print(f"Saved debug frame: {debug_path}")
        return counts
    def Generate_Video_Offsets(
        self,
        filename="ccf_offsets.avi",
        fps=10,
        width=1400,
        height=900,
        samples_per_edge=15,
        hold_frames=2,
    ):
        """
        Visualize actual offset fibre toolpaths.

        Red  = lane 0
        Blue = lane 1
        Green = already printed
        Yellow = nozzle position

        Uses:
            get_oriented_pass_curve()
        instead of centerlines.
        """

        import cv2
        import numpy as np
        from pathlib import Path

        if self.euler_edges is None:
            raise RuntimeError("Run optimization first.")

        if self.edge_pass_curves_xyz is None:
            raise RuntimeError(
                "edge_pass_curves_xyz not available."
            )

        route = self.route_with_lane_assignment()

        # ----------------------------------------------------
        # Collect ALL offset curves
        # ----------------------------------------------------

        all_curves = []

        for item in route:
            all_curves.append(
                self.get_oriented_pass_curve(item)
            )

        pts = np.vstack(all_curves)

        # ----------------------------------------------------
        # PCA Projection
        # ----------------------------------------------------

        center = pts.mean(axis=0)

        _, _, vh = np.linalg.svd(
            pts - center,
            full_matrices=False
        )

        basis = vh[:2].T

        projected = (pts - center) @ basis

        low = projected.min(axis=0)
        high = projected.max(axis=0)

        span = high - low

        margin = 50

        scale = min(
            (width - 2 * margin) / span[0],
            (height - 2 * margin) / span[1],
        )

        def project_xyz(curve):

            xy = (curve - center) @ basis

            result = np.zeros((len(xy), 2))

            result[:, 0] = (
                (xy[:, 0] - low[0]) * scale
                + margin
            )

            result[:, 1] = (
                height
                - (
                    (xy[:, 1] - low[1]) * scale
                    + margin
                )
            )

            return np.rint(result).astype(np.int32)

        projected_curves = [
            project_xyz(c)
            for c in all_curves
        ]

        # ----------------------------------------------------
        # Writer
        # ----------------------------------------------------

        filename = Path(filename)

        writer = cv2.VideoWriter(
            str(filename),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )

        printed = np.zeros(
            len(route),
            dtype=bool
        )

        # ----------------------------------------------------
        # frames
        # ----------------------------------------------------

        for traversal_id, item in enumerate(route):

            curve_xyz = self.get_oriented_pass_curve(item)

            curve_px = project_xyz(curve_xyz)

            lane = item["offset_lane"]

            lane_color = (
                (0,0,255)      # red lane
                if lane == 0
                else
                (255,0,0)      # blue lane
            )

            n = len(curve_xyz)

            indices = np.linspace(
                1,
                n-1,
                samples_per_edge
            ).astype(int)

            for idx in indices:

                frame = np.full(
                    (height,width,3),
                    255,
                    np.uint8
                )

                # -----------------------------------
                # draw all curves
                # -----------------------------------

                for rid, route_item in enumerate(route):

                    poly = projected_curves[rid]

                    route_lane = route_item["offset_lane"]

                    base_color = (
                        (0,0,180)
                        if route_lane == 0
                        else
                        (180,0,0)
                    )

                    if printed[rid]:

                        if route_lane == 0:
                            color = (0,255,0)      # bright green
                        else:
                            color = (0,255,255)    # yellow

                    else:

                        if route_lane == 0:
                            color = (0,0,180)      # dark red
                        else:
                            color = (180,0,0)      # dark blue

                    cv2.polylines(
                        frame,
                        [poly.reshape(-1,1,2)],
                        False,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                # -----------------------------------
                # current path
                # -----------------------------------

                partial = curve_px[:idx+1]

                cv2.polylines(
                    frame,
                    [partial.reshape(-1,1,2)],
                    False,
                    lane_color,
                    6,
                    cv2.LINE_AA
                )

                nozzle = tuple(partial[-1])

                cv2.circle(
                    frame,
                    nozzle,
                    8,
                    (0,255,255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Step {traversal_id+1}/{len(route)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Edge {item['edge_id']}  Lane {lane}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,0),
                    2
                )

                writer.write(frame)

            printed[traversal_id] = True

            for _ in range(hold_frames):
                writer.write(frame)

        writer.release()

        print("Saved:", filename)

        return str(filename)
