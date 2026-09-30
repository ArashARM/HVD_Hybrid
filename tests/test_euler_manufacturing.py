import json
import numpy as np
import pytest
import torch

from Utils.EulerGraph import (
    EulerCCFToolpath,
    ManufacturingGeometryError,
    ManufacturingInfeasibilityError,
    MeshSurfaceProjector,
    boundary_lane_offset_distances,
    calculate_uniform_even_fibre_count,
    coerce_surface_mesh,
    compute_edge_normals_from_cad,
    generate_manufacturing_offset_passes,
    generate_surface_offset_passes,
    prepare_manufacturing_physical_graph,
    symmetric_lane_offsets,
    _edge_spacing_stats,
)


class CurvedPlaneCad:
    def eval_uv_norm_batch(self, uv, return_inside_mask=False):
        uv = np.asarray(uv, dtype=float)
        u = uv[:, 0]
        v = uv[:, 1]
        xyz = np.column_stack((u, v, u * u + 0.5 * v))
        xu = np.column_stack((np.ones_like(u), np.zeros_like(u), 2.0 * u))
        xv = np.column_stack((np.zeros_like(v), np.ones_like(v), 0.5 * np.ones_like(v)))
        out = {"xyz": xyz, "Xu": xu, "Xv": xv}
        if return_inside_mask:
            out["inside"] = np.ones(len(uv), dtype=bool)
        return out

    def __call__(self, uv):
        return self.eval_uv_norm_batch(uv, return_inside_mask=True)


class TorchXyzOnlyCurvedPlaneCad(CurvedPlaneCad):
    def eval_uv_norm_batch_torch(self, uv):
        xyz = super().eval_uv_norm_batch(uv.detach().cpu().numpy())["xyz"]
        return {"xyz": torch.as_tensor(xyz, dtype=uv.dtype, device=uv.device)}


class SlightlyDifferentBatchXyzCad(TorchXyzOnlyCurvedPlaneCad):
    def eval_uv_norm_batch(self, uv, return_inside_mask=False):
        out = super().eval_uv_norm_batch(uv, return_inside_mask=return_inside_mask)
        out["xyz"] = out["xyz"].copy()
        out["xyz"][:, 2] += 5.0e-7
        return out


class PlaneCad:
    def __call__(self, uv):
        uv = np.asarray(uv, dtype=float)
        xyz = np.column_stack((uv[:, 0], uv[:, 1], np.zeros(len(uv))))
        xu = np.tile(np.asarray([1.0, 0.0, 0.0]), (len(uv), 1))
        xv = np.tile(np.asarray([0.0, 1.0, 0.0]), (len(uv), 1))
        return {"xyz": xyz, "Xu": xu, "Xv": xv, "inside": np.ones(len(uv), dtype=bool)}


class TrimmedPlaneCad:
    def __init__(self, inside_fn, z_fn=None):
        self.inside_fn = inside_fn
        self.z_fn = (lambda uv: np.zeros(len(uv))) if z_fn is None else z_fn

    def __call__(self, uv):
        uv = np.asarray(uv, dtype=float)
        z = np.asarray(self.z_fn(uv), dtype=float)
        xyz = np.column_stack((uv[:, 0], uv[:, 1], z))
        if z.ndim == 0 or np.allclose(z, 0.0):
            dzdu = np.zeros(len(uv))
            dzdv = np.zeros(len(uv))
        else:
            dzdu = 0.2 * uv[:, 0]
            dzdv = 0.1 * np.ones(len(uv))
        xu = np.column_stack((np.ones(len(uv)), np.zeros(len(uv)), dzdu))
        xv = np.column_stack((np.zeros(len(uv)), np.ones(len(uv)), dzdv))
        return {"xyz": xyz, "Xu": xu, "Xv": xv, "inside": np.asarray(self.inside_fn(uv), dtype=bool)}


def normals_like(curves, normal=(0.0, 0.0, 1.0)):
    return [np.tile(np.asarray(normal, dtype=float), (len(curve), 1)) for curve in curves]


def square_tool():
    edge_index = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        dtype=np.int64,
    )
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    return EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals_like(curves),
    )


def plane_mesh():
    vertices = np.asarray(
        [
            [-1.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def half_plane_mesh():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def curved_plane_mesh(cad, grid=5):
    uv_values = []
    for v in np.linspace(0.0, 1.0, grid):
        for u in np.linspace(0.0, 1.0, grid):
            uv_values.append([u, v])
    uv = np.asarray(uv_values, dtype=float)
    vertices = cad.eval_uv_norm_batch(uv, return_inside_mask=True)["xyz"]
    faces = []
    for j in range(grid - 1):
        for i in range(grid - 1):
            a = j * grid + i
            b = a + 1
            c = a + grid
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return vertices, np.asarray(faces, dtype=np.int64)


def curved_single_edge_tool():
    cad = CurvedPlaneCad()
    u = np.linspace(0.1, 0.9, 17)
    center_uv = np.column_stack((u, np.full_like(u, 0.50)))
    lane_uv = [
        np.column_stack((u, np.full_like(u, 0.44))),
        np.column_stack((u, np.full_like(u, 0.56))),
    ]
    center_eval = cad.eval_uv_norm_batch(center_uv, return_inside_mask=True)
    lane_eval = [cad.eval_uv_norm_batch(values, return_inside_mask=True) for values in lane_uv]
    normals = np.cross(center_eval["Xu"], center_eval["Xv"])
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    pass_normals = []
    for evaluated in lane_eval:
        lane_normals = np.cross(evaluated["Xu"], evaluated["Xv"])
        pass_normals.append(lane_normals / np.linalg.norm(lane_normals, axis=1)[:, None])
    return cad, EulerCCFToolpath(
        np.asarray([[0, 1]], dtype=np.int64),
        [center_eval["xyz"]],
        fibres_per_edge=2,
        edge_normals_xyz=[normals],
        edge_curves_uv=[center_uv],
        edge_pass_curves_xyz=[[lane_eval[0]["xyz"], lane_eval[1]["xyz"]]],
        edge_pass_normals_xyz=[[pass_normals[0], pass_normals[1]]],
        edge_pass_curves_uv=[lane_uv],
        edge_pass_offsets=[np.asarray([-0.06, 0.06])],
    )


def single_edge_manufactured_tool():
    edge_index = np.asarray([[0, 1]], dtype=np.int64)
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    normals = normals_like(curves)
    passes = [
        [
            np.asarray([[0.0, -0.1, 0.0], [1.0, -0.1, 0.0]]),
            np.asarray([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]]),
        ]
    ]
    pass_normals = [[normals[0].copy(), normals[0].copy()]]
    return EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals,
        edge_pass_curves_xyz=passes,
        edge_pass_normals_xyz=pass_normals,
        edge_pass_offsets=[-0.1, 0.1],
    )


def two_edge_crossing_tool(off_junction=True):
    edge_index = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    if off_junction:
        curves = [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ]
    else:
        curves = [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ]
        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    normals = normals_like(curves)
    return EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals,
        edge_pass_curves_xyz=[[curve.copy(), curve.copy()] for curve in curves],
        edge_pass_normals_xyz=[[normals[i].copy(), normals[i].copy()] for i in range(len(curves))],
        edge_pass_offsets=[np.asarray([0.0, 0.0]) for _ in curves],
    )


def multi_lane_single_edge_tool(fibres_per_edge):
    edge_index = np.asarray([[0, 1]], dtype=np.int64)
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    normals = normals_like(curves)
    offsets = symmetric_lane_offsets(fibres_per_edge, fibre_line_width=0.1, fibre_gap=0.02)
    passes = [[curve + np.asarray([0.0, offset, 0.0]) for offset in offsets] for curve in curves]
    pass_normals = [[normals[0].copy() for _ in offsets]]
    return EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=fibres_per_edge,
        edge_normals_xyz=normals,
        edge_pass_curves_xyz=passes,
        edge_pass_normals_xyz=pass_normals,
        edge_pass_offsets=[offsets],
    )


def test_uniform_even_fibre_count_and_symmetric_offsets():
    assert calculate_uniform_even_fibre_count(
        strut_thickness=0.76,
        fibre_line_width=0.1,
        fibre_gap=0.02,
        boundary_margin=0.02,
    ) == 6
    assert calculate_uniform_even_fibre_count(
        strut_thickness=0.52,
        fibre_line_width=0.1,
        fibre_gap=0.02,
        boundary_margin=0.0,
        fibres_per_edge=4,
    ) == 4
    with pytest.raises(ManufacturingGeometryError):
        calculate_uniform_even_fibre_count(
            strut_thickness=0.5,
            fibre_line_width=0.1,
            fibres_per_edge=3,
        )
    assert np.allclose(symmetric_lane_offsets(4, 0.1, 0.02), [-0.18, -0.06, 0.06, 0.18])


@pytest.mark.parametrize("fibres_per_edge", [2, 4, 6])
def test_expanded_graph_has_uniform_even_physical_instances(fibres_per_edge):
    tool = multi_lane_single_edge_tool(fibres_per_edge)
    graph = tool.build_multigraph()
    assert graph.number_of_edges() == fibres_per_edge * len(tool.edge_index)
    route = [
        {"u": 0, "v": 1, "key": i, "edge_id": 0, "fibre_id": i}
        for i in range(fibres_per_edge)
    ]
    mapping = tool.assign_route_lanes(route)
    assert sorted(item["fibre_instance_id"] for item in mapping) == list(range(fibres_per_edge))
    assert sorted(item["offset_lane"] for item in mapping) == list(range(fibres_per_edge))


def test_incomplete_geometry_refuses_ordered_path_by_default():
    tool = square_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()

    report = tool.check_manufacturing_constraints(
        route,
        settings={"endpoint_tolerance": 1.0e-9},
        verbose=False,
    )
    assert report["feasible"]
    assert report["max_endpoint_gap"] == pytest.approx(0.0)

    with pytest.raises((ManufacturingGeometryError, ManufacturingInfeasibilityError)):
        tool.export_ordered_path(route)


def test_compute_edge_normals_from_cad_preserves_uv_sample_order():
    cad = CurvedPlaneCad()
    curves_uv = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        ],
        dtype=float,
    )
    curves_xyz = cad.eval_uv_norm_batch(curves_uv.reshape(-1, 2))["xyz"].reshape(2, 3, 3)

    normals, diagnostics = compute_edge_normals_from_cad(
        cad,
        {"edge_curves_uv": curves_uv, "edge_curves_xyz": curves_xyz},
        nozzle_approach="-normal",
    )

    expected = np.cross(
        cad.eval_uv_norm_batch(curves_uv.reshape(-1, 2))["Xu"],
        cad.eval_uv_norm_batch(curves_uv.reshape(-1, 2))["Xv"],
    )
    expected = expected / np.linalg.norm(expected, axis=1)[:, None]
    assert diagnostics["nozzle_approach"] == "-normal"
    assert diagnostics["xyz_matches_supplied"]
    assert np.allclose(np.asarray(normals), expected.reshape(2, 3, 3))


def test_compute_edge_normals_falls_back_when_torch_evaluator_returns_xyz_only():
    cad = TorchXyzOnlyCurvedPlaneCad()
    curves_uv = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    curves_xyz = torch.as_tensor(
        cad.eval_uv_norm_batch(curves_uv.reshape(-1, 2).numpy())["xyz"].reshape(2, 3, 3),
        dtype=curves_uv.dtype,
    )

    normals, diagnostics = compute_edge_normals_from_cad(
        cad,
        {"edge_curves_uv": curves_uv, "edge_curves_xyz": curves_xyz},
    )

    assert diagnostics["xyz_matches_supplied"]
    assert np.asarray(normals).shape == (2, 3, 3)


def test_compute_edge_normals_keeps_torch_xyz_when_batch_xyz_differs_slightly():
    cad = SlightlyDifferentBatchXyzCad()
    curves_uv = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    curves_xyz = cad.eval_uv_norm_batch_torch(curves_uv.reshape(-1, 2))["xyz"].reshape(2, 3, 3)

    normals, diagnostics = compute_edge_normals_from_cad(
        cad,
        {"edge_curves_uv": curves_uv, "edge_curves_xyz": curves_xyz},
        xyz_tolerance=1.0e-7,
    )

    assert diagnostics["xyz_matches_supplied"]
    assert diagnostics["max_reevaluated_xyz_error"] == pytest.approx(0.0)
    assert np.asarray(normals).shape == (2, 3, 3)


def test_endpoint_gap_reports_location_edges_value_and_limit():
    edge_index = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.1, 0.0, 0.0], [0.5, 1.0, 0.0]]),
        np.asarray([[0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    tool = EulerCCFToolpath(edge_index, curves, edge_normals_xyz=normals_like(curves))
    tool.build_multigraph()
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 2, "key": 0, "edge_id": 1, "fibre_id": 0},
        {"u": 2, "v": 0, "key": 0, "edge_id": 2, "fibre_id": 0},
        {"u": 0, "v": 1, "key": 1, "edge_id": 0, "fibre_id": 1},
        {"u": 1, "v": 2, "key": 1, "edge_id": 1, "fibre_id": 1},
        {"u": 2, "v": 0, "key": 1, "edge_id": 2, "fibre_id": 1},
    ]
    tool.validate_route(route, raise_on_error=False, verbose=False)

    report = tool.check_manufacturing_constraints(
        route,
        settings={"endpoint_tolerance": 1.0e-6},
        verbose=False,
    )
    violation = next(
        item
        for item in report["violations"]
        if item["kind"] == "endpoint_gap"
        and abs(item["measured_value"] - 0.1) < 1.0e-9
    )
    assert set(violation["edge_ids"]) == {0, 1}
    assert violation["measured_value"] == pytest.approx(0.1)
    assert violation["limit"] == pytest.approx(1.0e-6)
    assert "transition_index" in violation["location"]


def test_internal_bend_requires_edge_geometry_change():
    edge_index = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.05, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]),
        np.asarray([[0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    tool = EulerCCFToolpath(edge_index, curves, edge_normals_xyz=normals_like(curves))
    tool.build_multigraph()
    route = tool.compute_euler_circuit()

    report = tool.check_manufacturing_constraints(
        route,
        settings={"max_internal_bend_degrees": 45.0, "min_bend_radius": 0.5},
        verbose=False,
    )
    kinds = {item["kind"] for item in report["requires_graph_or_geometry_change"]}
    assert "internal_bend_angle" in kinds
    assert "bend_radius" in kinds


def test_coincident_fibre_passes_violate_spacing():
    edge_index = np.asarray([[0, 1]], dtype=np.int64)
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    tool = EulerCCFToolpath(edge_index, curves, edge_normals_xyz=normals_like(curves))
    tool.build_multigraph()
    route = tool.compute_euler_circuit()

    report = tool.check_manufacturing_constraints(
        route,
        settings={"min_pass_spacing": 0.1},
        verbose=False,
    )
    static = next(
        item
        for item in report["requires_graph_or_geometry_change"]
        if item["kind"] == "coincident_fibre_pass_geometry"
    )
    assert static["edge_ids"] == [0]
    assert static["measured_value"] == pytest.approx(0.0)
    assert report["requires_graph_or_geometry_change"]


def test_feasible_routes_rank_ahead_of_infeasible_without_penalty():
    tool = square_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()

    feasible = tool.route_score(
        route,
        manufacturing_settings={"endpoint_tolerance": 1.0e-9},
        return_breakdown=True,
    )
    infeasible = tool.route_score(
        route,
        manufacturing_settings={"min_pass_spacing": 0.1},
        return_breakdown=True,
    )

    assert feasible["manufacturing_ready"]
    assert not infeasible["manufacturing_ready"]
    assert feasible["rank_tuple"] < infeasible["rank_tuple"]
    assert "hard_constraint_violations" not in infeasible["terms"]


def test_infeasible_policy_can_raise_dedicated_error():
    tool = square_tool()
    tool.build_multigraph()
    tool.compute_euler_circuit()

    with pytest.raises(ManufacturingInfeasibilityError):
        tool.optimize_turn_aware_euler_circuit(
            trials=0,
            refinement_passes=0,
            manufacturing_settings={"min_pass_spacing": 0.1},
            infeasible_policy="raise",
            verbose=False,
        )


def test_min_bend_radius_reports_unknown_junction_radius_without_fillet():
    tool = square_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()

    report = tool.check_manufacturing_constraints(
        route,
        settings={"min_bend_radius": 0.5},
        verbose=False,
    )

    unknown = report["checks_awaiting_geometry_or_machine_data"]
    assert any(item["kind"] == "junction_bend_radius_unknown" for item in unknown)
    assert any(item["location"]["closing_junction"] for item in unknown)


def test_surface_offset_rejects_infeasible_width():
    vertices, faces = plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    with pytest.raises(ManufacturingGeometryError, match="Insufficient strut width"):
        generate_surface_offset_passes(
            edge_curves_xyz=curves,
            edge_curves_uv=None,
            edge_normals_xyz=normals_like(curves),
            surface_vertices=vertices,
            surface_faces=faces,
            strut_thickness=0.4,
            fibre_line_width=1.0,
        )


def test_planar_interior_edge_keeps_symmetric_offsets():
    vertices, faces = plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    passes, _, _, offsets, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_normals_xyz=normals_like(curves),
        surface_vertices=vertices,
        surface_faces=faces,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        fibres_per_edge=2,
    )
    assert np.allclose(offsets[0], [-0.05, 0.05])
    assert np.allclose(passes[0][0][:, 1], -0.05)
    assert np.allclose(passes[0][1][:, 1], 0.05)
    assert diagnostics["boundary_edge_ids"] == []


def test_planar_boundary_edge_uses_one_sided_inward_offsets_and_spacing():
    vertices, faces = half_plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    passes, _, _, offsets, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_normals_xyz=normals_like(curves),
        surface_vertices=vertices,
        surface_faces=faces,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        fibres_per_edge=2,
        edge_types=np.asarray([4]),
    )
    assert np.allclose(boundary_lane_offset_distances(2, 0.05, 0.05), [0.025, 0.125])
    assert np.allclose(offsets[0], [0.025, 0.125])
    assert np.all(passes[0][0][:, 1] >= 0.0)
    assert np.all(passes[0][1][:, 1] >= 0.0)
    assert np.linalg.norm(passes[0][1] - passes[0][0], axis=1).min() == pytest.approx(0.10)
    assert diagnostics["minimum_actual_spacing"] == pytest.approx(0.10)
    assert diagnostics["boundary_inward_signs"] == {0: 1.0}


def test_trimmed_mesh_selects_boundary_side_when_untrimmed_cad_accepts_both_signs():
    vertices, faces = half_plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    curves_uv = [curves[0][:, :2].copy()]
    passes, _, pass_uv, offsets, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_curves_uv=curves_uv,
        edge_normals_xyz=normals_like(curves),
        surface_vertices=vertices,
        surface_faces=faces,
        surface_evaluator=PlaneCad(),
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        fibres_per_edge=2,
        edge_types=np.asarray([4]),
    )
    assert pass_uv is not None
    assert np.allclose(offsets[0], [0.025, 0.125])
    assert np.all(passes[0][0][:, 1] >= 0.0)
    assert diagnostics["boundary_direction_candidates"][0]["candidates"][0]["method"] == "trimmed_mesh_projection"


def test_reversing_boundary_curve_keeps_same_physical_lane_positions():
    vertices, faces = half_plane_mesh()
    forward = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    reversed_curve = [forward[0][::-1].copy()]
    fwd_passes, _, _, _, fwd_diag = generate_surface_offset_passes(
        edge_curves_xyz=forward,
        edge_normals_xyz=normals_like(forward),
        surface_vertices=vertices,
        surface_faces=faces,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        fibres_per_edge=2,
        edge_types=np.asarray([4]),
    )
    rev_passes, _, _, rev_offsets, rev_diag = generate_surface_offset_passes(
        edge_curves_xyz=reversed_curve,
        edge_normals_xyz=normals_like(reversed_curve),
        surface_vertices=vertices,
        surface_faces=faces,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        fibres_per_edge=2,
        edge_types=np.asarray([4]),
    )
    assert np.allclose(rev_offsets[0], [-0.025, -0.125])
    assert rev_diag["boundary_inward_signs"] == {0: -1.0}
    for lane in range(2):
        assert np.allclose(fwd_passes[0][lane], rev_passes[0][lane][::-1])
    assert fwd_diag["minimum_actual_spacing"] == pytest.approx(rev_diag["minimum_actual_spacing"])


def _reported_boundary_incident_setup(reverse=False):
    start = np.asarray([0.0, 1.34144354, 0.0])
    end = np.asarray([0.65140545, 1.19173455, 0.0])
    curve = np.linspace(start, end, 64)
    if reverse:
        curve = curve[::-1].copy()
    uv = curve[:, :2].copy()
    boundary_y0 = 1.34144354
    boundary_slope = 0.184
    cad = TrimmedPlaneCad(lambda values: values[:, 1] <= boundary_y0 + boundary_slope * values[:, 0] + 1.0e-12)
    return cad, [curve], [uv]


def test_boundary_incident_type1_reported_planar_pattern_blends_and_keeps_multiplicity():
    cad, curves, curves_uv = _reported_boundary_incident_setup()
    offsets = symmetric_lane_offsets(2, fibre_line_width=0.05, fibre_gap=0.025)
    raw_inside = []
    for offset in offsets:
        raw_inside.append(cad(curves_uv[0] + np.asarray([0.0, offset]))["inside"])
    assert raw_inside[0].all()
    assert np.flatnonzero(~raw_inside[1]).tolist() == list(range(9))

    passes, normals, pass_uv, lane_offsets, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_curves_uv=curves_uv,
        edge_normals_xyz=normals_like(curves),
        surface_evaluator=cad,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.025,
        fibres_per_edge=2,
        edge_types=np.asarray([1]),
    )
    assert diagnostics["boundary_incident_edge_ids"] == [0]
    assert diagnostics["boundary_incident_blended_edge_ids"] == [0]
    edge_diag = diagnostics["boundary_incident_edge_diagnostics"][0]
    assert edge_diag["boundary_incident_start"]
    assert not edge_diag["boundary_incident_end"]
    assert edge_diag["start_boundary_incident"]
    assert not edge_diag["end_boundary_incident"]
    assert edge_diag["originally_outside_lane_ids"] == [1]
    assert edge_diag["originally_outside_sample_ids"] == list(range(9))
    assert edge_diag["all_samples_inside"]
    assert edge_diag["lane_order_preserved"]
    assert edge_diag["minimum_lane_spacing"] == pytest.approx(0.0)
    assert edge_diag["endpoint_compressed_sample_count"] > 0
    assert edge_diag["minimum_robust_spacing"] >= 0.075 - 1.0e-6
    assert not diagnostics["per_edge_lane_spacing"][0]["sustained_compression"]
    assert np.all(cad(pass_uv[0][0])["inside"])
    assert np.all(cad(pass_uv[0][1])["inside"])
    assert np.all(np.linalg.norm(np.diff(passes[0][0], axis=0), axis=1) > 0.0)
    assert np.all(np.linalg.norm(np.diff(passes[0][1], axis=0), axis=1) > 0.0)
    lane_spacing = np.linalg.norm(passes[0][1] - passes[0][0], axis=1)
    assert lane_spacing[0] == pytest.approx(0.0)
    assert np.all(lane_spacing[1:] > 0.0)
    assert lane_spacing[-1] == pytest.approx(0.075)
    assert np.allclose(lane_offsets[0], offsets)

    tool = EulerCCFToolpath(
        np.asarray([[0, 1]], dtype=np.int64),
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals_like(curves),
        edge_curves_uv=curves_uv,
        edge_pass_curves_xyz=passes,
        edge_pass_normals_xyz=normals,
        edge_pass_curves_uv=pass_uv,
        edge_pass_offsets=lane_offsets,
    )
    graph = tool.build_multigraph()
    assert graph.number_of_edges() == 2
    route = tool.compute_euler_circuit()
    assert len(route) == 2
    transition_report = tool.build_junction_transitions(route, surface_evaluator=cad, transition_samples=9)
    assert transition_report["junctions_with_no_valid_transition"] == 0


def test_boundary_incident_reversed_curve_is_detected_at_end():
    cad, curves, curves_uv = _reported_boundary_incident_setup(reverse=True)
    _, _, pass_uv, _, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_curves_uv=curves_uv,
        edge_normals_xyz=normals_like(curves),
        surface_evaluator=cad,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.025,
        fibres_per_edge=2,
        edge_types=np.asarray([1]),
    )
    edge_diag = diagnostics["boundary_incident_edge_diagnostics"][0]
    assert not edge_diag["boundary_incident_start"]
    assert edge_diag["boundary_incident_end"]
    assert not edge_diag["start_boundary_incident"]
    assert edge_diag["end_boundary_incident"]
    assert np.all(cad(pass_uv[0][0])["inside"])
    assert np.all(cad(pass_uv[0][1])["inside"])


def test_boundary_incident_blends_end_and_both_ends():
    x = np.linspace(0.0, 1.0, 64)
    y = -0.2 * np.minimum(x, 1.0 - x)
    curve = np.column_stack((x, y, np.zeros_like(x)))
    cad = TrimmedPlaneCad(lambda uv: uv[:, 1] <= 0.0 + 1.0e-12)
    passes, _, pass_uv, _, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=[curve],
        edge_curves_uv=[curve[:, :2].copy()],
        edge_normals_xyz=normals_like([curve]),
        surface_evaluator=cad,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.025,
        fibres_per_edge=2,
        edge_types=np.asarray([1]),
    )
    edge_diag = diagnostics["boundary_incident_edge_diagnostics"][0]
    assert edge_diag["boundary_incident_start"]
    assert edge_diag["boundary_incident_end"]
    assert all(np.all(cad(lane_uv)["inside"]) for lane_uv in pass_uv[0])
    assert edge_diag["minimum_robust_spacing"] >= 0.075 - 1.0e-6
    assert np.linalg.norm(passes[0][1][0] - passes[0][0][0]) == pytest.approx(0.0)
    assert np.linalg.norm(passes[0][1][-1] - passes[0][0][-1]) == pytest.approx(0.0)

    y_end = -0.2 * (1.0 - x)
    end_curve = np.column_stack((x, y_end, np.zeros_like(x)))
    _, _, _, _, end_diag = generate_surface_offset_passes(
        edge_curves_xyz=[end_curve],
        edge_curves_uv=[end_curve[:, :2].copy()],
        edge_normals_xyz=normals_like([end_curve]),
        surface_evaluator=cad,
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.025,
        fibres_per_edge=2,
        edge_types=np.asarray([1]),
    )
    assert end_diag["boundary_incident_edge_diagnostics"][0]["boundary_incident_end"]


def test_boundary_incident_curved_surface_hole_and_more_than_two_lanes():
    x = np.linspace(0.0, 1.0, 80)
    y = -0.18 * x
    curve = np.column_stack((x, y, 0.1 * x * x + 0.1 * y))
    cad = TrimmedPlaneCad(
        lambda uv: (uv[:, 1] <= 0.0 + 1.0e-12) & (((uv[:, 0] - 0.55) ** 2 + (uv[:, 1] + 0.08) ** 2) >= 0.04 ** 2),
        z_fn=lambda uv: 0.1 * uv[:, 0] * uv[:, 0] + 0.1 * uv[:, 1],
    )
    passes, _, pass_uv, _, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=[curve],
        edge_curves_uv=[curve[:, :2].copy()],
        edge_normals_xyz=normals_like([curve]),
        surface_evaluator=cad,
        strut_thickness=0.7,
        fibre_line_width=0.04,
        fibre_gap=0.02,
        fibres_per_edge=4,
        edge_types=np.asarray([1]),
    )
    assert diagnostics["boundary_incident_blended_edge_ids"] == [0]
    assert all(np.all(cad(lane_uv)["inside"]) for lane_uv in pass_uv[0])
    edge_diag = diagnostics["boundary_incident_edge_diagnostics"][0]
    assert edge_diag["minimum_robust_spacing"] >= 0.06 - 1.0e-6
    assert not diagnostics["per_edge_lane_spacing"][0]["sustained_compression"]


def test_boundary_incident_interior_domain_exit_fails():
    x = np.linspace(0.0, 1.0, 64)
    curve = np.column_stack((x, np.full_like(x, -0.1), np.zeros_like(x)))
    cad = TrimmedPlaneCad(lambda uv: ~((uv[:, 0] > 0.4) & (uv[:, 0] < 0.6) & (uv[:, 1] > -0.2)))
    with pytest.raises(ManufacturingGeometryError, match="edge interior"):
        generate_surface_offset_passes(
            edge_curves_xyz=[curve],
            edge_curves_uv=[curve[:, :2].copy()],
            edge_normals_xyz=normals_like([curve]),
            surface_evaluator=cad,
            strut_thickness=0.4,
            fibre_line_width=0.05,
            fibre_gap=0.025,
            fibres_per_edge=2,
            edge_types=np.asarray([1]),
        )


def test_type3_type4_coincident_pair_preserves_boundary_copy_only():
    edge_index = np.asarray([[12, 13], [12, 13], [1, 2]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 1.0e-6, 0.0], [0.0, 1.0e-6, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    ]
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_types=np.asarray([3, 4, 3]),
        coincident_edge_tolerance=2.0e-5,
    )
    assert filtered["edge_types"].tolist() == [4, 3]
    assert filtered["kept_original_physical_edge_ids"].tolist() == [1, 2]
    assert filtered["old_to_new_edge_id"] == {0: 0, 1: 0, 2: 1}
    assert filtered["new_to_old_edge_id"] == {0: 1, 1: 2}
    assert filtered["diagnostics"]["merged_pairs"][0]["preserved_original_edge_id"] == 1
    assert filtered["diagnostics"]["merged_pairs"][0]["removed_original_edge_id"] == 0
    assert filtered["diagnostics"]["merged_pairs"][0]["removed_maps_to_filtered_edge_id"] == 0


def test_prepare_physical_graph_accepts_e2_and_2e_edge_index_forms_identically():
    edge_index_e2 = np.asarray([[12, 13], [12, 13], [1, 2]], dtype=np.int64)
    edge_index_2e = edge_index_e2.T
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 1.0e-6, 0.0], [0.0, 1.0e-6, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    ]
    kwargs = {
        "edge_curves_xyz": curves,
        "edge_types": np.asarray([3, 4, 3]),
        "coincident_edge_tolerance": 2.0e-5,
    }
    filtered_e2 = prepare_manufacturing_physical_graph(edge_index=edge_index_e2, **kwargs)
    filtered_2e = prepare_manufacturing_physical_graph(edge_index=edge_index_2e, **kwargs)
    assert filtered_e2["edge_index"].tolist() == filtered_2e["edge_index"].tolist()
    assert filtered_e2["old_to_new_edge_id"] == filtered_2e["old_to_new_edge_id"]


def test_prepare_physical_graph_rejects_mismatched_parallel_arrays():
    edge_index = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ]
    with pytest.raises(ManufacturingGeometryError, match="one value per original physical edge"):
        prepare_manufacturing_physical_graph(
            edge_index=edge_index,
            edge_curves_xyz=curves,
            edge_types=np.asarray([3, 4]),
            edge_normals_xyz=np.zeros((3, 2, 3)),
            coincident_edge_tolerance=2.0e-5,
        )


def test_noncoincident_parallel_edge_is_not_removed_and_ambiguous_duplicates_raise():
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.01, 0.0], [1.0, 0.01, 0.0]]),
    ]
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_types=np.asarray([3, 4]),
        coincident_edge_tolerance=2.0e-5,
    )
    assert filtered["kept_original_physical_edge_ids"].tolist() == [0, 1]
    with pytest.raises(ManufacturingGeometryError, match="Ambiguous coincident"):
        prepare_manufacturing_physical_graph(
            edge_index=edge_index,
            edge_curves_xyz=[curves[0], curves[0].copy()],
            edge_types=np.asarray([3, 3]),
            coincident_edge_tolerance=2.0e-5,
        )


def test_short_manufacturing_edge_is_removed_and_nodes_contract():
    edge_index = np.asarray([[10, 11], [11, 12], [12, 13]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]),
        np.asarray([[1.01, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    uv = [curve[:, :2].copy() for curve in curves]
    normals = normals_like(curves)
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_curves_uv=uv,
        edge_normals_xyz=normals,
        edge_types=np.asarray([1, 1, 1]),
        strut_thickness=np.asarray([0.2, 0.2, 0.2]),
        labels=np.asarray([100, 101, 102]),
        coincident_edge_tolerance=1.0e-6,
        minimum_manufacturable_edge_length=0.05,
    )
    assert filtered["edge_index"].tolist() == [[10, 11], [11, 13]]
    assert filtered["old_to_new_edge_id"] == {0: 0, 1: None, 2: 1}
    assert filtered["new_to_old_edge_id"] == {0: 0, 1: 2}
    assert filtered["kept_original_physical_edge_ids"].tolist() == [0, 2]
    assert filtered["strut_thickness"].tolist() == [0.2, 0.2]
    assert filtered["labels"].tolist() == [100, 102]
    assert filtered["diagnostics"]["short_edge_original_ids"] == [1]
    assert filtered["diagnostics"]["removed_short_edge_count"] == 1
    assert filtered["diagnostics"]["graph_connected_after_contraction"]
    assert filtered["diagnostics"]["contracted_node_map"] == {11: 11, 12: 11}
    assert filtered["diagnostics"]["maximum_endpoint_snap_distance"] == pytest.approx(0.01)

    assert np.allclose(filtered["edge_curves_xyz"][0][-1], filtered["edge_curves_xyz"][1][0])
    assert np.allclose(filtered["edge_curves_xyz"][1][0], [1.0, 0.0, 0.0])
    assert np.allclose(filtered["edge_curves_uv"][1][0], filtered["edge_curves_uv"][0][-1])
    assert np.allclose(filtered["edge_normals_xyz"][1][0], filtered["edge_normals_xyz"][0][-1])
    assert np.linalg.norm(filtered["edge_normals_xyz"][1][0]) == pytest.approx(1.0)


def test_multiple_adjacent_short_edges_contract_as_one_component():
    edge_index = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]),
        np.asarray([[1.01, 0.0, 0.0], [1.02, 0.0, 0.0]]),
        np.asarray([[1.02, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_types=np.ones(4, dtype=np.int64),
        coincident_edge_tolerance=1.0e-6,
        minimum_manufacturable_edge_length=0.05,
    )
    assert filtered["edge_index"].tolist() == [[0, 1], [1, 4]]
    assert filtered["old_to_new_edge_id"] == {0: 0, 1: None, 2: None, 3: 1}
    assert filtered["diagnostics"]["contracted_node_components"] == [[1, 2, 3]]
    assert filtered["diagnostics"]["contracted_component_representatives"] == [1]
    assert filtered["diagnostics"]["removed_short_edge_count"] == 2
    assert np.allclose(filtered["edge_curves_xyz"][0][-1], filtered["edge_curves_xyz"][1][0])


def test_minimum_manufacturable_edge_length_default_preserves_existing_behavior():
    edge_index = np.asarray([[10, 11], [11, 12], [12, 13]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]),
        np.asarray([[1.01, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_types=np.ones(3, dtype=np.int64),
        coincident_edge_tolerance=1.0e-6,
    )
    assert filtered["edge_index"].tolist() == edge_index.tolist()
    assert filtered["old_to_new_edge_id"] == {0: 0, 1: 1, 2: 2}
    assert "minimum_manufacturable_edge_length" not in filtered["diagnostics"]


def test_short_edge_detection_uses_xyz_arc_length_not_endpoint_distance():
    edge_index = np.asarray([[0, 1]], dtype=np.int64)
    curve = np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.0, 0.0]])
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=[curve],
        edge_types=np.asarray([1]),
        coincident_edge_tolerance=1.0e-6,
        minimum_manufacturable_edge_length=0.05,
    )
    assert filtered["edge_index"].tolist() == [[0, 1]]
    assert filtered["old_to_new_edge_id"] == {0: 0}
    assert filtered["diagnostics"]["short_edge_original_ids"] == []


def test_short_edge_contraction_allows_uv_scale_different_from_xyz():
    edge_index = np.asarray([[10, 11], [11, 12], [12, 13]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]),
        np.asarray([[1.01, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    uv = [
        np.asarray([[0.0, 0.0], [100.0, 200.0]]),
        np.asarray([[5000.0, 6000.0], [7000.0, 8000.0]]),
        np.asarray([[9000.0, 10000.0], [2.0, 0.0]]),
    ]
    normals = normals_like(curves)
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_curves_uv=uv,
        edge_normals_xyz=normals,
        edge_types=np.ones(3, dtype=np.int64),
        coincident_edge_tolerance=1.0e-6,
        minimum_manufacturable_edge_length=0.05,
    )
    spreads = filtered["diagnostics"]["contracted_component_endpoint_spreads"]
    assert filtered["diagnostics"]["maximum_endpoint_snap_distance"] == pytest.approx(0.01)
    assert spreads[11]["uv"] > 1000.0
    assert np.allclose(filtered["edge_curves_xyz"][1][0], [1.0, 0.0, 0.0])
    assert np.allclose(filtered["edge_curves_uv"][1][0], [100.0, 200.0])
    assert np.allclose(filtered["edge_normals_xyz"][1][0], [0.0, 0.0, 1.0])

    with pytest.raises(ManufacturingGeometryError, match="maximum_contracted_uv_spread"):
        prepare_manufacturing_physical_graph(
            edge_index=edge_index,
            edge_curves_xyz=curves,
            edge_curves_uv=uv,
            edge_normals_xyz=normals,
            edge_types=np.ones(3, dtype=np.int64),
            coincident_edge_tolerance=1.0e-6,
            minimum_manufacturable_edge_length=0.05,
            maximum_contracted_uv_spread=10.0,
        )


def test_unsafe_short_edge_contraction_raises_for_wide_endpoint_spread():
    edge_index = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    curves = [
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    with pytest.raises(ManufacturingGeometryError, match="widely separated XYZ"):
        prepare_manufacturing_physical_graph(
            edge_index=edge_index,
            edge_curves_xyz=curves,
            edge_types=np.ones(3, dtype=np.int64),
            coincident_edge_tolerance=1.0e-6,
            minimum_manufacturable_edge_length=0.05,
        )


def test_short_edge_contraction_rejects_retained_self_loops():
    edge_index = np.asarray([[1, 2], [1, 2]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
    ]
    with pytest.raises(ManufacturingGeometryError, match="self-loop"):
        prepare_manufacturing_physical_graph(
            edge_index=edge_index,
            edge_curves_xyz=curves,
            edge_types=np.asarray([1, 2]),
            coincident_edge_tolerance=1.0e-6,
            minimum_manufacturable_edge_length=0.05,
        )


def test_contracted_graph_produces_exact_even_euler_circuit():
    edge_index = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    curves = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]),
        np.asarray([[1.01, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    filtered = prepare_manufacturing_physical_graph(
        edge_index=edge_index,
        edge_curves_xyz=curves,
        edge_normals_xyz=normals_like(curves),
        edge_types=np.ones(3, dtype=np.int64),
        coincident_edge_tolerance=1.0e-6,
        minimum_manufacturable_edge_length=0.05,
    )
    tool = EulerCCFToolpath(
        filtered["edge_index"],
        filtered["edge_curves_xyz"],
        fibres_per_edge=2,
        edge_normals_xyz=filtered["edge_normals_xyz"],
    )
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    validation = tool.validate_route(route, raise_on_error=False, verbose=False)
    assert validation["valid"]
    assert len(route) == 2 * len(filtered["edge_index"])


def test_original_physical_edge_ids_survive_validation_and_export():
    vertices, faces = plane_mesh()
    tool = single_edge_manufactured_tool()
    tool.original_physical_edge_ids = np.asarray([57], dtype=np.int64)
    tool.physical_edge_metadata = {"deduplicated": True}
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    report = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        verbose=False,
    )
    exported = tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    assert report["original_physical_edge_ids"].tolist() == [57]
    assert exported["metadata"]["original_physical_edge_ids"] == [57]
    assert {row["original_physical_edge_id"] for row in exported["points"]} == {57}


def test_uniform_automatic_fibre_count_respects_boundary_half_width():
    assert calculate_uniform_even_fibre_count(
        strut_thickness=0.4,
        fibre_line_width=0.05,
        fibre_gap=0.05,
        boundary_edge_mask=np.asarray([False, True]),
    ) == 2
    with pytest.raises(ManufacturingGeometryError):
        calculate_uniform_even_fibre_count(
            strut_thickness=0.4,
            fibre_line_width=0.05,
            fibre_gap=0.05,
            fibres_per_edge=4,
            boundary_edge_mask=np.asarray([True]),
        )


def test_projection_collapse_is_reported_and_rejected():
    vertices, faces = half_plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    with pytest.raises(ManufacturingGeometryError, match="collapsed|compressed"):
        generate_surface_offset_passes(
            edge_curves_xyz=curves,
            edge_normals_xyz=normals_like(curves),
            surface_vertices=vertices,
            surface_faces=faces,
            strut_thickness=0.4,
            fibre_line_width=0.05,
            fibre_gap=0.05,
            fibres_per_edge=2,
            edge_types=np.asarray([4]),
            boundary_inward_signs=np.asarray([-1.0]),
        )


def test_endpoint_only_spacing_compression_is_reported_but_not_sustained():
    lane_a = np.column_stack((np.linspace(0.0, 1.0, 64), np.zeros(64), np.zeros(64)))
    lane_b = lane_a.copy()
    lane_b[:, 1] = 0.10
    lane_b[0, 1] = 0.09014
    stats = _edge_spacing_stats([lane_a, lane_b], requested_pitch=0.10, tolerance=1.0e-6)
    assert stats["raw_minimum_spacing"] == pytest.approx(0.09014)
    assert stats["endpoint_compression"]
    assert stats["compressed_sample_count"] == 1
    assert stats["maximum_compressed_run"] == 1
    assert not stats["sustained_compression"]


def test_full_curve_spacing_collapse_is_sustained():
    lane_a = np.column_stack((np.linspace(0.0, 1.0, 64), np.zeros(64), np.zeros(64)))
    lane_b = lane_a.copy()
    lane_b[:, 1] = 0.05
    stats = _edge_spacing_stats([lane_a, lane_b], requested_pitch=0.10, tolerance=1.0e-6)
    assert stats["raw_minimum_spacing"] == pytest.approx(0.05)
    assert stats["compressed_sample_count"] == 64
    assert stats["maximum_compressed_run"] == 64
    assert stats["sustained_compression"]


def test_pyvista_face_conversion_validates_shape_and_range():
    vertices, _ = plane_mesh()
    pv_faces = np.asarray([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64)
    out_vertices, out_faces = coerce_surface_mesh(vertices, pv_faces)
    assert out_vertices.shape == (4, 3)
    assert out_faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    with pytest.raises(ManufacturingGeometryError):
        coerce_surface_mesh(vertices, np.asarray([[0, 1, 99]], dtype=np.int64))


def test_opposite_and_same_direction_lane_assignment():
    tool = single_edge_manufactured_tool()
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    mapping = tool.assign_route_lanes(route)
    assert [item["offset_lane"] for item in mapping] == [0, 1]
    same_direction = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 0, "v": 1, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    mapping = tool.assign_route_lanes(same_direction)
    assert [item["offset_lane"] for item in mapping] == [0, 1]


def test_closure_transition_is_constructed():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    report = tool.build_junction_transitions(route, surface_vertices=vertices, surface_faces=faces)
    assert len(report["transitions"]) == len(route)
    assert report["transitions"][-1]["closing"]
    assert report["transitions"][-1]["points"].shape[1] == 3
    assert "trim_distance_from_previous_lane" in report["transitions"][0]
    assert "transition_arc_length" in report["transitions"][0]


def test_manufactured_xyz_uses_offset_lanes_and_transitions():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    manufactured = tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
    )
    structural = tool.build_xyz_toolpath(route)
    assert not np.allclose(manufactured[: structural.shape[0]], structural)
    assert np.max(np.abs(manufactured[:, 1])) == pytest.approx(0.1)


def test_geometry_only_export_succeeds_without_machine_ready():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    exported = tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    assert exported["metadata"]["frame_valid"]
    assert any(row["point_type"] == "junction_transition" for row in exported["points"])


def test_machine_ready_export_refuses_missing_required_checks():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    with pytest.raises(ManufacturingInfeasibilityError):
        tool.export_ordered_path(
            route,
            surface_vertices=vertices,
            surface_faces=faces,
            fibre_line_width=0.2,
            strut_thickness=1.0,
            min_pass_spacing=0.2,
            require_machine_ready=True,
        )


def test_readiness_tracks_missing_required_checks():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    report = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        verbose=False,
    )
    assert report["route_ordering_ready"]
    assert not report["machine_ready"]
    assert not report["manufacturing_ready"]
    assert "self_intersections" not in report["missing_data"]
    assert report["self_intersections"] == []


def test_mesh_fallback_returns_no_uv_passes():
    vertices, faces = plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    _, _, pass_uv, offsets, diagnostics = generate_surface_offset_passes(
        edge_curves_xyz=curves,
        edge_curves_uv=[np.asarray([[0.0, 0.0], [1.0, 0.0]])],
        edge_normals_xyz=normals_like(curves),
        surface_vertices=vertices,
        surface_faces=faces,
        strut_thickness=1.0,
        fibre_line_width=0.2,
        surface_evaluator=None,
    )
    assert pass_uv is None
    assert len(offsets[0]) == 4
    assert "tangent_plane_mesh_projection" in diagnostics["method"]


def test_unvalidated_manufactured_xyz_refuses_invalid_paths():
    tool = single_edge_manufactured_tool()
    tool.edge_pass_curves_xyz[0][0] = tool.edge_pass_curves_xyz[0][0] + np.asarray([0.0, 0.0, 1.0])
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    with pytest.raises(ManufacturingInfeasibilityError):
        tool.build_manufactured_xyz_toolpath(
            route,
            surface_vertices=vertices,
            surface_faces=faces,
        )
    path = tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        allow_incomplete_geometry=True,
    )
    assert path.shape[1] == 3


def test_closing_transition_continuity_is_validated():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    report = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        verbose=False,
    )
    assert report["offset_geometry_ready"]
    assert report["max_path_discontinuity"] >= 0.0
    assert report["junction_transitions"][-1]["closing"]


def test_junction_tangent_and_radius_checks_are_reported():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    transition_report = tool.build_junction_transitions(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        min_bend_radius=1000.0,
    )
    assert transition_report["violations"]
    first = transition_report["transitions"][0]
    assert "maximum_joined_turn_angle" in first
    assert "minimum_realised_bend_radius" in first
    assert "maximum_projection_correction" in first


def test_final_manufactured_path_metrics_are_stable_across_transition_samples():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    reports = [
        tool.validate_manufactured_paths(
            route,
            surface_vertices=vertices,
            surface_faces=faces,
            fibre_line_width=0.2,
            strut_thickness=1.0,
            min_pass_spacing=0.2,
            transition_samples=samples,
            verbose=False,
        )
        for samples in (9, 17, 33)
    ]
    turns = [item["manufactured_maximum_turn_angle"] for item in reports]
    radii = [item["manufactured_minimum_bend_radius"] for item in reports]
    turns_above_120 = [item["manufactured_turns_above_120"] for item in reports]
    assert max(turns) - min(turns) <= 1.0
    finite_radii = [value for value in radii if value is not None]
    assert min(finite_radii) > 0.0
    assert max(finite_radii) / min(finite_radii) <= 1.05
    assert len(set(turns_above_120)) == 1


def test_transition_projection_degeneration_metrics_detect_collapsed_segments():
    tool = single_edge_manufactured_tool()
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    metrics = tool._transition_geometry_metrics(
        unprojected_points=points,
        projected_points=points,
        tangent_in=np.asarray([1.0, 0.0, 0.0]),
        tangent_out=np.asarray([1.0, 0.0, 0.0]),
        transition_tolerance=1.0e-6,
    )
    assert metrics["collapsed_projected_segments"]
    assert metrics["projection_degenerate"]


def test_trimmed_lane_portions_are_used_in_assembled_output():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    route_lanes = tool.route_with_lane_assignment(route)
    transitions = tool.build_junction_transitions(route, transition_samples=9)["transitions"]
    points = tool._assembled_manufactured_points(route_lanes, transitions, tolerance=1.0e-9)
    assert transitions[0]["trim_distance_from_previous_lane"] > 0.0
    assert np.allclose(points[0], tool.get_oriented_pass_curve(route_lanes[0])[0])
    assert not np.any(np.all(np.isclose(points, tool.get_oriented_pass_curve(route_lanes[0])[-1]), axis=1))


def test_failed_transition_is_not_assembled_and_blocks_readiness():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    report = tool.build_junction_transitions(route, min_bend_radius=1.0e6)
    assert any(not item.get("accepted", True) for item in report["transitions"])
    route_lanes = tool.route_with_lane_assignment(route)
    with pytest.raises(ManufacturingGeometryError):
        tool._assembled_manufactured_points(route_lanes, report["transitions"])
    validation = tool.validate_manufactured_paths(
        route,
        surface_vertices=plane_mesh()[0],
        surface_faces=plane_mesh()[1],
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        min_bend_radius=1.0e6,
        verbose=False,
    )
    assert not validation["offset_geometry_ready"]
    assert validation["junctions_with_no_valid_transition"] > 0


def test_canonical_lane_curve_applies_start_and_end_trims():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    route_lanes = tool.route_with_lane_assignment(route)
    transitions = tool.build_junction_transitions(route, transition_samples=9)["transitions"]
    lane = tool.get_oriented_pass_curve(route_lanes[0])
    manufactured_lane = tool._manufactured_lane_curve_for_transition(route_lanes, transitions, 0)
    assert not np.allclose(manufactured_lane[0], lane[0])
    assert not np.allclose(manufactured_lane[-1], lane[-1])
    assert manufactured_lane.shape[0] >= 2


def test_uv_trim_corresponds_to_xyz_arc_length_trim():
    curve_xyz = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    curve_uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    trimmed_xyz, trimmed_uv = EulerCCFToolpath.trim_corresponding_xyz_uv(
        curve_xyz,
        curve_uv,
        start_distance=0.5,
        end_distance=1.0,
    )
    assert np.allclose(trimmed_xyz[0], [0.5, 0.0, 0.0])
    assert np.allclose(trimmed_uv[0], [0.25, 0.0])
    assert np.allclose(trimmed_xyz[-1], [2.0, 1.0, 0.0])
    assert np.allclose(trimmed_uv[-1], [1.0, 0.5])
    assert len(trimmed_xyz) == len(trimmed_uv)


def test_joined_turn_angles_include_lane_transition_boundaries_and_closure():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    transitions = tool.build_junction_transitions(route, transition_samples=9)["transitions"]
    assert transitions[-1]["closing"]
    assert all(item["joined_turn_angles"].size > 0 for item in transitions)
    assert all(item.get("accepted", True) for item in transitions)


def test_export_and_xyz_construction_use_same_trimmed_geometry():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    exported = tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        transition_samples=9,
    )
    path = tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        transition_samples=9,
    )
    exported_xyz = np.asarray([[row["x"], row["y"], row["z"]] for row in exported["points"]])
    assert np.allclose(exported_xyz, path)


def test_validate_manufactured_paths_forwards_junction_trim_distance(monkeypatch):
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    trim_distance = 0.075
    forwarded = []
    original = tool.build_junction_transitions

    def spy_build_junction_transitions(*args, **kwargs):
        forwarded.append(kwargs.get("junction_trim_distance"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tool, "build_junction_transitions", spy_build_junction_transitions)
    tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
        verbose=False,
    )
    assert forwarded == [trim_distance]


def test_export_ordered_path_forwards_junction_trim_distance(monkeypatch):
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    trim_distance = 0.075
    validation_trim_distances = []
    transition_trim_distances = []
    original_validate = tool.validate_manufactured_paths
    original_build = tool.build_junction_transitions

    def spy_validate_manufactured_paths(*args, **kwargs):
        validation_trim_distances.append(kwargs.get("junction_trim_distance"))
        return original_validate(*args, **kwargs)

    def spy_build_junction_transitions(*args, **kwargs):
        transition_trim_distances.append(kwargs.get("junction_trim_distance"))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(tool, "validate_manufactured_paths", spy_validate_manufactured_paths)
    monkeypatch.setattr(tool, "build_junction_transitions", spy_build_junction_transitions)
    tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
    )
    assert validation_trim_distances == [trim_distance]
    assert transition_trim_distances == [trim_distance, trim_distance]


def test_build_manufactured_xyz_toolpath_forwards_junction_trim_distance(monkeypatch):
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    trim_distance = 0.075
    validation_trim_distances = []
    transition_trim_distances = []
    original_validate = tool.validate_manufactured_paths
    original_build = tool.build_junction_transitions

    def spy_validate_manufactured_paths(*args, **kwargs):
        validation_trim_distances.append(kwargs.get("junction_trim_distance"))
        return original_validate(*args, **kwargs)

    def spy_build_junction_transitions(*args, **kwargs):
        transition_trim_distances.append(kwargs.get("junction_trim_distance"))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(tool, "validate_manufactured_paths", spy_validate_manufactured_paths)
    monkeypatch.setattr(tool, "build_junction_transitions", spy_build_junction_transitions)
    tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
    )
    assert validation_trim_distances == [trim_distance]
    assert transition_trim_distances == [trim_distance, trim_distance]


def test_manufacturing_apis_remain_valid_without_junction_trim_distance():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    validation = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        verbose=False,
    )
    exported = tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    path = tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    assert validation["offset_geometry_ready"]
    assert exported["metadata"]["frame_valid"]
    assert path.shape[1] == 3


def test_validation_export_and_xyz_select_same_transition_candidate():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    trim_distance = 0.075
    validation = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
        transition_samples=9,
        verbose=False,
    )
    exported = tool.export_ordered_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
        transition_samples=9,
    )
    export_transitions = tool.junction_transitions
    path = tool.build_manufactured_xyz_toolpath(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        junction_trim_distance=trim_distance,
        transition_samples=9,
    )
    xyz_transitions = tool.junction_transitions
    for validation_transition, exported_transition, xyz_transition in zip(
        validation["junction_transitions"],
        export_transitions,
        xyz_transitions,
    ):
        assert validation_transition["trim_distance_from_previous_lane"] == pytest.approx(
            exported_transition["trim_distance_from_previous_lane"]
        )
        assert validation_transition["trim_distance_from_previous_lane"] == pytest.approx(
            xyz_transition["trim_distance_from_previous_lane"]
        )
        assert validation_transition["selected_handle_factor"] == pytest.approx(
            exported_transition["selected_handle_factor"]
        )
        assert validation_transition["selected_handle_factor"] == pytest.approx(
            xyz_transition["selected_handle_factor"]
        )
        assert np.allclose(validation_transition["points"], exported_transition["points"])
        assert np.allclose(validation_transition["points"], xyz_transition["points"])
    exported_xyz = np.asarray([[row["x"], row["y"], row["z"]] for row in exported["points"]])
    assert np.allclose(exported_xyz, path)


def test_valid_planar_transition_has_no_180_turn_or_degenerate_projection():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    transitions = tool.build_junction_transitions(route, transition_samples=17)["transitions"]
    assert all(item.get("accepted", True) for item in transitions)
    assert all(not item["projection_degenerate"] for item in transitions)
    assert all(float(np.max(item["joined_turn_angles"])) < 179.0 for item in transitions)


def test_curved_cad_transitions_accept_coarse_trimmed_mesh_deviation():
    cad, tool = curved_single_edge_tool()
    vertices, faces = curved_plane_mesh(cad, grid=4)
    dense_uv = np.asarray([[u, v] for v in np.linspace(0.1, 0.9, 9) for u in np.linspace(0.1, 0.9, 9)])
    dense_xyz = cad.eval_uv_norm_batch(dense_uv, return_inside_mask=True)["xyz"]
    _, _, mesh_distances = MeshSurfaceProjector(vertices, faces).project(dense_xyz)
    assert float(mesh_distances.max()) > 1.0e-5

    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    tables = {}
    choices = {}
    for with_mesh in (False, True):
        reports = []
        transition_choices = []
        for samples in (9, 17, 33):
            kwargs = {
                "surface_evaluator": cad,
                "fibre_line_width": 0.05,
                "strut_thickness": 0.4,
                "min_pass_spacing": 0.05,
                "transition_samples": samples,
                "verbose": False,
                "require_surface_check": True,
            }
            if with_mesh:
                kwargs.update({"surface_vertices": vertices, "surface_faces": faces})
            report = tool.validate_manufactured_paths(route, **kwargs)
            reports.append(report)
            assert report["junctions_with_no_valid_transition"] == 0
            assert all(item.get("accepted", True) for item in report["junction_transitions"])
            assert all(not item["projection_degenerate"] for item in report["junction_transitions"])
            transition = report["junction_transitions"][0]
            assert transition["points"].shape[0] == samples
            assert transition["normals"].shape[0] == samples
            assert transition["uv"].shape[0] == samples
            route_lanes = tool.route_with_lane_assignment(route)
            for transition_id, item in enumerate(report["junction_transitions"]):
                lane = tool._manufactured_lane_curve_for_transition(route_lanes, report["junction_transitions"], transition_id)
                next_lane = tool._manufactured_lane_curve_for_transition(route_lanes, report["junction_transitions"], (transition_id + 1) % len(route_lanes))
                assert np.linalg.norm(lane[-1] - item["points"][0]) <= 1.0e-6
                assert np.linalg.norm(item["points"][-1] - next_lane[0]) <= 1.0e-6
            transition_choices.append((
                transition["trim_distance_from_previous_lane"],
                transition["selected_handle_factor"],
            ))
        turns = [item["manufactured_maximum_turn_angle"] for item in reports]
        radii = [item["manufactured_minimum_bend_radius"] for item in reports]
        local_turns = [
            max(float(np.max(transition["joined_turn_angles"])) for transition in item["junction_transitions"])
            for item in reports
        ]
        assert max(turns) - min(turns) <= 1.0
        assert max(radii) / min(radii) <= 1.05
        assert len({item["manufactured_turns_above_120"] for item in reports}) == 1
        assert all(abs(turn - local) <= 1.0 for turn, local in zip(turns, local_turns))
        assert max(turns) <= 135.0
        tables[with_mesh] = (turns, radii, [item["manufactured_turns_above_120"] for item in reports])
        choices[with_mesh] = transition_choices
    assert choices[False] == choices[True]


def test_article_geodesic_offset_mode_is_out_of_scope():
    vertices, faces = plane_mesh()
    curves = [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    with pytest.raises(NotImplementedError, match="out of scope"):
        generate_manufacturing_offset_passes(
            offset_method="voronoi_geodesic_isocurve",
            edge_curves_xyz=curves,
            edge_curves_uv=None,
            edge_normals_xyz=normals_like(curves),
            surface_vertices=vertices,
            surface_faces=faces,
            strut_thickness=1.0,
            fibre_line_width=0.2,
        )


def test_minimal_xyz_normal_export_has_six_columns():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    data = tool.export_minimal_xyz_normal_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    assert data.ndim == 2
    assert data.shape[1] == 6


def test_full_export_uses_fibre_lane_and_closure_transition_types(tmp_path):
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    csv_path = tmp_path / "ordered.csv"
    json_path = tmp_path / "ordered.json"
    exported = tool.export_ordered_path(
        route,
        filename=csv_path,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    point_types = {row["point_type"] for row in exported["points"]}
    assert "fibre_lane" in point_types
    assert "closure_transition" in point_types
    assert all("fibre_instance_id" in row and "lane_id" in row and "fibre_on" in row for row in exported["points"])
    assert csv_path.read_text().splitlines()[0].startswith("path_index,traversal_id,physical_edge_id")
    tool.export_ordered_path(
        route,
        filename=json_path,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
    )
    payload = json.loads(json_path.read_text())
    assert payload["points"][0]["point_type"] == "fibre_lane"


def test_minus_normal_reverses_exported_normals_only():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    plus = tool.export_minimal_xyz_normal_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        nozzle_approach="+normal",
    )
    minus = tool.export_minimal_xyz_normal_path(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        nozzle_approach="-normal",
    )
    assert np.allclose(plus[:, :3], minus[:, :3])
    assert np.allclose(plus[:, 3:6], -minus[:, 3:6])


def test_exact_mesh_projection_not_centroid_limited():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [30.0, 30.0, 1.0],
            [31.0, 30.0, 1.0],
            [30.0, 31.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[3, 4, 5], [0, 1, 2]], dtype=np.int64)
    projected, _, distances = MeshSurfaceProjector(vertices, faces).project(np.asarray([[0.1, 0.1, 0.2]]))
    assert np.allclose(projected[0], [0.1, 0.1, 0.0])
    assert distances[0] == pytest.approx(0.2)


def test_cad_transition_endpoint_tangents_follow_surface_jacobian():
    edge_index = np.asarray([[0, 1]], dtype=np.int64)
    curves = [np.asarray([[0.0, -0.2, 0.0], [1.0, -0.2, 0.0]])]
    normals = normals_like(curves)
    passes = [[
        np.asarray([[0.0, -0.2, 0.0], [1.0, -0.2, 0.0]]),
        np.asarray([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]]),
    ]]
    pass_uv = [[curve[:, :2].copy() for curve in passes[0]]]
    tool = EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals,
        edge_pass_curves_xyz=passes,
        edge_pass_normals_xyz=[[normals[0].copy(), normals[0].copy()]],
        edge_pass_curves_uv=pass_uv,
        edge_pass_offsets=[np.asarray([-0.2, 0.2])],
    )
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    report = tool.build_junction_transitions(route, surface_evaluator=PlaneCad(), transition_samples=101)
    assert report["transitions"][0]["method"] == "cad_uv_transition"
    assert report["transitions"][0]["start_tangent_angle"] < 10.0
    assert report["transitions"][0]["end_tangent_angle"] < 10.0


def test_allowed_point_intersection_at_physical_junction():
    tool = two_edge_crossing_tool(off_junction=False)
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 2, "key": 0, "edge_id": 1, "fibre_id": 0},
        {"u": 2, "v": 1, "key": 1, "edge_id": 1, "fibre_id": 1},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    tool.edge_pass_curves_xyz = [
        [tool.edge_curves_xyz[0].copy(), tool.edge_curves_xyz[0].copy() + np.asarray([0.0, 0.05, 0.0])],
        [tool.edge_curves_xyz[1].copy(), tool.edge_curves_xyz[1].copy() + np.asarray([-0.05, 0.0, 0.0])],
    ]
    transitions = [{"points": np.asarray([tool.get_oriented_pass_curve(item)[-1], tool.get_oriented_pass_curve(tool.route_with_lane_assignment(route)[(i + 1) % len(route)])[0]]), "closing": i == len(route) - 1} for i, item in enumerate(tool.route_with_lane_assignment(route))]
    enriched = tool.route_with_lane_assignment(route)
    report = tool._validate_manufactured_intersections(enriched, transitions, junction_region_radius=0.25)
    assert report["violations"] == []


def test_forbidden_off_junction_intersection_and_finite_overlap():
    tool = two_edge_crossing_tool(off_junction=True)
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 2, "v": 3, "key": 0, "edge_id": 1, "fibre_id": 0},
        {"u": 3, "v": 2, "key": 1, "edge_id": 1, "fibre_id": 1},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    enriched = tool.route_with_lane_assignment(route)
    transitions = [{"points": np.asarray([tool.get_oriented_pass_curve(item)[-1], tool.get_oriented_pass_curve(enriched[(i + 1) % len(enriched)])[0]]), "closing": i == len(enriched) - 1} for i, item in enumerate(enriched)]
    report = tool._validate_manufactured_intersections(enriched, transitions, junction_region_radius=0.05)
    kinds = {item["kind"] for item in report["violations"]}
    assert "off_junction_intersection" in kinds or "lane_crossover" in kinds
    assert "finite_segment_overlap" in kinds


def test_adjacent_path_segments_are_not_false_self_intersections():
    tool = single_edge_manufactured_tool()
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    enriched = tool.route_with_lane_assignment(route)
    transitions = [{"points": np.asarray([tool.get_oriented_pass_curve(item)[-1], tool.get_oriented_pass_curve(enriched[(i + 1) % len(enriched)])[0]]), "closing": i == len(enriched) - 1} for i, item in enumerate(enriched)]
    report = tool._validate_manufactured_intersections(enriched, transitions)
    assert report["violations"] == []


def test_discontinuity_report_excludes_transition_span():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    report = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        verbose=False,
    )
    assert report["maximum_transition_span"] > 0.0
    assert report["maximum_unbridged_discontinuity"] == pytest.approx(0.0)


def test_lane_refinement_for_more_than_eight_lanes_is_bijective():
    n = 10
    tool = multi_lane_single_edge_tool(n)
    route = [{"u": 0, "v": 1, "key": i, "edge_id": 0, "fibre_id": i} for i in reversed(range(n))]
    mapping = tool.assign_route_lanes(route)
    assert sorted(item["offset_lane"] for item in mapping) == list(range(n))


def test_manufactured_hard_turn_limit_blocks_offset_geometry():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    report = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=1.0,
        min_pass_spacing=0.2,
        max_manufactured_turn_degrees=1.0,
        verbose=False,
    )
    assert not report["offset_geometry_ready"]
    assert any(item["kind"] == "manufactured_turn_angle" for item in report["violations"])


def test_per_edge_strut_thickness_arrays_are_validated():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    vertices, faces = plane_mesh()
    ok = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=np.asarray([1.0]),
        min_pass_spacing=0.2,
        verbose=False,
    )
    assert ok["inside_strut_with_half_width"]
    bad = tool.validate_manufactured_paths(
        route,
        surface_vertices=vertices,
        surface_faces=faces,
        fibre_line_width=0.2,
        strut_thickness=np.asarray([0.2]),
        min_pass_spacing=0.2,
        verbose=False,
    )
    assert not bad["offset_geometry_ready"]
    assert any(item["kind"] == "deposited_half_width_outside_strut" for item in bad["violations"])


def test_route_solver_euler_exact_even_alias():
    tool = square_tool()
    tool.build_multigraph()
    route = tool.compute_route(route_solver="euler_exact_even")
    assert tool.validate_route(route, raise_on_error=False, verbose=False)["valid"]


def test_3d_segment_detection_ignores_equal_xy_at_different_z():
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0, 0, 0], [1, 0, 0],
        [0, 0, 1], [1, 0, 1],
        tolerance=1.0e-6,
    )
    assert hit["type"] == "none"
    assert hit["distance"] == pytest.approx(1.0)


def test_3d_segment_detection_tilted_and_near_vertical_intersections():
    tilted = EulerCCFToolpath._segment_intersection_3d(
        [0, 0, 0], [1, 1, 1],
        [0, 1, 0], [1, 0, 1],
        tolerance=1.0e-6,
    )
    assert tilted["type"] == "point"
    assert np.allclose(tilted["point_first"], [0.5, 0.5, 0.5])
    vertical = EulerCCFToolpath._segment_intersection_3d(
        [0, 0, 0], [0.001, 0, 1],
        [0.001, -0.5, 0.5], [0.0, 0.5, 0.5],
        tolerance=1.0e-6,
    )
    assert vertical["type"] == "point"
    assert vertical["distance"] <= 1.0e-6


def assert_intersection_distance_within_tolerance(hit, tolerance):
    if hit["type"] in ("point", "overlap"):
        assert hit["distance"] <= tolerance


def test_3d_segment_detection_overlap_and_endpoint_contact():
    overlap = EulerCCFToolpath._segment_intersection_3d(
        [0, 0, 0], [2, 0, 0],
        [1, 0, 0], [3, 0, 0],
        tolerance=1.0e-6,
    )
    assert overlap["type"] == "overlap"
    assert overlap["distance"] <= 1.0e-6
    assert overlap["overlap_length"] > 1.0e-6
    assert "overlap_interval_first" in overlap
    assert "overlap_interval_second" in overlap
    endpoint = EulerCCFToolpath._segment_intersection_3d(
        [0, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 1, 0],
        tolerance=1.0e-6,
    )
    assert endpoint["type"] == "point"
    assert endpoint["distance"] <= 1.0e-6
    assert endpoint["parameter_first"] == pytest.approx(1.0)
    assert endpoint["parameter_second"] == pytest.approx(0.0)


def test_short_consecutive_segments_share_endpoint_as_point_not_overlap():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0],
        [1.0e-4, 0.0, 0.0], [2.0e-4, 1.0e-5, 0.0],
        tolerance=tol,
    )
    assert hit["type"] == "point"
    assert_intersection_distance_within_tolerance(hit, tol)


def test_short_nearly_parallel_segments_separated_above_tolerance_are_none():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0],
        [0.0, 8.5e-5, 0.0], [1.0e-4, 8.6e-5, 0.0],
        tolerance=tol,
    )
    assert hit["type"] == "none"
    assert hit["distance"] > tol


def test_short_nonparallel_segments_do_not_use_length_scaled_parallel_rule():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [2.0e-4, 0.0, 0.0],
        [0.0, 6.36e-4, 0.0], [2.0e-4, 6.56e-4, 0.0],
        tolerance=tol,
    )
    assert hit["type"] != "overlap"
    assert hit["type"] == "none"
    assert hit["distance"] > tol


def test_collinear_endpoint_touch_is_point_not_finite_overlap():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        tolerance=tol,
    )
    assert hit["type"] == "point"
    assert hit["overlap_length"] == pytest.approx(0.0)
    assert_intersection_distance_within_tolerance(hit, tol)


def test_parallel_separated_segments_are_none():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0e-5, 0.0], [1.0, 2.0e-5, 0.0],
        tolerance=tol,
    )
    assert hit["type"] == "none"
    assert hit["distance"] > tol


def test_skew_3d_segments_use_true_3d_closest_distance():
    tol = 1.0e-5
    hit = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.5, -0.5, 2.0e-5], [0.5, 0.5, 2.0e-5],
        tolerance=tol,
    )
    assert hit["type"] == "none"
    assert hit["distance"] == pytest.approx(2.0e-5)


def test_segment_intersection_scale_invariance():
    small_tol = 1.0e-5
    small = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0],
        [2.5e-5, -5.0e-5, 0.0], [2.5e-5, 5.0e-5, 0.0],
        tolerance=small_tol,
    )
    large = EulerCCFToolpath._segment_intersection_3d(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.25, -0.5, 0.0], [0.25, 0.5, 0.0],
        tolerance=0.1,
    )
    assert small["type"] == large["type"] == "point"
    assert_intersection_distance_within_tolerance(small, small_tol)
    assert_intersection_distance_within_tolerance(large, 0.1)


def test_segment_intersection_hits_always_respect_spatial_tolerance():
    cases = [
        (([0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]), 1.0e-6),
        (([0, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0]), 1.0e-6),
        (([0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]), 1.0e-6),
        (([0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0], [0.0, 8.5e-5, 0.0], [1.0e-4, 8.6e-5, 0.0]), 1.0e-5),
        (([0, 0, 0], [1, 0, 0], [0, 2.0e-5, 0], [1, 2.0e-5, 0]), 1.0e-5),
    ]
    for (a, b, c, d), tolerance in cases:
        hit = EulerCCFToolpath._segment_intersection_3d(a, b, c, d, tolerance=tolerance)
        assert_intersection_distance_within_tolerance(hit, tolerance)


def test_physically_scaled_junction_region_accepts_shared_node_crossing():
    tool = two_edge_crossing_tool(off_junction=False)
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 1, "v": 2, "key": 0, "edge_id": 1, "fibre_id": 0},
        {"u": 2, "v": 1, "key": 1, "edge_id": 1, "fibre_id": 1},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    tool.edge_pass_curves_xyz = [
        [np.asarray([[0.0, 0.0, 0.0], [1.0, 0.05, 0.0]]), np.asarray([[0.0, -0.05, 0.0], [1.0, 0.0, 0.0]])],
        [np.asarray([[1.0, 0.0, 0.0], [1.05, 1.0, 0.0]]), np.asarray([[0.95, 0.0, 0.0], [1.0, 1.0, 0.0]])],
    ]
    enriched = tool.route_with_lane_assignment(route)
    transitions = [
        {"points": np.asarray([tool.get_oriented_pass_curve(item)[-1], tool.get_oriented_pass_curve(enriched[(i + 1) % len(enriched)])[0]]), "closing": i == len(enriched) - 1, "edge_ids": [item["edge_id"], enriched[(i + 1) % len(enriched)]["edge_id"]]}
        for i, item in enumerate(enriched)
    ]
    report = tool._validate_manufactured_intersections(
        enriched,
        transitions,
        tolerance=1.0e-6,
        fibre_line_width=0.1,
        strut_thickness=0.4,
    )
    assert report["accepted_junction_intersection_count"] >= 1
    assert all(item["node_id"] == 1 for item in report["accepted_junction_intersections"])


def test_unrelated_segment_spacing_violation_is_reported():
    tool = two_edge_crossing_tool(off_junction=True)
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0},
        {"u": 2, "v": 3, "key": 0, "edge_id": 1, "fibre_id": 0},
        {"u": 3, "v": 2, "key": 1, "edge_id": 1, "fibre_id": 1},
        {"u": 1, "v": 0, "key": 1, "edge_id": 0, "fibre_id": 1},
    ]
    tool.edge_pass_curves_xyz[1] = [
        np.asarray([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]]),
        np.asarray([[1.0, 0.3, 0.0], [0.0, 0.3, 0.0]]),
    ]
    enriched = tool.route_with_lane_assignment(route)
    transitions = [
        {"points": np.asarray([tool.get_oriented_pass_curve(item)[-1], tool.get_oriented_pass_curve(enriched[(i + 1) % len(enriched)])[0]]), "closing": i == len(enriched) - 1, "edge_ids": [item["edge_id"], enriched[(i + 1) % len(enriched)]["edge_id"]]}
        for i, item in enumerate(enriched)
    ]
    report = tool._validate_manufactured_intersections(
        enriched,
        transitions,
        tolerance=1.0e-6,
        min_unrelated_segment_spacing=0.25,
    )
    assert report["minimum_distance_between_unrelated_segments"] is not None
    assert any(item["kind"] == "unrelated_segment_spacing" for item in report["violations"])


def test_assembled_manufactured_points_contains_each_traversal_once():
    tool = single_edge_manufactured_tool()
    tool.build_multigraph()
    route = tool.compute_euler_circuit()
    route_lanes = tool.route_with_lane_assignment(route)
    transitions = tool.build_junction_transitions(route, transition_samples=2)["transitions"]
    points = tool._assembled_manufactured_points(route_lanes, transitions, tolerance=1.0e-9)
    assert len(route_lanes) == 2
    assert points.shape[1] == 3
    assert points.shape[0] >= 5
    assert np.allclose(points[0], points[-1])


def test_closed_square_manufactured_metrics_count_closing_turn_once():
    tool = square_tool()
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    metrics = tool._manufactured_path_metrics(points, tolerance=1.0e-9)
    assert len(metrics["manufactured_turn_angles"]) == 4
    assert np.allclose(metrics["manufactured_turn_angles"], 90.0)
    assert metrics["manufactured_mean_turn_angle"] == pytest.approx(90.0)
    assert metrics["manufactured_median_turn_angle"] == pytest.approx(90.0)
    assert metrics["manufactured_maximum_turn_angle"] == pytest.approx(90.0)


def parallel_spacing_tool(shared_node=False):
    if shared_node:
        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        curves = [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.2, 0.0], [2.0, 0.2, 0.0]]),
        ]
    else:
        edge_index = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
        curves = [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]]),
        ]
    normals = normals_like(curves)
    passes = [[curve.copy(), curve.copy() + np.asarray([0.0, 1.0, 0.0])] for curve in curves]
    return EulerCCFToolpath(
        edge_index,
        curves,
        fibres_per_edge=2,
        edge_normals_xyz=normals,
        edge_pass_curves_xyz=passes,
        edge_pass_normals_xyz=[[normals[i].copy(), normals[i].copy()] for i in range(len(curves))],
        edge_pass_offsets=[np.asarray([0.0, 1.0]) for _ in curves],
    )


def test_unrelated_spacing_broad_phase_uses_requested_clearance():
    tool = parallel_spacing_tool(shared_node=False)
    route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0, "offset_lane": 0},
        {"u": 2, "v": 3, "key": 0, "edge_id": 1, "fibre_id": 0, "offset_lane": 0},
    ]
    transitions = [{"points": np.asarray([tool.get_oriented_pass_curve(item)[-1]])} for item in route]
    bad = tool._validate_manufactured_intersections(route, transitions, min_unrelated_segment_spacing=0.3)
    assert bad["minimum_distance_between_unrelated_segments"] == pytest.approx(0.2)
    assert any(item["kind"] == "unrelated_segment_spacing" for item in bad["violations"])
    ok = tool._validate_manufactured_intersections(route, transitions, min_unrelated_segment_spacing=0.1)
    assert ok["minimum_distance_between_unrelated_segments"] == pytest.approx(0.2)
    assert not any(item["kind"] == "unrelated_segment_spacing" for item in ok["violations"])


def test_close_spacing_inside_same_junction_is_exempt_but_unrelated_is_not():
    shared = parallel_spacing_tool(shared_node=True)
    shared_route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0, "offset_lane": 0},
        {"u": 1, "v": 2, "key": 0, "edge_id": 1, "fibre_id": 0, "offset_lane": 0},
    ]
    transitions = [{"points": np.asarray([shared.get_oriented_pass_curve(item)[-1]])} for item in shared_route]
    shared_report = shared._validate_manufactured_intersections(
        shared_route,
        transitions,
        fibre_line_width=0.1,
        strut_thickness=0.4,
        min_unrelated_segment_spacing=0.3,
    )
    assert not any(item["kind"] == "unrelated_segment_spacing" for item in shared_report["violations"])

    unrelated = parallel_spacing_tool(shared_node=False)
    unrelated_route = [
        {"u": 0, "v": 1, "key": 0, "edge_id": 0, "fibre_id": 0, "offset_lane": 0},
        {"u": 2, "v": 3, "key": 0, "edge_id": 1, "fibre_id": 0, "offset_lane": 0},
    ]
    transitions = [{"points": np.asarray([unrelated.get_oriented_pass_curve(item)[-1]])} for item in unrelated_route]
    unrelated_report = unrelated._validate_manufactured_intersections(
        unrelated_route,
        transitions,
        fibre_line_width=0.1,
        strut_thickness=0.4,
        min_unrelated_segment_spacing=0.3,
    )
    assert any(item["kind"] == "unrelated_segment_spacing" for item in unrelated_report["violations"])


def test_transition_allows_only_exact_junction_node_not_other_endpoint():
    tool = parallel_spacing_tool(shared_node=True)
    tool.edge_pass_offsets = None
    fibre = {"node_ids": {0, 1}, "edge_ids": [0], "a": np.asarray([0, 0, 0]), "b": np.asarray([1, 0, 0])}
    transition = {"node_ids": {1}, "edge_ids": [0, 1], "a": np.asarray([0, 0, 0]), "b": np.asarray([1, 0, 0])}
    radii = tool._junction_radii(fibre_line_width=0.1, strut_thickness=0.4, user_supplied_junction_radius=0.2)
    assert tool._accepted_junction_node(np.asarray([0.0, 0.0, 0.0]), fibre, transition, radii) is None
    assert tool._accepted_junction_node(np.asarray([1.0, 0.0, 0.0]), fibre, transition, radii) == 1
