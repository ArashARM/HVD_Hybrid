import math

import numpy as np
import torch

from problems.ThickenShell import ThickenShell
from Utils.ExportAbaqus_VoxelBased import _material_bins_exact_or_logarithmic


def _face_areas(points, faces):
    tri = points[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )


def _tensors(points, faces, face_id=None, **extra):
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if face_id is None:
        face_id = np.zeros((points.shape[0],), dtype=np.int64)
    data = {
        "uv": points[:, :2].astype(np.float32),
        "points_xyz": points.astype(np.float32),
        "face_areas": _face_areas(points, faces).astype(np.float32),
        "Xu": np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (points.shape[0], 1)),
        "Xv": np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (points.shape[0], 1)),
        "faces_ijk": faces.astype(np.int64),
        "pv_faces": faces.astype(np.int64),
        "face_id": np.asarray(face_id, dtype=np.int64),
        "boundary_idx_ring1": np.array([], dtype=np.int64),
        "min_vol_frac": np.array([0.0], dtype=np.float32),
        "BBX": {
            "xmin": float(points[:, 0].min()),
            "xmax": float(points[:, 0].max()),
            "ymin": float(points[:, 1].min()),
            "ymax": float(points[:, 1].max()),
            "zmin": float(points[:, 2].min()),
            "zmax": float(points[:, 2].max()),
        },
    }
    data.update(extra)
    return data


def _annulus_with_boundary_tensors(n=32):
    outer_r = 2.0
    inner_r = 0.65
    points = []
    outer = []
    inner = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        outer.append([outer_r * math.cos(a), outer_r * math.sin(a), 0.0])
        inner.append([inner_r * math.cos(a), inner_r * math.sin(a), 0.0])
    points = np.asarray(outer + inner, dtype=np.float64)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    faces = np.asarray(faces, dtype=np.int64)

    outer_loop = np.asarray(outer + [outer[0]], dtype=np.float64)
    inner_loop = np.asarray(inner + [inner[0]], dtype=np.float64)
    curve_xyz = np.concatenate([outer_loop, inner_loop], axis=0)
    offsets = np.asarray([0, outer_loop.shape[0], curve_xyz.shape[0]], dtype=np.int64)
    return _tensors(
        points,
        faces,
        boundary_curve_xyz=curve_xyz,
        boundary_curve_uv=curve_xyz[:, :2].astype(np.float32),
        boundary_curve_offsets=offsets,
        boundary_curve_loop_id=np.asarray([0, 1], dtype=np.int64),
        boundary_curve_piece_id=np.asarray([0, 0], dtype=np.int64),
        boundary_curve_loop_kind=np.asarray([0, 1], dtype=np.int64),
        boundary_curve_loop_area=np.asarray([math.pi * outer_r * outer_r, -math.pi * inner_r * inner_r]),
        boundary_curve_length=np.asarray([2.0 * math.pi * outer_r, 2.0 * math.pi * inner_r]),
    )


def _c_bracket_with_boundary_tensors(nx=33, ny=25, hole_samples=48):
    x_values = np.linspace(-2.0, 2.0, nx)
    y_values = np.linspace(-1.5, 1.5, ny)
    hole_center = np.array([-0.75, 0.0, 0.0], dtype=np.float64)
    hole_r = 0.38
    notch_x = 0.65
    notch_half_height = 0.55

    point_index = {}
    points = []
    faces = []

    def include_cell(cx, cy):
        in_bbox = -2.0 <= cx <= 2.0 and -1.5 <= cy <= 1.5
        in_notch = cx > notch_x and abs(cy) < notch_half_height
        in_hole = np.linalg.norm([cx - hole_center[0], cy - hole_center[1]]) < hole_r
        return in_bbox and not in_notch and not in_hole

    def node_id(i, j):
        key = (i, j)
        if key not in point_index:
            point_index[key] = len(points)
            points.append([x_values[i], y_values[j], 0.0])
        return point_index[key]

    for i in range(nx - 1):
        for j in range(ny - 1):
            cx = 0.5 * (x_values[i] + x_values[i + 1])
            cy = 0.5 * (y_values[j] + y_values[j + 1])
            if not include_cell(cx, cy):
                continue
            p00 = node_id(i, j)
            p10 = node_id(i + 1, j)
            p11 = node_id(i + 1, j + 1)
            p01 = node_id(i, j + 1)
            faces.append([p00, p10, p11])
            faces.append([p00, p11, p01])

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    outer = np.asarray(
        [
            [-2.0, -1.5, 0.0],
            [2.0, -1.5, 0.0],
            [2.0, -notch_half_height, 0.0],
            [notch_x, -notch_half_height, 0.0],
            [notch_x, notch_half_height, 0.0],
            [2.0, notch_half_height, 0.0],
            [2.0, 1.5, 0.0],
            [-2.0, 1.5, 0.0],
            [-2.0, -1.5, 0.0],
        ],
        dtype=np.float64,
    )
    hole = []
    for i in range(hole_samples + 1):
        a = 2.0 * math.pi * i / hole_samples
        hole.append([
            hole_center[0] + hole_r * math.cos(a),
            hole_center[1] + hole_r * math.sin(a),
            0.0,
        ])
    hole = np.asarray(hole, dtype=np.float64)
    curve_xyz = np.concatenate([outer, hole], axis=0)
    offsets = np.asarray([0, outer.shape[0], curve_xyz.shape[0]], dtype=np.int64)

    tensors = _tensors(
        points,
        faces,
        boundary_curve_xyz=curve_xyz,
        boundary_curve_uv=curve_xyz[:, :2].astype(np.float32),
        boundary_curve_offsets=offsets,
        boundary_curve_loop_id=np.asarray([0, 1], dtype=np.int64),
        boundary_curve_piece_id=np.asarray([0, 0], dtype=np.int64),
        boundary_curve_loop_kind=np.asarray([0, 1], dtype=np.int64),
        boundary_curve_loop_area=np.asarray([12.0, -math.pi * hole_r * hole_r]),
        boundary_curve_length=np.asarray([17.4, 2.0 * math.pi * hole_r]),
    )
    return tensors, hole_center


def _shell(tensors, **kwargs):
    use_default_regions = kwargs.pop("use_default_regions", True)
    call_kwargs = dict(
        thickness=kwargs.pop("thickness", 1.0),
        BC_dir=kwargs.pop("BC_dir", "x"),
        Load_magnitude=kwargs.pop("Load_magnitude", -10.0),
        voxel_size=kwargs.pop("voxel_size", 0.5),
        extra_layers=kwargs.pop("extra_layers", 1),
        tensors=tensors,
        **kwargs,
    )
    if use_default_regions:
        call_kwargs["fixed_region"] = kwargs.pop(
            "fixed_region",
            {"extent_axis": "x", "extent_side": "min", "band": 0.375},
        )
        call_kwargs["load_region"] = kwargs.pop(
            "load_region",
            {"extent_axis": "x", "extent_side": "max", "band": 0.375, "direction": "z"},
        )
    return ThickenShell(**call_kwargs)


def test_triangle_distance_planar_plate_is_invariant_to_remesh():
    points_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    faces_a = np.array([[0, 1, 2], [0, 2, 3]])
    points_b = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces_b = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    shell_a = _shell(_tensors(points_a, faces_a))
    shell_b = _shell(_tensors(points_b, faces_b))

    assert np.array_equal(shell_a.elem_occupancy, shell_b.elem_occupancy)
    audit = shell_a.geometry_audit()
    assert audit["face_connected_component_count"] == 1
    assert audit["zero_face_neighbour_count"] == 0


def test_triangle_distance_preserves_open_hole_under_vertex_refinement():
    def annulus_segments(n):
        outer_r = 2.0
        inner_r = 0.65
        points = []
        for r in (outer_r, inner_r):
            for i in range(n):
                a = 2.0 * math.pi * i / n
                points.append([r * math.cos(a), r * math.sin(a), 0.0])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        return np.asarray(points), np.asarray(faces)

    coarse = _shell(_tensors(*annulus_segments(16)), voxel_size=0.4, thickness=0.4)
    fine = _shell(_tensors(*annulus_segments(48)), voxel_size=0.4, thickness=0.4)

    for shell in (coarse, fine):
        centers = shell.elem_centers.reshape(-1, 3)
        occ = shell.elem_occupancy.reshape(-1).astype(bool)
        hole_center_mask = np.linalg.norm(centers[:, :2], axis=1) < 0.35
        assert not np.any(occ & hole_center_mask)

    rel = abs(int(coarse.elem_occupancy.sum()) - int(fine.elem_occupancy.sum())) / int(fine.elem_occupancy.sum())
    assert rel < 0.08


def test_barycentric_transfer_uses_triangle_vertices_not_nearest_sample():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2]])),
        voxel_size=0.5,
        thickness=0.5,
    )
    rho = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
    fiber = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float32)
    fields = shell.build_fem_fields_from_decoder_torch(rho, fiber)
    active = np.flatnonzero(shell.elem_occupancy.reshape(-1).astype(bool))
    interior = active[np.argmin(np.linalg.norm(shell.elem_closest_point[active] - np.array([0.5, 0.5, 0.0]), axis=1))]
    assert 0.35 < float(fields["density"][interior]) < 0.65


def test_subvoxel_sampling_reports_partial_geometric_fraction():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        voxel_size=1.0,
        thickness=0.5,
        subvoxel_samples=2,
        min_geom_fraction=0.25,
    )
    active_fraction = shell.elem_geom_fraction[shell.elem_occupancy.reshape(-1).astype(bool)]
    assert np.any((active_fraction > 0.0) & (active_fraction < 1.0))
    assert active_fraction.min() >= 0.25


def test_surface_patch_load_is_area_weighted_and_conserved():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        voxel_size=0.5,
        thickness=0.5,
        load_region={"extent_axis": "x", "extent_side": "max", "band": 0.375, "direction": "y", "total_force": 12.5},
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), [0.0, 12.5, 0.0])
    fixed_nodes = np.unique(shell.boundaryCondition["fixed"] // 3)
    loaded_nodes = np.unique(np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0))
    assert np.intersect1d(fixed_nodes, loaded_nodes).size == 0
    assert shell.bc_report["loaded_face_count"] > 0


def test_extent_region_faces_stay_inside_requested_band():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    band = 0.25
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        voxel_size=0.5,
        thickness=0.5,
        fixed_region={"extent_axis": "x", "extent_side": "min", "band": band},
        load_region={"extent_axis": "x", "extent_side": "max", "band": band, "direction": "x", "total_force": 2.0},
    )
    fixed_faces = shell.select_exposed_faces_by_region(shell.fixed_region)
    load_faces = shell.select_exposed_faces_by_region(shell.load_region)
    assert max(shell._face_region_distance(face, shell.fixed_region) for face in fixed_faces) <= band + 1.0e-12
    assert max(shell._face_region_distance(face, shell.load_region) for face in load_faces) <= band + 1.0e-12
    assert shell.bc_report["fixed_patch_distance_max"] <= band + 1.0e-12
    assert shell.bc_report["loaded_patch_distance_max"] <= band + 1.0e-12


def test_zero_based_faces_are_not_shifted_when_vertex_zero_is_unused():
    points = np.array(
        [
            [-10.0, -10.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    shell = _shell(
        _tensors(points, np.array([[1, 2, 3]])),
        voxel_size=0.5,
        thickness=0.5,
        faces_index_base=0,
    )
    assert np.array_equal(shell.faces_ijk, np.array([[1, 2, 3]]))


def test_material_bins_are_exact_when_unique_values_fit():
    stiffness = np.array([0.25, 1.0, 0.25, 0.5, 1.0], dtype=np.float64)
    bin_ids, representatives, report = _material_bins_exact_or_logarithmic(stiffness, requested_bins=3)
    reconstructed = representatives[bin_ids]
    assert report["used_exact_stiffness_bins"] is True
    assert report["max_abs_stiffness_quantization_error"] == 0.0
    assert report["max_rel_stiffness_quantization_error"] == 0.0
    assert np.array_equal(reconstructed, stiffness)


def test_surface_patch_tensile_compression_uses_opposite_end_patches():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        use_default_regions=False,
        load_case="tensile_compression",
        BC_dir="x",
        Load_magnitude=2.0,
        voxel_size=0.5,
        thickness=0.5,
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), [2.0, 0.0, 0.0])
    assert shell.bc_report["load_case"] == "tensile_compression"
    assert shell.bc_report["fixed_cad_extent_reference"] == 0.0
    assert shell.bc_report["loaded_cad_extent_reference"] == 2.0
    assert shell.bc_report["overlap_node_count"] == 0


def test_surface_patch_fixed_side_loading_respects_force_side_and_direction_sign():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        use_default_regions=False,
        load_case="fixed_side_loading",
        BC_dir="x",
        fixed_side="max",
        force_side="min",
        load_dir="z",
        load_direction_side="min",
        Load_magnitude=7.0,
        voxel_size=0.5,
        thickness=0.5,
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), [0.0, 0.0, -7.0])
    assert shell.bc_report["fixed_cad_extent_reference"] == 2.0
    assert shell.bc_report["loaded_cad_extent_reference"] == 0.0


def test_surface_patch_three_point_bending_uses_two_supports_and_middle_load():
    points = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        use_default_regions=False,
        load_case="three_point_bending",
        BC_dir="x",
        load_dir="z",
        Load_magnitude=-3.0,
        voxel_size=0.5,
        thickness=0.5,
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    fixed_nodes = np.unique(shell.boundaryCondition["fixed"] // 3)
    loaded_nodes = np.unique(np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0))
    fixed_x = shell.node_coords.reshape(-1, 3)[fixed_nodes, 0]
    loaded_x = shell.node_coords.reshape(-1, 3)[loaded_nodes, 0]
    assert shell.bc_report["load_case"] == "three_point_bending"
    assert shell.bc_report["support_region_count"] == 2
    assert np.any(fixed_x < 0.5)
    assert np.any(fixed_x > 3.5)
    assert np.all(np.abs(loaded_x - 2.0) <= 0.75)
    assert np.allclose(force.sum(axis=0), [0.0, 0.0, -3.0])
    assert np.intersect1d(fixed_nodes, loaded_nodes).size == 0
    assert set(shell.boundaryCondition["fixed"] % 3) == {0, 1, 2}


def test_surface_patch_torsion_recovers_zero_resultant_and_requested_moment():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    shell = _shell(
        _tensors(points, np.array([[0, 1, 2], [0, 2, 3]])),
        use_default_regions=False,
        load_case="torsion",
        BC_dir="x",
        fixed_side="min",
        force_side="max",
        Load_magnitude=5.0,
        voxel_size=0.5,
        thickness=0.5,
    )
    assert np.allclose(shell.bc_report["recovered_resultant_force"], [0.0, 0.0, 0.0], atol=1.0e-10)
    assert np.isclose(shell.bc_report["recovered_torque"], 5.0)
    assert shell.bc_report["overlap_node_count"] == 0


def test_packed_boundary_curve_xyz_reconstructs_boundary_loops():
    shell = _shell(
        _annulus_with_boundary_tensors(),
        use_default_regions=False,
        load_case="tensile_compression",
        voxel_size=0.4,
        thickness=0.4,
    )
    assert 1 in shell.boundary_loops
    hole = shell.boundary_loops[1]
    assert hole["loop_kind"] == "hole"
    assert hole["closed"] is True
    assert hole["coordinates"].shape[1] == 3


def _pin_bearing_bracket_shell(contact_side, *, voxel_size=0.25, thickness=0.25, bearing_band=0.35):
    tensors, hole_center = _c_bracket_with_boundary_tensors()
    shell = _shell(
        tensors,
        use_default_regions=False,
        load_case="pin_bearing",
        BC_dir="x",
        fixed_side="max",
        bearing_contact_side=contact_side,
        load_direction_side=contact_side,
        Load_magnitude=11.0,
        voxel_size=voxel_size,
        thickness=thickness,
        bearing_band=bearing_band,
        fixed_patch_band=0.25,
    )
    return shell, hole_center


def test_pin_bearing_bracket_min_contact_loads_only_min_half_with_negative_resultant():
    shell, hole_center = _pin_bearing_bracket_shell("min")
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), [-11.0, 0.0, 0.0], atol=1.0e-10)
    loaded_nodes = np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0)
    fixed_nodes = np.unique(shell.boundaryCondition["fixed"] // 3)
    coords = shell.node_coords.reshape(-1, 3)
    assert loaded_nodes.size > 0
    assert np.all(force[loaded_nodes, 0] < 0.0)
    assert np.allclose(force[loaded_nodes, 1:], 0.0)
    assert np.all(coords[loaded_nodes, 0] <= hole_center[0] + 1.0e-12)
    peak_node = loaded_nodes[np.argmax(np.abs(force[loaded_nodes, 0]))]
    assert coords[peak_node, 0] <= coords[loaded_nodes, 0].min() + shell.voxel_size
    assert np.intersect1d(fixed_nodes, loaded_nodes).size == 0
    assert shell.bc_report["load_case"] == "pin_bearing_bracket"
    assert shell.bc_report["bearing_loop_id"] == 1
    assert set(shell.boundaryCondition["fixed"] % 3) == {0, 1, 2}
    fixed_x = coords[fixed_nodes, 0]
    fixed_y = coords[fixed_nodes, 1]
    assert np.all(fixed_x >= fixed_x.max() - shell.fixed_patch_band - shell.voxel_size)
    assert np.any(fixed_y > 0.55)
    assert np.any(fixed_y < -0.55)
    assert not np.any(np.abs(fixed_y) < 0.45)
    assert shell.bc_report["candidate_face_count"] >= shell.bc_report["contacting_face_count"] > 0
    assert shell.bc_report["loaded_face_count"] == shell.bc_report["contacting_face_count"]


def test_pin_bearing_bracket_max_contact_loads_only_max_half_with_positive_resultant():
    shell, hole_center = _pin_bearing_bracket_shell("max")
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), [11.0, 0.0, 0.0], atol=1.0e-10)
    loaded_nodes = np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0)
    fixed_nodes = np.unique(shell.boundaryCondition["fixed"] // 3)
    coords = shell.node_coords.reshape(-1, 3)
    assert loaded_nodes.size > 0
    assert np.all(force[loaded_nodes, 0] > 0.0)
    assert np.allclose(force[loaded_nodes, 1:], 0.0)
    assert np.all(coords[loaded_nodes, 0] >= hole_center[0] - 1.0e-12)
    peak_node = loaded_nodes[np.argmax(np.abs(force[loaded_nodes, 0]))]
    assert coords[peak_node, 0] >= coords[loaded_nodes, 0].max() - shell.voxel_size
    assert np.intersect1d(fixed_nodes, loaded_nodes).size == 0


def test_pin_bearing_bracket_through_thickness_load_has_zero_moment():
    shell, hole_center = _pin_bearing_bracket_shell(
        "min",
        voxel_size=0.2,
        thickness=0.8,
        bearing_band=0.35,
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    coords = shell.node_coords.reshape(-1, 3)
    loaded_nodes = np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0)
    loaded_coords = coords[loaded_nodes]
    loaded_force = force[loaded_nodes]
    q = loaded_force[:, 0] / loaded_force[:, 0].sum()

    assert np.all(q >= -1.0e-12)
    assert np.allclose(loaded_force.sum(axis=0), [-11.0, 0.0, 0.0], atol=1.0e-10)
    assert loaded_coords[:, 2].min() < hole_center[2] - 0.1
    assert loaded_coords[:, 2].max() > hole_center[2] + 0.1

    upper_force = loaded_force[loaded_coords[:, 2] > hole_center[2] + 1.0e-12, 0].sum()
    lower_force = loaded_force[loaded_coords[:, 2] < hole_center[2] - 1.0e-12, 0].sum()
    assert np.isclose(upper_force, lower_force, rtol=1.0e-10, atol=1.0e-10)

    force_centroid = np.sum(q[:, None] * loaded_coords, axis=0)
    assert np.isclose(force_centroid[1], hole_center[1], atol=1.0e-10)
    assert np.isclose(force_centroid[2], hole_center[2], atol=1.0e-10)

    moment = np.cross(loaded_coords - shell.bc_report["bearing_centre"], loaded_force).sum(axis=0)
    assert np.allclose(moment, [0.0, 0.0, 0.0], atol=1.0e-8)
    assert np.allclose(shell.bc_report["bearing_resultant_moment_about_centre"], [0.0, 0.0, 0.0], atol=1.0e-8)
    assert shell.bc_report["loaded_thickness_layer_count"] >= 3
    assert abs(shell.bc_report["loaded_normal_offset_weighted_mean"]) < 1.0e-10

    x = loaded_coords[:, 0]
    peak_band = x <= np.quantile(x, 0.25)
    edge_band = x >= np.quantile(x, 0.75)
    assert q[peak_band].mean() > q[edge_band].mean()


def test_pin_bearing_bracket_rejects_conflicting_load_direction_side():
    try:
        _shell(
            _c_bracket_with_boundary_tensors()[0],
            use_default_regions=False,
            load_case="pin_bearing",
            BC_dir="x",
            fixed_side="max",
            bearing_contact_side="max",
            load_direction_side="min",
            voxel_size=0.25,
            thickness=0.25,
        )
    except ValueError as exc:
        assert "load_direction_side" in str(exc)
    else:
        raise AssertionError("Conflicting bearing_contact_side/load_direction_side should fail")


def test_pin_bearing_bracket_rejects_mismatched_load_region_loop_id():
    try:
        _shell(
            _c_bracket_with_boundary_tensors()[0],
            use_default_regions=False,
            load_case="pin_bearing",
            BC_dir="x",
            fixed_side="max",
            bearing_loop_id=1,
            load_region={"boundary_loop_id": 0, "band": 0.35},
            voxel_size=0.25,
            thickness=0.25,
        )
    except ValueError as exc:
        assert "boundary_loop_id" in str(exc)
    else:
        raise AssertionError("Mismatched load_region boundary_loop_id should fail")


def test_pin_bearing_bracket_rejects_legacy_bbox_mapping():
    try:
        _shell(
            _annulus_with_boundary_tensors(),
            use_default_regions=False,
            load_case="bearing_bracket",
            bc_mapping_mode="legacy_bbox",
            voxel_size=0.4,
            thickness=0.4,
        )
    except ValueError as exc:
        assert "surface_patch" in str(exc)
    else:
        raise AssertionError("pin_bearing_bracket should reject legacy_bbox mapping")


def test_open_boundary_polyline_does_not_add_last_to_first_segment():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(_tensors(points, np.array([[0, 1, 2], [0, 2, 3]])))
    probe = np.array([[0.0, 0.5, 0.0]])
    open_distance = shell._distance_to_boundary_polyline(probe, points, closed=False)[0]
    closed_distance = shell._distance_to_boundary_polyline(probe, points, closed=True)[0]
    assert open_distance > 0.49
    assert closed_distance < 1.0e-12


def test_boundary_loop_includes_closing_segment_by_default():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(
            points,
            np.array([[0, 1, 2], [0, 2, 3]]),
            boundary_loops={"outer": {"vertex_ids": [0, 1, 2, 3]}},
        )
    )
    coords, ids, closed = shell._boundary_polyline({"boundary_loop_id": "outer"})
    assert closed is True
    assert np.array_equal(ids, np.array([0, 1, 2, 3]))
    assert shell._distance_to_boundary_polyline(np.array([[0.0, 0.5, 0.0]]), coords, closed=closed)[0] < 1.0e-12


def test_boundary_edge_metadata_schema_defaults_to_open_and_validates_ids():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(
            points,
            np.array([[0, 1, 2], [0, 2, 3]]),
            boundary_edges={"edge_a": {"point_indices": [0, 1, 2, 3], "cad_edge_id": 17}},
        )
    )
    coords, ids, closed = shell._boundary_polyline({"boundary_edge_id": "edge_a"})
    assert closed is False
    assert np.array_equal(ids, np.array([0, 1, 2, 3]))
    assert shell._distance_to_boundary_polyline(np.array([[0.0, 0.5, 0.0]]), coords, closed=closed)[0] > 0.49


def test_cad_extent_reference_is_stable_when_grid_padding_changes():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    tensors = _tensors(points, np.array([[0, 1, 2], [0, 2, 3]]))
    shell_a = _shell(tensors, use_default_regions=False, extra_layers=1, voxel_size=0.5, thickness=0.5)
    shell_b = _shell(tensors, use_default_regions=False, extra_layers=3, voxel_size=0.5, thickness=0.5)
    region = {"extent_axis": "x", "extent_side": "max", "band": 0.375}
    assert shell_a._region_extent_reference(region) == 2.0
    assert shell_b._region_extent_reference(region) == 2.0


def test_coordinate_boundary_polyline_is_validated_against_band():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(
            points,
            np.array([[0, 1, 2], [0, 2, 3]]),
            boundary_edges={"coord_edge": {"coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}},
        )
    )
    region = {"boundary_edge_id": "coord_edge", "band": 0.1}
    faces = [{"closest_point": np.array([0.5, 0.5, 0.0])}]
    try:
        shell._assert_faces_inside_region_band(faces, region, "loaded")
    except ValueError:
        pass
    else:
        raise AssertionError("Coordinate boundary polyline escaped band validation.")


def test_invalid_source_face_id_extent_raises_instead_of_falling_back():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(_tensors(points, np.array([[0, 1, 2], [0, 2, 3]]), face_id=np.zeros(4, dtype=np.int64)))
    try:
        shell._region_extent_reference({"extent_axis": "x", "extent_side": "min", "source_face_id": 99})
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid source_face_id did not raise.")


def test_edge_load_report_uses_only_actual_nonzero_loaded_nodes():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    shell = _shell(
        _tensors(
            points,
            np.array([[0, 1, 2], [0, 2, 3]]),
            boundary_edges={"right": {"coordinates": [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]}},
        ),
        use_default_regions=False,
        load_case="tensile_compression",
        BC_dir="x",
        Load_magnitude=4.0,
        voxel_size=0.5,
        thickness=0.5,
        fixed_region={"extent_axis": "x", "extent_side": "min", "band": 0.375},
        load_region={"boundary_edge_id": "right", "band": 0.26, "direction": "y", "total_force": 4.0},
    )
    force = shell.boundaryCondition["force"].reshape(-1, 3)
    actual_loaded_nodes = np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0)
    assert shell.bc_report["loaded_node_count"] == actual_loaded_nodes.size
    assert np.allclose(force.sum(axis=0), [0.0, 4.0, 0.0])
    assert shell.bc_report["loaded_bounds"] == shell._bounds_for_nodes(
        shell.node_coords.reshape(-1, 3),
        actual_loaded_nodes,
    )
