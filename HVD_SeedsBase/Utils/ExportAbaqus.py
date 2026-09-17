from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def export_abaqus_shell(
    fields,
    shell_problem,
    output_path,
    *,
    density_threshold=0.5,
    shell_thickness=None,
    material_name="HVD_LAMINA",
    material=None,  # (E1, E2, nu12, G12, G13, G23)
    bc_surface_radius=None,
):
    """
    Export an optimized HVD field as an Abaqus S3 shell model.

    Fixed and loaded shell nodes are transferred from the actual voxel
    boundary-condition node sets in shell_problem.boundaryCondition.

    Parameters
    ----------
    material:
        (E1, E2, nu12, G12, G13, G23)

    bc_surface_radius:
        Surface radius around projected voxel BC anchors. If None,
        0.75 times the largest voxel-grid spacing is used.
    """

    def as_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def write_id_list(
        stream,
        values,
        values_per_line=16,
    ):
        values = np.asarray(
            values,
            dtype=np.int64,
        ).reshape(-1)

        for start in range(
            0,
            values.size,
            values_per_line,
        ):
            current = values[
                start : start + values_per_line
            ]

            stream.write(
                ", ".join(
                    str(int(value))
                    for value in current
                )
                + "\n"
            )

    if shell_problem is None:
        raise ValueError(
            "shell_problem must be a configured "
            "ThickenShell instance."
        )

    if material is None or len(material) != 6:
        raise ValueError(
            "material must be:\n"
            "(E1, E2, nu12, G12, G13, G23)"
        )

    # ==============================================================
    # Geometry and optimized fields
    # ==============================================================

    face_tensor = fields["face_tensor"]

    xyz = as_numpy(
        face_tensor["points_xyz"]
    ).astype(np.float64)

    faces = as_numpy(
        face_tensor["faces_ijk"]
    ).astype(np.int64)

    rho = as_numpy(
        fields["rho"]
    ).reshape(-1).astype(np.float64)

    fiber = as_numpy(
        fields["fiber3d"]
    ).astype(np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape [N,3], "
            f"got {xyz.shape}."
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces_ijk must have shape [M,3], "
            f"got {faces.shape}."
        )

    if rho.shape[0] != xyz.shape[0]:
        raise ValueError(
            "rho must contain one value per surface node."
        )

    if fiber.shape != xyz.shape:
        raise ValueError(
            f"fiber3d must have shape {xyz.shape}, "
            f"got {fiber.shape}."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "points_xyz contains NaN or infinity."
        )

    if not np.all(np.isfinite(rho)):
        raise ValueError(
            "rho contains NaN or infinity."
        )

    if not np.all(np.isfinite(fiber)):
        raise ValueError(
            "fiber3d contains NaN or infinity."
        )

    # Convert one-based connectivity to zero-based.
    if faces.min() == 1:
        faces = faces - 1

    if (
        faces.min() < 0
        or faces.max() >= xyz.shape[0]
    ):
        raise ValueError(
            "faces_ijk contains invalid node indices."
        )

    # ==============================================================
    # Density thresholding
    # ==============================================================

    element_density = rho[faces].mean(axis=1)

    keep_mask = (
        element_density
        >= float(density_threshold)
    )

    kept_faces = faces[keep_mask]

    if kept_faces.shape[0] == 0:
        raise ValueError(
            "No elements survived density_threshold."
        )

    used_nodes = np.unique(
        kept_faces.reshape(-1)
    )

    old_to_new = -np.ones(
        xyz.shape[0],
        dtype=np.int64,
    )

    old_to_new[used_nodes] = np.arange(
        1,
        used_nodes.size + 1,
        dtype=np.int64,
    )

    shell_xyz = xyz[used_nodes]
    shell_faces = old_to_new[kept_faces]

    # ==============================================================
    # Element fibre orientations
    # ==============================================================

    p0 = xyz[kept_faces[:, 0]]
    p1 = xyz[kept_faces[:, 1]]
    p2 = xyz[kept_faces[:, 2]]

    normals = np.cross(
        p1 - p0,
        p2 - p0,
    )

    normal_norm = np.linalg.norm(
        normals,
        axis=1,
    )

    if np.any(normal_norm < 1.0e-12):
        raise ValueError(
            "Degenerate triangles were found "
            "in the retained mesh."
        )

    normals /= normal_norm[:, None]

    # Align nodal fibre signs because f and -f represent
    # the same orthotropic material direction.
    element_nodal_fibers = fiber[
        kept_faces
    ].copy()

    reference_fiber = (
        element_nodal_fibers[:, 0, :]
    )

    alignment = np.sum(
        element_nodal_fibers
        * reference_fiber[:, None, :],
        axis=2,
    )

    element_nodal_fibers[
        alignment < 0.0
    ] *= -1.0

    element_fiber = (
        element_nodal_fibers.mean(axis=1)
    )

    # Project the fibre onto the element tangent plane.
    orientation_1 = (
        element_fiber
        - np.sum(
            element_fiber * normals,
            axis=1,
            keepdims=True,
        )
        * normals
    )

    orientation_1_norm = np.linalg.norm(
        orientation_1,
        axis=1,
    )

    fallback_mask = (
        orientation_1_norm < 1.0e-12
    )

    if np.any(fallback_mask):
        fallback = p1 - p0

        fallback -= (
            np.sum(
                fallback * normals,
                axis=1,
                keepdims=True,
            )
            * normals
        )

        orientation_1[
            fallback_mask
        ] = fallback[fallback_mask]

        orientation_1_norm = np.linalg.norm(
            orientation_1,
            axis=1,
        )

    if np.any(
        orientation_1_norm < 1.0e-12
    ):
        raise ValueError(
            "Unable to construct valid element orientations."
        )

    orientation_1 /= (
        orientation_1_norm[:, None]
    )

    orientation_2 = np.cross(
        normals,
        orientation_1,
    )

    orientation_2_norm = np.linalg.norm(
        orientation_2,
        axis=1,
    )

    if np.any(
        orientation_2_norm < 1.0e-12
    ):
        raise ValueError(
            "Unable to construct the second "
            "material direction."
        )

    orientation_2 /= (
        orientation_2_norm[:, None]
    )

    tangent_error = float(
        np.max(
            np.abs(
                np.sum(
                    orientation_1 * normals,
                    axis=1,
                )
            )
        )
    )

    orthogonality_error = float(
        np.max(
            np.abs(
                np.sum(
                    orientation_1
                    * orientation_2,
                    axis=1,
                )
            )
        )
    )

    # ==============================================================
    # Actual voxel boundary conditions
    # ==============================================================

    boundary_condition = getattr(
        shell_problem,
        "boundaryCondition",
        None,
    )

    if boundary_condition is None:
        raise ValueError(
            "shell_problem.boundaryCondition is missing."
        )

    dof_per_node = int(
        boundary_condition.get(
            "numDOFPerNode",
            3,
        )
    )

    if dof_per_node != 3:
        raise ValueError(
            "Expected three translational "
            "voxel DOFs per node."
        )

    voxel_xyz = as_numpy(
        shell_problem.node_coords
    ).reshape(-1, 3).astype(np.float64)

    # Fixed voxel nodes.
    fixed_dofs = as_numpy(
        boundary_condition["fixed"]
    ).reshape(-1).astype(np.int64)

    if fixed_dofs.size == 0:
        raise ValueError(
            "The voxel fixed-DOF set is empty."
        )

    if (
        fixed_dofs.min() < 0
        or fixed_dofs.max()
        >= 3 * voxel_xyz.shape[0]
    ):
        raise ValueError(
            "The voxel fixed-DOF set "
            "contains invalid indices."
        )

    fixed_voxel_nodes = np.unique(
        fixed_dofs // 3
    )

    # Loaded voxel nodes and force.
    force_vector = as_numpy(
        boundary_condition["force"]
    ).reshape(-1).astype(np.float64)

    if force_vector.size != (
        3 * voxel_xyz.shape[0]
    ):
        raise ValueError(
            "The voxel force-vector size does "
            "not match node_coords."
        )

    nodal_force = force_vector.reshape(
        -1,
        3,
    )

    maximum_force_component = float(
        np.max(np.abs(nodal_force))
    )

    force_tolerance = max(
        1.0e-15,
        maximum_force_component * 1.0e-12,
    )

    loaded_voxel_nodes = np.flatnonzero(
        np.linalg.norm(
            nodal_force,
            axis=1,
        )
        > force_tolerance
    )

    if loaded_voxel_nodes.size == 0:
        raise ValueError(
            "No loaded voxel nodes were found."
        )

    total_force_vector = (
        nodal_force.sum(axis=0)
    )

    component_tolerance = max(
        1.0e-15,
        float(
            np.max(
                np.abs(total_force_vector)
            )
        )
        * 1.0e-12,
    )

    active_force_components = (
        np.flatnonzero(
            np.abs(total_force_vector)
            > component_tolerance
        )
    )

    if active_force_components.size != 1:
        raise ValueError(
            "The exporter supports one nonzero "
            "resultant force component. Detected "
            f"total force: {total_force_vector}."
        )

    load_direction = int(
        active_force_components[0]
    )

    total_load = float(
        total_force_vector[load_direction]
    )

    # ==============================================================
    # Confirm matching CAD surface samples
    # ==============================================================

    problem_surface_xyz = as_numpy(
        shell_problem.points_xyz
    ).astype(np.float64)

    if problem_surface_xyz.shape != xyz.shape:
        raise ValueError(
            "shell_problem.points_xyz and fields "
            "points_xyz have different shapes."
        )

    maximum_surface_difference = float(
        np.max(
            np.linalg.norm(
                problem_surface_xyz - xyz,
                axis=1,
            )
        )
    )

    geometry_scale = max(
        float(
            np.linalg.norm(
                np.ptp(xyz, axis=0)
            )
        ),
        1.0,
    )

    geometry_tolerance = max(
        1.0e-10 * geometry_scale,
        1.0e-10,
    )

    if (
        maximum_surface_difference
        > geometry_tolerance
    ):
        raise ValueError(
            "shell_problem and fields do not use "
            "the same ordered surface samples. "
            f"Maximum difference: "
            f"{maximum_surface_difference:.6e}; "
            f"tolerance: {geometry_tolerance:.6e}."
        )

    # ==============================================================
    # Project voxel BC nodes onto the CAD surface
    # ==============================================================

    nelx = int(shell_problem.mesh["nelx"])
    nely = int(shell_problem.mesh["nely"])
    nelz = int(shell_problem.mesh["nelz"])

    expected_voxel_node_count = (
        (nelz + 1)
        * (nelx + 1)
        * (nely + 1)
    )

    if (
        voxel_xyz.shape[0]
        != expected_voxel_node_count
    ):
        raise ValueError(
            "node_coords size is inconsistent "
            "with shell_problem.mesh."
        )

    voxel_grid_xyz = voxel_xyz.reshape(
        nelz + 1,
        nelx + 1,
        nely + 1,
        3,
    )

    hx = float(
        voxel_grid_xyz[0, 1, 0, 0]
        - voxel_grid_xyz[0, 0, 0, 0]
    )

    hy = float(
        voxel_grid_xyz[0, 0, 1, 1]
        - voxel_grid_xyz[0, 0, 0, 1]
    )

    hz = float(
        voxel_grid_xyz[1, 0, 0, 2]
        - voxel_grid_xyz[0, 0, 0, 2]
    )

    if min(hx, hy, hz) <= 0.0:
        raise ValueError(
            "The structured voxel grid "
            "has invalid spacing."
        )

    if bc_surface_radius is None:
        bc_surface_radius = (
            0.75 * max(hx, hy, hz)
        )
    else:
        bc_surface_radius = float(
            bc_surface_radius
        )

    if bc_surface_radius <= 0.0:
        raise ValueError(
            "bc_surface_radius must be positive."
        )

    full_surface_tree = cKDTree(xyz)

    def map_voxel_bc_to_surface(
        target_voxel_nodes,
    ):
        target_voxel_nodes = np.asarray(
            target_voxel_nodes,
            dtype=np.int64,
        ).reshape(-1)

        if target_voxel_nodes.size == 0:
            raise ValueError(
                "A voxel boundary-node set is empty."
            )

        if (
            target_voxel_nodes.min() < 0
            or target_voxel_nodes.max()
            >= expected_voxel_node_count
        ):
            raise ValueError(
                "A voxel boundary-node index "
                "is outside the grid."
            )

        # Project actual voxel BC nodes onto the CAD surface.
        projection_distance, anchor_ids = (
            full_surface_tree.query(
                voxel_xyz[target_voxel_nodes],
                k=1,
            )
        )

        anchor_ids = np.unique(
            np.asarray(
                anchor_ids,
                dtype=np.int64,
            )
        )

        if anchor_ids.size == 0:
            raise ValueError(
                "No surface anchors were found "
                "for a voxel BC set."
            )

        # Expand around anchors on the dense CAD mesh.
        anchor_tree = cKDTree(
            xyz[anchor_ids]
        )

        surface_distance, _ = (
            anchor_tree.query(
                xyz,
                k=1,
            )
        )

        surface_mask = (
            surface_distance
            <= bc_surface_radius
        )

        # Always include the projected anchors.
        surface_mask[anchor_ids] = True

        return (
            surface_mask,
            surface_distance,
            anchor_ids,
            projection_distance,
        )

    (
        fixed_surface_mask,
        fixed_surface_distance,
        fixed_anchor_ids,
        fixed_projection_distance,
    ) = map_voxel_bc_to_surface(
        fixed_voxel_nodes
    )

    (
        loaded_surface_mask,
        loaded_surface_distance,
        loaded_anchor_ids,
        loaded_projection_distance,
    ) = map_voxel_bc_to_surface(
        loaded_voxel_nodes
    )

    # Resolve any unlikely overlap by assigning each surface
    # point to the closest projected BC cloud.
    surface_overlap = (
        fixed_surface_mask
        & loaded_surface_mask
    )

    if np.any(surface_overlap):
        fixed_is_closer = (
            fixed_surface_distance
            <= loaded_surface_distance
        )

        fixed_surface_mask[
            surface_overlap
            & ~fixed_is_closer
        ] = False

        loaded_surface_mask[
            surface_overlap
            & fixed_is_closer
        ] = False

    # Transfer to retained thresholded shell nodes.
    fixed_nodes = (
        np.flatnonzero(
            fixed_surface_mask[used_nodes]
        ).astype(np.int64)
        + 1
    )

    loaded_nodes = (
        np.flatnonzero(
            loaded_surface_mask[used_nodes]
        ).astype(np.int64)
        + 1
    )

    if fixed_nodes.size == 0:
        raise ValueError(
            "No retained shell nodes map to the "
            "projected voxel fixed region. Increase "
            "bc_surface_radius."
        )

    if loaded_nodes.size == 0:
        raise ValueError(
            "No retained shell nodes map to the "
            "projected voxel loaded region. Increase "
            "bc_surface_radius."
        )

    overlap = np.intersect1d(
        fixed_nodes,
        loaded_nodes,
    )

    if overlap.size > 0:
        raise ValueError(
            f"{overlap.size} retained shell nodes "
            "map to both fixed and loaded regions."
        )

    # ==============================================================
    # Material, thickness and load
    # ==============================================================

    if shell_thickness is None:
        shell_thickness = float(
            shell_problem.thickness
        )
    else:
        shell_thickness = float(
            shell_thickness
        )

    if shell_thickness <= 0.0:
        raise ValueError(
            "shell_thickness must be positive."
        )

    load_per_node = (
        total_load
        / float(loaded_nodes.size)
    )

    E1, E2, nu12, G12, G13, G23 = [
        float(value)
        for value in material
    ]

    # ==============================================================
    # Write Abaqus input file
    # ==============================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as inp:
        inp.write("*HEADING\n")
        inp.write(
            "Optimized HVD shell with transferred "
            "ThickenShell boundary conditions\n"
        )
        inp.write("**\n")

        # Model-level keyword; must remain outside *PART.
        inp.write(
            "*DISTRIBUTION TABLE, "
            "NAME=FIBER_TABLE\n"
        )
        inp.write(
            "COORD3D, COORD3D\n"
        )
        inp.write("**\n")

        # ----------------------------------------------------------
        # Part
        # ----------------------------------------------------------

        inp.write(
            "*PART, NAME=HVD_SHELL\n"
        )

        inp.write("*NODE\n")

        for node_id, point in enumerate(
            shell_xyz,
            start=1,
        ):
            inp.write(
                f"{node_id}, "
                f"{point[0]:.12g}, "
                f"{point[1]:.12g}, "
                f"{point[2]:.12g}\n"
            )

        inp.write(
            "*ELEMENT, TYPE=S3, "
            "ELSET=NETWORK\n"
        )

        for element_id, connectivity in enumerate(
            shell_faces,
            start=1,
        ):
            inp.write(
                f"{element_id}, "
                f"{int(connectivity[0])}, "
                f"{int(connectivity[1])}, "
                f"{int(connectivity[2])}\n"
            )

        inp.write(
            "*NSET, NSET=FIXED\n"
        )
        write_id_list(
            inp,
            fixed_nodes,
        )

        inp.write(
            "*NSET, NSET=LOADED\n"
        )
        write_id_list(
            inp,
            loaded_nodes,
        )

        inp.write("**\n")

        inp.write(
            "*DISTRIBUTION, NAME=FIBER_FIELD, "
            "LOCATION=ELEMENT, TABLE=FIBER_TABLE\n"
        )

        # Abaqus requires a default distribution value.
        inp.write(
            ", 1.0, 0.0, 0.0, "
            "0.0, 1.0, 0.0\n"
        )

        for element_id, (a1, a2) in enumerate(
            zip(
                orientation_1,
                orientation_2,
            ),
            start=1,
        ):
            inp.write(
                f"{element_id}, "
                f"{a1[0]:.12g}, "
                f"{a1[1]:.12g}, "
                f"{a1[2]:.12g}, "
                f"{a2[0]:.12g}, "
                f"{a2[1]:.12g}, "
                f"{a2[2]:.12g}\n"
            )

        inp.write(
            "*ORIENTATION, "
            "NAME=FIBER_ORIENTATION, "
            "DEFINITION=COORDINATES, "
            "SYSTEM=RECTANGULAR\n"
        )
        inp.write(
            "FIBER_FIELD\n"
        )
        inp.write(
            "3, 0.0\n"
        )

        inp.write(
            "*SHELL SECTION, "
            "ELSET=NETWORK, "
            f"MATERIAL={material_name}, "
            "ORIENTATION=FIBER_ORIENTATION\n"
        )
        inp.write(
            f"{shell_thickness:.12g}, 5\n"
        )

        inp.write("*END PART\n")
        inp.write("**\n")

        # ----------------------------------------------------------
        # Material
        # ----------------------------------------------------------

        inp.write(
            f"*MATERIAL, NAME={material_name}\n"
        )
        inp.write(
            "*ELASTIC, TYPE=LAMINA\n"
        )
        inp.write(
            f"{E1:.12g}, "
            f"{E2:.12g}, "
            f"{nu12:.12g}, "
            f"{G12:.12g}, "
            f"{G13:.12g}, "
            f"{G23:.12g}\n"
        )
        inp.write("**\n")

        # ----------------------------------------------------------
        # Assembly
        # ----------------------------------------------------------

        inp.write(
            "*ASSEMBLY, NAME=ASSEMBLY\n"
        )
        inp.write(
            "*INSTANCE, NAME=HVD_SHELL-1, "
            "PART=HVD_SHELL\n"
        )
        inp.write(
            "*END INSTANCE\n"
        )
        inp.write(
            "*END ASSEMBLY\n"
        )
        inp.write("**\n")

        # ----------------------------------------------------------
        # Static analysis step
        # ----------------------------------------------------------

        inp.write(
            "*STEP, NAME=STATIC_LOAD, "
            "NLGEOM=NO\n"
        )
        inp.write(
            "*STATIC\n"
        )
        inp.write(
            "0.1, 1.0, 1.0e-06, 0.1\n"
        )

        inp.write(
            "*BOUNDARY\n"
        )
        inp.write(
            "HVD_SHELL-1.FIXED, "
            "1, 6, 0.0\n"
        )

        inp.write(
            "*CLOAD\n"
        )
        inp.write(
            f"HVD_SHELL-1.LOADED, "
            f"{load_direction + 1}, "
            f"{load_per_node:.12g}\n"
        )

        inp.write(
            "*OUTPUT, FIELD, FREQUENCY=1\n"
        )
        inp.write(
            "*NODE OUTPUT\n"
        )
        inp.write(
            "U, RF\n"
        )

        inp.write(
            "*ELEMENT OUTPUT, DIRECTIONS=YES\n"
        )
        inp.write(
            "S, E\n"
        )

        inp.write(
            "*OUTPUT, HISTORY, FREQUENCY=1\n"
        )
        inp.write(
            "*ENERGY OUTPUT\n"
        )
        inp.write(
            "ALLIE, ALLSE\n"
        )

        inp.write(
            "*END STEP\n"
        )

    # ==============================================================
    # Verify critical Abaqus syntax
    # ==============================================================

    expected_distribution = (
        "*DISTRIBUTION, NAME=FIBER_FIELD, "
        "LOCATION=ELEMENT, TABLE=FIBER_TABLE"
    )

    distribution_keyword = None

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as inp:
        for line in inp:
            if line.upper().startswith(
                "*DISTRIBUTION,"
            ):
                distribution_keyword = (
                    line.strip()
                )
                break

    if (
        distribution_keyword
        != expected_distribution
    ):
        raise RuntimeError(
            "Incorrect distribution keyword generated: "
            f"{distribution_keyword!r}"
        )

    # ==============================================================
    # Diagnostics
    # ==============================================================

    direction_names = (
        "X",
        "Y",
        "Z",
    )

    print(f"Written: {output_path}")
    print(
        f"Shell nodes: "
        f"{shell_xyz.shape[0]}"
    )
    print(
        f"Shell elements: "
        f"{shell_faces.shape[0]}"
    )
    print(
        f"Voxel fixed nodes: "
        f"{fixed_voxel_nodes.size}"
    )
    print(
        f"Voxel loaded nodes: "
        f"{loaded_voxel_nodes.size}"
    )
    print(
        f"BC surface radius: "
        f"{bc_surface_radius:.12g}"
    )
    print(
        f"Fixed surface anchors: "
        f"{fixed_anchor_ids.size}"
    )
    print(
        f"Loaded surface anchors: "
        f"{loaded_anchor_ids.size}"
    )
    print(
        "Mapped surface fixed nodes "
        f"before threshold: "
        f"{int(fixed_surface_mask.sum())}"
    )
    print(
        "Mapped surface loaded nodes "
        f"before threshold: "
        f"{int(loaded_surface_mask.sum())}"
    )
    print(
        f"Retained Abaqus fixed nodes: "
        f"{fixed_nodes.size}"
    )
    print(
        f"Retained Abaqus loaded nodes: "
        f"{loaded_nodes.size}"
    )
    print(
        f"Maximum fixed projection distance: "
        f"{float(fixed_projection_distance.max()):.6g}"
    )
    print(
        f"Maximum loaded projection distance: "
        f"{float(loaded_projection_distance.max()):.6g}"
    )
    print(
        f"Load direction: "
        f"{direction_names[load_direction]}"
    )
    print(
        f"Signed total load: "
        f"{total_load:.12g}"
    )
    print(
        f"Load per Abaqus node: "
        f"{load_per_node:.12g}"
    )
    print(
        f"Distribution: "
        f"{distribution_keyword}"
    )
    print(
        f"Maximum tangency error: "
        f"{tangent_error:.3e}"
    )
    print(
        "Maximum orientation orthogonality error: "
        f"{orthogonality_error:.3e}"
    )

    return {
        "output_path": output_path,
        "fixed_nodes": fixed_nodes,
        "loaded_nodes": loaded_nodes,
        "fixed_surface_mask": fixed_surface_mask,
        "loaded_surface_mask": loaded_surface_mask,
        "fixed_anchor_ids": fixed_anchor_ids,
        "loaded_anchor_ids": loaded_anchor_ids,
        "bc_surface_radius": bc_surface_radius,
        "total_force_vector": total_force_vector,
        "load_per_node": load_per_node,
    }


# from Utils.ExportAbaqus import export_abaqus_shell

# abaqus_export =  export_abaqus_shell(
#     fields,
#     shell_problem,
#     "FreeForm3_HVD_shell_final.inp",
#     density_threshold=0.4,
#     material=(
#         70000.0,
#         7000.0,
#         0.30,
#         4500.0,
#         4500.0,
#         2600.0,
#     ),
# )