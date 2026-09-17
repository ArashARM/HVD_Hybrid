from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from Utils.VoxelMassProperties import calculate_voxel_mass_properties


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _write_id_list(stream: Any, values: np.ndarray, values_per_line: int = 16) -> None:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    for start in range(0, values.size, values_per_line):
        current = values[start : start + values_per_line]
        stream.write(", ".join(str(int(value)) for value in current) + "\n")


def _material_from_problem(
    shell_problem: Any,
    material: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, float]:
    """Return the nine engineering constants used by the internal 3D FEM."""
    source: Mapping[str, Any]
    if material is None:
        problem_material = getattr(shell_problem, "materialProperty", None)
        if not isinstance(problem_material, Mapping):
            raise ValueError(
                "material must be supplied when shell_problem.materialProperty "
                "is not a mapping."
            )
        source = problem_material
    elif isinstance(material, Mapping):
        source = material
    else:
        values = tuple(float(value) for value in material)
        if len(values) != 9:
            raise ValueError(
                "A material sequence must contain "
                "(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)."
            )
        return dict(
            zip(
                ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23"),
                values,
            )
        )

    def get_value(name: str, *aliases: str, default: float | None = None) -> float:
        for key in (name, *aliases):
            if key in source:
                return float(source[key])
        if default is not None:
            return float(default)
        raise KeyError(f"Material property {name!r} is missing.")

    E1 = get_value("material_E1", "material_E_longitudinal", "Ef", "E1")
    E2 = get_value("material_E2", "material_E_transverse", "Et", "E2")
    E3 = get_value("material_E3", "E3", default=E2)
    nu12 = get_value("material_nu12", "material_nu_longitudinal", "nuf", "nu12")
    nu13 = get_value("material_nu13", "nu13", default=nu12)
    nu23 = get_value("material_nu23", "material_nu_transverse", "nut", "nu23")
    G12 = get_value("material_G12", "material_shear_modulus", "Gf", "G12")
    G13 = get_value("material_G13", "G13", default=G12)
    G23 = get_value("material_G23", "G23")

    constants = {
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "nu12": nu12,
        "nu13": nu13,
        "nu23": nu23,
        "G12": G12,
        "G13": G13,
        "G23": G23,
    }
    for name, value in constants.items():
        if not math.isfinite(value):
            raise ValueError(f"Material property {name} must be finite, got {value}.")
        if name.startswith(("E", "G")) and value <= 0.0:
            raise ValueError(f"Material property {name} must be positive, got {value}.")
    return constants


def _structured_c3d8_connectivity(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Return zero-based C3D8 connectivity in the code's [z, x, y] cell order."""
    node_ids = np.arange(
        (nelz + 1) * (nelx + 1) * (nely + 1),
        dtype=np.int64,
    ).reshape(nelz + 1, nelx + 1, nely + 1)

    connectivity = np.empty((nelz * nelx * nely, 8), dtype=np.int64)
    for iz in range(nelz):
        for ix in range(nelx):
            for iy in range(nely):
                element_id = iy + ix * nely + iz * nelx * nely
                connectivity[element_id] = (
                    node_ids[iz, ix, iy],
                    node_ids[iz, ix + 1, iy],
                    node_ids[iz, ix + 1, iy + 1],
                    node_ids[iz, ix, iy + 1],
                    node_ids[iz + 1, ix, iy],
                    node_ids[iz + 1, ix + 1, iy],
                    node_ids[iz + 1, ix + 1, iy + 1],
                    node_ids[iz + 1, ix, iy + 1],
                )
    return connectivity


def _orientation_axes(phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the internal azimuth/polar-angle convention without an extra roll."""
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    sin_theta = np.sin(theta)
    axis_1 = np.column_stack(
        (
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        )
    )
    axis_2 = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    axis_1 /= np.clip(np.linalg.norm(axis_1, axis=1, keepdims=True), 1.0e-14, None)
    axis_2 /= np.clip(np.linalg.norm(axis_2, axis=1, keepdims=True), 1.0e-14, None)
    return axis_1, axis_2


def _material_bins_exact_or_logarithmic(
    stiffness: np.ndarray,
    requested_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """Return bin IDs, representatives, and quantization diagnostics."""
    values = np.asarray(stiffness, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Active stiffness factors must be finite and positive.")
    if int(requested_bins) < 1:
        raise ValueError("material_bins must be at least one.")

    unique_values, inverse = np.unique(values, return_inverse=True)
    if unique_values.size <= int(requested_bins):
        diagnostics = {
            "used_exact_stiffness_bins": True,
            "max_abs_stiffness_quantization_error": 0.0,
            "max_rel_stiffness_quantization_error": 0.0,
        }
        return inverse.astype(np.int64), unique_values.astype(np.float64), diagnostics

    log_values = np.log(values)
    log_min = float(log_values.min())
    log_max = float(log_values.max())
    if math.isclose(log_min, log_max, rel_tol=1.0e-12, abs_tol=1.0e-14):
        representative = np.asarray([float(values.mean())])
        error = np.abs(values - representative[0])
        diagnostics = {
            "used_exact_stiffness_bins": False,
            "max_abs_stiffness_quantization_error": float(error.max()),
            "max_rel_stiffness_quantization_error": float((error / np.abs(values)).max()),
        }
        return np.zeros(values.size, dtype=np.int64), representative, diagnostics

    count = min(int(requested_bins), values.size)
    edges = np.linspace(log_min, log_max, count + 1)
    bin_ids = np.searchsorted(edges[1:-1], log_values, side="right").astype(np.int64)

    remap = np.full(count, -1, dtype=np.int64)
    representatives: list[float] = []
    for old_id in range(count):
        mask = bin_ids == old_id
        if not np.any(mask):
            continue
        remap[old_id] = len(representatives)
        representatives.append(float(values[mask].mean()))
    representative_array = np.asarray(representatives, dtype=np.float64)
    remapped = remap[bin_ids]
    assigned = representative_array[remapped]
    abs_error = np.abs(values - assigned)
    rel_error = abs_error / np.clip(np.abs(values), 1.0e-300, None)
    diagnostics = {
        "used_exact_stiffness_bins": False,
        "max_abs_stiffness_quantization_error": float(abs_error.max()),
        "max_rel_stiffness_quantization_error": float(rel_error.max()),
    }
    return remapped, representative_array, diagnostics


def export_abaqus_voxel_fem(
    fields: Mapping[str, Any],
    shell_problem: Any,
    output_path: str | Path,
    *,
    fem_result: Mapping[str, Any] | None = None,
    material: Mapping[str, float] | Sequence[float] | None = None,
    penal: float = 3.0,
    rho_min_ratio: float = 1.0e-6,
    material_bins: int = 128,
    element_type: str = "C3D8",
    material_mass_density_kg_m3: float | None = None,
    unit_system: str | None = None,
) -> dict[str, Any]:
    """Export an Abaqus solid model closely matching the internal voxel FEM.

    Unlike the former S3 surface exporter, this exporter:

    * writes the same occupied structured voxels as 8-node solid elements;
    * uses the code's surface-to-voxel density and fibre transfer;
    * preserves the continuous SIMP stiffness factor through material bins;
    * preserves every active nodal force component without redistribution; and
    * preserves the original translational fixed-DOF set.

    Passing ``fem_result`` from ``trainer.loss_fem.evaluate`` is recommended.
    It supplies the exact density and scheduled SIMP stiffness fields used by
    that evaluation. Without it, those fields are reconstructed using
    ``penal`` and ``rho_min_ratio``.

    ``material_mass_density_kg_m3`` is optional. When supplied it must be the
    actual solid material density. For composite struts, supply the composite
    density of the carbon fibre and polymer matrix. It is separate from the
    dimensionless design density, ``rho_min_ratio`` and the stiffness factor.
    """
    if shell_problem is None:
        raise ValueError("shell_problem must be a configured ThickenShell instance.")
    if "rho" not in fields or "fiber3d" not in fields:
        raise KeyError("fields must contain 'rho' and 'fiber3d'.")
    if not math.isfinite(float(penal)) or float(penal) <= 0.0:
        raise ValueError("penal must be positive and finite.")
    if not 0.0 < float(rho_min_ratio) <= 1.0:
        raise ValueError("rho_min_ratio must lie in (0, 1].")
    if element_type.upper() not in {"C3D8", "C3D8R"}:
        raise ValueError("element_type must be 'C3D8' or 'C3D8R'.")

    rho_surface = fields["rho"]
    fiber_surface = fields["fiber3d"]
    if not isinstance(rho_surface, torch.Tensor):
        rho_surface = torch.as_tensor(rho_surface, dtype=torch.float32)
    if not isinstance(fiber_surface, torch.Tensor):
        fiber_surface = torch.as_tensor(
            fiber_surface,
            dtype=rho_surface.dtype,
            device=rho_surface.device,
        )
    else:
        fiber_surface = fiber_surface.to(device=rho_surface.device, dtype=rho_surface.dtype)

    expected_surface_count = int(np.asarray(shell_problem.points_xyz).reshape(-1, 3).shape[0])
    if rho_surface.numel() != expected_surface_count:
        raise ValueError(
            "The surface density length does not match shell_problem.points_xyz: "
            f"{rho_surface.numel()} != {expected_surface_count}."
        )
    if tuple(fiber_surface.shape) != (expected_surface_count, 3):
        raise ValueError(
            "fiber3d must have shape "
            f"({expected_surface_count}, 3), got {tuple(fiber_surface.shape)}."
        )

    with torch.no_grad():
        voxel_fields = shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
        )

    density = _as_numpy(voxel_fields["density"], np.float64).reshape(-1)
    phi = _as_numpy(voxel_fields["phi"], np.float64).reshape(-1)
    theta = _as_numpy(voxel_fields["theta"], np.float64).reshape(-1)
    orientation_matrix = None
    if "orientation_matrix" in voxel_fields:
        orientation_matrix = _as_numpy(voxel_fields["orientation_matrix"], np.float64).reshape(-1, 3, 3)
    orientation_axis_1 = None
    orientation_axis_2 = None
    orientation_axis_3 = None
    if "orientation_axis_1" in voxel_fields and "orientation_axis_2" in voxel_fields:
        orientation_axis_1 = _as_numpy(voxel_fields["orientation_axis_1"], np.float64).reshape(-1, 3)
        orientation_axis_2 = _as_numpy(voxel_fields["orientation_axis_2"], np.float64).reshape(-1, 3)
        if "orientation_axis_3" in voxel_fields:
            orientation_axis_3 = _as_numpy(voxel_fields["orientation_axis_3"], np.float64).reshape(-1, 3)
    geom_fraction = _as_numpy(
        voxel_fields.get("elem_geom_fraction", np.ones_like(density)),
        np.float64,
    ).reshape(-1)

    occupancy = np.asarray(shell_problem.elem_occupancy, dtype=bool).reshape(-1)
    if (
        density.size != occupancy.size
        or phi.size != occupancy.size
        or theta.size != occupancy.size
        or geom_fraction.size != occupancy.size
    ):
        raise ValueError("Voxel field sizes are inconsistent with elem_occupancy.")
    if orientation_matrix is not None and orientation_matrix.shape != (occupancy.size, 3, 3):
        raise ValueError("orientation_matrix size is inconsistent with elem_occupancy.")
    if orientation_axis_1 is not None and (
        orientation_axis_1.shape != (occupancy.size, 3)
        or orientation_axis_2.shape != (occupancy.size, 3)
    ):
        raise ValueError("orientation axis sizes are inconsistent with elem_occupancy.")

    if fem_result is not None:
        if "density_field" not in fem_result or "stiffness_factor_field" not in fem_result:
            raise KeyError(
                "fem_result must contain 'density_field' and 'stiffness_factor_field'."
            )
        density_from_fem = _as_numpy(fem_result["density_field"], np.float64).reshape(-1)
        stiffness_factor = _as_numpy(
            fem_result["stiffness_factor_field"], np.float64
        ).reshape(-1)
        if density_from_fem.size != occupancy.size or stiffness_factor.size != occupancy.size:
            raise ValueError("fem_result fields are inconsistent with elem_occupancy.")
        density = density_from_fem
        stiffness_source = "fem_result"
        density_field_source = "fem_result"
    else:
        fiber_density = np.clip(density, 0.0, 1.0)
        stiffness_inside = float(rho_min_ratio) + (
            1.0 - float(rho_min_ratio)
        ) * np.power(fiber_density, float(penal))
        stiffness_factor = (
            occupancy.astype(np.float64)
            * np.clip(geom_fraction, 0.0, 1.0)
            * stiffness_inside
        )
        stiffness_source = "reconstructed"
        density_field_source = "surface_to_voxel_mapping"

    active_element_ids = np.flatnonzero(occupancy).astype(np.int64)
    if active_element_ids.size == 0:
        raise ValueError("The occupied voxel domain is empty.")
    active_stiffness = stiffness_factor[active_element_ids]
    if not np.all(np.isfinite(active_stiffness)) or np.any(active_stiffness <= 0.0):
        raise ValueError("Active stiffness factors must be finite and positive.")

    mesh = shell_problem.mesh
    nelx = int(mesh["nelx"])
    nely = int(mesh["nely"])
    nelz = int(mesh["nelz"])
    expected_element_count = nelx * nely * nelz
    if occupancy.size != expected_element_count:
        raise ValueError(
            "elem_occupancy size is inconsistent with shell_problem.mesh: "
            f"{occupancy.size} != {expected_element_count}."
        )

    element_size = np.asarray(mesh["elemSize"], dtype=np.float64).reshape(3)
    resolved_unit_system = (
        str(unit_system)
        if unit_system is not None
        else str(
            getattr(
                shell_problem,
                "unit_system",
                getattr(shell_problem, "materialProperty", {}).get("unit_system", "N-mm-s"),
            )
        )
    )

    all_connectivity = _structured_c3d8_connectivity(nelx, nely, nelz)
    active_connectivity = all_connectivity[active_element_ids]
    all_node_xyz = np.asarray(shell_problem.node_coords, dtype=np.float64).reshape(-1, 3)
    expected_node_count = (nelx + 1) * (nely + 1) * (nelz + 1)
    if all_node_xyz.shape != (expected_node_count, 3):
        raise ValueError(
            "shell_problem.node_coords has an unexpected shape: "
            f"{all_node_xyz.shape} != ({expected_node_count}, 3)."
        )

    active_node_ids = np.unique(active_connectivity.reshape(-1))
    old_to_new_node = np.full(expected_node_count, -1, dtype=np.int64)
    old_to_new_node[active_node_ids] = np.arange(1, active_node_ids.size + 1)
    exported_connectivity = old_to_new_node[active_connectivity]
    exported_node_xyz = all_node_xyz[active_node_ids]
    exported_element_ids = np.arange(1, active_element_ids.size + 1, dtype=np.int64)

    boundary_condition = getattr(shell_problem, "boundaryCondition", None)
    if not isinstance(boundary_condition, Mapping):
        raise ValueError("shell_problem.boundaryCondition is missing or invalid.")
    if int(boundary_condition.get("numDOFPerNode", 3)) != 3:
        raise ValueError("The internal and exported solid models require three DOFs per node.")

    fixed_dofs = np.unique(
        _as_numpy(boundary_condition.get("fixed", np.empty(0)), np.int64).reshape(-1)
    )
    force_vector = _as_numpy(
        boundary_condition.get("force", np.empty(0)), np.float64
    ).reshape(-1)
    if force_vector.size != 3 * expected_node_count:
        raise ValueError("The force-vector size is inconsistent with the voxel node count.")
    if fixed_dofs.size == 0:
        raise ValueError("The fixed-DOF set is empty.")
    if fixed_dofs.min() < 0 or fixed_dofs.max() >= force_vector.size:
        raise ValueError("The fixed-DOF set contains an invalid index.")

    fixed_old_nodes = fixed_dofs // 3
    fixed_components = fixed_dofs % 3 + 1
    fixed_active_mask = old_to_new_node[fixed_old_nodes] > 0
    fixed_new_nodes = old_to_new_node[fixed_old_nodes[fixed_active_mask]]
    fixed_components = fixed_components[fixed_active_mask]
    if fixed_new_nodes.size == 0:
        raise ValueError("No fixed DOF belongs to an active voxel node.")

    loaded_dofs = np.flatnonzero(np.abs(force_vector) > 0.0).astype(np.int64)
    loaded_old_nodes = loaded_dofs // 3
    loaded_components = loaded_dofs % 3 + 1
    if np.any(old_to_new_node[loaded_old_nodes] <= 0):
        invalid = loaded_old_nodes[old_to_new_node[loaded_old_nodes] <= 0]
        raise ValueError(
            "A nonzero load acts on an inactive voxel node; first invalid IDs: "
            f"{invalid[:10].tolist()}."
        )
    loaded_new_nodes = old_to_new_node[loaded_old_nodes]
    loaded_values = force_vector[loaded_dofs]

    bin_ids, representative_stiffness, quantization_report = _material_bins_exact_or_logarithmic(
        active_stiffness,
        requested_bins=int(material_bins),
    )
    mass_report = calculate_voxel_mass_properties(
        design_density=density,
        geometric_fraction=geom_fraction,
        active_element_ids=active_element_ids,
        voxel_dimensions=element_size,
        material_mass_density_kg_m3=material_mass_density_kg_m3,
        unit_system=resolved_unit_system,
        density_field_source=density_field_source,
        bin_ids=bin_ids,
    )
    bin_mass_densities = mass_report["bin_mass_densities_model_units"]
    constants = _material_from_problem(shell_problem, material)
    if orientation_axis_1 is not None:
        axis_1 = orientation_axis_1[active_element_ids]
        axis_2 = orientation_axis_2[active_element_ids]
    elif orientation_matrix is not None:
        active_orientation = orientation_matrix[active_element_ids]
        axis_1 = active_orientation[:, :, 0]
        axis_2 = active_orientation[:, :, 1]
    else:
        axis_1, axis_2 = _orientation_axes(
            phi[active_element_ids],
            theta[active_element_ids],
        )
    axis_1 /= np.clip(np.linalg.norm(axis_1, axis=1, keepdims=True), 1.0e-14, None)
    axis_2 = axis_2 - np.sum(axis_2 * axis_1, axis=1, keepdims=True) * axis_1
    axis_2 /= np.clip(np.linalg.norm(axis_2, axis=1, keepdims=True), 1.0e-14, None)
    axis_3 = np.cross(axis_1, axis_2)
    det_orientation = np.linalg.det(np.stack((axis_1, axis_2, axis_3), axis=2))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    instance_name = "HVD_VOXEL-1"

    with output_path.open("w", encoding="utf-8", newline="\n") as inp:
        inp.write("*HEADING\n")
        inp.write("HVD voxel model matching the internal H8 FEM discretization\n")
        inp.write("**\n")
        inp.write("*DISTRIBUTION TABLE, NAME=HVD_ORIENTATION_TABLE\n")
        inp.write("COORD3D, COORD3D\n")
        inp.write("**\n")
        inp.write("*PART, NAME=HVD_VOXEL\n")
        inp.write("*NODE\n")
        for node_id, xyz in enumerate(exported_node_xyz, start=1):
            inp.write(
                f"{node_id}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}\n"
            )

        inp.write(f"*ELEMENT, TYPE={element_type.upper()}, ELSET=ALL_ACTIVE\n")
        for element_id, nodes in zip(exported_element_ids, exported_connectivity):
            inp.write(
                f"{int(element_id)}, "
                + ", ".join(str(int(node)) for node in nodes)
                + "\n"
            )

        inp.write(
            "*DISTRIBUTION, NAME=HVD_ORIENTATION_FIELD, "
            "LOCATION=ELEMENT, TABLE=HVD_ORIENTATION_TABLE\n"
        )
        inp.write(", 1., 0., 0., 0., 1., 0.\n")
        for element_id, first, second in zip(exported_element_ids, axis_1, axis_2):
            inp.write(
                f"{int(element_id)}, "
                f"{first[0]:.12g}, {first[1]:.12g}, {first[2]:.12g}, "
                f"{second[0]:.12g}, {second[1]:.12g}, {second[2]:.12g}\n"
            )

        inp.write(
            "*ORIENTATION, NAME=HVD_ORIENTATION, DEFINITION=COORDINATES, "
            "SYSTEM=RECTANGULAR\n"
        )
        inp.write("HVD_ORIENTATION_FIELD\n")
        inp.write("3, 0.\n")

        for bin_id, stiffness_value in enumerate(representative_stiffness):
            set_name = f"HVD_BIN_{bin_id:03d}"
            material_name = f"HVD_MAT_{bin_id:03d}"
            members = exported_element_ids[bin_ids == bin_id]
            inp.write(f"*ELSET, ELSET={set_name}\n")
            _write_id_list(inp, members)
            inp.write(
                f"*SOLID SECTION, ELSET={set_name}, MATERIAL={material_name}, "
                "ORIENTATION=HVD_ORIENTATION\n"
            )
            inp.write(",\n")
        inp.write("*END PART\n")
        inp.write("**\n")

        for bin_id, stiffness_value in enumerate(representative_stiffness):
            material_name = f"HVD_MAT_{bin_id:03d}"
            inp.write(f"*MATERIAL, NAME={material_name}\n")
            inp.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            inp.write(
                f"{constants['E1'] * stiffness_value:.12g}, "
                f"{constants['E2'] * stiffness_value:.12g}, "
                f"{constants['E3'] * stiffness_value:.12g}, "
                f"{constants['nu12']:.12g}, "
                f"{constants['nu13']:.12g}, "
                f"{constants['nu23']:.12g}, "
                f"{constants['G12'] * stiffness_value:.12g}, "
                f"{constants['G13'] * stiffness_value:.12g}\n"
            )
            inp.write(f"{constants['G23'] * stiffness_value:.12g}\n")
            if bin_mass_densities is not None:
                inp.write("*DENSITY\n")
                inp.write(f"{float(bin_mass_densities[bin_id]):.12g}\n")
        inp.write("**\n")

        inp.write("*ASSEMBLY, NAME=ASSEMBLY\n")
        inp.write(f"*INSTANCE, NAME={instance_name}, PART=HVD_VOXEL\n")
        inp.write("*END INSTANCE\n")
        inp.write("*END ASSEMBLY\n")
        inp.write("**\n")

        inp.write("*STEP, NAME=STATIC_LOAD, NLGEOM=NO\n")
        inp.write("*STATIC\n")
        inp.write("0.1, 1.0, 1.0e-08, 0.1\n")
        inp.write("*BOUNDARY\n")
        for node_id, component in zip(fixed_new_nodes, fixed_components):
            inp.write(
                f"{instance_name}.{int(node_id)}, {int(component)}, "
                f"{int(component)}, 0.\n"
            )

        inp.write("*CLOAD\n")
        for node_id, component, value in zip(
            loaded_new_nodes,
            loaded_components,
            loaded_values,
        ):
            inp.write(
                f"{instance_name}.{int(node_id)}, {int(component)}, {value:.12g}\n"
            )

        inp.write("*OUTPUT, FIELD, FREQUENCY=1\n")
        inp.write("*NODE OUTPUT\n")
        inp.write("U, RF,S\n")
        inp.write(
            "*ELEMENT OUTPUT, "
            "POSITION=INTEGRATION POINTS, "
            "DIRECTIONS=YES\n"
        )
        inp.write("S\n")
        inp.write("*OUTPUT, HISTORY, FREQUENCY=1\n")
        inp.write("*ENERGY OUTPUT\n")
        inp.write("ALLIE, ALLSE\n")
        inp.write("*END STEP\n")

    generated_text = output_path.read_text(encoding="utf-8")
    required_keywords = (
        f"*ELEMENT, TYPE={element_type.upper()}, ELSET=ALL_ACTIVE",
        "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
        (
            "*ELEMENT OUTPUT, "
            "POSITION=INTEGRATION POINTS, "
            "DIRECTIONS=YES"
        ),
        "S",
    )
    missing_keywords = [keyword for keyword in required_keywords if keyword not in generated_text]
    if missing_keywords:
        raise RuntimeError(f"Generated input is missing keywords: {missing_keywords}.")

    total_force = np.zeros(3, dtype=np.float64)
    for component, value in zip(loaded_components, loaded_values):
        total_force[int(component) - 1] += float(value)

    diagnostics = {
        "output_path": output_path,
        "element_type": element_type.upper(),
        "num_active_elements": int(active_element_ids.size),
        "expected_integration_point_stress_records": int(active_element_ids.size * 8),
        "num_active_nodes": int(active_node_ids.size),
        "num_material_bins": int(representative_stiffness.size),
        "stiffness_source": stiffness_source,
        "density_field_source": density_field_source,
        "mass_available": mass_report["mass_available"],
        "material_mass_density_kg_m3": mass_report["material_mass_density_kg_m3"],
        "material_mass_density_units": mass_report["material_mass_density_units"],
        "unit_system": mass_report["unit_system"],
        "model_mass_unit": mass_report["model_mass_unit"],
        "model_density_unit": mass_report["model_density_unit"],
        "material_mass_density_model_units": mass_report[
            "material_mass_density_model_units"
        ],
        "active_voxel_count": mass_report["active_voxel_count"],
        "voxel_dimensions": mass_report["voxel_dimensions"],
        "voxel_volume": mass_report["voxel_volume"],
        "full_active_voxel_volume": mass_report["full_active_voxel_volume"],
        "geometry_weighted_shell_volume": mass_report[
            "geometry_weighted_shell_volume"
        ],
        "material_volume": mass_report["material_volume"],
        "embedded_total_mass_model_units": mass_report[
            "embedded_total_mass_model_units"
        ],
        "embedded_total_mass_grams": mass_report["embedded_total_mass_grams"],
        "expected_exported_total_mass_model_units": mass_report[
            "expected_exported_total_mass_model_units"
        ],
        "expected_exported_total_mass_grams": mass_report[
            "expected_exported_total_mass_grams"
        ],
        "mass_difference_grams": mass_report["mass_difference_grams"],
        "mass_difference_relative": mass_report["mass_difference_relative"],
        "bin_mass_densities_model_units": bin_mass_densities,
        "bin_expected_mass_model_units": mass_report["bin_expected_mass_model_units"],
        "mass_density_bin_averaging_note": mass_report[
            "mass_density_bin_averaging_note"
        ],
        "abaqus_density_limitation": (
            "Zero-mass bins are written as *DENSITY, 0.0 when physical mass "
            "export is enabled. Local Abaqus documentation/runtime verification "
            "was not available in this environment."
        ),
        "active_density_min": float(density[active_element_ids].min()),
        "active_density_mean": float(density[active_element_ids].mean()),
        "active_density_max": float(density[active_element_ids].max()),
        "active_geom_fraction_min": float(geom_fraction[active_element_ids].min()),
        "active_geom_fraction_mean": float(geom_fraction[active_element_ids].mean()),
        "active_geom_fraction_max": float(geom_fraction[active_element_ids].max()),
        "active_stiffness_min": float(active_stiffness.min()),
        "active_stiffness_mean": float(active_stiffness.mean()),
        "active_stiffness_max": float(active_stiffness.max()),
        "num_fixed_dofs": int(fixed_new_nodes.size),
        "num_loaded_dofs": int(loaded_new_nodes.size),
        "total_force_vector": total_force,
        "material_bin_representatives": representative_stiffness,
        "orientation_axis_1": axis_1,
        "orientation_axis_2": axis_2,
        "orientation_axis_3": axis_3,
        "orientation_matrix": np.stack((axis_1, axis_2, axis_3), axis=2),
        "orientation_det_min": float(det_orientation.min()),
        "orientation_det_max": float(det_orientation.max()),
        **quantization_report,
    }

    print(f"Written: {output_path}")
    print(f"Element type: {diagnostics['element_type']}")
    print(f"Active elements: {diagnostics['num_active_elements']}")
    print(f"Expected raw integration-point S records: {diagnostics['expected_integration_point_stress_records']}")
    print(f"Active nodes: {diagnostics['num_active_nodes']}")
    print(f"Material bins: {diagnostics['num_material_bins']}")
    print(f"Exact stiffness bins: {diagnostics['used_exact_stiffness_bins']}")
    print(
        "Max stiffness quantization error abs/rel: "
        f"{diagnostics['max_abs_stiffness_quantization_error']:.6g} / "
        f"{diagnostics['max_rel_stiffness_quantization_error']:.6g}"
    )
    print(f"Stiffness source: {diagnostics['stiffness_source']}")
    print(f"Density-field source: {diagnostics['density_field_source']}")
    print(
        "Active stiffness range: "
        f"{diagnostics['active_stiffness_min']:.6g} to "
        f"{diagnostics['active_stiffness_max']:.6g}"
    )
    print(f"Fixed DOFs: {diagnostics['num_fixed_dofs']}")
    print(f"Loaded DOFs: {diagnostics['num_loaded_dofs']}")
    print(f"Total force: {total_force.tolist()}")
    print(
        "Orientation determinant range: "
        f"{diagnostics['orientation_det_min']:.12g} to "
        f"{diagnostics['orientation_det_max']:.12g}"
    )
    print(f"Model unit system: {diagnostics['unit_system']}")
    if diagnostics["mass_available"]:
        print(
            "Material mass density: "
            f"{diagnostics['material_mass_density_kg_m3']:.12g} "
            f"{diagnostics['material_mass_density_units']} "
            f"= {diagnostics['material_mass_density_model_units']:.12g} "
            f"{diagnostics['model_density_unit']}"
        )
        print(f"Active voxel count: {diagnostics['active_voxel_count']}")
        print(f"Full active voxel volume: {diagnostics['full_active_voxel_volume']:.12g}")
        print(
            "Geometry-weighted shell volume: "
            f"{diagnostics['geometry_weighted_shell_volume']:.12g}"
        )
        print(f"Material volume: {diagnostics['material_volume']:.12g}")
        print(
            "Embedded total mass: "
            f"{diagnostics['embedded_total_mass_grams']:.12g} g "
            f"({diagnostics['embedded_total_mass_model_units']:.12g} "
            f"{diagnostics['model_mass_unit']})"
        )
        print(
            "Expected exported total mass: "
            f"{diagnostics['expected_exported_total_mass_grams']:.12g} g "
            f"({diagnostics['expected_exported_total_mass_model_units']:.12g} "
            f"{diagnostics['model_mass_unit']})"
        )
        print(
            "Mass difference abs/rel: "
            f"{diagnostics['mass_difference_grams']:.6g} g / "
            f"{diagnostics['mass_difference_relative']:.6g}"
        )
    else:
        print("Physical mass: unavailable; material_mass_density_kg_m3 was not supplied.")
    return diagnostics


# Example
# -------
# from Utils.ExportAbaqusVoxelFEM import export_abaqus_voxel_fem
#
# abaqus_export = export_abaqus_voxel_fem(
#     fields=honeycomb_fields_fem,
#     shell_problem=shell_problem,
#     output_path="Uniform_Honeycomb_voxel_fem.inp",
#     fem_result=fem_honeycomb,
#     material=shell_problem.materialProperty,
#     penal=cfg.fem_penal,
#     rho_min_ratio=cfg.fem_rho_min_end,
#     material_bins=128,
#     element_type="C3D8",
# )
