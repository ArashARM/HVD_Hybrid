"""General Embedded-H8 FEM runner and Abaqus C3D8 exporter.

This module is executed in the normal project Python environment (PyTorch),
not in the Abaqus Python interpreter.  The generated ``.inp`` file is solved
in Abaqus and the companion ODB audit script is then run in Abaqus/CAE.

Authoritative stress quantity
-----------------------------
The stress used here is the largest raw von Mises value over every active
element and every C3D8 integration point.  Centroidal and nodally averaged
stress values are deliberately not used for constraint validation.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from networkx import density
import numpy as np
import torch
import torch.nn.functional as F

from neuraltomo_fem.run_fem_loss import NeuralTOMOFEM
from Utils.ExportAbaqus_VoxelBased import export_abaqus_voxel_fem
from Utils.VoxelMassProperties import calculate_voxel_mass_properties


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _format_vector(value: Any) -> str:
    values = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
    return "[" + ", ".join("%.12g" % float(v) for v in values) + "]"


def _safe_case_name(case_name: str) -> str:
    name = str(case_name).strip()
    if not name:
        raise ValueError("case_name must not be empty.")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("case_name must be a filename-safe name, not a path.")
    return name


def _global_to_material_abaqus_order(
    sigma_global: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    orientation_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert global stress to local Abaqus order.

    Input order:  ``[Sxx,Syy,Szz,Syz,Sxz,Sxy]``.
    Output order: ``[S11,S22,S33,S12,S13,S23]``.
    """

    sxx, syy, szz, syz, sxz, sxy = sigma_global
    stress_global = torch.stack(
        (
            torch.stack((sxx, sxy, sxz)),
            torch.stack((sxy, syy, syz)),
            torch.stack((sxz, syz, szz)),
        )
    )

    if orientation_matrix is None:
        zero = torch.zeros_like(phi)
        axis_1 = torch.stack(
            (
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            )
        )
        axis_2 = torch.stack((-torch.sin(phi), torch.cos(phi), zero))
        axis_1 = F.normalize(axis_1, dim=0, eps=1.0e-12)
        axis_2 = F.normalize(axis_2, dim=0, eps=1.0e-12)
        axis_3 = F.normalize(
            torch.cross(axis_1, axis_2, dim=0), dim=0, eps=1.0e-12
        )
        rotation = torch.stack((axis_1, axis_2, axis_3), dim=1)
    else:
        rotation = orientation_matrix.to(device=sigma_global.device, dtype=sigma_global.dtype)
    stress_local = rotation.transpose(0, 1) @ stress_global @ rotation

    return torch.stack(
        (
            stress_local[0, 0],
            stress_local[1, 1],
            stress_local[2, 2],
            stress_local[0, 1],
            stress_local[0, 2],
            stress_local[1, 2],
        )
    )


class VoxelFEMAbaqusRunner:
    """Run and export comparable Embedded FEM/Abaqus voxel cases.

    Parameters
    ----------
    face_mesh:
        Surface field dictionary. It must contain ``points_xyz``.
    shell_problem:
        The already-created ``ThickenShell`` (or compatible) problem. The
        Embedded FEM and Abaqus exporter intentionally share this exact object.
    device:
        Torch device used by the Embedded FEM.
    output_dir:
        Destination for Embedded audit text files, Abaqus input files and the
        optional multi-case summary CSV.
    isotropic:
        Passed to ``NeuralTOMOFEM`` when this class constructs the solver.
    embedded_fem:
        Optional existing solver. Supplying it is useful in notebooks and
        avoids constructing a second solver.
    """

    def __init__(
        self,
        face_mesh: Mapping[str, Any],
        *,
        shell_problem: Any,
        device: Any,
        output_dir: Any = "FEM_Verification",
        isotropic: bool = False,
        embedded_fem: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        if shell_problem is None:
            raise ValueError("shell_problem is required.")
        if "points_xyz" not in face_mesh:
            raise KeyError("face_mesh must contain 'points_xyz'.")

        self.face_mesh = face_mesh
        self.shell_problem = shell_problem
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = bool(verbose)

        self.embedded_fem = embedded_fem
        if self.embedded_fem is None:
            self.embedded_fem = NeuralTOMOFEM(
                problem=self.shell_problem,
                device=self.device,
                isotropic=bool(isotropic),
            )

        self.geometry_report: Optional[Dict[str, Any]] = None
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.solid_cases = self.cases  # legacy notebook attribute

        # Convenient references to the most recently executed case.
        self.latest_fields: Optional[Dict[str, Any]] = None
        self.latest_fem_result: Optional[Dict[str, Any]] = None
        self.latest_metrics: Optional[Dict[str, Any]] = None
        self.latest_export_report: Optional[Any] = None
        self.solid_fields: Optional[Dict[str, Any]] = None
        self.solid_fem_result: Optional[Dict[str, Any]] = None
        self.solid_metrics: Optional[Dict[str, Any]] = None
        self.solid_export_report: Optional[Any] = None

    def audit_common_geometry(
        self,
        expected_total_force: Optional[Sequence[float]] = None,
        *,
        force_tolerance: float = 1.0e-10,
    ) -> Dict[str, Any]:
        """Check that mesh, supports and loads share one active domain."""

        problem = self.shell_problem
        mesh_data = problem.mesh
        fe_mesh = self.embedded_fem.fe.mesh

        nelx = int(mesh_data["nelx"])
        nely = int(mesh_data["nely"])
        nelz = int(mesh_data["nelz"])
        element_size = np.asarray(mesh_data["elemSize"], dtype=np.float64).reshape(3)

        occupancy = np.asarray(problem.elem_occupancy, dtype=bool).reshape(-1)
        active_element_ids = np.flatnonzero(occupancy).astype(np.int64)
        active_nodes = np.asarray(fe_mesh.active_node_ids, dtype=np.int64)

        force = np.asarray(
            problem.boundaryCondition["force"], dtype=np.float64
        ).reshape(-1, 3)
        total_force = force.sum(axis=0)
        loaded_nodes = np.flatnonzero(
            np.linalg.norm(force, axis=1) > 1.0e-12
        ).astype(np.int64)

        fixed_dofs = np.asarray(
            problem.boundaryCondition["fixed"], dtype=np.int64
        ).reshape(-1)
        fixed_nodes = np.unique(fixed_dofs // 3)

        if active_element_ids.size == 0:
            raise ValueError("The occupied voxel domain is empty.")
        if fixed_nodes.size == 0:
            raise ValueError("No fixed nodes were identified.")
        if loaded_nodes.size == 0:
            raise ValueError("No loaded nodes were identified.")
        if not np.all(np.isin(fixed_nodes, active_nodes)):
            raise ValueError("Some fixed nodes are outside the active domain.")
        if not np.all(np.isin(loaded_nodes, active_nodes)):
            raise ValueError("Some loaded nodes are outside the active domain.")

        if expected_total_force is not None:
            expected = np.asarray(expected_total_force, dtype=np.float64).reshape(3)
            if not np.allclose(
                total_force, expected, rtol=0.0, atol=float(force_tolerance)
            ):
                raise ValueError(
                    "Assembled total force %s does not match expected force %s."
                    % (_format_vector(total_force), _format_vector(expected))
                )

        voxel_volume = float(np.prod(element_size))
        occupied_volume = float(active_element_ids.size * voxel_volume)

        face_areas = getattr(problem, "face_areas", None)
        thickness = getattr(problem, "thickness", None)
        surface_area = None
        nominal_shell_volume = None
        relative_volume_error = None
        if face_areas is not None:
            surface_area = float(np.asarray(face_areas, dtype=np.float64).sum())
        if surface_area is not None and thickness is not None:
            nominal_shell_volume = surface_area * float(thickness)
            if nominal_shell_volume > 0.0:
                relative_volume_error = (
                    occupied_volume - nominal_shell_volume
                ) / nominal_shell_volume

        report = {
            "grid": (nelx, nely, nelz),
            "element_size": element_size,
            "total_structured_elements": nelx * nely * nelz,
            "active_elements": int(active_element_ids.size),
            "active_nodes": int(active_nodes.size),
            "voxel_volume": voxel_volume,
            "occupied_voxel_volume": occupied_volume,
            "surface_area": surface_area,
            "nominal_shell_volume": nominal_shell_volume,
            "relative_volume_error": relative_volume_error,
            "fixed_nodes": int(fixed_nodes.size),
            "fixed_dofs": int(fixed_dofs.size),
            "loaded_nodes": int(loaded_nodes.size),
            "loaded_dofs": int(np.count_nonzero(np.abs(force) > 1.0e-12)),
            "total_force_vector": total_force,
        }
        self.geometry_report = report

        print("\n=== COMMON VOXEL MODEL AUDIT ===")
        print("Grid:", "%d x %d x %d" % report["grid"])
        print("Element size:", element_size)
        print("Total structured elements:", report["total_structured_elements"])
        print("Active elements:", report["active_elements"])
        print("Active nodes:", report["active_nodes"])
        if surface_area is not None:
            print("Surface area:", surface_area)
        if nominal_shell_volume is not None:
            print("Nominal shell volume:", nominal_shell_volume)
            print("Occupied voxel volume:", occupied_volume)
            print("Relative volume error:", relative_volume_error)
        print("Fixed nodes:", report["fixed_nodes"])
        print("Fixed DOFs:", report["fixed_dofs"])
        print("Loaded nodes:", report["loaded_nodes"])
        print("Loaded DOFs:", report["loaded_dofs"])
        print("Total force:", total_force)
        return report

    def run_uniform_case(
        self,
        fibre_vector: Sequence[float],
        case_name: str,
        *,
        density_value: float = 1.0,
        uniform_active_stiffness: Optional[float] = None,
        material_mass_density_kg_m3: Optional[float] = None,
        unit_system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a uniform density and uniform global-fibre benchmark."""

        reference = self.face_mesh["points_xyz"]
        fibre = torch.as_tensor(
            fibre_vector, dtype=reference.dtype, device=self.device
        ).reshape(3)
        if not torch.isfinite(fibre).all() or float(torch.linalg.norm(fibre)) <= 0.0:
            raise ValueError("fibre_vector must be finite and non-zero.")
        fibre = F.normalize(fibre, dim=0, eps=1.0e-12)

        if not 0.0 <= float(density_value) <= 1.0:
            raise ValueError("density_value must lie in [0, 1].")
        number_of_nodes = int(reference.shape[0])
        rho_surface = torch.full(
            (number_of_nodes,),
            float(density_value),
            dtype=reference.dtype,
            device=self.device,
        )
        fibre_surface = fibre.reshape(1, 3).repeat(number_of_nodes, 1)

        return self.run_surface_fields(
            rho_surface=rho_surface,
            fibre_surface=fibre_surface,
            case_name=case_name,
            stiffness_factor_override=uniform_active_stiffness,
            material_mass_density_kg_m3=material_mass_density_kg_m3,
            unit_system=unit_system,
            **kwargs,
        )

    def run_fields(
        self,
        fields: Mapping[str, Any],
        case_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a decoder/optimization field dictionary directly."""

        rho = fields.get("rho", fields.get("density"))
        fibre = fields.get("fiber3d", fields.get("fiber"))
        if rho is None:
            raise KeyError("fields must contain 'rho' or 'density'.")
        if fibre is None:
            raise KeyError("fields must contain 'fiber3d' or 'fiber'.")
        return self.run_surface_fields(
            rho_surface=rho,
            fibre_surface=fibre,
            case_name=case_name,
            source_fields=fields,
            **kwargs,
        )

    def run_surface_fields(
        self,
        rho_surface: Any,
        fibre_surface: Any,
        case_name: str,
        *,
        rho_min_ratio: float = 1.0e-4,
        penal: float = 3.0,
        material_bins: int = 128,
        element_type: str = "C3D8",
        stiffness_factor_override: Optional[Any] = None,
        source_fields: Optional[Mapping[str, Any]] = None,
        material_mass_density_kg_m3: Optional[float] = None,
        unit_system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Embedded FEM, audit raw IP values, and export the same case."""

        case_name = _safe_case_name(case_name)
        if str(element_type).upper() != "C3D8":
            raise ValueError(
                "This validated max-IP workflow requires full-integration C3D8."
            )
        if not 0.0 <= float(rho_min_ratio) <= 1.0:
            raise ValueError("rho_min_ratio must lie in [0, 1].")
        if float(penal) <= 0.0:
            raise ValueError("penal must be positive.")

        reference = self.face_mesh["points_xyz"]
        dtype = reference.dtype
        number_of_surface_nodes = int(reference.shape[0])
        rho_surface = torch.as_tensor(
            rho_surface, dtype=dtype, device=self.device
        ).reshape(-1)
        fibre_surface = torch.as_tensor(
            fibre_surface, dtype=dtype, device=self.device
        ).reshape(-1, 3)

        if rho_surface.numel() != number_of_surface_nodes:
            raise ValueError(
                "rho_surface has %d values; expected %d."
                % (rho_surface.numel(), number_of_surface_nodes)
            )
        if fibre_surface.shape[0] != number_of_surface_nodes:
            raise ValueError(
                "fibre_surface has %d rows; expected %d."
                % (fibre_surface.shape[0], number_of_surface_nodes)
            )
        if not torch.isfinite(rho_surface).all():
            raise ValueError("rho_surface contains non-finite values.")
        if torch.any((rho_surface < 0.0) | (rho_surface > 1.0)):
            raise ValueError("rho_surface must lie in [0, 1].")
        if not torch.isfinite(fibre_surface).all():
            raise ValueError("fibre_surface contains non-finite values.")
        fibre_norm = torch.linalg.norm(fibre_surface, dim=1, keepdim=True)
        if torch.any(fibre_norm <= 1.0e-12):
            raise ValueError("Every surface fibre vector must be non-zero.")
        fibre_surface = fibre_surface / fibre_norm

        problem = self.shell_problem
        resolved_unit_system = (
            str(unit_system)
            if unit_system is not None
            else str(getattr(problem, "unit_system", problem.materialProperty.get("unit_system", "N-mm-s")))
        )
        voxel_fields = problem.build_fem_fields_from_decoder_torch(
            rho_surface=rho_surface,
            fiber_surface=fibre_surface,
        )
        density = voxel_fields["density"].reshape(-1)
        occupancy = voxel_fields["shell_occupancy"].reshape(-1).to(
            device=density.device, dtype=density.dtype
        )
        phi = voxel_fields["phi"].reshape(-1)
        theta = voxel_fields["theta"].reshape(-1)
        orientation_matrix = voxel_fields.get("orientation_matrix")
        if orientation_matrix is not None:
            orientation_matrix = orientation_matrix.to(device=density.device, dtype=density.dtype)
        active_mask = occupancy > 0.5
        if not torch.any(active_mask):
            raise ValueError("Transferred voxel domain has no active elements.")
        elem_geom_fraction = voxel_fields.get("elem_geom_fraction")
        if elem_geom_fraction is None:
            elem_geom_fraction = torch.ones_like(density)
        else:
            elem_geom_fraction = elem_geom_fraction.to(
                device=density.device,
                dtype=density.dtype,
            ).reshape(-1)

        stiffness_inside = float(rho_min_ratio) + (
            1.0 - float(rho_min_ratio)
        ) * density.clamp(0.0, 1.0).pow(float(penal))
        stiffness_factor = occupancy * stiffness_inside

        if stiffness_factor_override is not None:
            override = torch.as_tensor(
                stiffness_factor_override,
                dtype=density.dtype,
                device=density.device,
            )
            if override.numel() == 1:
                if not torch.isfinite(override) or float(override) <= 0.0:
                    raise ValueError("Uniform active stiffness must be finite and > 0.")
                stiffness_factor = occupancy * override.reshape(())
            else:
                override = override.reshape(-1)
                if override.shape != density.shape:
                    raise ValueError(
                        "stiffness_factor_override must be scalar or have one "
                        "value per structured element."
                    )
                if not torch.isfinite(override).all() or torch.any(override < 0.0):
                    raise ValueError("stiffness_factor_override must be finite and >= 0.")
                stiffness_factor = occupancy * override

        _, compliance_returned = self.embedded_fem(
            stiffness_factor,
            phi,
            theta,
            penal=1.0,
            orientation_matrix=orientation_matrix,
        )
        fe_solver = self.embedded_fem.fe
        fe_mesh = fe_solver.mesh

        required_stress_fields = ("sigma_ip_active", "stress_vm_ip_active")
        missing = [name for name in required_stress_fields if not hasattr(fe_solver, name)]
        if missing:
            raise AttributeError(
                "Embedded solver is missing max-IP recovery fields: %s"
                % ", ".join(missing)
            )
        sigma_ip = fe_solver.sigma_ip_active
        stress_vm_ip = fe_solver.stress_vm_ip_active
        if sigma_ip.ndim != 3 or sigma_ip.shape[-1] != 6:
            raise ValueError("sigma_ip_active must have shape [elements, IPs, 6].")
        if stress_vm_ip.shape != sigma_ip.shape[:2]:
            raise ValueError("stress_vm_ip_active shape does not match sigma_ip_active.")
        if int(stress_vm_ip.shape[1]) != 8:
            raise ValueError("Full-integration C3D8 must provide 8 IPs per element.")

        u = fe_solver.u[:, 0]
        u_nodes = u.reshape(fe_mesh.numNodes, 3)
        force_np = np.asarray(
            problem.boundaryCondition["force"], dtype=np.float64
        ).reshape(-1, 3)
        force = torch.as_tensor(force_np, dtype=u.dtype, device=u.device)
        loaded_nodes_np = np.flatnonzero(
            np.linalg.norm(force_np, axis=1) > 1.0e-12
        )
        if loaded_nodes_np.size == 0:
            raise ValueError("No loaded nodes were identified.")
        loaded_nodes = torch.as_tensor(
            loaded_nodes_np, dtype=torch.long, device=u.device
        )
        active_nodes = torch.as_tensor(
            fe_mesh.active_node_ids, dtype=torch.long, device=u.device
        )
        total_force = force.sum(dim=0)
        load_axis = int(torch.argmax(torch.abs(total_force)).item())

        loaded_directional = u_nodes[loaded_nodes, load_axis]
        loaded_magnitude = torch.linalg.norm(u_nodes[loaded_nodes], dim=1)
        active_magnitude = torch.linalg.norm(u_nodes[active_nodes], dim=1)
        compliance = torch.sum(force.reshape(-1) * u)
        strain_energy = 0.5 * compliance

        embedded_audit = self._write_embedded_audit(
            case_name=case_name,
            phi=phi,
            theta=theta,
            orientation_matrix=orientation_matrix,
            compliance=compliance,
            strain_energy=strain_energy,
        )

        surface_fields = dict(source_fields or {})
        surface_fields.update(
            {
                "rho": rho_surface,
                "density": rho_surface,
                "fiber3d": fibre_surface,
                "fiber": fibre_surface,
                "face_tensor": surface_fields.get("face_tensor", self.face_mesh),
            }
        )
        fem_result = {
            "density_field": density.detach(),
            "stiffness_factor_field": stiffness_factor.detach(),
        }
        active_element_ids = np.flatnonzero(_as_numpy(occupancy).reshape(-1) > 0.5)
        density = density.clamp(0.0, 1.0)
        embedded_mass_report = calculate_voxel_mass_properties(
            design_density=density.detach(),
            geometric_fraction=elem_geom_fraction.detach(),
            active_element_ids=active_element_ids,
            voxel_dimensions=problem.mesh["elemSize"],
            material_mass_density_kg_m3=material_mass_density_kg_m3,
            unit_system=resolved_unit_system,
            density_field_source="fem_result",
        )
        input_path = self.output_dir / (case_name + ".inp")
        export_arguments = {
            "fields": surface_fields,
            "shell_problem": problem,
            "output_path": input_path,
            "fem_result": fem_result,
            "material": problem.materialProperty,
            "penal": float(penal),
            "rho_min_ratio": float(rho_min_ratio),
            "material_bins": int(material_bins),
            "element_type": "C3D8",
            "material_mass_density_kg_m3": material_mass_density_kg_m3,
            "unit_system": resolved_unit_system,
        }
        if self.verbose:
            export_report = export_abaqus_voxel_fem(**export_arguments)
        else:
            # The exporter has useful diagnostic prints, but they make a
            # multi-case notebook difficult to read. They remain available
            # by constructing this runner with verbose=True.
            with redirect_stdout(io.StringIO()):
                export_report = export_abaqus_voxel_fem(**export_arguments)

        returned_compliance = float(compliance_returned.detach().cpu())
        direct_compliance = float(compliance.detach().cpu())
        metrics = {
            "case_name": case_name,
            "active_elements": int(stress_vm_ip.shape[0]),
            "active_nodes": int(active_nodes.numel()),
            "integration_point_records": int(stress_vm_ip.numel()),
            "total_force": _as_numpy(total_force),
            "load_axis": load_axis,
            "loaded_mean_directional_displacement": float(
                loaded_directional.mean().detach().cpu()
            ),
            "loaded_max_directional_displacement": float(
                loaded_directional.abs().max().detach().cpu()
            ),
            "loaded_max_displacement_magnitude": float(
                loaded_magnitude.max().detach().cpu()
            ),
            "global_max_displacement_magnitude": embedded_audit[
                "maximum_displacement_magnitude"
            ],
            "global_max_displacement_coordinates": embedded_audit[
                "maximum_displacement_coordinates"
            ],
            "compliance": direct_compliance,
            "returned_compliance": returned_compliance,
            "compliance_difference": abs(returned_compliance - direct_compliance),
            "strain_energy": float(strain_energy.detach().cpu()),
            "maximum_integration_point_von_mises": embedded_audit[
                "maximum_integration_point_von_mises"
            ],
            "critical_stress_coordinates": embedded_audit[
                "critical_integration_point_coordinates"
            ],
            "critical_abaqus_element_label": embedded_audit[
                "critical_abaqus_element_label"
            ],
            "critical_abaqus_integration_point": embedded_audit[
                "critical_abaqus_integration_point"
            ],
            "phi_range": (
                float(phi[active_mask].min().detach().cpu()),
                float(phi[active_mask].max().detach().cpu()),
            ),
            "theta_range": (
                float(theta[active_mask].min().detach().cpu()),
                float(theta[active_mask].max().detach().cpu()),
            ),
            "density_range": (
                float(density[active_mask].min().detach().cpu()),
                float(density[active_mask].max().detach().cpu()),
            ),
            "stiffness_range": (
                float(stiffness_factor[active_mask].min().detach().cpu()),
                float(stiffness_factor[active_mask].max().detach().cpu()),
            ),
            "mass_available": embedded_mass_report["mass_available"],
            "material_mass_density_kg_m3": embedded_mass_report[
                "material_mass_density_kg_m3"
            ],
            "material_mass_density_units": embedded_mass_report[
                "material_mass_density_units"
            ],
            "unit_system": embedded_mass_report["unit_system"],
            "model_mass_unit": embedded_mass_report["model_mass_unit"],
            "model_density_unit": embedded_mass_report["model_density_unit"],
            "density_field_source": embedded_mass_report["density_field_source"],
            "active_voxel_count": embedded_mass_report["active_voxel_count"],
            "full_active_voxel_volume": embedded_mass_report[
                "full_active_voxel_volume"
            ],
            "geometry_weighted_shell_volume": embedded_mass_report[
                "geometry_weighted_shell_volume"
            ],
            "material_volume": embedded_mass_report["material_volume"],
            "material_mass_density_model_units": embedded_mass_report[
                "material_mass_density_model_units"
            ],
            "embedded_total_mass_model_units": embedded_mass_report[
                "embedded_total_mass_model_units"
            ],
            "embedded_total_mass_grams": embedded_mass_report[
                "embedded_total_mass_grams"
            ],
            "expected_exported_total_mass_model_units": export_report[
                "expected_exported_total_mass_model_units"
            ],
            "expected_exported_total_mass_grams": export_report[
                "expected_exported_total_mass_grams"
            ],
            "mass_difference_grams": export_report["mass_difference_grams"],
            "mass_difference_relative": export_report["mass_difference_relative"],
            "abaqus_input_path": str(input_path.resolve()),
            "embedded_audit_path": embedded_audit["output_path"],
            "embedded_audit": embedded_audit,
            "export_report": export_report,
        }
        summary_json_path = self.output_dir / (case_name + "_summary.json")
        metrics["summary_json_path"] = str(summary_json_path.resolve())
        with summary_json_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self._json_safe(metrics), stream, indent=2, sort_keys=True)

        case = {
            "metrics": metrics,
            "fields": surface_fields,
            "voxel_fields": voxel_fields,
            "fem_result": fem_result,
            "export_report": export_report,
        }
        self.cases[case_name] = case
        self.latest_fields = surface_fields
        self.latest_fem_result = fem_result
        self.latest_metrics = metrics
        self.latest_export_report = export_report
        self.solid_fields = surface_fields
        self.solid_fem_result = fem_result
        self.solid_metrics = metrics
        self.solid_export_report = export_report

        print("\n=== %s ===" % case_name)
        print(
            "Maximum displacement magnitude: %.12g at %s"
            % (
                metrics["global_max_displacement_magnitude"],
                _format_vector(metrics["global_max_displacement_coordinates"]),
            )
        )
        print("Compliance F^T U: %.12g" % metrics["compliance"])
        print(
            "Maximum raw IP von Mises stress: %.12g at %s"
            % (
                metrics["maximum_integration_point_von_mises"],
                _format_vector(metrics["critical_stress_coordinates"]),
            )
        )
        
        print("Abaqus input: %s" % metrics["abaqus_input_path"])
        #print("Summary JSON: %s" % metrics["summary_json_path"])
        if metrics["mass_available"]:
            print(
                "Embedded mass: %.12g g"
                % (
                    metrics["embedded_total_mass_grams"],
                )
            )
        else:
            print("Physical mass: unavailable; material_mass_density_kg_m3 was not supplied.")
        return metrics

    def run_uniform_cases(
        self,
        cases: Mapping[str, Sequence[float]],
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Run several named uniform-fibre cases and write a summary CSV."""

        results = {
            name: self.run_uniform_case(vector, name, **kwargs)
            for name, vector in cases.items()
        }
        #self.write_summary_csv()
        return results

    def write_summary_csv(self, filename: str = "embedded_case_summary.csv") -> str:
        """Write one concise row for every case executed by this runner."""

        path = self.output_dir / filename
        columns = [
            "case",
            "active_elements",
            "active_nodes",
            "integration_point_records",
            "compliance",
            "strain_energy",
            "maximum_displacement_magnitude",
            "maximum_displacement_x",
            "maximum_displacement_y",
            "maximum_displacement_z",
            "maximum_ip_von_mises",
            "critical_stress_x",
            "critical_stress_y",
            "critical_stress_z",
            "critical_abaqus_element",
            "critical_abaqus_ip",
            "abaqus_input",
            "embedded_audit",
            "summary_json",
            "material_mass_density_kg_m3",
            "unit_system",
            "density_field_source",
            "active_voxel_count",
            "full_active_voxel_volume",
            "geometry_weighted_shell_volume",
            "material_volume",
            "embedded_total_mass_grams",
            "expected_exported_total_mass_grams",
            "mass_difference_grams",
            "mass_difference_relative",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for case_name, case in self.cases.items():
                m = case["metrics"]
                u_xyz = np.asarray(m["global_max_displacement_coordinates"])
                s_xyz = np.asarray(m["critical_stress_coordinates"])
                writer.writerow(
                    {
                        "case": case_name,
                        "active_elements": m["active_elements"],
                        "active_nodes": m["active_nodes"],
                        "integration_point_records": m["integration_point_records"],
                        "compliance": m["compliance"],
                        "strain_energy": m["strain_energy"],
                        "maximum_displacement_magnitude": m[
                            "global_max_displacement_magnitude"
                        ],
                        "maximum_displacement_x": u_xyz[0],
                        "maximum_displacement_y": u_xyz[1],
                        "maximum_displacement_z": u_xyz[2],
                        "maximum_ip_von_mises": m[
                            "maximum_integration_point_von_mises"
                        ],
                        "critical_stress_x": s_xyz[0],
                        "critical_stress_y": s_xyz[1],
                        "critical_stress_z": s_xyz[2],
                        "critical_abaqus_element": m[
                            "critical_abaqus_element_label"
                        ],
                        "critical_abaqus_ip": m[
                            "critical_abaqus_integration_point"
                        ],
                        "abaqus_input": m["abaqus_input_path"],
                        "embedded_audit": m["embedded_audit_path"],
                        "summary_json": m.get("summary_json_path"),
                        "material_mass_density_kg_m3": m.get(
                            "material_mass_density_kg_m3"
                        ),
                        "unit_system": m.get("unit_system"),
                        "density_field_source": m.get("density_field_source"),
                        "active_voxel_count": m.get("active_voxel_count"),
                        "full_active_voxel_volume": m.get(
                            "full_active_voxel_volume"
                        ),
                        "geometry_weighted_shell_volume": m.get(
                            "geometry_weighted_shell_volume"
                        ),
                        "material_volume": m.get("material_volume"),
                        "embedded_total_mass_grams": m.get(
                            "embedded_total_mass_grams"
                        ),
                        "expected_exported_total_mass_grams": m.get(
                            "expected_exported_total_mass_grams"
                        ),
                        "mass_difference_grams": m.get("mass_difference_grams"),
                        "mass_difference_relative": m.get(
                            "mass_difference_relative"
                        ),
                    }
                )
        if self.verbose:
            print("Written:", path.resolve())
        return str(path.resolve())

    def write_summary_json(self, filename: str = "embedded_case_summary.json") -> str:
        path = self.output_dir / filename
        payload = {
            case_name: case["metrics"]
            for case_name, case in self.cases.items()
        }
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self._json_safe(payload), stream, indent=2, sort_keys=True)
        if self.verbose:
            print("Written:", path.resolve())
        return str(path.resolve())

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return VoxelFEMAbaqusRunner._json_safe(value.detach().cpu().numpy())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): VoxelFEMAbaqusRunner._json_safe(current)
                for key, current in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [VoxelFEMAbaqusRunner._json_safe(current) for current in value]
        return value

    def _write_embedded_audit(
        self,
        *,
        case_name: str,
        phi: torch.Tensor,
        theta: torch.Tensor,
        orientation_matrix: torch.Tensor | None,
        compliance: torch.Tensor,
        strain_energy: torch.Tensor,
    ) -> Dict[str, Any]:
        """Print and write maxima from the current Embedded FEM solution."""

        fe_solver = self.embedded_fem.fe
        fe_mesh = fe_solver.mesh
        device = fe_solver.u.device
        dtype = fe_solver.u.dtype

        active_nodes = torch.as_tensor(
            fe_mesh.active_node_ids, dtype=torch.long, device=device
        )
        active_elements = torch.as_tensor(
            fe_mesh.active_element_ids, dtype=torch.long, device=device
        )
        u_nodes = fe_solver.u[:, 0].reshape(fe_mesh.numNodes, 3)
        active_u = u_nodes[active_nodes]
        active_u_magnitude = torch.linalg.norm(active_u, dim=1)
        max_u_rank = torch.argmax(active_u_magnitude)
        max_u_internal_node = active_nodes[max_u_rank]
        max_u_abaqus_node = max_u_rank + 1

        node_xyz = torch.as_tensor(fe_mesh.nodeXYZ, dtype=dtype, device=device)
        max_u_coordinates = node_xyz[max_u_internal_node]
        max_u_vector = u_nodes[max_u_internal_node]
        max_u_magnitude = active_u_magnitude[max_u_rank]

        stress_vm_ip = fe_solver.stress_vm_ip_active
        sigma_ip = fe_solver.sigma_ip_active
        number_of_ip = int(stress_vm_ip.shape[1])
        flat_index = torch.argmax(stress_vm_ip.reshape(-1))
        critical_active_rank = torch.div(
            flat_index, number_of_ip, rounding_mode="floor"
        )
        critical_ip_index = flat_index.remainder(number_of_ip)
        critical_internal_element = active_elements[critical_active_rank]
        critical_abaqus_element = critical_active_rank + 1
        critical_abaqus_ip = critical_ip_index + 1
        max_stress = stress_vm_ip[critical_active_rank, critical_ip_index]
        stress_global = sigma_ip[critical_active_rank, critical_ip_index]

        stress_material = _global_to_material_abaqus_order(
            stress_global,
            phi[critical_internal_element],
            theta[critical_internal_element],
            (
                None
                if orientation_matrix is None
                else orientation_matrix[critical_internal_element]
            ),
        )
        element_centroid = torch.as_tensor(
            fe_mesh.elemCenters[int(critical_internal_element.detach().cpu())],
            dtype=dtype,
            device=device,
        )
        if not hasattr(fe_solver, "critical_ip_coordinates"):
            raise AttributeError(
                "FE solver must expose critical_ip_coordinates for location audit."
            )
        critical_ip_coordinates = fe_solver.critical_ip_coordinates

        audit_path = self.output_dir / (
            case_name + "_embedded_general_audit.txt"
        )
        lines = [
            "=== EMBEDDED FEM GENERAL VOXEL AUDIT ===",
            "Case: %s" % case_name,
            "Active nodes: %d" % int(active_nodes.numel()),
            "Active elements: %d" % int(active_elements.numel()),
            "Raw integration-point stress records: %d" % int(stress_vm_ip.numel()),
            "",
            "--- MAXIMUM DISPLACEMENT MAGNITUDE ---",
            "Internal node ID (zero based): %d" % int(max_u_internal_node),
            "Expected Abaqus node label: %d" % int(max_u_abaqus_node),
            "Coordinates: %s" % _format_vector(max_u_coordinates),
            "Displacement vector [U1,U2,U3]: %s" % _format_vector(max_u_vector),
            "Maximum displacement magnitude: %.12g" % float(max_u_magnitude),
            "",
            "--- COMPLIANCE ---",
            "Compliance F^T U: %.12g" % float(compliance),
            "Strain energy 0.5 F^T U: %.12g" % float(strain_energy),
            "",
            "--- MAXIMUM RAW INTEGRATION-POINT STRESS ---",
            "Structured element ID (zero based): %d"
            % int(critical_internal_element),
            "Expected Abaqus element label: %d" % int(critical_abaqus_element),
            "Expected Abaqus integration point: %d" % int(critical_abaqus_ip),
            "Element centroid: %s" % _format_vector(element_centroid),
            "Integration-point coordinates: %s"
            % _format_vector(critical_ip_coordinates),
            "Global components [Sxx,Syy,Szz,Syz,Sxz,Sxy]: %s"
            % _format_vector(stress_global),
            "Material components in Abaqus order [S11,S22,S33,S12,S13,S23]: %s"
            % _format_vector(stress_material),
            "Maximum integration-point von Mises: %.12g" % float(max_stress),
        ]
        text = "\n".join(lines) + "\n"
        #audit_path.write_text(text, encoding="utf-8")
        if self.verbose:
            print("\n" + text.rstrip())
            print("Written:", audit_path.resolve())

        return {
            "output_path": str(audit_path.resolve()),
            "maximum_displacement_magnitude": float(max_u_magnitude.detach().cpu()),
            "maximum_displacement_internal_node_id": int(max_u_internal_node),
            "maximum_displacement_abaqus_node_label": int(max_u_abaqus_node),
            "maximum_displacement_coordinates": _as_numpy(max_u_coordinates),
            "maximum_displacement_vector": _as_numpy(max_u_vector),
            "compliance": float(compliance.detach().cpu()),
            "strain_energy": float(strain_energy.detach().cpu()),
            "maximum_integration_point_von_mises": float(max_stress.detach().cpu()),
            "critical_internal_element_id": int(critical_internal_element),
            "critical_abaqus_element_label": int(critical_abaqus_element),
            "critical_abaqus_integration_point": int(critical_abaqus_ip),
            "critical_element_centroid": _as_numpy(element_centroid),
            "critical_integration_point_coordinates": _as_numpy(
                critical_ip_coordinates
            ),
            "critical_stress_global": _as_numpy(stress_global),
            "critical_stress_material_abaqus_order": _as_numpy(stress_material),
        }


# Backward-compatible name for notebooks that already imported the old class.
EmbeddedVoxelAbaqusVerification = VoxelFEMAbaqusRunner


def run_solid_orientation_case(
    verification: Any,
    fibre_vector: Sequence[float],
    case_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around :meth:`run_uniform_case`."""

    if isinstance(verification, VoxelFEMAbaqusRunner):
        return verification.run_uniform_case(
            fibre_vector=fibre_vector, case_name=case_name, **kwargs
        )

    runner = VoxelFEMAbaqusRunner(
        face_mesh=verification.face_mesh,
        shell_problem=verification.shell_problem,
        device=verification.device,
        embedded_fem=verification.embedded_fem,
    )
    metrics = runner.run_uniform_case(
        fibre_vector=fibre_vector, case_name=case_name, **kwargs
    )
    if not hasattr(verification, "solid_cases"):
        verification.solid_cases = {}
    verification.solid_cases[case_name] = runner.cases[case_name]
    return metrics


__all__ = [
    "VoxelFEMAbaqusRunner",
    "EmbeddedVoxelAbaqusVerification",
    "run_solid_orientation_case",
]
