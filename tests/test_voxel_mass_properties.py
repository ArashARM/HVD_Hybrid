import math
import re
from types import SimpleNamespace

import numpy as np
import torch

from Utils.ExportAbaqus_VoxelBased import export_abaqus_voxel_fem
from Utils.VoxelMassProperties import (
    calculate_voxel_mass_properties,
    kg_m3_to_model_density,
    model_mass_to_grams,
)


def test_mass_fully_solid_voxels_and_unit_conversions():
    report = calculate_voxel_mass_properties(
        design_density=np.ones(2),
        geometric_fraction=np.ones(2),
        active_element_ids=np.array([0, 1]),
        voxel_dimensions=np.array([2.0, 3.0, 4.0]),
        material_mass_density_kg_m3=1600.0,
        unit_system="N-mm-s",
        density_field_source="fem_result",
    )

    assert report["material_volume"] == 48.0
    assert report["material_mass_density_model_units"] == 1600.0e-12
    assert report["embedded_total_mass_model_units"] == 48.0 * 1600.0e-12
    assert report["embedded_total_mass_grams"] == 0.0768
    assert kg_m3_to_model_density(1600.0, "N-mm-s") == 1600.0e-12
    assert model_mass_to_grams(1.0e-9, "N-mm-s") == 1.0e-3


def test_mass_mixed_density_and_partial_geometry():
    report = calculate_voxel_mass_properties(
        design_density=np.array([0.0, 0.5, 1.0]),
        geometric_fraction=np.array([1.0, 0.25, 0.5]),
        active_element_ids=np.array([0, 1, 2]),
        voxel_dimensions=np.array([10.0, 2.0, 1.0]),
        material_mass_density_kg_m3=2000.0,
        unit_system="N-mm-s",
    )

    assert report["geometry_weighted_shell_volume"] == 35.0
    assert report["material_volume"] == 12.5
    assert report["embedded_total_mass_grams"] == 0.025


def test_omitted_physical_density_preserves_unavailable_mass():
    report = calculate_voxel_mass_properties(
        design_density=np.array([1.0]),
        geometric_fraction=np.array([1.0]),
        active_element_ids=np.array([0]),
        voxel_dimensions=np.array([1.0, 1.0, 1.0]),
        material_mass_density_kg_m3=None,
    )

    assert report["mass_available"] is False
    assert report["embedded_total_mass_grams"] is None
    assert report["expected_exported_total_mass_grams"] is None


def test_multiple_material_bins_preserve_total_mass():
    report = calculate_voxel_mass_properties(
        design_density=np.array([0.0, 0.5, 1.0, 0.25]),
        geometric_fraction=np.array([1.0, 0.5, 1.0, 0.5]),
        active_element_ids=np.array([0, 1, 2, 3]),
        voxel_dimensions=np.array([1.0, 2.0, 3.0]),
        material_mass_density_kg_m3=1000.0,
        unit_system="N-mm-s",
        bin_ids=np.array([0, 1, 1, 0]),
    )

    assert report["mass_difference_grams"] < 1.0e-15
    assert report["mass_difference_relative"] < 1.0e-12
    assert report["bin_mass_densities_model_units"][0] == 0.0625e-9
    assert report["bin_mass_densities_model_units"][1] == 0.625e-9


def _tiny_problem(density, geom_fraction):
    density = torch.as_tensor(density, dtype=torch.float64)
    geom_fraction = torch.as_tensor(geom_fraction, dtype=torch.float64)
    occupancy = np.ones(int(density.numel()), dtype=bool)
    nelx = int(density.numel())
    nely = 1
    nelz = 1
    node_coords = []
    for iz in range(nelz + 1):
        for ix in range(nelx + 1):
            for iy in range(nely + 1):
                node_coords.append((float(ix), float(iy), float(iz)))
    force = np.zeros(((nelx + 1) * (nely + 1) * (nelz + 1), 3))
    force[-1, 0] = 1.0

    def build_fem_fields_from_decoder_torch(rho_surface, fiber_surface):
        count = density.numel()
        phi = torch.zeros(count, dtype=torch.float64)
        theta = torch.full((count,), math.pi / 2.0, dtype=torch.float64)
        axes = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(count, 1, 1)
        return {
            "density": density,
            "shell_occupancy": torch.ones(count, dtype=torch.float64),
            "elem_geom_fraction": geom_fraction,
            "phi": phi,
            "theta": theta,
            "orientation_matrix": axes,
        }

    return SimpleNamespace(
        points_xyz=np.zeros((1, 3), dtype=np.float64),
        elem_occupancy=occupancy,
        node_coords=np.asarray(node_coords, dtype=np.float64),
        mesh={"nelx": nelx, "nely": nely, "nelz": nelz, "elemSize": np.array([1.0, 1.0, 1.0])},
        boundaryCondition={
            "force": force,
            "fixed": np.array([0, 1, 2], dtype=np.int64),
            "numDOFPerNode": 3,
        },
        materialProperty={
            "unit_system": "N-mm-s",
            "material_E1": 10.0,
            "material_E2": 20.0,
            "material_E3": 30.0,
            "material_nu12": 0.1,
            "material_nu13": 0.2,
            "material_nu23": 0.3,
            "material_G12": 4.0,
            "material_G13": 5.0,
            "material_G23": 6.0,
        },
        build_fem_fields_from_decoder_torch=build_fem_fields_from_decoder_torch,
    )


def _material_blocks(text):
    return re.findall(r"(\*MATERIAL, NAME=HVD_MAT_\d{3}.*?)(?=\*MATERIAL|\*\*)", text, flags=re.S)


def _reconstruct_mass_from_inp(text, voxel_volume=1.0):
    set_members = {}
    for name, body in re.findall(
        r"\*ELSET, ELSET=(HVD_BIN_\d{3})\n(.*?)(?=\*SOLID SECTION)",
        text,
        flags=re.S,
    ):
        set_members[name] = [
            int(value)
            for value in re.findall(r"\d+", body)
        ]
    material_density = {}
    for material_name, body in re.findall(
        r"\*MATERIAL, NAME=(HVD_MAT_\d{3})\n(.*?)(?=\*MATERIAL|\*\*)",
        text,
        flags=re.S,
    ):
        density_match = re.search(r"\*DENSITY\n([0-9.eE+-]+)", body)
        if density_match:
            material_density[material_name] = float(density_match.group(1))
    reconstructed = 0.0
    for bin_name, members in set_members.items():
        material_name = bin_name.replace("BIN", "MAT")
        reconstructed += material_density[material_name] * len(members) * voxel_volume
    return reconstructed


def test_export_omitted_density_writes_no_density_keyword(tmp_path):
    problem = _tiny_problem([1.0, 0.5], [1.0, 1.0])
    output = tmp_path / "no_mass.inp"

    report = export_abaqus_voxel_fem(
        fields={"rho": torch.ones(1), "fiber3d": torch.tensor([[1.0, 0.0, 0.0]])},
        shell_problem=problem,
        output_path=output,
        material_bins=1,
    )

    text = output.read_text(encoding="utf-8")
    assert "*DENSITY" not in text
    assert report["mass_available"] is False
    assert report["embedded_total_mass_grams"] is None


def test_export_density_bins_preserve_mass_and_elastic_assignments(tmp_path):
    problem = _tiny_problem([0.0, 0.5, 1.0], [1.0, 0.5, 1.0])
    output = tmp_path / "mass.inp"
    fem_result = {
        "density_field": torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        "stiffness_factor_field": torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64),
    }

    report = export_abaqus_voxel_fem(
        fields={"rho": torch.ones(1), "fiber3d": torch.tensor([[1.0, 0.0, 0.0]])},
        shell_problem=problem,
        output_path=output,
        fem_result=fem_result,
        material_bins=2,
        material_mass_density_kg_m3=1000.0,
        unit_system="N-mm-s",
    )

    text = output.read_text(encoding="utf-8")
    blocks = _material_blocks(text)
    assert len(blocks) == 2
    assert "*DENSITY" in text
    assert "1, 2\n" in text
    assert "3\n" in text
    assert "1.5, 3, 4.5, 0.1, 0.2, 0.3, 0.6, 0.75" in text
    assert "4, 8, 12, 0.1, 0.2, 0.3, 1.6, 2" in text

    reconstructed = _reconstruct_mass_from_inp(text)
    for density_value in report["bin_mass_densities_model_units"]:
        assert density_value >= 0.0
    assert np.isclose(
        reconstructed,
        report["expected_exported_total_mass_model_units"],
        rtol=0.0,
        atol=1.0e-18,
    )
    assert report["embedded_total_mass_grams"] == report["expected_exported_total_mass_grams"]
    assert report["mass_difference_grams"] < 1.0e-15
