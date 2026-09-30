from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


SUPPORTED_UNIT_SYSTEMS = {"N-mm-s"}


def as_numpy_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def kg_m3_to_model_density(material_mass_density_kg_m3: float, unit_system: str) -> float:
    unit_system = str(unit_system)
    if unit_system not in SUPPORTED_UNIT_SYSTEMS:
        raise ValueError(
            f"Unsupported unit_system {unit_system!r}; supported values: "
            f"{sorted(SUPPORTED_UNIT_SYSTEMS)}."
        )
    density = float(material_mass_density_kg_m3)
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("material_mass_density_kg_m3 must be finite and positive.")
    if unit_system == "N-mm-s":
        return density * 1.0e-12
    raise AssertionError("Unhandled supported unit system.")


def model_mass_to_grams(model_mass: float, unit_system: str) -> float:
    unit_system = str(unit_system)
    if unit_system not in SUPPORTED_UNIT_SYSTEMS:
        raise ValueError(
            f"Unsupported unit_system {unit_system!r}; supported values: "
            f"{sorted(SUPPORTED_UNIT_SYSTEMS)}."
        )
    if unit_system == "N-mm-s":
        return float(model_mass) * 1.0e6
    raise AssertionError("Unhandled supported unit system.")


def calculate_voxel_mass_properties(
    *,
    design_density: Any,
    geometric_fraction: Any,
    active_element_ids: Any,
    voxel_dimensions: Any,
    material_mass_density_kg_m3: float | None,
    unit_system: str = "N-mm-s",
    density_field_source: str = "unavailable",
    bin_ids: Any | None = None,
) -> dict[str, Any]:
    """Calculate physical mass from relaxed voxel material volume.

    ``material_mass_density_kg_m3`` is the density of the actual solid material.
    For composite struts this should be the composite density of the carbon
    fibre plus matrix system, not the dimensionless design density.
    """

    unit_system = str(unit_system)
    if unit_system not in SUPPORTED_UNIT_SYSTEMS:
        raise ValueError(
            f"Unsupported unit_system {unit_system!r}; supported values: "
            f"{sorted(SUPPORTED_UNIT_SYSTEMS)}."
        )

    active_ids = as_numpy_array(active_element_ids, np.int64).reshape(-1)
    density = as_numpy_array(design_density, np.float64).reshape(-1)
    geom = as_numpy_array(geometric_fraction, np.float64).reshape(-1)
    voxel_dims = as_numpy_array(voxel_dimensions, np.float64).reshape(3)

    if density.size != geom.size:
        raise ValueError("design_density and geometric_fraction sizes differ.")
    if active_ids.size == 0:
        raise ValueError("The active shell element set is empty.")
    if active_ids.min() < 0 or active_ids.max() >= density.size:
        raise ValueError("active_element_ids contains an out-of-range element index.")
    if not np.all(np.isfinite(density)):
        raise ValueError("design_density contains non-finite values.")
    if not np.all(np.isfinite(geom)):
        raise ValueError("geometric_fraction contains non-finite values.")
    # if np.any((density < 0.0) | (density > 1.0)):
    #     raise ValueError("design_density must lie in [0, 1].")
    if np.any((geom < 0.0) | (geom > 1.0)):
        raise ValueError("geometric_fraction must lie in [0, 1].")
    if not np.all(np.isfinite(voxel_dims)) or np.any(voxel_dims <= 0.0):
        raise ValueError("voxel_dimensions must contain three finite positive values.")

    voxel_volume = float(np.prod(voxel_dims))
    active_density = density[active_ids]
    active_geom = geom[active_ids]
    full_active_voxel_volume = float(active_ids.size * voxel_volume)
    geometry_weighted_shell_volume = float(np.sum(active_geom * voxel_volume))
    material_volume = float(np.sum(active_density * active_geom * voxel_volume))

    report: dict[str, Any] = {
        "mass_available": material_mass_density_kg_m3 is not None,
        "material_mass_density_kg_m3": (
            None if material_mass_density_kg_m3 is None else float(material_mass_density_kg_m3)
        ),
        "material_mass_density_units": "kg/m^3",
        "unit_system": unit_system,
        "model_mass_unit": "tonne" if unit_system == "N-mm-s" else None,
        "model_density_unit": "tonne/mm^3" if unit_system == "N-mm-s" else None,
        "density_field_source": str(density_field_source),
        "active_voxel_count": int(active_ids.size),
        "voxel_dimensions": voxel_dims,
        "voxel_volume": voxel_volume,
        "full_active_voxel_volume": full_active_voxel_volume,
        "geometry_weighted_shell_volume": geometry_weighted_shell_volume,
        "material_volume": material_volume,
        "mass_density_bin_averaging_note": (
            "Averaging density within stiffness-material bins preserves total "
            "mass but approximates spatial mass distribution; centres of mass "
            "and inertias are not guaranteed exact."
        ),
    }

    if material_mass_density_kg_m3 is None:
        report.update(
            {
                "material_mass_density_model_units": None,
                "embedded_total_mass_model_units": None,
                "embedded_total_mass_grams": None,
                "expected_exported_total_mass_model_units": None,
                "expected_exported_total_mass_grams": None,
                "mass_difference_grams": None,
                "mass_difference_relative": None,
                "active_effective_mass_density_model_units": None,
                "bin_mass_densities_model_units": None,
                "bin_expected_mass_model_units": None,
            }
        )
        return report

    solid_density_model = kg_m3_to_model_density(
        float(material_mass_density_kg_m3),
        unit_system,
    )
    active_effective_density = solid_density_model * active_density * active_geom
    embedded_mass_model = float(np.sum(active_effective_density * voxel_volume))
    embedded_mass_grams = model_mass_to_grams(embedded_mass_model, unit_system)

    report.update(
        {
            "material_mass_density_model_units": solid_density_model,
            "embedded_total_mass_model_units": embedded_mass_model,
            "embedded_total_mass_grams": embedded_mass_grams,
            "active_effective_mass_density_model_units": active_effective_density,
        }
    )

    if bin_ids is None:
        report.update(
            {
                "bin_mass_densities_model_units": None,
                "bin_expected_mass_model_units": None,
                "expected_exported_total_mass_model_units": embedded_mass_model,
                "expected_exported_total_mass_grams": embedded_mass_grams,
                "mass_difference_grams": 0.0,
                "mass_difference_relative": 0.0,
            }
        )
        return report

    active_bin_ids = as_numpy_array(bin_ids, np.int64).reshape(-1)
    if active_bin_ids.size != active_ids.size:
        raise ValueError("bin_ids must contain one value per active element.")
    if active_bin_ids.size == 0 or active_bin_ids.min() < 0:
        raise ValueError("bin_ids must be non-empty and non-negative.")
    bin_count = int(active_bin_ids.max()) + 1
    bin_density = np.zeros(bin_count, dtype=np.float64)
    bin_mass = np.zeros(bin_count, dtype=np.float64)
    for bin_id in range(bin_count):
        mask = active_bin_ids == bin_id
        if not np.any(mask):
            continue
        bin_mass[bin_id] = float(np.sum(active_effective_density[mask] * voxel_volume))
        bin_volume = float(np.sum(np.full(np.count_nonzero(mask), voxel_volume)))
        bin_density[bin_id] = 0.0 if bin_mass[bin_id] == 0.0 else bin_mass[bin_id] / bin_volume

    exported_mass_model = float(np.sum(bin_density[active_bin_ids] * voxel_volume))
    exported_mass_grams = model_mass_to_grams(exported_mass_model, unit_system)
    diff_grams = abs(exported_mass_grams - embedded_mass_grams)
    rel_diff = diff_grams / max(abs(embedded_mass_grams), 1.0e-300)
    report.update(
        {
            "bin_mass_densities_model_units": bin_density,
            "bin_expected_mass_model_units": bin_mass,
            "expected_exported_total_mass_model_units": exported_mass_model,
            "expected_exported_total_mass_grams": exported_mass_grams,
            "mass_difference_grams": diff_grams,
            "mass_difference_relative": rel_diff,
        }
    )
    return report
