"""Embedded H8 integration-point stress audit for the solid Abaqus benchmarks.

Paste/import this helper before ``run_solid_orientation_case`` and call it
immediately after the embedded FEM solve.  The CSV uses the same sequential
active-element labels written by ``export_abaqus_voxel_fem``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _von_mises_from_embedded_components(sigma):
    """Return von Mises for [S11,S22,S33,S23,S13,S12]."""
    s11, s22, s33 = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    s23, s13, s12 = sigma[..., 3], sigma[..., 4], sigma[..., 5]
    vm2 = (
        0.5
        * (
            (s11 - s22).square()
            + (s22 - s33).square()
            + (s33 - s11).square()
        )
        + 3.0 * (s12.square() + s23.square() + s13.square())
    )
    return torch.sqrt(torch.clamp(vm2, min=0.0))


def embedded_to_abaqus_stress_order(sigma):
    """Convert [S11,S22,S33,S23,S13,S12] to Abaqus [S11,S22,S33,S12,S13,S23]."""
    if isinstance(sigma, torch.Tensor):
        return sigma[..., [0, 1, 2, 5, 4, 3]]
    return np.asarray(sigma)[..., [0, 1, 2, 5, 4, 3]]


def _soft_p_norm(values, p=12.0, eps=1.0e-12):
    """Same scaled mean p-norm used by Training/Loss_FEM.py."""
    flat = values.reshape(-1).abs()
    if flat.numel() == 0:
        raise ValueError("The active stress field is empty.")
    scale = flat.detach().max().clamp_min(float(eps))
    return scale * (flat / scale).pow(float(p)).mean().pow(1.0 / float(p))


def export_embedded_integration_point_stress_audit(
    *,
    fe_solver,
    density,
    stiffness_factor,
    phi,
    theta,
    case_name,
    output_dir="FEM_Verification",
    p_norm=12.0,
):
    """Audit and export embedded stresses at all eight C3D8 integration points.

    This function must be called before another FEM case overwrites
    ``fe_solver.u``, ``fe_solver.sigma_ip`` and ``fe_solver.stress_vm_ip``.
    """
    mesh = fe_solver.mesh
    required = ("sigma_ip", "stress_vm_ip", "stress_vm_element_max", "u")
    missing = [name for name in required if not hasattr(fe_solver, name)]
    if missing:
        raise AttributeError(
            "Run the embedded FEM solve first. Missing FE fields: %s"
            % ", ".join(missing)
        )

    active_ids_np = np.asarray(mesh.active_element_ids, dtype=np.int64)
    if active_ids_np.size == 0:
        raise ValueError("The FEM mesh has no active elements.")

    device = fe_solver.sigmaElem.device
    active_ids = torch.as_tensor(active_ids_np, dtype=torch.long, device=device)

    sigma = fe_solver.sigma_ip[active_ids]
    vm_stored = fe_solver.stress_vm_ip[active_ids]
    vm_recomputed = _von_mises_from_embedded_components(sigma)
    vm_difference = (vm_stored - vm_recomputed).abs().max()

    if not torch.allclose(vm_stored, vm_recomputed, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(
            "Stored and independently recomputed von Mises fields disagree; "
            "maximum absolute difference = %.9g" % float(vm_difference.detach().cpu())
        )

    maximum_flat = int(torch.argmax(vm_stored.reshape(-1)).item())
    maximum_active_rank = maximum_flat // 8
    maximum_ip_zero_based = maximum_flat % 8
    maximum_structured_id = int(active_ids_np[maximum_active_rank])

    # The voxel exporter writes active elements in this same order with labels
    # 1, 2, ..., number_of_active_elements.
    abaqus_labels = np.arange(1, active_ids_np.size + 1, dtype=np.int64)
    maximum_abaqus_label = int(abaqus_labels[maximum_active_rank])

    elem_nodes = np.asarray(mesh.elemNodes, dtype=np.int64)[active_ids_np]
    node_xyz = np.asarray(mesh.nodeXYZ, dtype=np.float64)
    centroids = node_xyz[elem_nodes].mean(axis=1)
    gauss_half = fe_solver.gauss_points.detach().cpu().numpy()
    elem_size = np.asarray(mesh.elemSize, dtype=np.float64).reshape(1, 1, 3)
    ip_coordinates = centroids[:, None, :] + gauss_half[None, :, :] * elem_size

    sigma_np = sigma.detach().cpu().numpy()
    sigma_abaqus_order_np = embedded_to_abaqus_stress_order(sigma).detach().cpu().numpy()
    vm_np = vm_stored.detach().cpu().numpy()
    density_np = density.reshape(-1)[active_ids].detach().cpu().numpy()
    stiffness_np = (
        stiffness_factor.reshape(-1)[active_ids].detach().cpu().numpy()
    )
    phi_np = phi.reshape(-1)[active_ids].detach().cpu().numpy()
    theta_np = theta.reshape(-1)[active_ids].detach().cpu().numpy()

    rows = []
    for active_rank in range(active_ids_np.size):
        for ip in range(8):
            rows.append(
                {
                    "abaqus_element_label": int(abaqus_labels[active_rank]),
                    "abaqus_integration_point": ip + 1,
                    "embedded_active_rank_zero_based": active_rank,
                    "structured_element_id_zero_based": int(active_ids_np[active_rank]),
                    "centroid_x": centroids[active_rank, 0],
                    "centroid_y": centroids[active_rank, 1],
                    "centroid_z": centroids[active_rank, 2],
                    "ip_x": ip_coordinates[active_rank, ip, 0],
                    "ip_y": ip_coordinates[active_rank, ip, 1],
                    "ip_z": ip_coordinates[active_rank, ip, 2],
                    "S11_embedded": sigma_np[active_rank, ip, 0],
                    "S22_embedded": sigma_np[active_rank, ip, 1],
                    "S33_embedded": sigma_np[active_rank, ip, 2],
                    "S23_embedded": sigma_np[active_rank, ip, 3],
                    "S13_embedded": sigma_np[active_rank, ip, 4],
                    "S12_embedded": sigma_np[active_rank, ip, 5],
                    "S11_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 0],
                    "S22_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 1],
                    "S33_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 2],
                    "S12_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 3],
                    "S13_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 4],
                    "S23_abaqus_order": sigma_abaqus_order_np[active_rank, ip, 5],
                    "von_mises_integration_point": vm_np[active_rank, ip],
                    "density": density_np[active_rank],
                    "stiffness_factor": stiffness_np[active_rank],
                    "phi": phi_np[active_rank],
                    "theta": theta_np[active_rank],
                }
            )

    table = pd.DataFrame(rows)
    output_path = Path(output_dir) / ("%s_embedded_ip_stress.csv" % case_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    stress_p_norm_mean_diagnostic = _soft_p_norm(vm_stored, p=p_norm)
    maximum_sigma = sigma[maximum_active_rank, maximum_ip_zero_based]
    maximum_centroid = centroids[maximum_active_rank]
    maximum_ip_coordinates = ip_coordinates[maximum_active_rank, maximum_ip_zero_based]

    result = {
        "recovery_position": "8 C3D8 integration points per active element",
        "component_order": ["S11", "S22", "S33", "S23", "S13", "S12"],
        "abaqus_component_order": ["S11", "S22", "S33", "S12", "S13", "S23"],
        "number_of_active_elements": int(active_ids_np.size),
        "number_of_integration_point_records": int(active_ids_np.size * 8),
        "maximum_integration_point_von_mises": float(vm_stored.reshape(-1)[maximum_flat].detach().cpu()),
        "stress_p_norm_mean_diagnostic": float(stress_p_norm_mean_diagnostic.detach().cpu()),
        "p_norm_exponent": float(p_norm),
        "maximum_structured_element_id_zero_based": maximum_structured_id,
        "maximum_active_rank_zero_based": maximum_active_rank,
        "maximum_abaqus_element_label": maximum_abaqus_label,
        "maximum_abaqus_integration_point": maximum_ip_zero_based + 1,
        "maximum_element_centroid": maximum_centroid.copy(),
        "maximum_integration_point_coordinates": maximum_ip_coordinates.copy(),
        "maximum_stress_components": maximum_sigma.detach().cpu().numpy(),
        "maximum_stress_components_abaqus_order": embedded_to_abaqus_stress_order(
            maximum_sigma
        ).detach().cpu().numpy(),
        "stored_vm_recalculation_max_difference": float(
            vm_difference.detach().cpu()
        ),
        "csv_path": str(output_path.resolve()),
    }

    print("\n=== EMBEDDED INTEGRATION-POINT STRESS AUDIT ===")
    print("Case:", case_name)
    print("Recovery position:", result["recovery_position"])
    print("Maximum IP von Mises:", result["maximum_integration_point_von_mises"])
    print("Mean p-norm diagnostic (p=%g):" % p_norm, result["stress_p_norm_mean_diagnostic"])
    print(
        "Structured element ID (zero based):",
        result["maximum_structured_element_id_zero_based"],
    )
    print("Expected Abaqus element label:", maximum_abaqus_label)
    print("Expected Abaqus integration point:", result["maximum_abaqus_integration_point"])
    print("Element centroid:", result["maximum_element_centroid"])
    print("Integration-point coordinates:", result["maximum_integration_point_coordinates"])
    print("Components [S11,S22,S33,S23,S13,S12]:")
    print(result["maximum_stress_components"])
    print("Components in Abaqus order [S11,S22,S33,S12,S13,S23]:")
    print(result["maximum_stress_components_abaqus_order"])
    print(
        "Stored/recomputed von Mises maximum difference:",
        result["stored_vm_recalculation_max_difference"],
    )
    print("Full active element/IP CSV:", result["csv_path"])

    return result


def export_embedded_centroid_stress_audit(*args, **kwargs):
    """Deprecated alias; exports integration-point stresses."""
    return export_embedded_integration_point_stress_audit(*args, **kwargs)


def print_stress_case_summary(*metrics_dicts):
    """Print a compact summary after X, Y and XY45 cases have run."""
    rows = []
    for metrics in metrics_dicts:
        audit = metrics["stress_audit"]
        rows.append(
            {
                "case": Path(metrics["output_path"]).stem,
                "maximum_integration_point_von_mises": audit["maximum_integration_point_von_mises"],
                "stress_p_norm_mean_diagnostic": audit["stress_p_norm_mean_diagnostic"],
                "abaqus_element_label": audit["maximum_abaqus_element_label"],
                "abaqus_integration_point": audit["maximum_abaqus_integration_point"],
                "centroid_x": audit["maximum_element_centroid"][0],
                "centroid_y": audit["maximum_element_centroid"][1],
                "centroid_z": audit["maximum_element_centroid"][2],
                "ip_x": audit["maximum_integration_point_coordinates"][0],
                "ip_y": audit["maximum_integration_point_coordinates"][1],
                "ip_z": audit["maximum_integration_point_coordinates"][2],
            }
        )
    summary = pd.DataFrame(rows)
    print("\n=== EMBEDDED STRESS CASE SUMMARY ===")
    print(summary.to_string(index=False))
    return summary


# -------------------------------------------------------------------------
# Add this call inside run_solid_orientation_case immediately after
# `maximum_stress = ...` and before the next FEM case can run:
#
# stress_audit = export_embedded_integration_point_stress_audit(
#     fe_solver=fe_solver,
#     density=density,
#     stiffness_factor=stiffness_factor,
#     phi=phi,
#     theta=theta,
#     case_name=case_name,
#     p_norm=12.0,
# )
#
# Then add this entry to `metrics`:
#     "stress_audit": stress_audit,
#
# Finally, after rerunning all three cases:
# stress_summary = print_stress_case_summary(
#     corrected_x_metrics,
#     corrected_y_metrics,
#     corrected_xy45_metrics,
# )
