"""General Abaqus/CAE audit for an HVD C3D8 voxel model.

The script prints one concise summary in the Abaqus/CAE console and writes the
same summary to one text file.  It deliberately leaves the ODB open.

Run from Abaqus/CAE with File -> Run Script after editing the configuration
block below.  The code is compatible with the Python environment in older
Abaqus/CAE releases and avoids generator expressions in numerical reductions.
"""

from __future__ import print_function

import math
import os

from abaqus import session


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ODB_PATH = "IP_XFiber_Latest.odb"

# None selects the last step or automatically selects the model instance.
STEP_NAME = "STATIC_LOAD"
INSTANCE_NAME = None

# None creates <ODB base name>_general_audit.txt beside the ODB.
OUTPUT_PATH = None

# False: print only the requested comparison values.
# True: print the complete audit that is written to the text file.
VERBOSE = False


def select_last_mapping_value(mapping):
    keys = list(mapping.keys())
    if len(keys) == 0:
        raise ValueError("The requested Abaqus mapping is empty.")
    key = keys[-1]
    return mapping[key], key


def select_instance(assembly, requested_name):
    names = list(assembly.instances.keys())

    if requested_name is not None:
        if requested_name not in assembly.instances:
            raise KeyError(
                "Instance %s was not found. Available instances: %s"
                % (requested_name, names)
            )
        return assembly.instances[requested_name], requested_name

    if len(names) == 1:
        return assembly.instances[names[0]], names[0]

    preferred_names = (
        "HVD_VOXEL-1",
        "CUBE-1",
        "PART-1-1",
    )

    for name in preferred_names:
        if name in assembly.instances:
            return assembly.instances[name], name

    raise ValueError(
        "Multiple instances exist. Set INSTANCE_NAME to one of: %s"
        % names
    )


def open_or_reuse_odb(path):
    """Open through the CAE session and deliberately retain the ODB."""
    requested_path = os.path.normcase(os.path.abspath(path))

    for key in list(session.odbs.keys()):
        candidate = session.odbs[key]
        candidate_path = getattr(candidate, "path", key)
        candidate_path = os.path.normcase(os.path.abspath(candidate_path))
        if candidate_path == requested_path:
            return candidate, True

    return session.openOdb(
        name=os.path.abspath(path),
        readOnly=True,
    ), False


def vector_magnitude(values):
    total = 0.0
    for value in values:
        total += float(value) * float(value)
    return math.sqrt(max(total, 0.0))


def von_mises_from_components(components):
    """Abaqus 3-D stress order: S11,S22,S33,S12,S13,S23."""
    s11 = float(components[0])
    s22 = float(components[1])
    s33 = float(components[2])
    s12 = float(components[3])
    s13 = float(components[4])
    s23 = float(components[5])

    value = (
        0.5
        * (
            (s11 - s22) ** 2
            + (s22 - s33) ** 2
            + (s33 - s11) ** 2
        )
        + 3.0 * (
            s12 ** 2
            + s13 ** 2
            + s23 ** 2
        )
    )

    return math.sqrt(max(value, 0.0))


def format_vector(values):
    return "[%.12g, %.12g, %.12g]" % (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )


def format_stress(values):
    return "[%.12g, %.12g, %.12g, %.12g, %.12g, %.12g]" % (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def element_centroid(instance, element_label, node_coordinates):
    element = instance.getElementFromLabel(
        label=int(element_label)
    )

    centroid = [0.0, 0.0, 0.0]
    count = 0

    for node_label in element.connectivity:
        xyz = node_coordinates[int(node_label)]
        centroid[0] += float(xyz[0])
        centroid[1] += float(xyz[1])
        centroid[2] += float(xyz[2])
        count += 1

    if count == 0:
        raise ValueError(
            "Element %d has no connected nodes."
            % int(element_label)
        )

    centroid[0] /= float(count)
    centroid[1] /= float(count)
    centroid[2] /= float(count)

    return centroid


def c3d8_integration_point_coordinates(
    instance,
    element_label,
    integration_point,
    node_coordinates,
):
    """Interpolate the physical coordinate of an Abaqus C3D8 Gauss point."""
    element = instance.getElementFromLabel(
        label=int(element_label)
    )

    if len(element.connectivity) != 8:
        return None

    a = 1.0 / math.sqrt(3.0)

    # Abaqus C3D8 integration-point numbering.
    integration_points = (
        (-a, -a, -a),
        ( a, -a, -a),
        (-a,  a, -a),
        ( a,  a, -a),
        (-a, -a,  a),
        ( a, -a,  a),
        (-a,  a,  a),
        ( a,  a,  a),
    )

    ip_number = int(integration_point)
    if ip_number < 1 or ip_number > 8:
        return None

    xi, eta, zeta = integration_points[ip_number - 1]

    # Standard Abaqus C3D8 node natural coordinates.
    node_signs = (
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
    )

    xyz = [0.0, 0.0, 0.0]

    for index in range(8):
        sx, sy, sz = node_signs[index]
        shape_value = (
            0.125
            * (1.0 + sx * xi)
            * (1.0 + sy * eta)
            * (1.0 + sz * zeta)
        )

        node_label = int(element.connectivity[index])
        node_xyz = node_coordinates[node_label]

        xyz[0] += shape_value * float(node_xyz[0])
        xyz[1] += shape_value * float(node_xyz[1])
        xyz[2] += shape_value * float(node_xyz[2])

    return xyz


def find_history_value(step, output_name):
    """Return a final whole-model history value when available."""
    candidates = []

    for region_name in list(step.historyRegions.keys()):
        region = step.historyRegions[region_name]
        if output_name not in region.historyOutputs:
            continue

        data = region.historyOutputs[output_name].data
        if len(data) == 0:
            continue

        value = float(data[-1][1])
        preferred = 0
        upper_name = str(region_name).upper()
        if "ASSEMBLY" in upper_name or "WHOLE MODEL" in upper_name:
            preferred = 1

        candidates.append((preferred, region_name, value))

    if len(candidates) == 0:
        return None, None

    candidates.sort()
    selected = candidates[-1]
    return selected[2], selected[1]


def main():
    odb, reused_existing_odb = open_or_reuse_odb(ODB_PATH)

    if STEP_NAME is None:
        step, step_name = select_last_mapping_value(odb.steps)
    else:
        if STEP_NAME not in odb.steps:
            raise KeyError(
                "Step %s was not found. Available steps: %s"
                % (STEP_NAME, list(odb.steps.keys()))
            )
        step = odb.steps[STEP_NAME]
        step_name = STEP_NAME

    frame = step.frames[-1]

    if "U" not in frame.fieldOutputs:
        raise KeyError(
            "Displacement output U is absent from the final frame."
        )

    if "S" not in frame.fieldOutputs:
        raise KeyError(
            "Stress output S is absent from the final frame."
        )

    instance, instance_name = select_instance(
        odb.rootAssembly,
        INSTANCE_NAME,
    )

    node_coordinates = {}
    for node in instance.nodes:
        node_coordinates[int(node.label)] = node.coordinates

    # ---------------------------------------------------------------------
    # Maximum nodal displacement magnitude
    # ---------------------------------------------------------------------

    maximum_displacement = -1.0
    maximum_displacement_node = None
    maximum_displacement_vector = None

    displacement_count = 0

    for value in frame.fieldOutputs["U"].values:
        if value.instance.name != instance_name:
            continue

        magnitude = vector_magnitude(value.data)
        displacement_count += 1

        if magnitude > maximum_displacement:
            maximum_displacement = magnitude
            maximum_displacement_node = int(value.nodeLabel)
            maximum_displacement_vector = [
                float(value.data[0]),
                float(value.data[1]),
                float(value.data[2]),
            ]

    if maximum_displacement_node is None:
        raise ValueError(
            "No nodal displacement values were found for instance %s."
            % instance_name
        )

    maximum_displacement_coordinates = node_coordinates[
        maximum_displacement_node
    ]

    # ---------------------------------------------------------------------
    # Maximum raw integration-point von Mises stress
    # ---------------------------------------------------------------------

    maximum_stress = -1.0
    maximum_stress_element = None
    maximum_stress_ip = None
    maximum_stress_components = None
    stress_count = 0

    for value in frame.fieldOutputs["S"].values:
        if value.instance.name != instance_name:
            continue

        element_label = getattr(value, "elementLabel", None)
        integration_point = getattr(value, "integrationPoint", None)

        if element_label is None or integration_point is None:
            continue

        components = value.data
        if len(components) < 6:
            continue

        try:
            mises = float(value.mises)
        except Exception:
            mises = von_mises_from_components(components)

        stress_count += 1

        if mises > maximum_stress:
            maximum_stress = mises
            maximum_stress_element = int(element_label)
            maximum_stress_ip = int(integration_point)
            maximum_stress_components = [
                float(components[0]),
                float(components[1]),
                float(components[2]),
                float(components[3]),
                float(components[4]),
                float(components[5]),
            ]

    if maximum_stress_element is None:
        raise ValueError(
            "No integration-point stress values were found for instance %s."
            % instance_name
        )

    maximum_stress_centroid = element_centroid(
        instance,
        maximum_stress_element,
        node_coordinates,
    )

    maximum_stress_coordinates = c3d8_integration_point_coordinates(
        instance,
        maximum_stress_element,
        maximum_stress_ip,
        node_coordinates,
    )

    # ---------------------------------------------------------------------
    # Linear-static compliance from the final strain-energy history
    # ---------------------------------------------------------------------

    strain_energy, energy_region = find_history_value(
        step,
        "ALLSE",
    )

    compliance = None
    compliance_method = None

    if strain_energy is not None:
        compliance = 2.0 * strain_energy
        compliance_method = "2 * ALLSE (linear-static relation)"
    else:
        internal_energy, energy_region = find_history_value(
            step,
            "ALLIE",
        )
        if internal_energy is not None:
            strain_energy = internal_energy
            compliance = 2.0 * internal_energy
            compliance_method = (
                "2 * ALLIE fallback (valid for this linear-elastic static model)"
            )

    # ---------------------------------------------------------------------
    # Build one summary for both the console and the text file
    # ---------------------------------------------------------------------

    if OUTPUT_PATH is None:
        output_path = os.path.splitext(
            os.path.abspath(ODB_PATH)
        )[0] + "_general_audit.txt"
    else:
        output_path = os.path.abspath(OUTPUT_PATH)

    lines = []
    lines.append("=== ABAQUS GENERAL VOXEL AUDIT ===")
    lines.append("ODB: %s" % os.path.abspath(ODB_PATH))
    lines.append("ODB reused from CAE session: %s" % reused_existing_odb)
    lines.append("ODB remains open after audit: True")
    lines.append("Instance: %s" % instance_name)
    lines.append("Step: %s" % step_name)
    lines.append("Frame index: %d" % (len(step.frames) - 1))
    lines.append("")

    lines.append("--- MAXIMUM DISPLACEMENT MAGNITUDE ---")
    lines.append("Node label: %d" % maximum_displacement_node)
    lines.append(
        "Coordinates: %s"
        % format_vector(maximum_displacement_coordinates)
    )
    lines.append(
        "Displacement vector [U1,U2,U3]: %s"
        % format_vector(maximum_displacement_vector)
    )
    lines.append(
        "Maximum displacement magnitude: %.12g"
        % maximum_displacement
    )
    lines.append("")

    lines.append("--- COMPLIANCE ---")
    if compliance is None:
        lines.append("Compliance: unavailable")
        lines.append(
            "Reason: neither ALLSE nor ALLIE was found in history output."
        )
    else:
        lines.append("Strain energy: %.12g" % strain_energy)
        lines.append("Compliance: %.12g" % compliance)
        lines.append("Method: %s" % compliance_method)
        lines.append("History region: %s" % energy_region)
    lines.append("")

    lines.append("--- MAXIMUM RAW INTEGRATION-POINT STRESS ---")
    lines.append("Element label: %d" % maximum_stress_element)
    lines.append("Integration point: %d" % maximum_stress_ip)
    lines.append(
        "Element centroid: %s"
        % format_vector(maximum_stress_centroid)
    )
    if maximum_stress_coordinates is None:
        lines.append("Integration-point coordinates: unavailable")
    else:
        lines.append(
            "Integration-point coordinates: %s"
            % format_vector(maximum_stress_coordinates)
        )
    lines.append(
        "Stress components [S11,S22,S33,S12,S13,S23]: %s"
        % format_stress(maximum_stress_components)
    )
    lines.append(
        "Maximum integration-point von Mises: %.12g"
        % maximum_stress
    )
    lines.append("")

    lines.append("--- COUNTS ---")
    lines.append("Nodal displacement records: %d" % displacement_count)
    lines.append("Integration-point stress records: %d" % stress_count)
    lines.append(
        "Expected C3D8 stress records from element count: %d"
        % (8 * len(instance.elements))
    )

    output = "\n".join(lines)

    handle = open(output_path, "w")
    handle.write(output + "\n")
    handle.close()

    if VERBOSE:
        print("\n" + output)
        print("\nWritten: %s" % output_path)
        print("ODB left open in Abaqus/CAE session.")
    else:
        print("\n=== ABAQUS: %s ===" % os.path.basename(ODB_PATH))
        print(
            "Maximum displacement magnitude: %.12g at %s"
            % (
                maximum_displacement,
                format_vector(maximum_displacement_coordinates),
            )
        )
        if compliance is None:
            print("Compliance: unavailable")
        else:
            print("Compliance: %.12g" % compliance)

        if maximum_stress_coordinates is None:
            stress_location = "unavailable"
        else:
            stress_location = format_vector(maximum_stress_coordinates)

        print(
            "Maximum raw IP von Mises stress: %.12g at %s"
            % (maximum_stress, stress_location)
        )
        print("Audit report: %s" % output_path)

    # Return the handle and retain it at module scope below.  Do not close it.
    return odb


if __name__ == "__main__":
    OPEN_ODB_HANDLE = main()
