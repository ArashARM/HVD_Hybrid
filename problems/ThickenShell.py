import os.path
import math
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass
import torch

from neuraltomo_fem.anisotropicFE_new import orientation_frame_from_fiber
from .problemBase import problemBase


class ThickenShell(problemBase):
    problemName = 'ThickenShell'

    def __init__(
        self,
        thickness,
        BC_dir,
        Load_magnitude,
        voxel_size,
        extra_layers=1,
        tensors=None,
        tangential_tol=None,
        load_case="tensile_compression",
        load_dir=None,
        load_surface_dir=None,
        load_surface_side="max",
        fixed_side="min",
        force_side=None,
        load_direction_side=None,
        voxelization_mode="triangle_distance",
        subvoxel_samples=1,
        min_geom_fraction=0.25,
        fixed_region=None,
        load_region=None,
        support_regions=None,
        fixed_region_1=None,
        fixed_region_2=None,
        bc_mapping_mode="surface_patch",
        faces_index_base=None,
        bearing_loop_id=None,
        bearing_contact_side="min",
        bearing_pressure_exponent=1.0,
        bearing_band=None,
        fixed_patch_band=None,
    ):
        super().__init__()
        self.name = self.problemName

        self.brep_bbox = None
        self.thickness = float(thickness)
        self.face_bboxes = None
        self.samples_by_face = None
        self.voxel_size = float(voxel_size)
        self.extra_layers = int(extra_layers)
        self.tangential_tol = None if tangential_tol is None else float(tangential_tol)
        self.BC_dir = str(BC_dir).lower()
        self.Load_magnitude = float(Load_magnitude)
        self.load_case = str(load_case).lower()
        self.load_dir = None if load_dir is None else str(load_dir).lower()
        self.load_surface_dir = None if load_surface_dir is None else str(load_surface_dir).lower()
        self.load_surface_side = str(load_surface_side).lower()
        self.fixed_side = str(fixed_side).lower()
        self.force_side = None if force_side is None else str(force_side).lower()
        self.load_direction_side = None if load_direction_side is None else str(load_direction_side).lower()
        self.voxelization_mode = str(voxelization_mode).lower()
        self.subvoxel_samples = int(subvoxel_samples)
        self.min_geom_fraction = float(min_geom_fraction)
        self.fixed_region = fixed_region
        self.load_region = load_region
        self.support_regions = support_regions
        self.fixed_region_1 = fixed_region_1
        self.fixed_region_2 = fixed_region_2
        self.bc_mapping_mode = str(bc_mapping_mode).lower()
        self.faces_index_base = faces_index_base
        self.bearing_loop_id = bearing_loop_id
        self.bearing_contact_side = str(bearing_contact_side).lower()
        self.bearing_pressure_exponent = float(bearing_pressure_exponent)
        self.bearing_band = None if bearing_band is None else float(bearing_band)
        self.fixed_patch_band = None if fixed_patch_band is None else float(fixed_patch_band)

        if self.voxelization_mode not in ("triangle_distance", "legacy_samples"):
            raise ValueError(
                "voxelization_mode must be 'triangle_distance' or 'legacy_samples'."
            )
        if self.bc_mapping_mode not in ("surface_patch", "legacy_bbox"):
            raise ValueError("bc_mapping_mode must be 'surface_patch' or 'legacy_bbox'.")
        if self.bearing_contact_side not in ("min", "max"):
            raise ValueError("bearing_contact_side must be 'min' or 'max'.")
        if self.bearing_pressure_exponent < 0.0:
            raise ValueError("bearing_pressure_exponent must be >= 0.")
        if self.bearing_band is not None and self.bearing_band <= 0.0:
            raise ValueError("bearing_band must be positive.")
        if self.fixed_patch_band is not None and self.fixed_patch_band <= 0.0:
            raise ValueError("fixed_patch_band must be positive.")
        if self.bearing_band is None:
            self.bearing_band = 0.75 * self.voxel_size
        if self.fixed_patch_band is None:
            self.fixed_patch_band = 0.75 * self.voxel_size
        if self.subvoxel_samples < 1:
            raise ValueError("subvoxel_samples must be at least 1.")
        if not (0.0 < self.min_geom_fraction <= 1.0):
            raise ValueError("min_geom_fraction must lie in (0, 1].")
        if self.voxelization_mode == "triangle_distance" and tangential_tol is not None:
            warnings.warn(
                "tangential_tol is ignored in voxelization_mode='triangle_distance'; "
                "it is retained only for legacy_samples.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.grid_geom = None
        self.elem_centers = None
        self.node_coords = None

        self.elem_sample_idx = None
        self.sample_elem_idx = None
        self.elem_sample_count = None
        self.elem_fiber = None
        self.elem_phi = None
        self.elem_theta = None

        self.uv = None
        self.points_xyz = None
        self.face_areas = None
        self.Xu = None
        self.Xv = None
        self.faces_ijk = None
        self.pv_faces = None
        self.face_id = None
        self.boundary_idx_ring1 = None
        self.min_vol_frac = None
        self.sample_normals = None
        self.triangles = None
        self.triangle_face_id = None
        self.boundary_edges = {}
        self.boundary_loops = {}

        self.elem_occupancy = None
        self.elem_geom_fraction = None
        self.elem_closest_triangle_id = None
        self.elem_closest_point = None
        self.elem_barycentric = None
        self.elem_distance_to_surface = None
        self.elem_source_face_id = None
        self.elem_density = None
        self.bc_report = {}
        self._occupied_voxel_mesh_cache = None
        self._surface_cloud_cache = None

        if tensors is None:
            raise ValueError("tensors must be provided at this stage")

        # Get CAD info about the shell geometry and samples from the tensors. Sample: points_xyz, Xu, Xv, face_areas, etc. Also parse the bounding box of the BREP geometry.
        self.set_cad_samples(tensors)
   

        # Build a full structured voxel grid that covers the padded bounding box of the shell.
        # This creates a rectangular grid (mesh), initializes empty boundary conditions,
        # and defines material properties. At this stage NO voxels are trimmed or filtered
        # based on the shell geometry — the grid is still a complete box. The actual shell
        # shape is imposed later by voxelize_shell_from_samples(), which marks which voxels
        # belong to the shell thickness via elem_occupancy.
        self.mesh, self.boundaryCondition, self.materialProperty = self.shellSettings()



        if self.voxelization_mode == "legacy_samples":
            # Legacy sample/normal/tangential-tolerance voxelization. This is
            # kept for old studies only; new work should use triangle_distance.
            self.elem_occupancy,self.elem_sample_idx = self.voxelize_shell_from_samples(self.thickness,tangential_tol=self.tangential_tol)
            self.sample_elem_idx = self.surface_samples_to_element_indices()
            self.ensure_surface_sample_elements_are_occupied()
            self.elem_geom_fraction = self.elem_occupancy.reshape(-1).astype(np.float32)
        else:
            self.voxelize_shell_from_triangles(
                thickness=self.thickness,
                subvoxel_samples=self.subvoxel_samples,
                min_geom_fraction=self.min_geom_fraction,
            )
            self.sample_elem_idx = self.surface_samples_to_element_indices()
        self.elem_sample_count = self.count_surface_samples_per_element()


        # Physical material-density placeholder used for debugging / visualization.
        # It is binary shell occupancy, not a stiffness interpolation factor.
        # The actual density and fiber fields used in the FEM solve will later come
        # from the neural decoder and can be assigned via assign_decoder_fields().
        self.elem_density = self.elem_occupancy.reshape(-1).astype(np.float32)


        self.apply_load_case_boundary_conditions()

    def apply_load_case_boundary_conditions(self):
        """
        Dispatch boundary-condition construction by load-case category.

        Current implemented cases:
        - tensile_compression: fixed slab on the negative side of BC_dir and
          loaded slab on the positive side of BC_dir. The sign of
          Load_magnitude decides tension/compression.
        - fixed_side_loading: fixed slab on fixed_side of BC_dir and loaded
          slab on the opposite side. load_dir chooses the force axis and
          load_direction_side, when provided, chooses positive/negative force.
        """
        pin_bearing_cases = ("pin_bearing_bracket", "pin_bearing", "bearing_bracket")
        if self.bc_mapping_mode == "surface_patch":
            if self.load_case in pin_bearing_cases:
                self.apply_surface_patch_pin_bearing_bracket()
                return

            if self.load_case in ("tensile_compression", "tensile", "compression"):
                self.apply_surface_patch_tensile_compression()
                return

            if self.load_case in ("three_point_bending", "threepoint_bending", "3_point_bending"):
                self.apply_surface_patch_three_point_bending()
                return

            if self.load_case in ("torsion", "twist", "torque"):
                self.apply_surface_patch_torsion()
                return

            if self.load_case in (
                "fixed_side_loading",
                "fixed_side_load",
                "side_loading",
                "side_load",
                "fixed_opposite_loading",
            ):
                self.apply_surface_patch_fixed_side_loading()
                return

            raise ValueError(
                f"Unsupported load_case for surface_patch BC mapping: {self.load_case}. "
                "Currently supported: tensile_compression, three_point_bending, torsion, "
                "fixed_side_loading, pin_bearing_bracket"
            )

        if self.bc_mapping_mode != "legacy_bbox":
            raise ValueError("bc_mapping_mode must be 'surface_patch' or 'legacy_bbox'.")

        if self.load_case in pin_bearing_cases:
            raise ValueError("pin_bearing_bracket is supported only with bc_mapping_mode='surface_patch'.")

        if self.load_case in ("tensile_compression", "tensile", "compression"):
            self.apply_tensile_compression_boundary_conditions()
            return

        if self.load_case in ("three_point_bending", "threepoint_bending", "3_point_bending"):
            self.apply_three_point_bending_boundary_conditions()
            return

        if self.load_case in ("torsion", "twist", "torque"):
            self.apply_torsion_boundary_conditions()
            return

        if self.load_case in (
            "fixed_side_loading",
            "fixed_side_load",
            "side_loading",
            "side_load",
            "fixed_opposite_loading",
        ):
            self.apply_fixed_side_loading_boundary_conditions()
            return

        raise ValueError(
            f"Unsupported load_case: {self.load_case}. "
            "Currently supported: tensile_compression, three_point_bending, torsion, "
            "fixed_side_loading, pin_bearing_bracket"
        )

    def _axis_bounds_keys(self, axis):
        axis = str(axis).lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Unsupported BC_dir: {axis}")
        return f"{axis}min", f"{axis}max"

    def select_axis_end_slab_nodes(self, axis, side, bbox, tol):
        lo_key, hi_key = self._axis_bounds_keys(axis)
        side = str(side).lower()

        if side == "min":
            lo = bbox[lo_key]
            hi = bbox[lo_key] + tol
        elif side == "max":
            lo = bbox[hi_key] - tol
            hi = bbox[hi_key]
        else:
            raise ValueError(f"Unsupported slab side: {side}")

        kwargs = {lo_key: lo, hi_key: hi}
        return self.select_nodes_in_box(**kwargs)

    def select_axis_middle_slab_nodes(self, axis, bbox, tol):
        lo_key, hi_key = self._axis_bounds_keys(axis)
        center = 0.5 * (bbox[lo_key] + bbox[hi_key])
        half_width = 0.5 * tol
        kwargs = {
            lo_key: center - half_width,
            hi_key: center + half_width,
        }
        return self.select_nodes_in_box(**kwargs)

    def select_shell_axis_slab_nodes(self, shell_nodes, axis, side, tol, max_expand=8):
        axis = str(axis).lower()
        comp_map = {"x": 0, "y": 1, "z": 2}
        if axis not in comp_map:
            raise ValueError(f"Unsupported axis: {axis}")

        shell_nodes = np.asarray(shell_nodes, dtype=np.int64).reshape(-1)
        if shell_nodes.size == 0:
            return shell_nodes

        _node_ids, coords = self.get_flat_node_coords()
        values = coords[shell_nodes, comp_map[axis]]
        side = str(side).lower()
        lo = float(values.min())
        hi = float(values.max())

        for scale in range(1, int(max_expand) + 1):
            width = float(tol) * float(scale)
            if side == "min":
                mask = values <= lo + width
            elif side == "max":
                mask = values >= hi - width
            elif side == "middle":
                center = 0.5 * (lo + hi)
                mask = np.abs(values - center) <= 0.5 * width
            else:
                raise ValueError(f"Unsupported slab side: {side}")

            selected = shell_nodes[mask]
            if selected.size > 0:
                return selected

        if side == "min":
            target = lo
        elif side == "max":
            target = hi
        else:
            target = 0.5 * (lo + hi)

        nearest = np.argmin(np.abs(values - target))
        return shell_nodes[[nearest]]

    def filter_shell_nodes_by_surface(self, shell_nodes, axis, side, tol, max_expand=8):
        axis = str(axis).lower()
        comp_map = {"x": 0, "y": 1, "z": 2}
        if axis not in comp_map:
            raise ValueError(f"Unsupported surface axis: {axis}")

        shell_nodes = np.asarray(shell_nodes, dtype=np.int64).reshape(-1)
        if shell_nodes.size == 0:
            return shell_nodes

        _node_ids, coords = self.get_flat_node_coords()
        values = coords[shell_nodes, comp_map[axis]]
        side = str(side).lower()
        lo = float(values.min())
        hi = float(values.max())

        for scale in range(1, int(max_expand) + 1):
            width = float(tol) * float(scale)
            if side == "min":
                mask = values <= lo + width
            elif side == "max":
                mask = values >= hi - width
            else:
                raise ValueError(f"Unsupported surface side: {side}")

            selected = shell_nodes[mask]
            if selected.size > 0:
                return selected

        target = lo if side == "min" else hi
        nearest = np.argmin(np.abs(values - target))
        return shell_nodes[[nearest]]

    @staticmethod
    def _opposite_side(side):
        side = str(side).lower()
        if side == "min":
            return "max"
        if side == "max":
            return "min"
        raise ValueError(f"Unsupported side: {side}. Expected 'min' or 'max'.")

    @staticmethod
    def _direction_side_sign(side):
        side = str(side).lower()
        if side == "max":
            return 1.0
        if side == "min":
            return -1.0
        raise ValueError(f"Unsupported load_direction_side: {side}. Expected 'min' or 'max'.")

    def apply_tensile_compression_boundary_conditions(self):
        tol = 0.5* self.voxel_size
        shell_nodes = self.occupied_node_ids()

        fixed_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "min", tol)
        force_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "max", tol)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No fixed shell nodes selected for load_case={self.load_case}, BC_dir={self.BC_dir}"
            )
        if force_nodes.size == 0:
            raise ValueError(
                f"No force shell nodes selected for load_case={self.load_case}, BC_dir={self.BC_dir}"
            )

        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=self.BC_dir,
            force_value=self.Load_magnitude,
        )

    def apply_three_point_bending_boundary_conditions(self):
        load_dir = self.load_dir
        if load_dir is None:
            raise ValueError("load_dir must be provided for load_case='three_point_bending'")
        self._axis_bounds_keys(load_dir)

        tol = 0.5 * self.voxel_size
        shell_nodes = self.occupied_node_ids()

        min_support_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "min", tol)
        max_support_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "max", tol)
        force_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "middle", tol)
        if self.load_surface_dir is not None:
            force_nodes = self.filter_shell_nodes_by_surface(
                force_nodes,
                self.load_surface_dir,
                self.load_surface_side,
                tol,
            )
        fixed_nodes = np.union1d(min_support_nodes, max_support_nodes)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No support shell nodes selected for load_case={self.load_case}, span BC_dir={self.BC_dir}"
            )
        if force_nodes.size == 0:
            raise ValueError(
                f"No middle load shell nodes selected for load_case={self.load_case}, span BC_dir={self.BC_dir}"
            )

        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=load_dir,
            force_value=self.Load_magnitude,
        )

    def apply_torsion_boundary_conditions(self):
        self._axis_bounds_keys(self.BC_dir)
        fixed_side = self.fixed_side
        force_side = self.force_side if self.force_side is not None else self._opposite_side(fixed_side)
        if force_side == fixed_side:
            raise ValueError("force_side must be opposite to fixed_side for load_case='torsion'")

        tol = 0.5 * self.voxel_size
        shell_nodes = self.occupied_node_ids()

        fixed_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, fixed_side, tol)
        torque_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, force_side, tol)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No fixed shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, fixed_side={fixed_side}"
            )
        if torque_nodes.size == 0:
            raise ValueError(
                f"No torque shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, force_side={force_side}"
            )

        self.set_torsion_boundary_conditions(
            fixed_nodes=fixed_nodes,
            torque_nodes=torque_nodes,
            torque_axis=self.BC_dir,
            total_torque=self.Load_magnitude,
        )

    def apply_fixed_side_loading_boundary_conditions(self):
        self._axis_bounds_keys(self.BC_dir)
        fixed_side = self.fixed_side
        force_side = self.force_side if self.force_side is not None else self._opposite_side(fixed_side)
        if force_side == fixed_side:
            raise ValueError(
                "force_side must be opposite to fixed_side for load_case='fixed_side_loading'"
            )

        force_direction = self.load_dir if self.load_dir is not None else self.BC_dir
        self._axis_bounds_keys(force_direction)

        force_value = self.Load_magnitude
        if self.load_direction_side is not None:
            force_value = abs(force_value) * self._direction_side_sign(self.load_direction_side)

        tol = 0.5 * self.voxel_size
        shell_nodes = self.occupied_node_ids()

        fixed_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, fixed_side, tol)
        force_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, force_side, tol)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No fixed shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, fixed_side={fixed_side}"
            )
        if force_nodes.size == 0:
            raise ValueError(
                f"No force shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, force_side={force_side}"
            )

        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=force_direction,
            force_value=force_value,
        )

    def shellSettings(self):
        mesh, grid_geom, elem_centers, node_coords = self.build_voxel_grid_for_shell(
            self.brep_bbox,
            self.thickness,
            self.voxel_size,
            self.extra_layers
        )

        self.grid_geom = grid_geom
        self.elem_centers = elem_centers
        self.node_coords = node_coords

        # FEM unit convention:
        # length = millimetres (mm), force = newtons (N),
        # stress/Young's modulus = N/mm^2 = MPa, displacement = mm.
        # Replace these example CCF values with experimentally calibrated
        # anisotropic material data before interpreting feasibility.
        material_E_longitudinal = float(getattr(self, "material_E_longitudinal", 70000.0))
        material_E_transverse = float(getattr(self, "material_E_transverse", 7000.0))
        material_nu_longitudinal = float(getattr(self, "material_nu_longitudinal", 0.30))
        material_nu_transverse = float(getattr(self, "material_nu_transverse", 0.30))
        material_G12 = float(getattr(self, "material_G12", 4500.0))
        material_G23 = float(getattr(self, "material_G23", 2600.0))
        material_G23 = float(
    material_E_transverse
    / (2.0 * (1.0 + material_nu_transverse))
)
        material_G13 = float(getattr(self, "material_G13", material_G12))
        material_yield_strength = float(getattr(self, "material_yield_strength", 200.0))

        isotropic_test = False

        if isotropic_test:
            E_iso = 7000.0
            nu_iso = 0.30
            G_iso = E_iso / (2.0 * (1.0 + nu_iso))

            material_E_longitudinal = E_iso
            material_E_transverse = E_iso

            material_nu_longitudinal = nu_iso
            material_nu_transverse = nu_iso

            material_G12 = G_iso
            material_G13 = G_iso
            material_G23 = G_iso

        matProp = {
            'length_unit': 'mm',
            'force_unit': 'N',
            'stress_unit': 'MPa',
            'material_E1': material_E_longitudinal,
            'material_E2': material_E_transverse,
            'material_E3': material_E_transverse,
            'material_nu12': material_nu_longitudinal,
            'material_nu23': material_nu_transverse,
            'material_nu13': material_nu_longitudinal,
            'material_G12': material_G12,
            'material_G23': material_G23,
            'material_G13': material_G13,
            'material_E_longitudinal': material_E_longitudinal,
            'material_E_transverse': material_E_transverse,
            'material_nu_longitudinal': material_nu_longitudinal,
            'material_nu_transverse': material_nu_transverse,
            'material_shear_modulus': material_G12,
            'material_yield_strength': material_yield_strength,
            'E': material_E_transverse,
            'nu': material_nu_transverse,
            'Ef': material_E_longitudinal,
            'Et': material_E_transverse,
            'nuf': material_nu_longitudinal,
            'nut': material_nu_transverse,
            'Gf': material_G12,
            'penal': 3,
        }

        ndof = 3 * (mesh['nelx'] + 1) * (mesh['nely'] + 1) * (mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        fixed = np.array([], dtype=np.int64)

        bc = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

        return mesh, bc, matProp

    def to_numpy(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _normalize_single_bbox(self, bbox):
        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if not isinstance(bbox, dict) or not all(k in bbox for k in required):
            raise ValueError(f"Unsupported single-face BBX format: {bbox}")
        return {
            'xmin': float(bbox['xmin']),
            'xmax': float(bbox['xmax']),
            'ymin': float(bbox['ymin']),
            'ymax': float(bbox['ymax']),
            'zmin': float(bbox['zmin']),
            'zmax': float(bbox['zmax']),
        }
    def parse_bbox(self, bbox_raw):
        """
        Accept either:
        - {'xmin':..., 'xmax':..., 'ymin':..., 'ymax':..., 'zmin':..., 'zmax':...}
        - {0: {...}, 1: {...}, ...}

        Returns
        -------
        union_bbox : dict
            Global union bounding box across all faces.
        face_bboxes : dict[int, dict]
            Per-face bounding boxes. For a single-face input, face id 0 is used.
        """
        if not isinstance(bbox_raw, dict):
            raise ValueError(f"Unsupported BBX format: {bbox_raw}")

        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if all(k in bbox_raw for k in required):
            b = self._normalize_single_bbox(bbox_raw)
            return b, {0: b.copy()}

        face_bboxes = {}
        for fid, bbox in bbox_raw.items():
            if not isinstance(bbox, dict):
                raise ValueError(f"Unsupported BBX entry for face {fid}: {bbox}")
            face_bboxes[int(fid)] = self._normalize_single_bbox(bbox)

        if len(face_bboxes) == 0:
            raise ValueError(f"Unsupported empty BBX format: {bbox_raw}")

        union_bbox = {
            'xmin': min(b['xmin'] for b in face_bboxes.values()),
            'xmax': max(b['xmax'] for b in face_bboxes.values()),
            'ymin': min(b['ymin'] for b in face_bboxes.values()),
            'ymax': max(b['ymax'] for b in face_bboxes.values()),
            'zmin': min(b['zmin'] for b in face_bboxes.values()),
            'zmax': max(b['zmax'] for b in face_bboxes.values()),
        }
        return union_bbox, face_bboxes

    def set_cad_samples(self, tensors):
        self.uv = self.to_numpy(tensors["uv"])
        self.points_xyz = self.to_numpy(tensors["points_xyz"]).reshape(-1, 3)
        self.face_areas = self.to_numpy(tensors["face_areas"])
        self.Xu = self.to_numpy(tensors["Xu"]).reshape(-1, 3)
        self.Xv = self.to_numpy(tensors["Xv"]).reshape(-1, 3)
        self.faces_ijk = self.to_numpy(tensors["faces_ijk"])
        self.pv_faces = self.to_numpy(tensors["pv_faces"])
        self.face_id = self.to_numpy(tensors["face_id"]).reshape(-1).astype(np.int64)
        self.boundary_idx_ring1 = self.to_numpy(tensors["boundary_idx_ring1"])
        self.min_vol_frac = self.to_numpy(tensors["min_vol_frac"])

        self.sample_normals = self.compute_sample_normals(self.Xu, self.Xv)

        bbox_raw = tensors["BBX"]
        self.brep_bbox, self.face_bboxes = self.parse_bbox(bbox_raw)

        self.samples_by_face = {}
        if self.face_id.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("face_id must have same length as points_xyz")
        for fid in np.unique(self.face_id):
            self.samples_by_face[int(fid)] = np.flatnonzero(self.face_id == fid).astype(np.int64)

        faces = np.asarray(self.faces_ijk, dtype=np.int64).reshape(-1, 3)
        if faces.size == 0:
            raise ValueError("faces_ijk must contain at least one triangle.")
        faces_index_base = self.faces_index_base
        if faces_index_base is None and isinstance(tensors, dict):
            faces_index_base = tensors.get("faces_index_base", 0)
        faces_index_base = int(faces_index_base)
        if faces_index_base not in (0, 1):
            raise ValueError("faces_index_base must be 0 or 1.")
        if faces_index_base == 1:
            faces = faces - 1
        if faces.min() < 0 or faces.max() >= self.points_xyz.shape[0]:
            raise ValueError("faces_ijk contains invalid vertex indices.")
        self.faces_ijk = faces
        self.triangles = self.points_xyz[faces]

        tri_face_id = np.zeros((faces.shape[0],), dtype=np.int64)
        if self.face_id.shape[0] == self.points_xyz.shape[0]:
            for tri_id, tri_nodes in enumerate(faces):
                ids, counts = np.unique(self.face_id[tri_nodes], return_counts=True)
                tri_face_id[tri_id] = int(ids[np.argmax(counts)])
        self.triangle_face_id = tri_face_id
        self.faces_index_base = faces_index_base

        if isinstance(tensors, dict):
            self.boundary_edges = tensors.get("boundary_edges", {}) or {}
            self.boundary_loops = tensors.get("boundary_loops", {}) or {}
            self._unpack_boundary_topology_from_packed_curves(tensors)

    def _unpack_boundary_topology_from_packed_curves(self, tensors):
        if not isinstance(tensors, dict) or "boundary_curve_xyz" not in tensors:
            return

        existing_edges = self.boundary_edges if isinstance(self.boundary_edges, dict) else {}
        existing_loops = self.boundary_loops if isinstance(self.boundary_loops, dict) else {}
        need_edges = len(existing_edges) == 0
        need_loops = len(existing_loops) == 0
        if not need_edges and not need_loops:
            return

        required = (
            "boundary_curve_offsets",
            "boundary_curve_loop_id",
            "boundary_curve_piece_id",
            "boundary_curve_loop_kind",
            "boundary_curve_loop_area",
            "boundary_curve_length",
        )
        if any(key not in tensors for key in required):
            return

        xyz = self.to_numpy(tensors["boundary_curve_xyz"]).reshape(-1, 3).astype(np.float64, copy=False)
        offsets = self.to_numpy(tensors["boundary_curve_offsets"]).reshape(-1).astype(np.int64)
        loop_ids = self.to_numpy(tensors["boundary_curve_loop_id"]).reshape(-1).astype(np.int64)
        piece_ids = self.to_numpy(tensors["boundary_curve_piece_id"]).reshape(-1).astype(np.int64)
        loop_kind_ids = self.to_numpy(tensors["boundary_curve_loop_kind"]).reshape(-1).astype(np.int64)
        loop_areas = self.to_numpy(tensors["boundary_curve_loop_area"]).reshape(-1).astype(np.float64)
        lengths = self.to_numpy(tensors["boundary_curve_length"]).reshape(-1).astype(np.float64)

        num_pieces = loop_ids.size
        if offsets.size != num_pieces + 1:
            raise ValueError("boundary_curve_offsets must have one more entry than boundary curve pieces.")
        if offsets[-1] != xyz.shape[0]:
            raise ValueError("boundary_curve_xyz must align one-to-one with packed boundary curves.")

        pieces_by_loop = {}
        reconstructed_edges = {}
        for piece_index in range(num_pieces):
            start = int(offsets[piece_index])
            stop = int(offsets[piece_index + 1])
            coords = xyz[start:stop]
            loop_id = int(loop_ids[piece_index])
            piece_id = int(piece_ids[piece_index])
            loop_kind = "outer" if int(loop_kind_ids[piece_index]) == 0 else "hole"
            entry = {
                "loop_id": loop_id,
                "piece_id": piece_id,
                "loop_kind": loop_kind,
                "loop_area": float(loop_areas[piece_index]),
                "length": float(lengths[piece_index]),
                "coordinates": coords.copy(),
                "closed": False,
            }
            reconstructed_edges[(loop_id, piece_id)] = entry
            pieces_by_loop.setdefault(loop_id, []).append(entry)

        reconstructed_loops = {}
        for loop_id, pieces in pieces_by_loop.items():
            pieces = sorted(pieces, key=lambda item: int(item["piece_id"]))
            joined = []
            for piece in pieces:
                coords = np.asarray(piece["coordinates"], dtype=np.float64).reshape(-1, 3)
                if coords.shape[0] == 0:
                    continue
                if not joined:
                    joined.extend(coords)
                    continue
                if np.linalg.norm(np.asarray(joined[-1]) - coords[0]) <= 1.0e-10:
                    joined.extend(coords[1:])
                else:
                    joined.extend(coords)
            if len(joined) == 0:
                continue
            loop_coords = np.asarray(joined, dtype=np.float64).reshape(-1, 3)
            if loop_coords.shape[0] > 1 and np.linalg.norm(loop_coords[0] - loop_coords[-1]) <= 1.0e-10:
                loop_coords = loop_coords[:-1]
            first = pieces[0]
            reconstructed_loops[int(loop_id)] = {
                "loop_id": int(loop_id),
                "loop_kind": str(first["loop_kind"]),
                "loop_area": float(first["loop_area"]),
                "coordinates": loop_coords,
                "closed": True,
            }

        if need_edges:
            self.boundary_edges = reconstructed_edges
        if need_loops:
            self.boundary_loops = reconstructed_loops

    def occupied_node_ids(self):
        occ = self.elem_occupancy.astype(bool)   # shape (nelz, nelx, nely)
        nelz, nelx, nely = occ.shape

        node_mask = np.zeros((nelz + 1, nelx + 1, nely + 1), dtype=bool)

        for k in range(nelz):
            for i in range(nelx):
                for j in range(nely):
                    if occ[k, i, j]:
                        node_mask[k:k+2, i:i+2, j:j+2] = True

        node_ids = np.arange(node_mask.size, dtype=np.int64).reshape(node_mask.shape)
        return node_ids[node_mask]    
    def intersect_node_sets(self, a, b):
        return np.intersect1d(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))

    def compute_sample_normals(self, Xu, Xv, eps=1e-12):
        normals = np.cross(Xu, Xv)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norm, eps, None)
        return normals

    def voxel_center_is_near_bbox(self, center, bbox, margin):
        return (
            (bbox['xmin'] - margin <= center[0] <= bbox['xmax'] + margin) and
            (bbox['ymin'] - margin <= center[1] <= bbox['ymax'] + margin) and
            (bbox['zmin'] - margin <= center[2] <= bbox['zmax'] + margin)
        )

    def candidate_face_ids_for_center(self, center, margin):
        if not self.face_bboxes:
            return []
        return [
            fid for fid, bbox in self.face_bboxes.items()
            if self.voxel_center_is_near_bbox(center, bbox, margin)
        ]

    @staticmethod
    def _closest_point_on_triangle(point, tri, eps=1.0e-14):
        """Return closest point and barycentric weights on one 3D triangle."""
        p = np.asarray(point, dtype=np.float64)
        a, b, c = np.asarray(tri, dtype=np.float64)
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a, np.array([1.0, 0.0, 0.0], dtype=np.float64)

        bp = p - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b, np.array([0.0, 1.0, 0.0], dtype=np.float64)

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / max(d1 - d3, eps)
            return a + v * ab, np.array([1.0 - v, v, 0.0], dtype=np.float64)

        cp = p - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c, np.array([0.0, 0.0, 1.0], dtype=np.float64)

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / max(d2 - d6, eps)
            return a + w * ac, np.array([1.0 - w, 0.0, w], dtype=np.float64)

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / max((d4 - d3) + (d5 - d6), eps)
            return b + w * (c - b), np.array([0.0, 1.0 - w, w], dtype=np.float64)

        denom = max(va + vb + vc, eps)
        v = vb / denom
        w = vc / denom
        u = 1.0 - v - w
        return a + ab * v + ac * w, np.array([u, v, w], dtype=np.float64)

    @staticmethod
    def _normal_from_triangle(tri, fallback=None):
        a, b, c = np.asarray(tri, dtype=np.float64)
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 1.0e-14:
            return normal / norm
        if fallback is not None:
            fallback = np.asarray(fallback, dtype=np.float64)
            fallback_norm = np.linalg.norm(fallback)
            if fallback_norm > 1.0e-14:
                return fallback / fallback_norm
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _subvoxel_offsets(self, samples):
        n = int(samples)
        hx, hy, hz = self.grid_geom["hx"], self.grid_geom["hy"], self.grid_geom["hz"]
        if n == 1:
            return np.zeros((1, 3), dtype=np.float64)
        local = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5
        dz, dx, dy = np.meshgrid(local * hz, local * hx, local * hy, indexing="ij")
        return np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)

    def _closest_triangle_for_point(self, point, triangle_ids):
        best_dist2 = float("inf")
        best_tri_id = -1
        best_point = np.zeros(3, dtype=np.float64)
        best_bary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        for tri_id in np.asarray(triangle_ids, dtype=np.int64).reshape(-1):
            closest, bary = self._closest_point_on_triangle(point, self.triangles[int(tri_id)])
            diff = np.asarray(point, dtype=np.float64) - closest
            dist2 = float(np.dot(diff, diff))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_tri_id = int(tri_id)
                best_point = closest
                best_bary = bary
        return best_tri_id, best_point, best_bary, math.sqrt(best_dist2)

    def _triangle_candidate_element_ids(self, margin):
        centers = self.elem_centers
        nelz, nelx, nely = centers.shape[:3]
        xs = centers[0, :, 0, 0]
        ys = centers[0, 0, :, 1]
        zs = centers[:, 0, 0, 2]
        candidates = [set() for _ in range(centers.reshape(-1, 3).shape[0])]

        for tri_id, tri in enumerate(self.triangles):
            lower = tri.min(axis=0) - float(margin)
            upper = tri.max(axis=0) + float(margin)
            ix0 = max(0, int(np.searchsorted(xs, lower[0], side="left")))
            ix1 = min(nelx, int(np.searchsorted(xs, upper[0], side="right")))
            iy0 = max(0, int(np.searchsorted(ys, lower[1], side="left")))
            iy1 = min(nely, int(np.searchsorted(ys, upper[1], side="right")))
            iz0 = max(0, int(np.searchsorted(zs, lower[2], side="left")))
            iz1 = min(nelz, int(np.searchsorted(zs, upper[2], side="right")))
            for iz in range(iz0, iz1):
                base_z = iz * nelx * nely
                for ix in range(ix0, ix1):
                    base = base_z + ix * nely
                    for iy in range(iy0, iy1):
                        candidates[base + iy].add(tri_id)

        return candidates

    def voxelize_shell_from_triangles(self, thickness, subvoxel_samples=1, min_geom_fraction=0.25):
        """
        Voxelize by exact distance to the triangulated midsurface.

        ``extra_layers`` only pads the grid. It does not thicken occupancy.
        Active elements satisfy ``elem_geom_fraction >= min_geom_fraction``.
        """
        centers = self.elem_centers.reshape(-1, 3).astype(np.float64)
        half_t = 0.5 * float(thickness)
        hx, hy, hz = self.grid_geom["hx"], self.grid_geom["hy"], self.grid_geom["hz"]
        candidate_margin = half_t + 0.5 * math.sqrt(hx * hx + hy * hy + hz * hz)
        candidate_tris = self._triangle_candidate_element_ids(candidate_margin)
        offsets = self._subvoxel_offsets(subvoxel_samples)
        total_subpoints = int(offsets.shape[0])

        num_elems = centers.shape[0]
        geom_fraction = np.zeros((num_elems,), dtype=np.float32)
        closest_tri = -np.ones((num_elems,), dtype=np.int64)
        closest_point = np.full((num_elems, 3), np.nan, dtype=np.float32)
        barycentric = np.full((num_elems, 3), np.nan, dtype=np.float32)
        distance = np.full((num_elems,), np.inf, dtype=np.float32)
        elem_sample_idx = -np.ones((num_elems,), dtype=np.int64)

        for elem_id, tris in enumerate(candidate_tris):
            if not tris:
                continue
            tri_ids = np.fromiter(tris, dtype=np.int64)
            tri_id, cp, bary, dist = self._closest_triangle_for_point(centers[elem_id], tri_ids)
            closest_tri[elem_id] = tri_id
            closest_point[elem_id] = cp.astype(np.float32)
            barycentric[elem_id] = bary.astype(np.float32)
            distance[elem_id] = float(dist)
            if tri_id >= 0:
                tri_nodes = self.faces_ijk[tri_id]
                elem_sample_idx[elem_id] = int(tri_nodes[int(np.argmax(bary))])

            inside_count = 0
            for offset in offsets:
                _sid, _cp, _bary, sub_dist = self._closest_triangle_for_point(
                    centers[elem_id] + offset,
                    tri_ids,
                )
                if sub_dist <= half_t:
                    inside_count += 1
            geom_fraction[elem_id] = inside_count / total_subpoints

        occ = geom_fraction >= float(min_geom_fraction)
        inactive = ~occ
        closest_tri[inactive] = -1
        closest_point[inactive] = np.nan
        barycentric[inactive] = np.nan
        distance[inactive] = np.inf
        elem_sample_idx[inactive] = -1

        self.elem_occupancy = occ.reshape(self.elem_centers.shape[:3]).astype(np.uint8)
        self.elem_geom_fraction = geom_fraction
        self.elem_closest_triangle_id = closest_tri
        self.elem_closest_point = closest_point
        self.elem_barycentric = barycentric
        self.elem_distance_to_surface = distance
        self.elem_sample_idx = elem_sample_idx.reshape(self.elem_centers.shape[:3])
        source_face = -np.ones((num_elems,), dtype=np.int64)
        active_tri = closest_tri >= 0
        source_face[active_tri] = self.triangle_face_id[closest_tri[active_tri]]
        self.elem_source_face_id = source_face

        if np.any(occ):
            active_fraction = geom_fraction[occ]
            self.min_active_geom_fraction = float(active_fraction.min())
        else:
            self.min_active_geom_fraction = float("nan")

        return self.elem_occupancy, self.elem_closest_triangle_id
    def voxelize_shell_from_samples(self, thickness, tangential_tol=None):
        centers = self.elem_centers.reshape(-1, 3)
        points = self.points_xyz
        normals = self.sample_normals

        if tangential_tol is None:
            tangential_tol = 0.35 * self.voxel_size

        half_t = 0.5 * thickness
        max_euclid = np.sqrt(half_t * half_t + tangential_tol * tangential_tol)
        bbox_margin = half_t + tangential_tol + self.voxel_size

        occ = np.zeros((centers.shape[0],), dtype=np.uint8)
        sample_idx = -np.ones((centers.shape[0],), dtype=np.int64)

        for i, x in enumerate(centers):
            candidate_face_ids = self.candidate_face_ids_for_center(x, bbox_margin)

            if candidate_face_ids:
                candidate_idx = np.concatenate([
                    self.samples_by_face[fid] for fid in candidate_face_ids if fid in self.samples_by_face
                ])
            else:
                candidate_idx = np.arange(points.shape[0], dtype=np.int64)

            if candidate_idx.size == 0:
                continue

            candidate_points = points[candidate_idx]
            diff = candidate_points - x[None, :]
            dist2 = np.einsum('ij,ij->i', diff, diff)
            local_j = np.argmin(dist2)
            j = candidate_idx[local_j]

            p = points[j]
            n = normals[j]

            r = x - p
            de = np.linalg.norm(r)
            dn = abs(np.dot(r, n))
            rt = r - np.dot(r, n) * n
            dt = np.linalg.norm(rt)

            if dn <= half_t and dt <= tangential_tol and de <= max_euclid:
                occ[i] = 1
                sample_idx[i] = j

        occ = occ.reshape(self.elem_centers.shape[:3])
        sample_idx = sample_idx.reshape(self.elem_centers.shape[:3])

        return occ, sample_idx

    def padded_bbox_from_midsurface(self, bbox, thickness, voxel_size, extra_layers=1):
        pad = thickness / 2.0 + extra_layers * voxel_size

        return {
            'xmin': bbox['xmin'] - pad,
            'xmax': bbox['xmax'] + pad,
            'ymin': bbox['ymin'] - pad,
            'ymax': bbox['ymax'] + pad,
            'zmin': bbox['zmin'] - pad,
            'zmax': bbox['zmax'] + pad,
        }

    def structured_grid_from_bbox(self, bbox, voxel_size):
        hx = hy = hz = float(voxel_size)

        lx = bbox['xmax'] - bbox['xmin']
        ly = bbox['ymax'] - bbox['ymin']
        lz = bbox['zmax'] - bbox['zmin']

        nelx = int(math.ceil(lx / hx))
        nely = int(math.ceil(ly / hy))
        nelz = int(math.ceil(lz / hz))

        mesh = {
            'nelx': nelx,
            'nely': nely,
            'nelz': nelz,
            'elemSize': np.array([hx, hy, hz], dtype=float),
            'type': 'grid'
        }

        grid_geom = {
            'xmin': bbox['xmin'],
            'ymin': bbox['ymin'],
            'zmin': bbox['zmin'],
            'hx': hx,
            'hy': hy,
            'hz': hz
        }

        return mesh, grid_geom

    def element_centers(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + (np.arange(nelx) + 0.5) * hx
        ys = ymin + (np.arange(nely) + 0.5) * hy
        zs = zmin + (np.arange(nelz) + 0.5) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        centers = np.stack([X, Y, Z], axis=-1)
        return centers

    def node_coordinates(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + np.arange(nelx + 1) * hx
        ys = ymin + np.arange(nely + 1) * hy
        zs = zmin + np.arange(nelz + 1) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)
        return coords

    def build_voxel_grid_for_shell(self, brep_bbox, thickness, voxel_size, extra_layers=1):
        padded = self.padded_bbox_from_midsurface(
            brep_bbox,
            thickness=thickness,
            voxel_size=voxel_size,
            extra_layers=extra_layers
        )

        mesh, grid_geom = self.structured_grid_from_bbox(padded, voxel_size)
        elem_centers = self.element_centers(mesh, grid_geom)
        node_coords = self.node_coordinates(mesh, grid_geom)

        return mesh, grid_geom, elem_centers, node_coords

    def surface_samples_to_element_indices(self):
        """
        Assign each midsurface sample to exactly one structured FEA element.

        Exact ties on voxel faces/edges/corners are broken toward the element
        with the larger grid index on the tied axis, so every sample still has
        exactly one owner.
        """
        points = self.points_xyz
        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        hx, hy, hz = self.grid_geom['hx'], self.grid_geom['hy'], self.grid_geom['hz']
        xmin, ymin, zmin = self.grid_geom['xmin'], self.grid_geom['ymin'], self.grid_geom['zmin']

        center_x0 = xmin + 0.5 * hx
        center_y0 = ymin + 0.5 * hy
        center_z0 = zmin + 0.5 * hz

        ix = np.floor((points[:, 0] - center_x0) / hx + 0.5).astype(np.int64)
        iy = np.floor((points[:, 1] - center_y0) / hy + 0.5).astype(np.int64)
        iz = np.floor((points[:, 2] - center_z0) / hz + 0.5).astype(np.int64)

        valid = (
            (0 <= ix) & (ix < nelx) &
            (0 <= iy) & (iy < nely) &
            (0 <= iz) & (iz < nelz)
        )

        elem_idx = -np.ones(points.shape[0], dtype=np.int64)
        elem_idx[valid] = iz[valid] * (nelx * nely) + ix[valid] * nely + iy[valid]
        return elem_idx

    def ensure_surface_sample_elements_are_occupied(self):
        """
        The midsurface/core layer is defined by the elements that own surface
        samples. Force those elements into the shell and keep one representative
        sample index for fallback/visualization.
        """
        sample_elem_idx = np.asarray(self.sample_elem_idx, dtype=np.int64).reshape(-1)
        valid = sample_elem_idx >= 0
        if not np.any(valid):
            return

        occ_flat = self.elem_occupancy.reshape(-1)
        elem_sample_flat = self.elem_sample_idx.reshape(-1)
        sample_ids = np.arange(sample_elem_idx.shape[0], dtype=np.int64)

        occ_flat[sample_elem_idx[valid]] = 1
        missing_rep = elem_sample_flat < 0
        for elem_idx, sample_id in zip(sample_elem_idx[valid], sample_ids[valid]):
            if missing_rep[elem_idx]:
                elem_sample_flat[elem_idx] = sample_id
                missing_rep[elem_idx] = False

    def count_surface_samples_per_element(self):
        num_elems = int(np.prod(self.elem_centers.shape[:3]))
        sample_elem_idx = np.asarray(self.sample_elem_idx, dtype=np.int64).reshape(-1)
        valid = sample_elem_idx >= 0
        return np.bincount(sample_elem_idx[valid], minlength=num_elems).astype(np.int64)

    def shell_layer_core_element_indices(self):
        """
        Map each occupied shell element to the midsurface/core element whose
        density and fiber it should inherit.

        Core elements map to themselves. Offset inner/outer layer elements map
        through their nearest surface sample to that sample's core element.
        """
        sample_idx = self.elem_sample_idx.reshape(-1)
        sample_elem_idx = self.sample_elem_idx.reshape(-1)
        occ = self.elem_occupancy.reshape(-1).astype(bool)

        num_elems = sample_idx.shape[0]
        counts = self.elem_sample_count
        if counts is None:
            counts = self.count_surface_samples_per_element()

        core_elem_idx = -np.ones((num_elems,), dtype=np.int64)
        has_core_samples = counts > 0
        core_elem_idx[occ & has_core_samples] = np.flatnonzero(occ & has_core_samples)

        layer = occ & (~has_core_samples) & (sample_idx >= 0)
        layer_sample_idx = sample_idx[layer]
        valid_layer_sample = (
            (layer_sample_idx >= 0)
            & (layer_sample_idx < sample_elem_idx.shape[0])
            & (sample_elem_idx[layer_sample_idx] >= 0)
        )
        layer_elem_ids = np.flatnonzero(layer)
        core_elem_idx[layer_elem_ids[valid_layer_sample]] = sample_elem_idx[layer_sample_idx[valid_layer_sample]]

        return core_elem_idx
    
    def assign_surface_fields_to_voxels(self, rho_surface, fiber_surface, rho_void=0.0):
        rho_surface = self.to_numpy(rho_surface).reshape(-1)
        fiber_surface = self.to_numpy(fiber_surface).reshape(-1, 3)

        if rho_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("rho_surface must have same length as points_xyz")

        if fiber_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("fiber_surface must have same length as points_xyz")

        if self.voxelization_mode == "triangle_distance":
            return self.assign_surface_fields_to_voxels_barycentric(
                rho_surface=rho_surface,
                fiber_surface=fiber_surface,
                rho_void=rho_void,
            )

        sample_elem_idx = self.sample_elem_idx.reshape(-1)
        occ = self.elem_occupancy.reshape(-1)

        num_elems = occ.shape[0]

        elem_density = np.full((num_elems,), rho_void, dtype=np.float32)
        elem_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))

        valid_samples = sample_elem_idx >= 0
        counts = np.bincount(sample_elem_idx[valid_samples], minlength=num_elems).astype(np.float32)
        rho_sum = np.bincount(
            sample_elem_idx[valid_samples],
            weights=rho_surface[valid_samples],
            minlength=num_elems,
        ).astype(np.float32)

        fiber_sum = np.zeros((num_elems, 3), dtype=np.float32)
        np.add.at(fiber_sum, sample_elem_idx[valid_samples], fiber_surface[valid_samples].astype(np.float32))

        has_samples = counts > 0

        core_density = np.full((num_elems,), rho_void, dtype=np.float32)
        core_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))
        core_density[has_samples] = rho_sum[has_samples] / counts[has_samples]
        core_fiber[has_samples] = fiber_sum[has_samples] / counts[has_samples, None]

        core_norms = np.linalg.norm(core_fiber[has_samples], axis=1, keepdims=True)
        core_fiber[has_samples] /= np.clip(core_norms, 1e-12, None)

        source_core_elem_idx = self.shell_layer_core_element_indices()
        filled = (occ > 0) & (source_core_elem_idx >= 0)
        filled_ids = np.flatnonzero(filled)
        filled[filled_ids] = has_samples[source_core_elem_idx[filled_ids]]

        elem_density[filled] = core_density[source_core_elem_idx[filled]]
        elem_fiber[filled] = core_fiber[source_core_elem_idx[filled]]

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi, self.elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        return elem_density, elem_fiber

    def assign_surface_fields_to_voxels_barycentric(self, rho_surface, fiber_surface, rho_void=0.0):
        occ = self.elem_occupancy.reshape(-1).astype(bool)
        num_elems = occ.shape[0]
        elem_density = np.full((num_elems,), rho_void, dtype=np.float32)
        elem_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))

        tri_ids = np.asarray(self.elem_closest_triangle_id, dtype=np.int64).reshape(-1)
        bary = np.asarray(self.elem_barycentric, dtype=np.float64).reshape(-1, 3)
        active_ids = np.flatnonzero(occ & (tri_ids >= 0))
        if active_ids.size == 0:
            self.elem_density = elem_density
            self.elem_fiber = elem_fiber
            self.elem_phi, self.elem_theta = self.fiber_vectors_to_angles(elem_fiber)
            return elem_density, elem_fiber

        tri_nodes = self.faces_ijk[tri_ids[active_ids]]
        weights = bary[active_ids]
        elem_density[active_ids] = np.sum(rho_surface[tri_nodes] * weights, axis=1).astype(np.float32)

        nodal_fiber = fiber_surface[tri_nodes].astype(np.float64)
        reference = nodal_fiber[:, 0, :]
        alignment = np.sum(nodal_fiber * reference[:, None, :], axis=2)
        nodal_fiber[alignment < 0.0] *= -1.0
        fiber = np.sum(nodal_fiber * weights[:, :, None], axis=1)

        for local, elem_id in enumerate(active_ids):
            tri = self.triangles[tri_ids[elem_id]]
            normal = self._normal_from_triangle(tri)
            fiber[local] = fiber[local] - np.dot(fiber[local], normal) * normal
            norm = np.linalg.norm(fiber[local])
            if norm <= 1.0e-12:
                fallback = tri[1] - tri[0]
                fallback = fallback - np.dot(fallback, normal) * normal
                fiber[local] = fallback
                norm = np.linalg.norm(fiber[local])
            fiber[local] = fiber[local] / max(norm, 1.0e-12)

        elem_fiber[active_ids] = fiber.astype(np.float32)
        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi, self.elem_theta = self.fiber_vectors_to_angles(elem_fiber)
        return elem_density, elem_fiber
    
    def fiber_vectors_to_angles(self, fiber_vec):
        fiber_vec = self.to_numpy(fiber_vec).reshape(-1, 3)

        norms = np.linalg.norm(fiber_vec, axis=1, keepdims=True)
        v = fiber_vec / np.clip(norms, 1e-12, None)

        ax = v[:, 0]
        ay = v[:, 1]
        az = v[:, 2]

        phi = np.arctan2(ay, ax).astype(np.float32)
        theta = np.arccos(np.clip(az, -1.0, 1.0)).astype(np.float32)

        return phi, theta
    def assign_decoder_fields(self, rho_surface, fiber_surface, rho_void=0.0):
        elem_density, elem_fiber = self.assign_surface_fields_to_voxels(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )
        elem_phi, elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi = elem_phi
        self.elem_theta = elem_theta

        return elem_density, elem_phi, elem_theta
    def show_voxels_and_surface(self):
        surface = self.points_xyz

        plotter = pv.Plotter()

        mesh = self.occupied_voxel_mesh()
        if mesh is not None:
            plotter.add_mesh(mesh, color="lightblue", opacity=0.6)
        else:
            print("No occupied voxels to display")

        cloud = self.surface_cloud()
        if cloud is not None:
            plotter.add_mesh(cloud, color="red", point_size=10, render_points_as_spheres=True)

        plotter.show()

    def surface_cloud(self):
        if self.points_xyz is None or self.points_xyz.shape[0] == 0:
            return None

        if self._surface_cloud_cache is None:
            self._surface_cloud_cache = pv.PolyData(self.points_xyz)

        return self._surface_cloud_cache

    def occupied_voxel_mesh(self, use_cache=True):
        """
        Build a PyVista mesh for all occupied voxels in one shot.

        This is much faster than creating and merging one pv.Cube per voxel.
        The result is cached because occupancy is fixed after shell voxelization.
        """
        if use_cache and self._occupied_voxel_mesh_cache is not None:
            return self._occupied_voxel_mesh_cache

        if self.elem_occupancy is None or not np.any(self.elem_occupancy):
            return None

        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        hx, hy, hz = self.grid_geom['hx'], self.grid_geom['hy'], self.grid_geom['hz']
        xmin, ymin, zmin = self.grid_geom['xmin'], self.grid_geom['ymin'], self.grid_geom['zmin']

        grid_cls = getattr(pv, "ImageData", None)
        if grid_cls is None:
            grid_cls = pv.UniformGrid

        grid = grid_cls(
            dimensions=(nelx + 1, nely + 1, nelz + 1),
            spacing=(hx, hy, hz),
            origin=(xmin, ymin, zmin),
        )

        occ_xyz = np.transpose(self.elem_occupancy.astype(np.uint8), (1, 2, 0))
        grid.cell_data["occupied"] = occ_xyz.ravel(order="F")
        mesh = grid.threshold(value=0.5, scalars="occupied")

        if use_cache:
            self._occupied_voxel_mesh_cache = mesh

        return mesh

    def get_flat_node_coords(self):
        """
        Returns
        -------
        node_ids : ndarray, shape (num_nodes,)
            Flat global node ids: 0, 1, 2, ..., num_nodes-1

        coords : ndarray, shape (num_nodes, 3)
            Flat node coordinates [x, y, z] for each node id.
        """
        coords = self.node_coords.reshape(-1, 3)
        node_ids = np.arange(coords.shape[0], dtype=np.int64)
        return node_ids, coords    
    def node_ids_to_dofs(self, node_ids, components=(0, 1, 2)):
        """
        Convert node ids to global DOF ids.

        Parameters
        ----------
        node_ids : array-like
            Global node ids.
        components : tuple
            Which displacement components to include:
            0 -> ux, 1 -> uy, 2 -> uz

        Returns
        -------
        dofs : ndarray
            Flat array of global DOF ids.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)

        dofs = []
        for c in components:
            dofs.append(3 * node_ids + int(c))

        if len(dofs) == 0:
            return np.array([], dtype=np.int64)

        return np.concatenate(dofs).astype(np.int64)
    
    def make_empty_force(self):
        """
        Create an empty global force vector of shape (ndof, 1).
        """
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        return np.zeros((ndof, 1), dtype=float)

    def element_ijk_from_flat(self, elem_id):
        elem_id = int(elem_id)
        nelx, nely = int(self.mesh["nelx"]), int(self.mesh["nely"])
        iz = elem_id // (nelx * nely)
        rem = elem_id - iz * nelx * nely
        ix = rem // nely
        iy = rem - ix * nely
        return iz, ix, iy

    def exposed_voxel_faces(self):
        occ = np.asarray(self.elem_occupancy, dtype=bool)
        nelz, nelx, nely = occ.shape
        nodes = np.arange((nelz + 1) * (nelx + 1) * (nely + 1), dtype=np.int64).reshape(
            nelz + 1,
            nelx + 1,
            nely + 1,
        )
        hx, hy, hz = self.grid_geom["hx"], self.grid_geom["hy"], self.grid_geom["hz"]
        side_defs = (
            ("x", "min", (0, -1), hy * hz, lambda z, x, y: [nodes[z, x, y], nodes[z + 1, x, y], nodes[z + 1, x, y + 1], nodes[z, x, y + 1]]),
            ("x", "max", (0, 1), hy * hz, lambda z, x, y: [nodes[z, x + 1, y], nodes[z, x + 1, y + 1], nodes[z + 1, x + 1, y + 1], nodes[z + 1, x + 1, y]]),
            ("y", "min", (1, -1), hx * hz, lambda z, x, y: [nodes[z, x, y], nodes[z, x + 1, y], nodes[z + 1, x + 1, y], nodes[z + 1, x, y]]),
            ("y", "max", (1, 1), hx * hz, lambda z, x, y: [nodes[z, x, y + 1], nodes[z + 1, x, y + 1], nodes[z + 1, x + 1, y + 1], nodes[z, x + 1, y + 1]]),
            ("z", "min", (2, -1), hx * hy, lambda z, x, y: [nodes[z, x, y], nodes[z, x, y + 1], nodes[z, x + 1, y + 1], nodes[z, x + 1, y]]),
            ("z", "max", (2, 1), hx * hy, lambda z, x, y: [nodes[z + 1, x, y], nodes[z + 1, x + 1, y], nodes[z + 1, x + 1, y + 1], nodes[z + 1, x, y + 1]]),
        )
        faces = []
        coords = self.node_coords.reshape(-1, 3)
        for iz in range(nelz):
            for ix in range(nelx):
                for iy in range(nely):
                    if not occ[iz, ix, iy]:
                        continue
                    elem_id = iy + ix * nely + iz * nelx * nely
                    for axis, side, (axis_idx, step), area, node_fn in side_defs:
                        nz, nx, ny = iz, ix, iy
                        if axis_idx == 0:
                            nx += step
                        elif axis_idx == 1:
                            ny += step
                        else:
                            nz += step
                        exposed = (
                            nx < 0 or nx >= nelx or ny < 0 or ny >= nely or nz < 0 or nz >= nelz
                            or not occ[nz, nx, ny]
                        )
                        if not exposed:
                            continue
                        face_nodes = np.asarray(node_fn(iz, ix, iy), dtype=np.int64)
                        faces.append(
                            {
                                "element_id": elem_id,
                                "axis": axis,
                                "side": side,
                                "face_axis": axis,
                                "face_side": side,
                                "nodes": face_nodes,
                                "area": float(area),
                                "centroid": coords[face_nodes].mean(axis=0),
                                "closest_point": (
                                    self.elem_closest_point[elem_id].astype(np.float64)
                                    if self.elem_closest_point is not None
                                    else self.elem_centers.reshape(-1, 3)[elem_id].astype(np.float64)
                                ),
                                "distance_to_surface": (
                                    float(self.elem_distance_to_surface[elem_id])
                                    if self.elem_distance_to_surface is not None
                                    else float("nan")
                                ),
                                "source_face_id": (
                                    int(self.elem_source_face_id[elem_id])
                                    if self.elem_source_face_id is not None
                                    else -1
                                ),
                            }
                        )
        return faces

    def _region_has_location_selector(self, region):
        if callable(region):
            return True
        if not isinstance(region, dict):
            return False
        keys = (
            "extent_axis",
            "bounds",
            "boundary_vertex_ids",
            "boundary_edge_id",
            "boundary_loop_id",
            "predicate",
        )
        return any(key in region for key in keys)

    @staticmethod
    def _axis_index(axis):
        axis = str(axis).lower()
        comp_map = {"x": 0, "y": 1, "z": 2}
        if axis not in comp_map:
            raise ValueError(f"Unsupported axis: {axis}")
        return comp_map[axis]

    def _active_closest_points(self):
        if self.elem_closest_point is not None:
            occ = np.asarray(self.elem_occupancy, dtype=bool).reshape(-1)
            pts = np.asarray(self.elem_closest_point, dtype=np.float64).reshape(-1, 3)
            valid = occ & np.all(np.isfinite(pts), axis=1)
            if np.any(valid):
                return pts[valid]
        return self.elem_centers[np.asarray(self.elem_occupancy, dtype=bool)].reshape(-1, 3)

    def _region_extent_reference(self, region):
        axis = region.get("extent_axis", None)
        if axis is None:
            return None
        axis_idx = self._axis_index(axis)
        pts = np.asarray(self.points_xyz, dtype=np.float64).reshape(-1, 3)
        source_face_id = region.get("source_face_id", region.get("cad_face_id", None))
        if source_face_id is not None and self.face_id is not None:
            allowed = set(int(v) for v in np.asarray(source_face_id, dtype=np.int64).reshape(-1))
            face_ids = np.asarray(self.face_id, dtype=np.int64).reshape(-1)
            mask = np.asarray([int(v) in allowed for v in face_ids], dtype=bool)
            if not np.any(mask):
                raise ValueError(
                    f"No CAD surface points match source_face_id/cad_face_id={sorted(allowed)}."
                )
            pts = pts[mask]
        if pts.size == 0:
            raise ValueError("Cannot compute region extent for an empty CAD surface.")
        side = str(region.get("extent_side", "min")).lower()
        if side == "min":
            return float(np.min(pts[:, axis_idx]))
        if side == "max":
            return float(np.max(pts[:, axis_idx]))
        raise ValueError("extent_side must be 'min' or 'max'.")

    def _selected_closest_extent(self, faces, region):
        if not isinstance(region, dict) or region.get("extent_axis", None) is None or len(faces) == 0:
            return None
        axis_idx = self._axis_index(region["extent_axis"])
        values = np.asarray([face["closest_point"][axis_idx] for face in faces], dtype=np.float64)
        side = str(region.get("extent_side", "min")).lower()
        return float(values.min() if side == "min" else values.max())

    @staticmethod
    def _point_segment_distance(points, a, b):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        a = np.asarray(a, dtype=np.float64).reshape(3)
        b = np.asarray(b, dtype=np.float64).reshape(3)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-20:
            return np.linalg.norm(points - a[None, :], axis=1)
        t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
        closest = a[None, :] + t[:, None] * ab[None, :]
        return np.linalg.norm(points - closest, axis=1)

    def _coerce_boundary_polyline_entry(self, entry, *, default_closed):
        closed = bool(default_closed)
        coords = None
        ids = None

        if isinstance(entry, dict):
            closed = bool(entry.get("closed", closed))
            for key in ("vertex_ids", "point_indices", "indices"):
                if key in entry:
                    ids = np.asarray(entry[key], dtype=np.int64).reshape(-1)
                    break
            for key in ("polyline", "polyline_coordinates", "coordinates", "points", "points_xyz"):
                if key in entry:
                    coords = np.asarray(entry[key], dtype=np.float64).reshape(-1, 3)
                    break
        else:
            array = np.asarray(entry)
            if array.ndim == 2 and array.shape[1] == 3:
                coords = array.astype(np.float64, copy=False)
            else:
                ids = np.asarray(entry, dtype=np.int64).reshape(-1)

        if ids is not None:
            if ids.size == 0:
                raise ValueError("Boundary polyline vertex list must not be empty.")
            if ids.min() < 0 or ids.max() >= self.points_xyz.shape[0]:
                raise ValueError("Boundary polyline contains invalid vertex indices.")
            coords = self.points_xyz[ids].astype(np.float64, copy=False)

        if coords is None or coords.shape[0] == 0:
            raise ValueError(
                "Boundary polyline entries must provide vertex_ids, point_indices, "
                "indices, polyline coordinates, or points."
            )
        return coords, ids, closed

    def _boundary_polyline(self, region):
        vertex_ids = region.get("boundary_vertex_ids", None)
        if vertex_ids is not None:
            if "closed" not in region:
                raise ValueError("boundary_vertex_ids regions must explicitly set closed=True or closed=False.")
            return self._coerce_boundary_polyline_entry(
                {"vertex_ids": vertex_ids, "closed": bool(region["closed"])},
                default_closed=False,
            )

        edge_id = region.get("boundary_edge_id", None)
        if edge_id is not None:
            key = edge_id if edge_id in self.boundary_edges else str(edge_id)
            if key not in self.boundary_edges:
                raise ValueError(f"boundary_edge_id {edge_id!r} is not present in tensors['boundary_edges'].")
            coords, ids, closed = self._coerce_boundary_polyline_entry(
                self.boundary_edges[key],
                default_closed=False,
            )
            if "closed" in region:
                closed = bool(region["closed"])
            return coords, ids, closed

        loop_id = region.get("boundary_loop_id", None)
        if loop_id is not None:
            key = loop_id if loop_id in self.boundary_loops else str(loop_id)
            if key not in self.boundary_loops:
                raise ValueError(f"boundary_loop_id {loop_id!r} is not present in tensors['boundary_loops'].")
            coords, ids, closed = self._coerce_boundary_polyline_entry(
                self.boundary_loops[key],
                default_closed=True,
            )
            if "closed" in region:
                closed = bool(region["closed"])
            return coords, ids, closed

        return None

    def _distance_to_boundary_polyline(self, points, polyline, closed=False):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        polyline = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
        if polyline.shape[0] == 1:
            return np.linalg.norm(points - polyline[0][None, :], axis=1)
        distances = np.full((points.shape[0],), np.inf, dtype=np.float64)
        for a, b in zip(polyline[:-1], polyline[1:]):
            distances = np.minimum(
                distances,
                self._point_segment_distance(points, a, b),
            )
        if bool(closed) and polyline.shape[0] > 2:
            distances = np.minimum(
                distances,
                self._point_segment_distance(points, polyline[-1], polyline[0]),
            )
        return distances

    def _face_region_distance(self, face, region):
        point = np.asarray(face["closest_point"], dtype=np.float64).reshape(1, 3)
        polyline = self._boundary_polyline(region) if isinstance(region, dict) else None
        if polyline is not None:
            coords, _ids, closed = polyline
            return float(self._distance_to_boundary_polyline(point, coords, closed=closed)[0])

        if isinstance(region, dict) and region.get("extent_axis", None) is not None:
            axis_idx = self._axis_index(region["extent_axis"])
            ref = self._region_extent_reference(region)
            return abs(float(point[0, axis_idx]) - float(ref))

        return float("nan")

    def _region_contains_face(self, region, face):
        if region is None:
            return False
        if callable(region):
            return bool(region(face))
        if not isinstance(region, dict):
            raise ValueError("Boundary regions must be dicts or callables.")

        source_face_id = region.get("source_face_id", region.get("cad_face_id", None))
        if source_face_id is not None:
            if not self._region_has_location_selector(region):
                raise ValueError(
                    "source_face_id alone is not a boundary region. Add extent_axis, "
                    "bounds, boundary_vertex_ids, boundary_edge_id, boundary_loop_id, "
                    "or predicate."
                )
            allowed = np.asarray(source_face_id, dtype=np.int64).reshape(-1)
            if int(face["source_face_id"]) not in set(int(v) for v in allowed):
                return False

        face_axis = region.get("face_axis", None)
        face_side = region.get("face_side", None)
        if face_axis is not None:
            face_axis = str(face_axis).lower()
            if face["face_axis"] != face_axis:
                return False
            if face_side is not None and face["face_side"] != str(face_side).lower():
                return False

        bounds = region.get("bounds", None)
        if bounds is not None:
            c = np.asarray(face["centroid"], dtype=float)
            for idx, key in enumerate(("x", "y", "z")):
                lo = bounds.get(f"{key}min", None)
                hi = bounds.get(f"{key}max", None)
                if lo is not None and c[idx] < float(lo):
                    return False
                if hi is not None and c[idx] > float(hi):
                    return False

        extent_axis = region.get("extent_axis", None)
        if extent_axis is not None:
            axis_idx = self._axis_index(extent_axis)
            ref = self._region_extent_reference(region)
            band = float(region.get("band", 0.75 * self.voxel_size))
            if band < 0.0:
                raise ValueError("region band must be non-negative.")
            value = float(np.asarray(face["closest_point"], dtype=np.float64)[axis_idx])
            if abs(value - ref) > band + 1.0e-12:
                return False

        polyline = self._boundary_polyline(region)
        if polyline is not None:
            band = float(region.get("band", self.voxel_size))
            coords, _ids, closed = polyline
            distance = self._distance_to_boundary_polyline(
                np.asarray(face["closest_point"], dtype=np.float64).reshape(1, 3),
                coords,
                closed=closed,
            )[0]
            if distance > band + 1.0e-12:
                return False

        predicate = region.get("predicate", None)
        if predicate is not None and not bool(predicate(face)):
            return False

        return True

    def select_exposed_faces_by_region(self, region):
        return [face for face in self.exposed_voxel_faces() if self._region_contains_face(region, face)]

    def _surface_point_bounds_for_faces(self, faces):
        if len(faces) == 0:
            return None
        pts = np.asarray([face["closest_point"] for face in faces], dtype=np.float64)
        return {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "centroid": pts.mean(axis=0).tolist(),
        }

    def _region_distance_range_for_faces(self, faces, region):
        if len(faces) == 0:
            return None, None
        distances = np.asarray(
            [self._face_region_distance(face, region) for face in faces],
            dtype=np.float64,
        )
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            return None, None
        return float(finite.min()), float(finite.max())

    def _assert_faces_inside_region_band(self, faces, region, label):
        if callable(region) or not isinstance(region, dict):
            return
        if "extent_axis" not in region and self._boundary_polyline(region) is None:
            return
        band = float(region.get("band", 0.75 * self.voxel_size if "extent_axis" in region else self.voxel_size))
        _dmin, dmax = self._region_distance_range_for_faces(faces, region)
        if dmax is not None and dmax > band + 1.0e-10:
            raise ValueError(
                f"{label} region selection escaped its requested band: "
                f"max distance {dmax} > band {band}."
            )

    @staticmethod
    def _bounds_for_nodes(coords, nodes):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if nodes.size == 0:
            return None
        pts = coords[nodes]
        return {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "centroid": pts.mean(axis=0).tolist(),
        }

    def apply_area_weighted_face_force(self, force, faces, direction, total_value):
        if len(faces) == 0:
            raise ValueError("No exposed faces selected for force application.")
        comp_map = {'x': 0, 'y': 1, 'z': 2}
        direction = str(direction).lower()
        if direction not in comp_map:
            raise ValueError(f"Unsupported force direction: {direction}")
        comp = comp_map[direction]
        total_area = float(sum(face["area"] for face in faces))
        if total_area <= 0.0:
            raise ValueError("Selected load faces have zero total area.")
        for face in faces:
            face_force = float(total_value) * float(face["area"]) / total_area
            dofs = 3 * np.asarray(face["nodes"], dtype=np.int64) + comp
            force[dofs, 0] += face_force / 4.0
        recovered = float(force[comp::3, 0].sum())
        if not np.isclose(recovered, float(total_value), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(
                f"Recovered nodal force {recovered} does not match requested {total_value}."
            )
        return force, total_area

    def _region_is_boundary_tag(self, region):
        return isinstance(region, dict) and any(
            key in region for key in ("boundary_vertex_ids", "boundary_edge_id", "boundary_loop_id")
        )

    def apply_edge_weighted_force(self, force, faces, region, direction, total_value):
        if len(faces) == 0:
            raise ValueError("No exposed faces selected for edge force application.")
        polyline = self._boundary_polyline(region)
        if polyline is None:
            return self.apply_area_weighted_face_force(force, faces, direction, total_value)

        coords, _ids, closed = polyline
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if coords.shape[0] < 2:
            raise ValueError("An edge load needs at least two boundary points.")

        comp = self._axis_index(direction)
        load_nodes = np.unique(np.concatenate([face["nodes"] for face in faces])).astype(np.int64)
        node_coords = self.node_coords.reshape(-1, 3)[load_nodes]
        weights = np.zeros((load_nodes.size,), dtype=np.float64)

        segment_pairs = list(zip(coords[:-1], coords[1:]))
        if closed and coords.shape[0] > 2:
            segment_pairs.append((coords[-1], coords[0]))

        band = float(region.get("band", self.voxel_size)) if isinstance(region, dict) else self.voxel_size
        for a, b in segment_pairs:
            length = float(np.linalg.norm(b - a))
            if length <= 1.0e-14:
                continue
            distances = self._point_segment_distance(node_coords, a, b)
            segment_nodes = np.flatnonzero(distances <= band + 1.0e-12)
            if segment_nodes.size == 0:
                segment_nodes = np.asarray([int(np.argmin(distances))], dtype=np.int64)
            weights[segment_nodes] += length / segment_nodes.size

        if not np.any(weights > 0.0):
            raise ValueError("Unable to assign edge-length weights to loaded nodes.")

        nodal_values = float(total_value) * weights / weights.sum()
        force[3 * load_nodes + comp, 0] += nodal_values
        actual_loaded_nodes = load_nodes[weights > 0.0]
        recovered = float(force[comp::3, 0].sum())
        if not np.isclose(recovered, float(total_value), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(
                f"Recovered edge load {recovered} does not match requested {total_value}."
            )
        return force, float(weights.sum()), actual_loaded_nodes

    def _surface_region_default(self, axis, side, *, direction=None, total_force=None, band=None):
        region = {
            "extent_axis": str(axis).lower(),
            "extent_side": str(side).lower(),
            "band": float(0.75 * self.voxel_size if band is None else band),
        }
        if direction is not None:
            region["direction"] = str(direction).lower()
        if total_force is not None:
            region["total_force"] = float(total_force)
        return region

    def _surface_region_middle(self, span_axis, *, direction, total_force):
        axis_idx = self._axis_index(span_axis)
        pts = np.asarray(self.points_xyz, dtype=np.float64).reshape(-1, 3)
        lo = float(pts[:, axis_idx].min())
        hi = float(pts[:, axis_idx].max())
        center = 0.5 * (lo + hi)
        band = 0.75 * self.voxel_size
        key = str(span_axis).lower()
        bounds = {f"{key}min": center - 0.5 * band, f"{key}max": center + 0.5 * band}
        region = {
            "bounds": bounds,
            "direction": str(direction).lower(),
            "total_force": float(total_force),
            "band": band,
        }
        if self.load_surface_dir is not None:
            region["extent_axis"] = self.load_surface_dir
            region["extent_side"] = self.load_surface_side
        return region

    def _select_faces_and_nodes(self, region, label):
        faces = self.select_exposed_faces_by_region(region)
        if len(faces) == 0:
            raise ValueError(f"No exposed voxel faces matched {label}_region={region!r}.")
        self._assert_faces_inside_region_band(faces, region, label)
        nodes = np.unique(np.concatenate([face["nodes"] for face in faces])).astype(np.int64)
        return faces, nodes

    def _set_surface_patch_report(
        self,
        *,
        load_case,
        fixed_regions,
        load_region,
        fixed_faces,
        load_faces,
        fixed_nodes,
        load_nodes,
        selected_measure,
        requested_total,
        overlap_before,
        overlap_removed,
        extra=None,
    ):
        node_coords = self.node_coords.reshape(-1, 3)
        force_array = self.boundaryCondition["force"].reshape(-1, 3)
        loaded_force = force_array.sum(axis=0)
        fixed_dist_min, fixed_dist_max = self._region_distance_range_for_faces(fixed_faces, fixed_regions[0])
        load_dist_min, load_dist_max = self._region_distance_range_for_faces(load_faces, load_region)
        report = {
            "mode": "surface_patch",
            "bc_mapping_mode": self.bc_mapping_mode,
            "load_case": load_case,
            "fixed_face_count": int(len(fixed_faces)),
            "loaded_face_count": int(len(load_faces)),
            "fixed_node_count": int(fixed_nodes.size),
            "loaded_node_count": int(load_nodes.size),
            "fixed_bounds": self._bounds_for_nodes(node_coords, fixed_nodes),
            "loaded_bounds": self._bounds_for_nodes(node_coords, load_nodes),
            "fixed_closest_surface_point_bounds": self._surface_point_bounds_for_faces(fixed_faces),
            "loaded_closest_surface_point_bounds": self._surface_point_bounds_for_faces(load_faces),
            "fixed_patch_distance_min": fixed_dist_min,
            "fixed_patch_distance_max": fixed_dist_max,
            "loaded_patch_distance_min": load_dist_min,
            "loaded_patch_distance_max": load_dist_max,
            "fixed_cad_extent_reference": (
                self._region_extent_reference(fixed_regions[0])
                if isinstance(fixed_regions[0], dict) and fixed_regions[0].get("extent_axis") is not None
                else None
            ),
            "loaded_cad_extent_reference": (
                self._region_extent_reference(load_region)
                if isinstance(load_region, dict) and load_region.get("extent_axis") is not None
                else None
            ),
            "fixed_selected_closest_extent": self._selected_closest_extent(fixed_faces, fixed_regions[0]),
            "loaded_selected_closest_extent": self._selected_closest_extent(load_faces, load_region),
            "selected_loaded_area": None if self._region_is_boundary_tag(load_region) else float(selected_measure),
            "selected_loaded_edge_length_weight": float(selected_measure) if self._region_is_boundary_tag(load_region) else None,
            "requested_total_force": float(requested_total),
            "recovered_total_force_vector": loaded_force.tolist(),
            "selected_resultant_load": loaded_force.tolist(),
            "overlap_node_count": int(overlap_before),
            "overlap_node_count_removed": int(overlap_removed),
        }
        if len(fixed_regions) > 1:
            report["support_region_count"] = int(len(fixed_regions))
            report["support_face_counts"] = [
                int(len(self.select_exposed_faces_by_region(region))) for region in fixed_regions
            ]
        if extra:
            report.update(extra)
        self.bc_report = report

    def _apply_surface_patch_force_bc(self, fixed_regions, load_region, *, force_direction, force_value, load_case):
        fixed_face_parts = []
        fixed_node_parts = []
        for idx, region in enumerate(fixed_regions):
            faces, nodes = self._select_faces_and_nodes(region, f"fixed_{idx}")
            fixed_face_parts.extend(faces)
            fixed_node_parts.append(nodes)
        load_faces, load_nodes = self._select_faces_and_nodes(load_region, "loaded")

        fixed_nodes = np.unique(np.concatenate(fixed_node_parts)).astype(np.int64)
        overlap = np.intersect1d(fixed_nodes, load_nodes)
        overlap_before = int(overlap.size)
        allow_overlap = bool(isinstance(load_region, dict) and load_region.get("allow_overlap", False))
        if overlap.size > 0 and not allow_overlap:
            load_faces = [
                face for face in load_faces
                if np.intersect1d(face["nodes"], overlap).size == 0
            ]
            load_nodes = (
                np.unique(np.concatenate([face["nodes"] for face in load_faces])).astype(np.int64)
                if load_faces
                else np.array([], dtype=np.int64)
            )
        if load_nodes.size == 0:
            raise ValueError("Loaded region overlaps the fixed region and no loaded nodes remain.")

        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        if self._region_is_boundary_tag(load_region):
            force, selected_measure, load_nodes = self.apply_edge_weighted_force(
                force,
                load_faces,
                load_region,
                force_direction,
                force_value,
            )
        else:
            force, selected_measure = self.apply_area_weighted_face_force(
                force,
                load_faces,
                force_direction,
                force_value,
            )
        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        reported_overlap = int(np.intersect1d(fixed_nodes, load_nodes).size)
        reported_overlap_removed = int(max(0, overlap_before - reported_overlap))
        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }
        self._set_surface_patch_report(
            load_case=load_case,
            fixed_regions=fixed_regions,
            load_region=load_region,
            fixed_faces=fixed_face_parts,
            load_faces=load_faces,
            fixed_nodes=fixed_nodes,
            load_nodes=load_nodes,
            selected_measure=selected_measure,
            requested_total=force_value,
            overlap_before=reported_overlap,
            overlap_removed=0 if allow_overlap else reported_overlap_removed,
        )

    def apply_surface_patch_tensile_compression(self):
        fixed_region = self.fixed_region or self._surface_region_default(
            self.BC_dir,
            "min",
        )
        load_region = self.load_region or self._surface_region_default(
            self.BC_dir,
            "max",
            direction=self.BC_dir,
            total_force=self.Load_magnitude,
        )
        force_direction = (
            load_region.get("direction", self.BC_dir)
            if isinstance(load_region, dict)
            else self.BC_dir
        )
        force_value = (
            load_region.get("total_force", self.Load_magnitude)
            if isinstance(load_region, dict)
            else self.Load_magnitude
        )
        self._apply_surface_patch_force_bc(
            [fixed_region],
            load_region,
            force_direction=force_direction,
            force_value=float(force_value),
            load_case="tensile_compression",
        )

    def apply_surface_patch_fixed_side_loading(self):
        self._axis_bounds_keys(self.BC_dir)
        fixed_side = self.fixed_side
        force_side = self.force_side if self.force_side is not None else self._opposite_side(fixed_side)
        if force_side == fixed_side:
            raise ValueError(
                "force_side must be opposite to fixed_side for load_case='fixed_side_loading'"
            )

        force_direction = self.load_dir if self.load_dir is not None else self.BC_dir
        self._axis_bounds_keys(force_direction)
        force_value = self.Load_magnitude
        if self.load_direction_side is not None:
            force_value = abs(force_value) * self._direction_side_sign(self.load_direction_side)

        fixed_region = self.fixed_region or self._surface_region_default(self.BC_dir, fixed_side)
        load_region = self.load_region or self._surface_region_default(
            self.BC_dir,
            force_side,
            direction=force_direction,
            total_force=force_value,
        )
        force_direction = (
            load_region.get("direction", force_direction)
            if isinstance(load_region, dict)
            else force_direction
        )
        force_value = (
            load_region.get("total_force", force_value)
            if isinstance(load_region, dict)
            else force_value
        )
        self._apply_surface_patch_force_bc(
            [fixed_region],
            load_region,
            force_direction=force_direction,
            force_value=float(force_value),
            load_case="fixed_side_loading",
        )

    def apply_surface_patch_three_point_bending(self):
        load_dir = self.load_dir
        if load_dir is None:
            raise ValueError("load_dir must be provided for load_case='three_point_bending'")
        self._axis_bounds_keys(load_dir)

        if self.support_regions is not None:
            fixed_regions = list(self.support_regions)
        else:
            region_min = self.fixed_region_1 or self._surface_region_default(self.BC_dir, "min")
            region_max = self.fixed_region_2 or self._surface_region_default(self.BC_dir, "max")
            fixed_regions = [region_min, region_max]

        load_region = self.load_region or self._surface_region_middle(
            self.BC_dir,
            direction=load_dir,
            total_force=self.Load_magnitude,
        )
        force_direction = (
            load_region.get("direction", load_dir)
            if isinstance(load_region, dict)
            else load_dir
        )
        force_value = (
            load_region.get("total_force", self.Load_magnitude)
            if isinstance(load_region, dict)
            else self.Load_magnitude
        )
        self._apply_surface_patch_force_bc(
            fixed_regions,
            load_region,
            force_direction=force_direction,
            force_value=float(force_value),
            load_case="three_point_bending",
        )

    def _pin_bearing_loop(self):
        if not isinstance(self.boundary_loops, dict) or len(self.boundary_loops) == 0:
            raise ValueError("pin_bearing_bracket requires a closed CAD hole boundary.")

        def normalize_loop(loop_id, entry):
            if not isinstance(entry, dict):
                return {
                    "loop_id": int(loop_id),
                    "loop_kind": None,
                    "coordinates": np.asarray(entry, dtype=np.float64).reshape(-1, 3),
                    "closed": True,
                }
            out = dict(entry)
            if "loop_id" not in out:
                out["loop_id"] = int(loop_id)
            return out

        loops = {}
        for key, entry in self.boundary_loops.items():
            loop = normalize_loop(key, entry)
            loops[int(loop["loop_id"])] = loop

        if self.bearing_loop_id is not None:
            loop_id = int(self.bearing_loop_id)
            if loop_id not in loops:
                raise ValueError(f"bearing_loop_id={loop_id} is not present in boundary_loops.")
            loop = loops[loop_id]
            if str(loop.get("loop_kind", "")).lower() != "hole":
                raise ValueError(f"bearing_loop_id={loop_id} must identify a CAD hole boundary.")
            return loop_id, loop

        hole_ids = sorted(
            loop_id
            for loop_id, loop in loops.items()
            if str(loop.get("loop_kind", "")).lower() == "hole"
        )
        if len(hole_ids) == 0:
            raise ValueError("pin_bearing_bracket requires a closed CAD hole boundary.")
        if len(hole_ids) > 1:
            raise ValueError(
                "pin_bearing_bracket found multiple CAD hole boundaries "
                f"{hole_ids}; set bearing_loop_id to choose one."
            )
        loop_id = hole_ids[0]
        return loop_id, loops[loop_id]

    def _bearing_weights_for_nodes(
        self,
        candidate_nodes,
        loop_coordinates,
        *,
        axis,
        contact_side,
        band,
        exponent,
    ):
        candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
        if candidate_nodes.size == 0:
            raise ValueError("No candidate nodes are available for pin-bearing force application.")
        coords = np.asarray(loop_coordinates, dtype=np.float64).reshape(-1, 3)
        if coords.shape[0] < 3:
            raise ValueError("pin_bearing_bracket requires a closed CAD hole boundary with at least three points.")
        if np.linalg.norm(coords[0] - coords[-1]) <= 1.0e-10:
            coords = coords[:-1]
        if coords.shape[0] < 3:
            raise ValueError("pin_bearing_bracket requires a closed CAD hole boundary with at least three unique points.")

        segment_pairs = list(zip(coords, np.roll(coords, -1, axis=0)))
        lengths = np.asarray([np.linalg.norm(b - a) for a, b in segment_pairs], dtype=np.float64)
        valid_lengths = lengths > 1.0e-14
        if not np.any(valid_lengths):
            raise ValueError("CAD hole boundary has zero perimeter.")

        midpoints = np.asarray([(a + b) * 0.5 for a, b in segment_pairs], dtype=np.float64)
        centre = (lengths[valid_lengths, None] * midpoints[valid_lengths]).sum(axis=0) / lengths[valid_lengths].sum()
        centred_loop = coords - centre[None, :]
        _u, _s, vh = np.linalg.svd(centred_loop, full_matrices=False)
        plane_normal = np.asarray(vh[-1], dtype=np.float64)
        plane_normal_norm = float(np.linalg.norm(plane_normal))
        if plane_normal_norm <= 1.0e-14:
            raise ValueError("Unable to determine pin-bearing CAD hole plane.")
        plane_normal /= plane_normal_norm

        axis_idx = self._axis_index(axis)
        sign = -1.0 if str(contact_side).lower() == "min" else 1.0
        node_coords = self.node_coords.reshape(-1, 3)[candidate_nodes].astype(np.float64, copy=False)
        relative = node_coords - centre[None, :]
        normal_offset = relative @ plane_normal
        projected_node_coords = node_coords - normal_offset[:, None] * plane_normal[None, :]
        contacting_node_mask = sign * (projected_node_coords[:, axis_idx] - centre[axis_idx]) >= -1.0e-12
        candidate_nodes = candidate_nodes[contacting_node_mask]
        node_coords = node_coords[contacting_node_mask]
        projected_node_coords = projected_node_coords[contacting_node_mask]
        normal_offset = normal_offset[contacting_node_mask]
        if candidate_nodes.size == 0:
            raise ValueError("No candidate nodes remain on the contacting half of the pin-bearing hole.")
        weights = np.zeros((candidate_nodes.size,), dtype=np.float64)

        for (a, b), length, midpoint in zip(segment_pairs, lengths, midpoints):
            if length <= 1.0e-14:
                continue
            radial = midpoint - centre
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm <= 1.0e-14:
                continue
            cosine = sign * float(radial[axis_idx]) / radial_norm
            if cosine <= 0.0:
                continue
            segment_weight = float(length) * (float(cosine) ** float(exponent))
            if segment_weight <= 0.0:
                continue
            distances = self._point_segment_distance(projected_node_coords, a, b)
            local = np.flatnonzero(distances <= float(band) + 1.0e-12)
            if local.size == 0:
                local = np.asarray([int(np.argmin(distances))], dtype=np.int64)
            weights[local] += segment_weight / float(local.size)

        positive = weights > 0.0
        if not np.any(positive):
            raise ValueError("Unable to assign cosine pin-bearing weights to candidate nodes.")
        loaded_nodes = candidate_nodes[positive]
        loaded_node_coords = node_coords[positive]
        raw_weights = weights[positive]
        q = self._project_bearing_weights_to_zero_moment(raw_weights, loaded_node_coords, centre)
        positive_corrected = q > 1.0e-14
        if not np.any(positive_corrected):
            raise ValueError("Corrected pin-bearing weights are all zero.")
        loaded_nodes = loaded_nodes[positive_corrected]
        loaded_node_coords = loaded_node_coords[positive_corrected]
        q = q[positive_corrected]
        q = q / float(q.sum())
        corrected_normal_offset = (loaded_node_coords - centre[None, :]) @ plane_normal
        force_centroid = np.sum(q[:, None] * loaded_node_coords, axis=0)
        metadata = {
            "centre": centre,
            "plane_normal": plane_normal,
            "normal_offset": corrected_normal_offset,
            "force_centroid": force_centroid,
            "characteristic_radius": float(np.mean(np.linalg.norm(coords - centre[None, :], axis=1))),
        }
        return loaded_nodes, q, centre, metadata

    @staticmethod
    def _project_bearing_weights_to_zero_moment(raw_weights, node_coords, centre):
        raw_weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        centre = np.asarray(centre, dtype=np.float64).reshape(3)
        if raw_weights.size == 0 or raw_weights.size != node_coords.shape[0]:
            raise ValueError("Pin-bearing weight projection received inconsistent node weights.")
        if np.any(raw_weights < 0.0):
            raise ValueError("Pin-bearing raw weights must be non-negative.")
        total = float(raw_weights.sum())
        if total <= 0.0:
            raise ValueError("Pin-bearing raw weights must have positive sum.")

        q0_full = raw_weights / total
        free = np.arange(q0_full.size, dtype=np.int64)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tol = 1.0e-12

        while True:
            if free.size == 0:
                raise ValueError("Pin-bearing zero-moment constraints are infeasible after non-negative projection.")
            coords_free = node_coords[free]
            A = np.vstack(
                [
                    np.ones(free.size, dtype=np.float64),
                    coords_free[:, 1] - centre[1],
                    coords_free[:, 2] - centre[2],
                ]
            )
            q0 = q0_full[free]
            gram = A @ A.T
            correction = A.T @ (np.linalg.pinv(gram, rcond=1.0e-12) @ (A @ q0 - b))
            q_free = q0 - correction
            if np.linalg.norm(A @ q_free - b, ord=np.inf) > 1.0e-9:
                raise ValueError("Pin-bearing zero-moment constraints are infeasible for the selected nodes.")
            negative = q_free < -tol
            if not np.any(negative):
                q = np.zeros_like(q0_full)
                q_free = np.where(q_free < 0.0, 0.0, q_free)
                q[free] = q_free
                constraint_error = A @ q[free] - b
                if np.linalg.norm(constraint_error, ord=np.inf) > 1.0e-9:
                    raise ValueError("Pin-bearing zero-moment projection failed to satisfy constraints.")
                q_sum = float(q.sum())
                if q_sum <= 0.0:
                    raise ValueError("Pin-bearing zero-moment projection produced zero total weight.")
                return q / q_sum
            free = free[~negative]

    def _apply_cosine_pin_bearing_force(
        self,
        force,
        candidate_nodes,
        loop_coordinates,
        *,
        axis,
        contact_side,
        total_force,
        band,
        exponent,
        fixed_nodes=None,
    ):
        candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
        fixed_nodes = (
            np.array([], dtype=np.int64)
            if fixed_nodes is None
            else np.asarray(fixed_nodes, dtype=np.int64).reshape(-1)
        )

        loaded_nodes, weights, centre, metadata = self._bearing_weights_for_nodes(
            candidate_nodes,
            loop_coordinates,
            axis=axis,
            contact_side=contact_side,
            band=band,
            exponent=exponent,
        )
        overlap = np.intersect1d(loaded_nodes, fixed_nodes)
        overlap_removed = int(overlap.size)
        if overlap_removed > 0:
            remaining_candidates = np.setdiff1d(candidate_nodes, fixed_nodes, assume_unique=False)
            if remaining_candidates.size == 0:
                raise ValueError("Loaded pin-bearing nodes overlap the fixed region and no loaded nodes remain.")
            loaded_nodes, weights, centre, metadata = self._bearing_weights_for_nodes(
                remaining_candidates,
                loop_coordinates,
                axis=axis,
                contact_side=contact_side,
                band=band,
                exponent=exponent,
            )
            if loaded_nodes.size == 0:
                raise ValueError("Loaded pin-bearing nodes overlap the fixed region and no loaded nodes remain.")

        comp = self._axis_index(axis)
        signed_total_force = float(total_force)
        nodal_values = signed_total_force * weights
        force[3 * loaded_nodes + comp, 0] += nodal_values

        loaded_force = force.reshape(-1, 3)[loaded_nodes]
        loaded_coords = self.node_coords.reshape(-1, 3)[loaded_nodes].astype(np.float64, copy=False)
        resultant = loaded_force.sum(axis=0)
        expected = np.zeros(3, dtype=np.float64)
        expected[comp] = signed_total_force
        if not np.allclose(resultant, expected, rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(
                f"Recovered pin-bearing force {resultant.tolist()} does not match requested {expected.tolist()}."
            )
        moment = np.cross(loaded_coords - centre[None, :], loaded_force).sum(axis=0)
        characteristic_radius = float(metadata["characteristic_radius"])
        moment_tolerance = max(1.0e-10, 1.0e-8 * abs(signed_total_force) * characteristic_radius)
        if not np.allclose(moment, np.zeros(3), rtol=0.0, atol=moment_tolerance):
            raise ValueError(
                f"Pin-bearing resultant moment {moment.tolist()} about hole centre exceeds "
                f"tolerance {moment_tolerance}."
            )
        metadata = dict(metadata)
        metadata["resultant"] = resultant
        metadata["moment"] = moment
        metadata["moment_tolerance"] = moment_tolerance
        metadata["weights"] = weights
        return force, float(weights.sum()), loaded_nodes, centre, overlap_removed, metadata

    def apply_surface_patch_pin_bearing_bracket(self):
        if self.load_dir is not None and self.load_dir != "x":
            raise ValueError("pin_bearing_bracket applies force only in global X; load_dir must be 'x' if provided.")
        if self.load_direction_side is not None and self.load_direction_side != self.bearing_contact_side:
            raise ValueError(
                "pin_bearing_bracket requires load_direction_side to match bearing_contact_side "
                "('min' gives negative X, 'max' gives positive X)."
            )

        loop_id, loop = self._pin_bearing_loop()
        loop_coordinates = np.asarray(loop.get("coordinates"), dtype=np.float64).reshape(-1, 3)
        requested_force = abs(self.Load_magnitude) * self._direction_side_sign(self.bearing_contact_side)

        fixed_region = self.fixed_region or self._surface_region_default(
            self.BC_dir,
            self.fixed_side,
            band=self.fixed_patch_band,
        )
        load_region = self.load_region or {
            "boundary_loop_id": int(loop_id),
            "band": float(self.bearing_band),
        }
        if isinstance(load_region, dict) and "boundary_loop_id" in load_region:
            requested_loop_id = int(load_region["boundary_loop_id"])
            if requested_loop_id != int(loop_id):
                raise ValueError(
                    f"load_region boundary_loop_id={requested_loop_id} does not match "
                    f"selected bearing_loop_id={int(loop_id)}."
                )
        fixed_faces, fixed_nodes = self._select_faces_and_nodes(fixed_region, "fixed")
        load_faces, candidate_nodes = self._select_faces_and_nodes(load_region, "loaded")

        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        force, selected_measure, load_nodes, centre, overlap_removed, bearing_metadata = self._apply_cosine_pin_bearing_force(
            force,
            candidate_nodes,
            loop_coordinates,
            axis="x",
            contact_side=self.bearing_contact_side,
            total_force=requested_force,
            band=self.bearing_band,
            exponent=self.bearing_pressure_exponent,
            fixed_nodes=fixed_nodes,
        )
        contacting_load_faces = [
            face for face in load_faces
            if np.intersect1d(np.asarray(face["nodes"], dtype=np.int64), load_nodes).size > 0
        ]
        normal_offset = np.asarray(bearing_metadata["normal_offset"], dtype=np.float64).reshape(-1)
        weights = np.asarray(bearing_metadata["weights"], dtype=np.float64).reshape(-1)
        if normal_offset.size != weights.size:
            raise ValueError("Internal pin-bearing report metadata has inconsistent normal offsets.")
        unique_offsets = np.unique(np.round(normal_offset, decimals=10))

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }
        self._set_surface_patch_report(
            load_case="pin_bearing_bracket",
            fixed_regions=[fixed_region],
            load_region=load_region,
            fixed_faces=fixed_faces,
            load_faces=contacting_load_faces,
            fixed_nodes=fixed_nodes,
            load_nodes=load_nodes,
            selected_measure=selected_measure,
            requested_total=requested_force,
            overlap_before=overlap_removed,
            overlap_removed=overlap_removed,
            extra={
                "candidate_face_count": int(len(load_faces)),
                "contacting_face_count": int(len(contacting_load_faces)),
                "bearing_loop_id": int(loop_id),
                "bearing_contact_side": self.bearing_contact_side,
                "bearing_pressure_exponent": float(self.bearing_pressure_exponent),
                "bearing_band": float(self.bearing_band),
                "fixed_patch_band": float(self.fixed_patch_band),
                "bearing_centre": centre.tolist(),
                "bearing_plane_normal": np.asarray(bearing_metadata["plane_normal"], dtype=float).tolist(),
                "bearing_force_centroid": np.asarray(bearing_metadata["force_centroid"], dtype=float).tolist(),
                "bearing_resultant_moment_about_centre": np.asarray(bearing_metadata["moment"], dtype=float).tolist(),
                "loaded_normal_offset_min": float(normal_offset.min()),
                "loaded_normal_offset_max": float(normal_offset.max()),
                "loaded_normal_offset_weighted_mean": float(np.dot(weights, normal_offset)),
                "loaded_thickness_layer_count": int(unique_offsets.size),
            },
        )

    def _force_resultant_and_moment(self, node_ids, axis, center=None):
        axis_vecs = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        axis = str(axis).lower()
        if axis not in axis_vecs:
            raise ValueError(f"Unsupported torque axis: {axis}")
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        coords = self.node_coords.reshape(-1, 3)[node_ids].astype(float, copy=False)
        forces = self.boundaryCondition["force"].reshape(-1, 3)[node_ids].astype(float, copy=False)
        if center is None:
            center = coords.mean(axis=0)
        resultant = forces.sum(axis=0)
        moment = np.cross(coords - center[None, :], forces).sum(axis=0)
        return resultant, moment, float(np.dot(moment, axis_vecs[axis])), center

    def apply_surface_patch_torsion(self):
        self._axis_bounds_keys(self.BC_dir)
        fixed_side = self.fixed_side
        force_side = self.force_side if self.force_side is not None else self._opposite_side(fixed_side)
        if force_side == fixed_side:
            raise ValueError("force_side must be opposite to fixed_side for load_case='torsion'")

        fixed_region = self.fixed_region or self._surface_region_default(self.BC_dir, fixed_side)
        torque_region = self.load_region or self._surface_region_default(self.BC_dir, force_side)
        fixed_faces, fixed_nodes = self._select_faces_and_nodes(fixed_region, "fixed")
        torque_faces, torque_nodes = self._select_faces_and_nodes(torque_region, "torque")
        overlap = np.intersect1d(fixed_nodes, torque_nodes)
        if overlap.size > 0:
            torque_faces = [
                face for face in torque_faces
                if np.intersect1d(face["nodes"], overlap).size == 0
            ]
            torque_nodes = (
                np.unique(np.concatenate([face["nodes"] for face in torque_faces])).astype(np.int64)
                if torque_faces
                else np.array([], dtype=np.int64)
            )
        if torque_nodes.size == 0:
            raise ValueError("Torque region overlaps the fixed region and no torque nodes remain.")

        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_torque(force, torque_nodes, self.BC_dir, self.Load_magnitude)
        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }
        resultant, moment, recovered_torque, center = self._force_resultant_and_moment(
            torque_nodes,
            self.BC_dir,
        )
        if not np.allclose(resultant, np.zeros(3), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(f"Torsion load has unintended resultant force {resultant.tolist()}.")
        if not np.isclose(recovered_torque, self.Load_magnitude, rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(
                f"Recovered torque {recovered_torque} does not match requested {self.Load_magnitude}."
            )
        self._set_surface_patch_report(
            load_case="torsion",
            fixed_regions=[fixed_region],
            load_region=torque_region,
            fixed_faces=fixed_faces,
            load_faces=torque_faces,
            fixed_nodes=fixed_nodes,
            load_nodes=torque_nodes,
            selected_measure=float(len(torque_nodes)),
            requested_total=self.Load_magnitude,
            overlap_before=int(overlap.size),
            overlap_removed=int(overlap.size),
            extra={
                "torque_axis": self.BC_dir,
                "torque_center": center.tolist(),
                "recovered_resultant_force": resultant.tolist(),
                "recovered_moment_vector": moment.tolist(),
                "recovered_torque": recovered_torque,
            },
        )

    def apply_nodal_force(self, force, node_ids, direction, total_value):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("No nodes selected for force application")

        comp_map = {'x': 0, 'y': 1, 'z': 2}
        c = comp_map[direction]

        val_per_node = total_value / node_ids.size
        dofs = 3 * node_ids + c
        force[dofs, 0] += val_per_node
        return force

    def apply_nodal_torque(self, force, node_ids, axis, total_torque):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("No nodes selected for torque application")

        axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        axis = str(axis).lower()
        if axis not in axis_map:
            raise ValueError(f"Unsupported torque axis: {axis}")

        _all_node_ids, coords = self.get_flat_node_coords()
        pts = coords[node_ids].astype(float, copy=False)
        axis_vec = axis_map[axis]

        center = pts.mean(axis=0)
        r = pts - center[None, :]
        r -= np.outer(r @ axis_vec, axis_vec)

        tangent = np.cross(axis_vec[None, :], r)
        radius_sq = np.sum(r * r, axis=1)
        denom = float(np.sum(radius_sq))
        if denom <= 1e-20:
            raise ValueError(
                f"Cannot apply torsion around axis={axis}: selected torque nodes have near-zero radius"
            )

        nodal_forces = (float(total_torque) / denom) * tangent
        for comp in range(3):
            dofs = 3 * node_ids + comp
            force[dofs, 0] += nodal_forces[:, comp]

        return force

    def set_boundary_conditions_from_regions(self, fixed_nodes, force_nodes, force_direction='z', force_value=-1.0):
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_force(force, force_nodes, force_direction, force_value)
        node_coords = self.node_coords.reshape(-1, 3)
        loaded_force = force.reshape(-1, 3).sum(axis=0)
        self.bc_report = {
            "mode": "legacy_bbox",
            "bc_mapping_mode": self.bc_mapping_mode,
            "fixed_face_count": None,
            "loaded_face_count": None,
            "fixed_node_count": int(np.asarray(fixed_nodes).reshape(-1).size),
            "loaded_node_count": int(np.asarray(force_nodes).reshape(-1).size),
            "fixed_bounds": self._bounds_for_nodes(node_coords, fixed_nodes),
            "loaded_bounds": self._bounds_for_nodes(node_coords, force_nodes),
            "fixed_closest_surface_point_bounds": None,
            "loaded_closest_surface_point_bounds": None,
            "fixed_patch_distance_min": None,
            "fixed_patch_distance_max": None,
            "loaded_patch_distance_min": None,
            "loaded_patch_distance_max": None,
            "selected_loaded_area": None,
            "requested_total_force": float(force_value),
            "recovered_total_force_vector": loaded_force.tolist(),
            "selected_resultant_load": loaded_force.tolist(),
            "overlap_node_count": int(np.intersect1d(fixed_nodes, force_nodes).size),
            "overlap_node_count_removed": 0,
        }

        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

    def set_torsion_boundary_conditions(self, fixed_nodes, torque_nodes, torque_axis='z', total_torque=1.0):
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_torque(force, torque_nodes, torque_axis, total_torque)
        node_coords = self.node_coords.reshape(-1, 3)
        loaded_force = force.reshape(-1, 3).sum(axis=0)
        self.bc_report = {
            "mode": "legacy_bbox",
            "bc_mapping_mode": self.bc_mapping_mode,
            "fixed_face_count": None,
            "loaded_face_count": None,
            "fixed_node_count": int(np.asarray(fixed_nodes).reshape(-1).size),
            "loaded_node_count": int(np.asarray(torque_nodes).reshape(-1).size),
            "fixed_bounds": self._bounds_for_nodes(node_coords, fixed_nodes),
            "loaded_bounds": self._bounds_for_nodes(node_coords, torque_nodes),
            "selected_loaded_area": None,
            "requested_total_force": float(total_torque),
            "recovered_total_force_vector": loaded_force.tolist(),
            "selected_resultant_load": loaded_force.tolist(),
            "overlap_node_count": int(np.intersect1d(fixed_nodes, torque_nodes).size),
            "overlap_node_count_removed": 0,
        }

        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

    def show_voxels_surface_and_bc(
        self,
        show_load_arrows=True,
        return_img=False,
        off_screen=False,
        window_size=None,
        show=False,
        show_window_size=(1040, 560),
        max_load_arrows=48,
    ):
        plotter_kwargs = {"off_screen": off_screen if not show else False}
        if window_size is not None:
            plotter_kwargs["window_size"] = window_size
        plotter = pv.Plotter(**plotter_kwargs)

        mesh = self.occupied_voxel_mesh()
        if mesh is not None:
            plotter.add_mesh(mesh, color="#94a3b8", opacity=0.22)

        cloud = self.surface_cloud()
        if cloud is not None:
            plotter.add_mesh(
                cloud,
                color="#ef4444",
                opacity=0.16,
                point_size=2,
                render_points_as_spheres=False,
            )

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="#00b894",
                point_size=7,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="#ffb000",
                point_size=7,
                render_points_as_spheres=True
            )
            if show_load_arrows:
                force_components = force.reshape(-1, 3)
                force_vecs = force_components[force_node_ids]
                force_norms = np.linalg.norm(force_vecs, axis=1)
                arrow_mask = force_norms > 0.0
                if np.any(arrow_mask):
                    arrow_pts = force_pts[arrow_mask]
                    arrow_vecs = force_vecs[arrow_mask] / force_norms[arrow_mask, None]
                    if arrow_pts.shape[0] > int(max_load_arrows):
                        keep_idx = np.linspace(
                            0,
                            arrow_pts.shape[0] - 1,
                            int(max_load_arrows),
                            dtype=np.int64,
                        )
                        arrow_pts = arrow_pts[keep_idx]
                        arrow_vecs = arrow_vecs[keep_idx]
                    arrow_cloud = pv.PolyData(arrow_pts)
                    arrow_cloud["vectors"] = arrow_vecs
                    arrows = arrow_cloud.glyph(
                        orient="vectors",
                        scale=False,
                        factor=4.2 * self.voxel_size,
                    )
                    plotter.add_mesh(arrows, color="#ff5a1f", opacity=1.0)

        plotter.show_axes()

        img = None
        if return_img:
            plotter.set_background("white")
            try:
                plotter.view_isometric()
                plotter.reset_camera()
                plotter.camera.zoom(1.12)
            except Exception:
                pass
            img = plotter.screenshot(return_img=True, transparent_background=False)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            legend_items = [
                ("Shell voxels", (71, 85, 105)),
                ("Surface samples", (239, 68, 68)),
                ("Fixed nodes", (0, 184, 148)),
                ("Loaded nodes", (255, 176, 0)),
                ("Load arrows", (255, 90, 31)),
            ]
            box_w = 132
            box_h = 18 + 16 * len(legend_items)
            margin = 10
            x0 = margin
            y0 = margin
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (x0, y0),
                (x0 + box_w, y0 + box_h),
                (255, 255, 255),
                thickness=-1,
            )
            img = cv2.addWeighted(overlay, 0.86, img, 0.14, 0)
            cv2.rectangle(
                img,
                (x0, y0),
                (x0 + box_w, y0 + box_h),
                (31, 41, 55),
                thickness=1,
            )
            cv2.putText(
                img,
                "Legend",
                (x0 + 8, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (17, 24, 39),
                1,
                cv2.LINE_AA,
            )
            for idx, (label, color) in enumerate(legend_items):
                y = y0 + 31 + idx * 16
                cv2.rectangle(img, (x0 + 8, y - 8), (x0 + 17, y + 1), color, thickness=-1)
                cv2.putText(
                    img,
                    label,
                    (x0 + 23, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.31,
                    (17, 24, 39),
                    1,
                    cv2.LINE_AA,
                )

        if show and show_window_size is not None:
            try:
                plotter.window_size = show_window_size
                plotter.ren_win.SetSize(int(show_window_size[0]), int(show_window_size[1]))
                plotter.reset_camera()
            except Exception:
                pass

        if show or not return_img:
            plotter.show()
        else:
            plotter.close()

        if return_img:
            return img

    def occupied_axis_bounds(self):
        occ = self.elem_occupancy.astype(bool)
        if not np.any(occ):
            return self.padded_bbox_from_midsurface(self.brep_bbox, self.thickness, self.voxel_size, self.extra_layers)

        occ_centers = self.elem_centers[occ]
        half = 0.5 * self.voxel_size
        return {
            'xmin': float(np.min(occ_centers[:, 0]) - half),
            'xmax': float(np.max(occ_centers[:, 0]) + half),
            'ymin': float(np.min(occ_centers[:, 1]) - half),
            'ymax': float(np.max(occ_centers[:, 1]) + half),
            'zmin': float(np.min(occ_centers[:, 2]) - half),
            'zmax': float(np.max(occ_centers[:, 2]) + half),
        }
    
    def select_nodes_in_box(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        """
        Select node ids whose coordinates lie inside a rectangular box.

        Any bound can be left as None to mean 'no restriction' in that direction.

        Returns
        -------
        node_ids : ndarray
            Flat array of selected global node ids.
        """
        node_ids, coords = self.get_flat_node_coords()

        mask = np.ones(coords.shape[0], dtype=bool)

        if xmin is not None:
            mask &= coords[:, 0] >= xmin
        if xmax is not None:
            mask &= coords[:, 0] <= xmax

        if ymin is not None:
            mask &= coords[:, 1] >= ymin
        if ymax is not None:
            mask &= coords[:, 1] <= ymax

        if zmin is not None:
            mask &= coords[:, 2] >= zmin
        if zmax is not None:
            mask &= coords[:, 2] <= zmax

        return node_ids[mask]
    def debug_voxel_stats(self):
        if self.elem_occupancy is None:
            print("elem_occupancy is None")
            return

        occ = self.elem_occupancy
        num_occ = int(occ.sum())
        num_total = int(occ.size)

        hx, hy, hz = self.mesh['elemSize']
        voxel_vol = hx * hy * hz
        vox_vol = num_occ * voxel_vol

        try:
            target_vol = float(np.sum(self.face_areas)) * self.thickness
        except Exception:
            target_vol = None

        print("=== Voxel Stats ===")
        print("brep_bbox:", self.brep_bbox)
        print("mesh:", self.mesh)
        print("elem_centers shape:", self.elem_centers.shape)
        print("node_coords shape:", self.node_coords.shape)
        print("occupied voxels:", num_occ)
        print("total voxels:", num_total)
        print("occupancy ratio:", num_occ / max(num_total, 1))
        if self.elem_sample_count is not None:
            num_with_samples = int(np.count_nonzero(self.elem_sample_count))
            occ_flat = self.elem_occupancy.reshape(-1).astype(bool)
            num_occ_with_samples = int(np.count_nonzero(self.elem_sample_count[occ_flat]))
            print("voxels with assigned surface samples:", num_with_samples)
            print("occupied voxels with assigned surface samples:", num_occ_with_samples)
        print("voxelized volume:", vox_vol)
        print("thickness:", self.thickness)
        print("voxel_size:", self.voxel_size)

        if target_vol is not None:
            print("target approx volume (sum(face_areas)*thickness):", target_vol)
            if target_vol > 0:
                print("volume ratio voxel/target:", vox_vol / target_vol)

    def geometry_audit(self):
        occ = np.asarray(self.elem_occupancy, dtype=bool)
        total = int(occ.size)
        active = int(occ.sum())
        hx, hy, hz = self.mesh["elemSize"]
        voxel_volume = float(hx * hy * hz)
        geom_fraction = (
            np.asarray(self.elem_geom_fraction, dtype=np.float64).reshape(-1)
            if self.elem_geom_fraction is not None
            else occ.reshape(-1).astype(np.float64)
        )
        occ_flat = occ.reshape(-1)
        sampled_band_volume = float(np.sum(geom_fraction) * voxel_volume)
        modelled_active_volume = float(np.sum(geom_fraction[occ_flat]) * voxel_volume)
        binary_active_volume = float(active * voxel_volume)
        reference_volume = float(np.sum(self.face_areas) * self.thickness)
        neighbour_counts = np.zeros((active,), dtype=np.int64)
        active_ids = np.argwhere(occ)
        active_lookup = {tuple(idx): rank for rank, idx in enumerate(active_ids)}
        for rank, (iz, ix, iy) in enumerate(active_ids):
            count = 0
            for dz, dx, dy in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                nz, nx, ny = int(iz + dz), int(ix + dx), int(iy + dy)
                if 0 <= nz < occ.shape[0] and 0 <= nx < occ.shape[1] and 0 <= ny < occ.shape[2] and occ[nz, nx, ny]:
                    count += 1
            neighbour_counts[rank] = count

        visited = set()
        component_sizes = []
        for idx_tuple in active_lookup:
            if idx_tuple in visited:
                continue
            stack = [idx_tuple]
            visited.add(idx_tuple)
            size = 0
            while stack:
                iz, ix, iy = stack.pop()
                size += 1
                for dz, dx, dy in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    nxt = (iz + dz, ix + dx, iy + dy)
                    if nxt in active_lookup and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            component_sizes.append(size)
        component_sizes.sort(reverse=True)

        exposed_faces = self.exposed_voxel_faces()
        audit = {
            "voxelization_mode": self.voxelization_mode,
            "bc_mapping_mode": self.bc_mapping_mode,
            "total_grid_elements": total,
            "active_elements": active,
            "active_nodes": int(self.occupied_node_ids().size) if active else 0,
            "sampled_band_volume": sampled_band_volume,
            "modelled_active_volume": modelled_active_volume,
            "binary_active_volume": binary_active_volume,
            "reference_surface_area_times_thickness": reference_volume,
            "sampled_band_volume_ratio": sampled_band_volume / reference_volume if reference_volume > 0.0 else float("nan"),
            "modelled_active_volume_ratio": modelled_active_volume / reference_volume if reference_volume > 0.0 else float("nan"),
            "binary_active_volume_ratio": binary_active_volume / reference_volume if reference_volume > 0.0 else float("nan"),
            "face_connected_component_count": len(component_sizes),
            "face_connected_component_sizes": component_sizes,
            "minimum_face_neighbour_count": int(neighbour_counts.min()) if neighbour_counts.size else 0,
            "zero_face_neighbour_count": int(np.count_nonzero(neighbour_counts == 0)),
            "one_face_neighbour_count": int(np.count_nonzero(neighbour_counts == 1)),
            "le_two_face_neighbour_count": int(np.count_nonzero(neighbour_counts <= 2)),
            "minimum_active_geometric_fraction": (
                float(geom_fraction[occ.reshape(-1)].min()) if active else float("nan")
            ),
            "exposed_face_count": int(len(exposed_faces)),
        }
        audit.update(getattr(self, "bc_report", {}) or {})
        return audit

    def write_geometry_audit_csv(self, output_path):
        audit = self.geometry_audit()
        with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("key,value\n")
            for key, value in audit.items():
                stream.write(f"{key},{value}\n")
        return audit
    def build_fem_fields_from_decoder(self, rho_surface, fiber_surface, rho_void=0.0):
        elem_density, elem_phi, elem_theta = self.assign_decoder_fields(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )

        return {
            'density': elem_density,
            'phi': elem_phi,
            'theta': elem_theta,
            'fixed': self.boundaryCondition['fixed'],
            'force': self.boundaryCondition['force'],
            'mesh': self.mesh,
            'materialProperty': self.materialProperty,
        }
    def build_fem_fields_from_decoder_torch(self, rho_surface, fiber_surface, rho_void=0.0):
        device = rho_surface.device
        fiber_surface = fiber_surface.to(device=device, dtype=rho_surface.dtype)

        if self.voxelization_mode == "triangle_distance":
            return self.build_fem_fields_from_decoder_torch_barycentric(
                rho_surface=rho_surface,
                fiber_surface=fiber_surface,
                rho_void=rho_void,
            )

        sample_idx = torch.as_tensor(self.elem_sample_idx.reshape(-1), device=device, dtype=torch.long)
        sample_elem_idx = torch.as_tensor(self.sample_elem_idx.reshape(-1), device=device, dtype=torch.long)
        occ = torch.as_tensor(self.elem_occupancy.reshape(-1), device=device, dtype=torch.bool)
        source_core_elem_idx = torch.as_tensor(
            self.shell_layer_core_element_indices(),
            device=device,
            dtype=torch.long,
        )

        num_elems = sample_idx.numel()

        density = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber[:, 0] = 1.0
        normal = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        normal[:, 2] = 1.0

        valid_samples = sample_elem_idx >= 0
        valid_sample_elem_idx = sample_elem_idx[valid_samples]

        counts = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        counts.index_add_(0, valid_sample_elem_idx, torch.ones_like(rho_surface[valid_samples]))

        rho_sum = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        rho_sum.index_add_(0, valid_sample_elem_idx, rho_surface[valid_samples])

        fiber_sum = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber_sum.index_add_(0, valid_sample_elem_idx, fiber_surface[valid_samples])
        sample_normals = torch.as_tensor(
            self.sample_normals,
            device=device,
            dtype=fiber_surface.dtype,
        )
        normal_sum = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        normal_sum.index_add_(0, valid_sample_elem_idx, sample_normals[valid_samples])

        has_samples = counts > 0

        core_density = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        core_fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        core_fiber[:, 0] = 1.0
        core_normal = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        core_normal[:, 2] = 1.0
        core_density[has_samples] = rho_sum[has_samples] / counts[has_samples]
        core_fiber[has_samples] = fiber_sum[has_samples] / counts[has_samples, None]
        core_normal[has_samples] = normal_sum[has_samples] / counts[has_samples, None]

        core_fiber_norm = torch.linalg.norm(core_fiber, dim=1, keepdim=True).clamp_min(1e-12)
        core_fiber = core_fiber / core_fiber_norm
        core_normal = core_normal / torch.linalg.norm(core_normal, dim=1, keepdim=True).clamp_min(1e-12)

        filled = occ & (source_core_elem_idx >= 0)
        filled_ids = torch.nonzero(filled, as_tuple=False).reshape(-1)
        if filled_ids.numel() > 0:
            valid_filled_ids = filled_ids[has_samples[source_core_elem_idx[filled_ids]]]
            density[valid_filled_ids] = core_density[source_core_elem_idx[valid_filled_ids]]
            fiber[valid_filled_ids] = core_fiber[source_core_elem_idx[valid_filled_ids]]
            normal[valid_filled_ids] = core_normal[source_core_elem_idx[valid_filled_ids]]

        angle_eps = 1e-6
        fiber_norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(angle_eps)
        fiber = fiber / fiber_norm

        ax = fiber[:, 0]
        ay = fiber[:, 1]
        az = fiber[:, 2]

        xy_norm = torch.linalg.norm(fiber[:, :2], dim=1)
        ax_safe = torch.where(xy_norm > angle_eps, ax, torch.ones_like(ax))
        ay_safe = torch.where(xy_norm > angle_eps, ay, torch.zeros_like(ay))
        phi = torch.atan2(ay_safe, ax_safe)
        theta = torch.acos(torch.clamp(az, -1.0 + angle_eps, 1.0 - angle_eps))
        orientation_matrix = orientation_frame_from_fiber(fiber, a3_reference=normal)

        return {
            "density": density,
            "shell_occupancy": occ.to(dtype=rho_surface.dtype),
            "fiber": fiber,
            "phi": phi,
            "theta": theta,
            "orientation_axis_1": orientation_matrix[..., :, 0],
            "orientation_axis_2": orientation_matrix[..., :, 1],
            "orientation_axis_3": orientation_matrix[..., :, 2],
            "orientation_matrix": orientation_matrix,
            "fixed": self.boundaryCondition["fixed"],
            "force": self.boundaryCondition["force"],
            "mesh": self.mesh,
            "materialProperty": self.materialProperty,
        }

    def build_fem_fields_from_decoder_torch_barycentric(self, rho_surface, fiber_surface, rho_void=0.0):
        device = rho_surface.device
        dtype = rho_surface.dtype
        fiber_surface = fiber_surface.to(device=device, dtype=dtype)
        occ = torch.as_tensor(self.elem_occupancy.reshape(-1), device=device, dtype=torch.bool)
        geom_fraction = torch.as_tensor(
            np.asarray(self.elem_geom_fraction, dtype=np.float32).reshape(-1),
            device=device,
            dtype=dtype,
        )
        tri_ids = torch.as_tensor(
            np.asarray(self.elem_closest_triangle_id, dtype=np.int64).reshape(-1),
            device=device,
            dtype=torch.long,
        )
        bary = torch.as_tensor(
            np.asarray(self.elem_barycentric, dtype=np.float32).reshape(-1, 3),
            device=device,
            dtype=dtype,
        )
        faces = torch.as_tensor(self.faces_ijk, device=device, dtype=torch.long)
        triangles = torch.as_tensor(self.triangles, device=device, dtype=dtype)

        num_elems = occ.numel()
        density = torch.full((num_elems,), float(rho_void), dtype=dtype, device=device)
        fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber[:, 0] = 1.0
        normal = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        normal[:, 2] = 1.0

        valid = occ & (tri_ids >= 0)
        valid_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_ids.numel() > 0:
            valid_tri_ids = tri_ids[valid_ids]
            tri_nodes = faces[valid_tri_ids]
            weights = bary[valid_ids]
            density[valid_ids] = torch.sum(rho_surface[tri_nodes] * weights, dim=1)

            nodal_fiber = fiber_surface[tri_nodes]
            reference = nodal_fiber[:, 0, :]
            alignment = torch.sum(nodal_fiber * reference[:, None, :], dim=2)
            nodal_fiber = torch.where(alignment[..., None] < 0.0, -nodal_fiber, nodal_fiber)
            interp_fiber = torch.sum(nodal_fiber * weights[:, :, None], dim=1)

            tri = triangles[valid_tri_ids]
            normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
            normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1.0e-12)
            interp_fiber = interp_fiber - torch.sum(interp_fiber * normals, dim=1, keepdim=True) * normals
            fallback = tri[:, 1] - tri[:, 0]
            fallback = fallback - torch.sum(fallback * normals, dim=1, keepdim=True) * normals
            interp_norm = torch.linalg.norm(interp_fiber, dim=1, keepdim=True)
            fallback_norm = torch.linalg.norm(fallback, dim=1, keepdim=True).clamp_min(1.0e-12)
            interp_fiber = torch.where(interp_norm > 1.0e-12, interp_fiber, fallback / fallback_norm)
            interp_fiber = interp_fiber / torch.linalg.norm(interp_fiber, dim=1, keepdim=True).clamp_min(1.0e-12)
            fiber[valid_ids] = interp_fiber
            normal[valid_ids] = normals

        angle_eps = 1e-6
        fiber_norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(angle_eps)
        fiber = fiber / fiber_norm

        ax = fiber[:, 0]
        ay = fiber[:, 1]
        az = fiber[:, 2]

        xy_norm = torch.linalg.norm(fiber[:, :2], dim=1)
        ax_safe = torch.where(xy_norm > angle_eps, ax, torch.ones_like(ax))
        ay_safe = torch.where(xy_norm > angle_eps, ay, torch.zeros_like(ay))
        phi = torch.atan2(ay_safe, ax_safe)
        theta = torch.acos(torch.clamp(az, -1.0 + angle_eps, 1.0 - angle_eps))
        orientation_matrix = orientation_frame_from_fiber(fiber, a3_reference=normal)

        return {
            "density": density,
            "shell_occupancy": occ.to(dtype=dtype),
            "elem_geom_fraction": geom_fraction,
            "fiber": fiber,
            "phi": phi,
            "theta": theta,
            "orientation_axis_1": orientation_matrix[..., :, 0],
            "orientation_axis_2": orientation_matrix[..., :, 1],
            "orientation_axis_3": orientation_matrix[..., :, 2],
            "orientation_matrix": orientation_matrix,
            "fixed": self.boundaryCondition["fixed"],
            "force": self.boundaryCondition["force"],
            "mesh": self.mesh,
            "materialProperty": self.materialProperty,
        }
    def show_voxels_surface_and_bc_NEW(self):
        occ = self.elem_occupancy
        centers = self.elem_centers
        surface = self.points_xyz

        vox_pts = centers[occ.astype(bool)]

        plotter = pv.Plotter()

        if vox_pts.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(vox_pts),
                color="lightblue",
                point_size=6,
                render_points_as_spheres=True,
            )

        if surface is not None and surface.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(surface),
                color="red",
                point_size=4,
                render_points_as_spheres=True,
            )

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="green",
                point_size=12,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="yellow",
                point_size=12,
                render_points_as_spheres=True
            )

    # Better text placement
        plotter.add_text("Yellow: Applied load", position="upper_right", font_size=12, color="yellow")
        plotter.add_text("Green: Fixed nodes", position="upper_left", font_size=12, color="green")
        plotter.add_text("Blue: Occupied voxels", position="lower_left", font_size=12, color="lightblue")
        plotter.add_text("Red: Surface points", position="lower_right", font_size=12, color="red")

   
        plotter.show_axes()
        plotter.show()

#if __name__ == '__main__':
    # expects `tensors` to already exist in the current scope
    # shell_problem = ThickenShell(
    #     thickness=2.0,
    #     voxel_size=1.0,
    #     extra_layers=1,
    #     tensors=tensors
    # )

    # shell_problem.debug_voxel_stats()

    # savePath = os.path.join('data', 'settings', '{}.npy'.format(shell_problem.name))
    # shell_problem.serialize(savePath)
    # print("saved to:", savePath)
