# Triangle-Distance Shell Voxelization

`ThickenShell` now accepts:

```python
voxelization_mode="triangle_distance"
subvoxel_samples=2
min_geom_fraction=0.25
fixed_region=None
load_region=None
bc_mapping_mode="surface_patch"
```

`triangle_distance` is the production path. It activates candidate voxels from
exact closest-point distance to the triangulated midsurface. `extra_layers` only
expands the grid padding and does not artificially thicken the occupied shell.
`tangential_tol` is ignored in this mode and is retained only for
`voxelization_mode="legacy_samples"`.

When `subvoxel_samples > 1`, each candidate voxel is evaluated on a regular
subpoint grid. `elem_geom_fraction` is the fraction of subpoints whose exact
triangle distance is within `thickness / 2`, while `elem_occupancy` remains the
binary active-element mask:

```python
elem_occupancy = elem_geom_fraction >= min_geom_fraction
stiffness_factor = elem_occupancy * elem_geom_fraction * simp_density_factor
```

Surface fields are transferred from the closest triangle using barycentric
coordinates. Density and fibre direction are interpolated from triangle
vertices; fibre directions are normalized and projected onto the triangle
tangent plane.

For explicit geometric boundary conditions, pass region dictionaries such as:

```python
fixed_region = {"extent_axis": "x", "extent_side": "min", "band": 0.75 * voxel_size}
load_region = {
    "extent_axis": "x",
    "extent_side": "max",
    "band": 0.75 * voxel_size,
    "direction": "z",
    "total_force": -100.0,
}
```

Region selection operates on exposed voxel faces found by six-neighbour
occupancy. `extent_axis` / `extent_side` selects closest midsurface points near
the requested global shell extent; `face_axis` / `face_side` can optionally
filter exposed-face orientation. `source_face_id` is only a filter and must be
combined with a location selector such as an extent, bounds, tagged boundary
vertices, edge, loop, or predicate.

`bc_mapping_mode="surface_patch"` is load-case aware:

- tensile/compression uses one fixed end and one opposite loaded end;
- fixed-side loading respects `fixed_side`, `force_side`, `load_dir`, and
  `load_direction_side`;
- three-point bending uses two support regions and one middle load region,
  with optional `load_surface_dir` / `load_surface_side` restriction;
- torsion uses one fixed region and tangential nodal forces on the opposite
  torque region, with zero-resultant and recovered-moment checks.

For three-point bending, explicit supports can be passed with
`support_regions=[region_min, region_max]` or with `fixed_region_1` and
`fixed_region_2`.

Tagged boundaries distinguish open and closed polylines. `boundary_edge_id`
defaults to open, `boundary_loop_id` defaults to closed, and
`boundary_vertex_ids` must explicitly set `closed=True` or `closed=False`.
Boundary metadata dictionaries may provide `vertex_ids`, `point_indices`,
`indices`, or explicit polyline coordinates.

Loads are distributed by exposed-face area and then shared to the four face
nodes for surface patches. Tagged boundary loads use edge-length nodal weights
instead of combined voxel-face area. `ThickenShell.geometry_audit()` reports active counts, connectivity
diagnostics, `sampled_band_volume`, `modelled_active_volume`,
`binary_active_volume`, selected BC counts and bounds, closest-surface-point
bounds, patch-distance ranges, and force recovery metadata.
