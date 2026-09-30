Integration-Point Stress Contract
=================================

Production FEM stress is recovered at all eight full-integration C3D8/H8
Gauss points for every active voxel element. Centroid stress is retained only
as `sigmaElem_legacy_centroid` for regression/debug comparisons.

Dependency map
--------------

* `neuraltomo_fem/anisotropicFE_new.py` defines the H8 Gauss rule, B-bar
  strain-displacement matrices, rotated constitutive tensor, and stiffness.
  `H8.B` has shape `[8, 6, 24]`, `H8.gauss_points_parent` stores the canonical
  `+/-1/sqrt(3)` parent coordinates, and `H8.gauss_points` stores the internal
  half-parent coordinates.
* `neuraltomo_fem/FE.py` solves displacements, recovers `sigma_ip` with shape
  `[num_elements, 8, 6]`, computes `stress_vm_ip` with shape
  `[num_elements, 8]`, computes `stress_vm_element_max` for voxel colouring,
  and records the exact global maximum IP stress/location metadata.
* `Training/Loss_FEM.py` masks to the active voxel domain and constructs the
  stress constraint from all active element/IP values. Final feasibility uses
  the hard maximum IP ratio. The old normalized mean p-norm is logged only as
  `stress_p_norm_mean_diagnostic`.
* `Training/FEMControl.py` consumes `physical_stress_ratio`; this now means the
  maximum integration-point ratio supplied by `Loss_FEM`.
* `Training/MainTrain.py` writes history/TensorBoard rows using maximum-IP
  stress names while preserving existing aliases (`stress_max`,
  `fem_stress_max`) as maximum-IP values.
* `Utils/TimelapseRecorder.py` labels stress maxima as maximum-IP stress.
  Spatial stress fields supplied by training are `stress_vm_element_max`, the
  element-wise maximum across the eight integration points.
* `Utils/ExportAbaqus_VoxelBased.py` requests raw integration-point `S` output
  and reports the expected `8 * active_elements` stress record count.
* `Utils/embedded_centroid_stress_audit.py` now exports embedded
  integration-point stresses. The deprecated centroid function name remains as
  an alias but writes the IP audit.

Stress component order
----------------------

Embedded tensors use `[S11, S22, S33, S23, S13, S12]`. Abaqus raw `S` records
use `[S11, S22, S33, S12, S13, S23]`. Use
`embedded_to_abaqus_stress_order()` at comparison/export boundaries.

Constraint modes
----------------

`fem_stress_constraint_mode` may be:

* `hard_max`: exact maximum over all active integration-point stress ratios.
* `ks_upper`: stable conservative KS/log-sum-exp upper bound over all active
  integration-point stress ratios.
* `p_norm_upper`: unnormalized p-norm upper measure over all active
  integration-point stress ratios.

`fem_training_safety_factor` is a training margin applied to the ratio. The
physical allowable stress itself is not reduced for final feasibility, so the
safety factor is not applied twice.

Known regression references
---------------------------

For the X-fibre benchmark, 5,406 active elements must produce 43,248 raw stress
records. The old embedded centroidal maximum von Mises was 21.183893 and
matched the Abaqus element-mean tensor value 21.183854, but this no longer
governs optimization or validation. The raw Abaqus integration-point target is
44.7933044434 at element label 5406, integration point 8. Abaqus extrapolated
or averaged contour maxima, including the approximately 67.32 contour value,
are visualization output and are not validation targets.
