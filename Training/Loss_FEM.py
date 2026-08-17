import math
import torch


class Loss_FEM:
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        max_displacement: float | None = None,
        yield_strength: float | None = None,
        constraint_weight: float = 100.0,
        baseline_weight: float = 0.05,
        violation_power: float = 4.0,
        stress_density_threshold: float | None = None,
        rho_min_ratio: float = 1.0e-5,
        penal: float = 3.0,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> torch.Tensor:
        return self.evaluate(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            max_displacement=max_displacement,
            yield_strength=yield_strength,
            constraint_weight=constraint_weight,
            baseline_weight=baseline_weight,
            violation_power=violation_power,
            stress_density_threshold=stress_density_threshold,
            rho_min_ratio=rho_min_ratio,
            penal=penal,
            eps=eps,
            save_debug_history=save_debug_history,
        )["fem_total"]

    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor | float | int) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).reshape(()).detach().item())
        return math.isfinite(float(x))

    def _record_invalid(self, debug: dict, reason: str, save_debug_history: bool):
        self.trainer._record_invalid_fem_debug(debug, reason, save_debug_history)

    @staticmethod
    def _constraint_excess(
        *,
        value: torch.Tensor,
        limit: float,
        power: float,
        reference: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value_scalar = value.reshape(())
        if not bool(torch.isfinite(value_scalar).detach().item()):
            raise ValueError("FEM constraint value must be finite.")
        limit_tensor = value_scalar.new_tensor(max(float(limit), float(eps)))
        ratio = value_scalar / limit_tensor
        excess = torch.relu(ratio - 1.0)
        loss = excess.pow(float(power))
        return loss, excess, ratio

    @staticmethod
    def _soft_p_norm(value: torch.Tensor, p: float, eps: float) -> torch.Tensor:
        flat = value.reshape(-1).abs()
        if flat.numel() == 0:
            raise ValueError("FEM p-norm field must not be empty.")
        p = max(float(p), float(eps))
        scale = flat.detach().max().clamp_min(float(eps))
        return scale * (flat / scale).pow(p).mean().pow(1.0 / p)

    @staticmethod
    def tensor_field_is_valid(field: torch.Tensor | None) -> bool:
        return (
            isinstance(field, torch.Tensor)
            and field.numel() > 0
            and bool(torch.isfinite(field).all().detach().item())
        )

    @staticmethod
    def _empty_invalid_output(
        *,
        reference: torch.Tensor,
        reason: str,
        density_field=None,
        stress_field=None,
        displacement_field=None,
        loaded_boundary_displacement_field=None,
    ) -> dict:
        nan_scalar = torch.full(
            (),
            float("nan"),
            dtype=reference.dtype,
            device=reference.device,
        )
        return {
            "fem_total": nan_scalar,
            "fem_valid": False,
            "failure_reason": reason,
            "density_field": density_field.detach() if isinstance(density_field, torch.Tensor) else density_field,
            "stiffness_factor_field": None,
            "stress_field": stress_field.detach() if isinstance(stress_field, torch.Tensor) else stress_field,
            "displacement_field": displacement_field.detach() if isinstance(displacement_field, torch.Tensor) else displacement_field,
            "loaded_boundary_displacement_field": (
                loaded_boundary_displacement_field.detach()
                if isinstance(loaded_boundary_displacement_field, torch.Tensor)
                else loaded_boundary_displacement_field
            ),
            "stress_constraint_loss": nan_scalar,
            "displacement_constraint_loss": nan_scalar,
            "baseline_fem_loss": nan_scalar,
            "violation_fem_loss": nan_scalar,
            "stress_constraint_excess": nan_scalar,
            "displacement_constraint_excess": nan_scalar,
            "stress_ratio": nan_scalar,
            "displacement_ratio": nan_scalar,
            "training_stress_ratio": nan_scalar,
            "training_displacement_ratio": nan_scalar,
            "physical_stress_ratio": nan_scalar,
            "physical_displacement_ratio": nan_scalar,
            "constraint_violation": nan_scalar,
            "training_feasible": False,
            "safety_margin_satisfied": False,
            "physical_feasible": False,
            "stress_max": nan_scalar,
            "displacement_max": nan_scalar,
        }
    def evaluate(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        max_displacement: float | None = None,
        yield_strength: float | None = None,
        constraint_weight: float = 100.0,
        baseline_weight: float = 0.05,
        violation_power: float = 4.0,
        stress_density_threshold: float | None = None,
        rho_min_ratio: float = 1.0e-5,
        penal: float = 3.0,
        eps: float = 1.0e-12,
        save_debug_history: bool = True,
    ) -> dict:
        device = rho_surface.device
        dtype = rho_surface.dtype

        # ------------------------------------------------------------------
        # Build FEM fields
        # ------------------------------------------------------------------
        fem_fields = (
            self.trainer.shell_problem.build_fem_fields_from_decoder_torch(
                rho_surface=rho_surface,
                fiber_surface=fiber_surface,
            )
        )

        density_raw = fem_fields["density"].to(
            device=device,
            dtype=dtype,
        )

        shell_occupancy = fem_fields.get("shell_occupancy")

        if isinstance(shell_occupancy, torch.Tensor):
            shell_occupancy = shell_occupancy.to(
                device=device,
                dtype=dtype,
            ).clamp(0.0, 1.0)
        else:
            shell_occupancy = torch.ones_like(density_raw)

        fiber_density = density_raw.clamp(0.0, 1.0)

        active_element_mask = (
            shell_occupancy.reshape(-1) > 0.5
        )

        rho = (
            shell_occupancy * fiber_density
        ).clamp(0.0, 1.0)

        stiffness_inside = (
            float(rho_min_ratio)
            + (1.0 - float(rho_min_ratio))
            * fiber_density.pow(float(penal))
        )

        stiffness_factor = (
            shell_occupancy * stiffness_inside
        )

        phi = fem_fields["phi"].to(
            device=device,
            dtype=dtype,
        )

        theta = fem_fields["theta"].to(
            device=device,
            dtype=dtype,
        )

        rho_flat = rho.reshape(-1)
        stiffness_flat = stiffness_factor.reshape(-1)

        inside_mask = active_element_mask
        outside_mask = ~inside_mask

        active_count = int(
            inside_mask.detach().sum().item()
        )

        total_count = int(inside_mask.numel())

        mean_inside = (
            rho_flat[inside_mask].mean()
            if active_count > 0
            else rho_flat.new_zeros(())
        )

        mean_outside = (
            rho_flat[outside_mask].mean()
            if bool(outside_mask.any().detach().item())
            else rho_flat.new_zeros(())
        )

        max_outside = (
            rho_flat[outside_mask].max()
            if bool(outside_mask.any().detach().item())
            else rho_flat.new_zeros(())
        )

        max_stiffness_outside = (
            stiffness_flat[outside_mask].max()
            if bool(outside_mask.any().detach().item())
            else stiffness_flat.new_zeros(())
        )

        fiber_norm = torch.linalg.norm(
            fiber_surface,
            dim=1,
        )

        # These are assigned later after the FEM solve.
        stress_field = None
        displacement_field = None
        loaded_boundary_displacement_field = None

        debug = {
            "rho_surface_shape": tuple(rho_surface.shape),
            "fiber_surface_shape": tuple(fiber_surface.shape),
            "density_shape": tuple(rho.shape),
            "phi_shape": tuple(phi.shape),
            "theta_shape": tuple(theta.shape),

            "rho_min_ratio": float(rho_min_ratio),
            "fem_penal": float(penal),

            "density_raw_min": float(
                density_raw.detach().min().item()
            ),
            "density_raw_mean": float(
                density_raw.detach().mean().item()
            ),
            "density_raw_max": float(
                density_raw.detach().max().item()
            ),

            "density_min": float(
                rho.detach().min().item()
            ),
            "density_mean": float(
                rho.detach().mean().item()
            ),
            "density_max": float(
                rho.detach().max().item()
            ),

            "stiffness_factor_min": float(
                stiffness_factor.detach().min().item()
            ),
            "stiffness_factor_mean": float(
                stiffness_factor.detach().mean().item()
            ),
            "stiffness_factor_max": float(
                stiffness_factor.detach().max().item()
            ),

            "occupied_voxels": active_count,
            "active_elements": active_count,
            "total_elements": total_count,

            "max_stiffness_outside_shell": float(
                max_stiffness_outside.detach().item()
            ),
            "mean_density_inside_shell": float(
                mean_inside.detach().item()
            ),
            "mean_density_outside_shell": float(
                mean_outside.detach().item()
            ),
            "max_density_outside_shell": float(
                max_outside.detach().item()
            ),

            "phi_has_nan": bool(
                torch.isnan(phi).any().detach().item()
            ),
            "phi_has_inf": bool(
                torch.isinf(phi).any().detach().item()
            ),
            "theta_has_nan": bool(
                torch.isnan(theta).any().detach().item()
            ),
            "theta_has_inf": bool(
                torch.isinf(theta).any().detach().item()
            ),
            "fiber_has_nan": bool(
                torch.isnan(fiber_surface).any().detach().item()
            ),
            "fiber_has_inf": bool(
                torch.isinf(fiber_surface).any().detach().item()
            ),

            "fiber_norm_min": float(
                fiber_norm.detach().min().item()
            ),
            "fiber_norm_mean": float(
                fiber_norm.detach().mean().item()
            ),
            "fiber_norm_max": float(
                fiber_norm.detach().max().item()
            ),

            "void_fraction_lt_1e_2_raw": float(
                (density_raw < 1.0e-2)
                .float()
                .mean()
                .detach()
                .item()
            ),
            "void_fraction_lt_5e_2_raw": float(
                (density_raw < 5.0e-2)
                .float()
                .mean()
                .detach()
                .item()
            ),
            "void_fraction_lt_floor_raw": float(
                (density_raw < float(rho_min_ratio))
                .float()
                .mean()
                .detach()
                .item()
            ),
            "void_fraction_lt_rho_min_raw": float(
                (density_raw < float(rho_min_ratio))
                .float()
                .mean()
                .detach()
                .item()
            ),
        }

        def invalid_output(reason: str) -> dict:
            self._record_invalid(
                debug,
                reason,
                save_debug_history,
            )

            return self._empty_invalid_output(
                reference=rho,
                reason=reason,
                density_field=rho,
                stress_field=stress_field,
                displacement_field=displacement_field,
                loaded_boundary_displacement_field=(
                    loaded_boundary_displacement_field
                ),
            )

        if active_count == 0:
            return invalid_output(
                "The FEM domain contains no active shell elements."
            )

        invalid_input_fields = (
            debug["phi_has_nan"]
            or debug["phi_has_inf"]
            or debug["theta_has_nan"]
            or debug["theta_has_inf"]
            or debug["fiber_has_nan"]
            or debug["fiber_has_inf"]
        )

        if invalid_input_fields:
            return invalid_output(
                "Invalid phi/theta/fiber fields before FEM solve."
            )

        # ------------------------------------------------------------------
        # FEM solve
        # ------------------------------------------------------------------
        try:
            _, solve_scalar = self.trainer.fem(
                stiffness_factor,
                phi,
                theta,
                penal=1.0,
            )
        except Exception as error:
            return invalid_output(
                f"FEM solve raised exception: {error!r}"
            )

        fe_solver = getattr(
            self.trainer.fem,
            "fe",
            None,
        )

        fe_mesh = getattr(
            fe_solver,
            "mesh",
            None,
        )

        stress_field = getattr(
            fe_solver,
            "stress_vm",
            None,
        )

        displacement_field = getattr(
            fe_solver,
            "displacement_mag_elem",
            None,
        )

        if displacement_field is None:
            displacement_field = getattr(
                fe_solver,
                "displacement_load_dir_elem",
                None,
            )

        loaded_boundary_displacement_field = getattr(
            fe_solver,
            "displacement_mag_loaded_boundary",
            None,
        )

        if loaded_boundary_displacement_field is None:
            loaded_boundary_displacement_field = getattr(
                fe_solver,
                "displacement_load_dir_loaded_boundary",
                None,
            )

        debug["fem_solve_scalar_is_finite"] = (
            self._scalar_tensor_is_finite(solve_scalar)
        )

        if fe_mesh is not None:
            debug.update(
                {
                    "fem_active_elements": len(
                        getattr(fe_mesh, "active_element_ids", [])
                    ),
                    "fem_total_elements": int(
                        getattr(fe_mesh, "numElems", 0)
                    ),
                    "fem_active_nodes": len(
                        getattr(fe_mesh, "active_node_ids", [])
                    ),
                    "fem_total_nodes": int(
                        getattr(fe_mesh, "numNodes", 0)
                    ),
                    "fem_active_free_dofs": len(
                        getattr(fe_mesh, "free", [])
                    ),
                    "fem_total_dofs": int(
                        getattr(fe_mesh, "ndof", 0)
                    ),
                }
            )

        if not debug["fem_solve_scalar_is_finite"]:
            return invalid_output(
                "Non-finite scalar returned by FEM solve."
            )

        # ------------------------------------------------------------------
        # Select and mask the displacement field
        # ------------------------------------------------------------------
        using_element_displacement = (
            isinstance(displacement_field, torch.Tensor)
            and displacement_field.numel() > 0
        )

        if using_element_displacement:
            displacement_for_loss = displacement_field
        else:
            displacement_for_loss = (
                loaded_boundary_displacement_field
            )

        if not self.tensor_field_is_valid(
            displacement_for_loss
        ):
            return invalid_output(
                "Displacement field is empty or contains NaN/Inf."
            )

        displacement_for_loss = (
            displacement_for_loss.reshape(-1)
        )

        # Remove outside-shell zeros from element displacement losses.
        if (
            using_element_displacement
            and displacement_for_loss.numel()
            == active_element_mask.numel()
        ):
            displacement_for_loss = displacement_for_loss[
                active_element_mask
            ]

        if not self.tensor_field_is_valid(
            displacement_for_loss
        ):
            return invalid_output(
                "Masked displacement field is empty or invalid."
            )

        debug["displacement_evaluated_element_count"] = int(
            displacement_for_loss.numel()
        )

        # ------------------------------------------------------------------
        # Select and mask the stress field
        # ------------------------------------------------------------------
        if not self.tensor_field_is_valid(stress_field):
            return invalid_output(
                "Stress field is empty or contains NaN/Inf."
            )

        stress_for_loss = stress_field.reshape(-1)

        if (
            stress_for_loss.numel()
            == active_element_mask.numel()
            and rho_flat.numel()
            == active_element_mask.numel()
        ):
            # Always exclude outside-shell elements.
            stress_mask = active_element_mask.clone()

            # Keep the existing hard density threshold for now.
            if stress_density_threshold is not None:
                stress_mask = (
                    stress_mask
                    & (
                        rho_flat.detach()
                        >= float(stress_density_threshold)
                    )
                )

            if not bool(stress_mask.any().detach().item()):
                return invalid_output(
                    "No active elements satisfy the FEM stress mask."
                )

            stress_for_loss = stress_for_loss[stress_mask]
            rho_for_stress = rho_flat[stress_mask]

            debug["stress_evaluated_element_count"] = int(
                stress_mask.detach().sum().item()
            )
            debug["stress_eval_density_min"] = float(
                rho_for_stress.detach().min().item()
            )
            debug["stress_eval_density_max"] = float(
                rho_for_stress.detach().max().item()
            )
        else:
            debug["stress_evaluated_element_count"] = int(
                stress_for_loss.numel()
            )

        if not self.tensor_field_is_valid(stress_for_loss):
            return invalid_output(
                "Masked stress field is empty or invalid."
            )

        # Equivalent von Mises stress must not be negative.
        if bool(
            (stress_for_loss < -float(eps))
            .any()
            .detach()
            .item()
        ):
            return invalid_output(
                "Equivalent stress field contains negative values."
            )

        # ------------------------------------------------------------------
        # Physical maxima and smooth p-norm measures
        # ------------------------------------------------------------------
        stress_max_tensor = stress_for_loss.max()

        displacement_max_tensor = (
            displacement_for_loss.abs().max()
        )

        p_norm = float(
            getattr(
                self.trainer.cfg,
                "fem_constraint_p_norm",
                12.0,
            )
        )

        stress_p_norm_tensor = self._soft_p_norm(
            stress_for_loss,
            p=p_norm,
            eps=eps,
        )

        displacement_p_norm_tensor = self._soft_p_norm(
            displacement_for_loss,
            p=p_norm,
            eps=eps,
        )

        safety_factor = float(
            getattr(
                self.trainer.cfg,
                "fem_training_safety_factor",
                0.95,
            )
        )

        margin_weight = float(
            getattr(
                self.trainer.cfg,
                "fem_safety_margin_weight",
                0.05,
            )
        )

        training_yield_strength = (
            None
            if yield_strength is None
            else safety_factor * float(yield_strength)
        )

        training_max_displacement = (
            None
            if max_displacement is None
            else safety_factor * float(max_displacement)
        )

        def normalized_ratio(
            value: torch.Tensor,
            limit: float | None,
        ) -> torch.Tensor:
            if limit is None:
                return value * 0.0

            limit_tensor = value.new_tensor(
                max(float(limit), float(eps))
            )

            return value / limit_tensor

        # Physical ratios use actual maxima.
        physical_stress_ratio = normalized_ratio(
            stress_max_tensor,
            yield_strength,
        )

        physical_displacement_ratio = normalized_ratio(
            displacement_max_tensor,
            max_displacement,
        )

        # Smooth loss ratios use active-domain p-norms.
        loss_stress_ratio = normalized_ratio(
            stress_p_norm_tensor,
            yield_strength,
        )

        loss_displacement_ratio = normalized_ratio(
            displacement_p_norm_tensor,
            max_displacement,
        )

        training_stress_ratio = normalized_ratio(
            stress_p_norm_tensor,
            training_yield_strength,
        )

        training_displacement_ratio = normalized_ratio(
            displacement_p_norm_tensor,
            training_max_displacement,
        )

        # ------------------------------------------------------------------
        # Constraint losses
        # ------------------------------------------------------------------

        # Hard physical violations use maximum values.
        stress_excess = torch.relu(
            physical_stress_ratio - 1.0
        )

        displacement_excess = torch.relu(
            physical_displacement_ratio - 1.0
        )

        stress_violation_loss = stress_excess.pow(
            float(violation_power)
        )

        displacement_violation_loss = (
            displacement_excess.pow(
                float(violation_power)
            )
        )

        # Smooth preventive margin begins at safety_factor × limit.
        training_stress_excess = torch.relu(
            loss_stress_ratio - safety_factor
        )

        training_displacement_excess = torch.relu(
            loss_displacement_ratio - safety_factor
        )

        stress_margin_loss = (
            margin_weight
            * training_stress_excess.pow(
                float(violation_power)
            )
        )

        displacement_margin_loss = (
            margin_weight
            * training_displacement_excess.pow(
                float(violation_power)
            )
        )

        stress_constraint_loss = (
            stress_margin_loss
            + stress_violation_loss
        )

        displacement_constraint_loss = (
            displacement_margin_loss
            + displacement_violation_loss
        )

        # Continuous gradient even when constraints are satisfied.
        baseline_fem_loss = float(baseline_weight) * (
            loss_stress_ratio
            + loss_displacement_ratio
        )

        violation_fem_loss = float(constraint_weight) * (
            stress_constraint_loss
            + displacement_constraint_loss
        )

        constraint_total = (
            baseline_fem_loss
            + violation_fem_loss
        )

        constraint_violation = torch.maximum(
            stress_excess,
            displacement_excess,
        )

        safety_margin_satisfied = bool(
            (training_stress_ratio <= 1.0).detach().item()
            and
            (
                training_displacement_ratio <= 1.0
            ).detach().item()
        )

        physical_feasible = bool(
            (physical_stress_ratio <= 1.0).detach().item()
            and
            (
                physical_displacement_ratio <= 1.0
            ).detach().item()
        )

        # Keep this alias for compatibility with existing logging.
        training_feasible = physical_feasible

        # ------------------------------------------------------------------
        # Debugging and logging
        # ------------------------------------------------------------------
        debug.update(
            {
                "stress_max": float(
                    stress_max_tensor.detach().item()
                ),
                "displacement_max": float(
                    displacement_max_tensor.detach().item()
                ),
                "stress_p_norm": float(
                    stress_p_norm_tensor.detach().item()
                ),
                "displacement_p_norm": float(
                    displacement_p_norm_tensor.detach().item()
                ),

                "fem_constraint_p_norm": p_norm,
                "fem_safety_margin_weight": margin_weight,

                "physical_stress_limit": (
                    None
                    if yield_strength is None
                    else float(yield_strength)
                ),
                "physical_displacement_limit": (
                    None
                    if max_displacement is None
                    else float(max_displacement)
                ),
                "training_stress_limit": (
                    None
                    if training_yield_strength is None
                    else float(training_yield_strength)
                ),
                "training_displacement_limit": (
                    None
                    if training_max_displacement is None
                    else float(training_max_displacement)
                ),

                "stress_constraint_loss_value": float(
                    stress_constraint_loss.detach().item()
                ),
                "displacement_constraint_loss_value": float(
                    displacement_constraint_loss.detach().item()
                ),
                "stress_margin_loss_value": float(
                    stress_margin_loss.detach().item()
                ),
                "displacement_margin_loss_value": float(
                    displacement_margin_loss.detach().item()
                ),
                "stress_violation_loss_value": float(
                    stress_violation_loss.detach().item()
                ),
                "displacement_violation_loss_value": float(
                    displacement_violation_loss.detach().item()
                ),
                "baseline_fem_loss_value": float(
                    baseline_fem_loss.detach().item()
                ),
                "violation_fem_loss_value": float(
                    violation_fem_loss.detach().item()
                ),

                "stress_constraint_excess_value": float(
                    stress_excess.detach().item()
                ),
                "displacement_constraint_excess_value": float(
                    displacement_excess.detach().item()
                ),
                "training_stress_excess_value": float(
                    training_stress_excess.detach().item()
                ),
                "training_displacement_excess_value": float(
                    training_displacement_excess.detach().item()
                ),

                "loss_stress_ratio_value": float(
                    loss_stress_ratio.detach().item()
                ),
                "loss_displacement_ratio_value": float(
                    loss_displacement_ratio.detach().item()
                ),
                "stress_ratio_value": float(
                    physical_stress_ratio.detach().item()
                ),
                "displacement_ratio_value": float(
                    physical_displacement_ratio.detach().item()
                ),
                "training_stress_ratio_value": float(
                    training_stress_ratio.detach().item()
                ),
                "training_displacement_ratio_value": float(
                    training_displacement_ratio.detach().item()
                ),
                "physical_stress_ratio_value": float(
                    physical_stress_ratio.detach().item()
                ),
                "physical_displacement_ratio_value": float(
                    physical_displacement_ratio.detach().item()
                ),

                "training_feasible": training_feasible,
                "safety_margin_satisfied": (
                    safety_margin_satisfied
                ),
                "physical_feasible": physical_feasible,

                "constraint_violation_value": float(
                    constraint_violation.detach().item()
                ),
                "fem_total_is_finite": (
                    self._scalar_tensor_is_finite(
                        constraint_total
                    )
                ),
                "fem_total_value": float(
                    constraint_total.detach().item()
                ),
            }
        )

        fem_valid = bool(
            debug["fem_solve_scalar_is_finite"]
            and debug["fem_total_is_finite"]
        )

        debug["fem_valid"] = fem_valid
        debug["failure_reason"] = (
            None
            if fem_valid
            else "Non-finite FEM constraint loss."
        )

        self.trainer.last_fem_debug = debug

        if save_debug_history:
            self.trainer.fem_debug_history.append(
                debug.copy()
            )

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        return {
            "fem_total": constraint_total,
            "fem_valid": fem_valid,
            "failure_reason": debug["failure_reason"],

            "density_field": rho.detach(),
            "stiffness_factor_field": (
                stiffness_factor.detach()
            ),
            "stress_field": (
                stress_field.detach()
                if isinstance(stress_field, torch.Tensor)
                else stress_field
            ),
            "displacement_field": (
                displacement_field.detach()
                if isinstance(displacement_field, torch.Tensor)
                else displacement_field
            ),
            "loaded_boundary_displacement_field": (
                loaded_boundary_displacement_field.detach()
                if isinstance(
                    loaded_boundary_displacement_field,
                    torch.Tensor,
                )
                else loaded_boundary_displacement_field
            ),

            "stress_constraint_loss": (
                stress_constraint_loss
            ),
            "displacement_constraint_loss": (
                displacement_constraint_loss
            ),
            "baseline_fem_loss": baseline_fem_loss,
            "violation_fem_loss": violation_fem_loss,

            "stress_constraint_excess": stress_excess,
            "displacement_constraint_excess": (
                displacement_excess
            ),

            "stress_margin_loss": stress_margin_loss,
            "displacement_margin_loss": (
                displacement_margin_loss
            ),
            "stress_violation_loss": (
                stress_violation_loss
            ),
            "displacement_violation_loss": (
                displacement_violation_loss
            ),

            "training_stress_excess": (
                training_stress_excess
            ),
            "training_displacement_excess": (
                training_displacement_excess
            ),

            "loss_stress_ratio": loss_stress_ratio,
            "loss_displacement_ratio": (
                loss_displacement_ratio
            ),

            "stress_ratio": physical_stress_ratio,
            "displacement_ratio": (
                physical_displacement_ratio
            ),
            "training_stress_ratio": (
                training_stress_ratio
            ),
            "training_displacement_ratio": (
                training_displacement_ratio
            ),
            "physical_stress_ratio": (
                physical_stress_ratio
            ),
            "physical_displacement_ratio": (
                physical_displacement_ratio
            ),

            "constraint_violation": constraint_violation,
            "training_feasible": training_feasible,
            "safety_margin_satisfied": (
                safety_margin_satisfied
            ),
            "physical_feasible": physical_feasible,

            "stress_max": stress_max_tensor.detach(),
            "displacement_max": (
                displacement_max_tensor.detach()
            ),
            "stress_p_norm": (
                stress_p_norm_tensor.detach()
            ),
            "displacement_p_norm": (
                displacement_p_norm_tensor.detach()
            ),
        }