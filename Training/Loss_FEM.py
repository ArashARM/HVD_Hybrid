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
        constraint_weight: float = 2.0,
        baseline_weight: float = 0.05,
        violation_power: float = 4.0,
        stress_density_threshold: float | None = None,
        rho_min_ratio: float = 1.0e-5,
        penal: float = 3.0,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        device = rho_surface.device
        dtype = rho_surface.dtype

        fem_fields = self.trainer.shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
        )
        density_raw = fem_fields["density"].to(device=device, dtype=dtype)
        shell_occupancy = fem_fields.get("shell_occupancy", None)
        if isinstance(shell_occupancy, torch.Tensor):
            shell_occupancy = shell_occupancy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        else:
            shell_occupancy = torch.ones_like(density_raw)
        fiber_density = density_raw.clamp(0.0, 1.0)
        rho = (shell_occupancy * fiber_density).clamp(0.0, 1.0)
        stiffness_factor = (
            float(rho_min_ratio)
            + (1.0 - float(rho_min_ratio)) * rho.pow(float(penal))
        )

        phi = fem_fields["phi"].to(device=device, dtype=dtype)
        theta = fem_fields["theta"].to(device=device, dtype=dtype)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)
        inside_mask = shell_occupancy.reshape(-1) > 0.5
        outside_mask = ~inside_mask
        mean_inside = rho[inside_mask].mean() if bool(inside_mask.detach().any().item()) else rho.new_zeros(())
        mean_outside = rho[outside_mask].mean() if bool(outside_mask.detach().any().item()) else rho.new_zeros(())
        max_outside = rho[outside_mask].max() if bool(outside_mask.detach().any().item()) else rho.new_zeros(())

        debug = {
            "rho_surface_shape": tuple(rho_surface.shape),
            "fiber_surface_shape": tuple(fiber_surface.shape),
            "density_shape": tuple(rho.shape),
            "phi_shape": tuple(phi.shape),
            "theta_shape": tuple(theta.shape),
            "rho_min_ratio": float(rho_min_ratio),
            "fem_penal": float(penal),
            "density_raw_min": float(density_raw.min().detach().item()),
            "density_raw_mean": float(density_raw.mean().detach().item()),
            "density_raw_max": float(density_raw.max().detach().item()),
            "density_min": float(rho.min().detach().item()),
            "density_mean": float(rho.mean().detach().item()),
            "density_max": float(rho.max().detach().item()),
            "stiffness_factor_min": float(stiffness_factor.min().detach().item()),
            "stiffness_factor_mean": float(stiffness_factor.mean().detach().item()),
            "stiffness_factor_max": float(stiffness_factor.max().detach().item()),
            "occupied_voxels": int(inside_mask.detach().sum().item()),
            "mean_density_inside_shell": float(mean_inside.detach().item()),
            "mean_density_outside_shell": float(mean_outside.detach().item()),
            "max_density_outside_shell": float(max_outside.detach().item()),
            "phi_has_nan": bool(torch.isnan(phi).any().detach().item()),
            "phi_has_inf": bool(torch.isinf(phi).any().detach().item()),
            "theta_has_nan": bool(torch.isnan(theta).any().detach().item()),
            "theta_has_inf": bool(torch.isinf(theta).any().detach().item()),
            "fiber_has_nan": bool(torch.isnan(fiber_surface).any().detach().item()),
            "fiber_has_inf": bool(torch.isinf(fiber_surface).any().detach().item()),
            "fiber_norm_min": float(fiber_norm.min().detach().item()),
            "fiber_norm_mean": float(fiber_norm.mean().detach().item()),
            "fiber_norm_max": float(fiber_norm.max().detach().item()),
            "void_fraction_lt_1e_2_raw": float((density_raw < 1e-2).float().mean().detach().item()),
            "void_fraction_lt_5e_2_raw": float((density_raw < 5e-2).float().mean().detach().item()),
            "void_fraction_lt_floor_raw": float((density_raw < float(rho_min_ratio)).float().mean().detach().item()),
            "void_fraction_lt_rho_min_raw": float((density_raw < float(rho_min_ratio)).float().mean().detach().item()),
        }

        if (
            debug["phi_has_nan"] or debug["phi_has_inf"] or
            debug["theta_has_nan"] or debug["theta_has_inf"] or
            debug["fiber_has_nan"] or debug["fiber_has_inf"]
        ):
            reason = "Invalid phi/theta/fiber fields before FEM solve"
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(reference=rho, reason=reason)

        try:
            _stress_unused, solve_scalar = self.trainer.fem(stiffness_factor, phi, theta, penal=1.0)
        except Exception as e:
            reason = f"FEM solve raised exception: {repr(e)}"
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(reference=rho, reason=reason)

        fe_solver = getattr(self.trainer.fem, "fe", None)
        stress_field = getattr(fe_solver, "stress_vm", None)
        displacement_field = getattr(fe_solver, "displacement_mag_elem", None)
        if displacement_field is None:
            displacement_field = getattr(fe_solver, "displacement_load_dir_elem", None)
        loaded_boundary_displacement_field = getattr(fe_solver, "displacement_mag_loaded_boundary", None)
        if loaded_boundary_displacement_field is None:
            loaded_boundary_displacement_field = getattr(fe_solver, "displacement_load_dir_loaded_boundary", None)

        debug.update({
            "fem_solve_scalar_is_finite": self._scalar_tensor_is_finite(solve_scalar),
        })

        if not debug["fem_solve_scalar_is_finite"]:
            reason = "Non-finite scalar returned by FEM solve"
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(
                reference=rho,
                reason=reason,
                density_field=rho,
                stress_field=stress_field,
                displacement_field=displacement_field,
                loaded_boundary_displacement_field=loaded_boundary_displacement_field,
            )

        displacement_for_loss = displacement_field
        if not isinstance(displacement_for_loss, torch.Tensor) or displacement_for_loss.numel() == 0:
            displacement_for_loss = loaded_boundary_displacement_field

        if not self.tensor_field_is_valid(displacement_for_loss):
            reason = "Displacement field is empty or contains NaN/Inf."
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(
                reference=rho,
                reason=reason,
                density_field=rho,
                stress_field=stress_field,
                displacement_field=displacement_field,
                loaded_boundary_displacement_field=loaded_boundary_displacement_field,
            )

        if not self.tensor_field_is_valid(stress_field):
            reason = "Stress field is empty or contains NaN/Inf."
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(
                reference=rho,
                reason=reason,
                density_field=rho,
                stress_field=stress_field,
                displacement_field=displacement_field,
                loaded_boundary_displacement_field=loaded_boundary_displacement_field,
            )

        stress_for_loss = stress_field
        if (
            isinstance(stress_for_loss, torch.Tensor)
            and stress_density_threshold is not None
            and isinstance(rho, torch.Tensor)
            and rho.numel() == stress_for_loss.numel()
        ):
            stress_mask = rho.detach().reshape(-1) >= float(stress_density_threshold)
            if not bool(stress_mask.any().detach().item()):
                reason = "No elements satisfy fem_stress_density_threshold."
                self._record_invalid(debug, reason, save_debug_history)
                return self._empty_invalid_output(
                    reference=rho,
                    reason=reason,
                    density_field=rho,
                    stress_field=stress_field,
                    displacement_field=displacement_field,
                    loaded_boundary_displacement_field=loaded_boundary_displacement_field,
                )
            stress_for_loss = stress_for_loss.reshape(-1)[stress_mask]
            rho_for_stress = rho.reshape(-1)[stress_mask]
            debug["stress_evaluated_element_count"] = int(stress_mask.detach().sum().item())
            debug["stress_eval_density_min"] = float(rho_for_stress.detach().min().item())
            debug["stress_eval_density_max"] = float(rho_for_stress.detach().max().item())

        stress_max_tensor = stress_for_loss.reshape(-1).max()
        if bool((stress_for_loss.reshape(-1) < -eps).any().detach().item()):
            reason = "Equivalent stress field contains negative values."
            self._record_invalid(debug, reason, save_debug_history)
            return self._empty_invalid_output(
                reference=rho,
                reason=reason,
                density_field=rho,
                stress_field=stress_field,
                displacement_field=displacement_field,
                loaded_boundary_displacement_field=loaded_boundary_displacement_field,
            )
        displacement_max_tensor = displacement_for_loss.reshape(-1).abs().max()
        safety_factor = float(getattr(self.trainer.cfg, "fem_training_safety_factor", 0.95))
        training_yield_strength = None if yield_strength is None else safety_factor * float(yield_strength)
        training_max_displacement = None if max_displacement is None else safety_factor * float(max_displacement)
        zero = rho.reshape(-1)[0] * 0.0
        physical_stress_ratio = (
            stress_max_tensor / stress_max_tensor.new_tensor(max(float(yield_strength), float(eps)))
            if yield_strength is not None else stress_max_tensor * 0.0
        )
        physical_displacement_ratio = (
            displacement_max_tensor / displacement_max_tensor.new_tensor(max(float(max_displacement), float(eps)))
            if max_displacement is not None else displacement_max_tensor * 0.0
        )
        training_stress_ratio = (
            stress_max_tensor / stress_max_tensor.new_tensor(max(float(training_yield_strength), float(eps)))
            if training_yield_strength is not None else stress_max_tensor * 0.0
        )
        training_displacement_ratio = (
            displacement_max_tensor / displacement_max_tensor.new_tensor(max(float(training_max_displacement), float(eps)))
            if training_max_displacement is not None else displacement_max_tensor * 0.0
        )
        stress_excess = torch.relu(physical_stress_ratio - 1.0)
        displacement_excess = torch.relu(physical_displacement_ratio - 1.0)
        #stress_constraint_loss = stress_excess.pow(float(violation_power))
        #displacement_constraint_loss = displacement_excess.pow(float(violation_power))
        stress_constraint_loss = stress_excess
        displacement_constraint_loss = displacement_excess
        baseline_fem_loss = float(baseline_weight) * (
            physical_stress_ratio.pow(1.0)
            + physical_displacement_ratio.pow(1.0)
        )
        violation_fem_loss = float(constraint_weight) * (stress_constraint_loss + displacement_constraint_loss)
        constraint_total = baseline_fem_loss + violation_fem_loss
        constraint_violation = torch.maximum(stress_excess, displacement_excess)
        training_feasible = bool(
            (training_stress_ratio <= 1.0).detach().item()
            and (training_displacement_ratio <= 1.0).detach().item()
        )
        physical_feasible = bool(
            (physical_stress_ratio <= 1.0).detach().item()
            and (physical_displacement_ratio <= 1.0).detach().item()
        )

        debug.update({
            "stress_max": float(stress_max_tensor.detach().item()),
            "displacement_max": float(displacement_max_tensor.detach().item()),
            "physical_stress_limit": None if yield_strength is None else float(yield_strength),
            "physical_displacement_limit": None if max_displacement is None else float(max_displacement),
            "training_stress_limit": None if training_yield_strength is None else float(training_yield_strength),
            "training_displacement_limit": None if training_max_displacement is None else float(training_max_displacement),
            "stress_constraint_loss_value": float(stress_constraint_loss.detach().item()),
            "displacement_constraint_loss_value": float(displacement_constraint_loss.detach().item()),
            "baseline_fem_loss_value": float(baseline_fem_loss.detach().item()),
            "violation_fem_loss_value": float(violation_fem_loss.detach().item()),
            "stress_constraint_excess_value": float(stress_excess.detach().item()),
            "displacement_constraint_excess_value": float(displacement_excess.detach().item()),
            "stress_ratio_value": float(physical_stress_ratio.detach().item()),
            "displacement_ratio_value": float(physical_displacement_ratio.detach().item()),
            "training_stress_ratio_value": float(training_stress_ratio.detach().item()),
            "training_displacement_ratio_value": float(training_displacement_ratio.detach().item()),
            "physical_stress_ratio_value": float(physical_stress_ratio.detach().item()),
            "physical_displacement_ratio_value": float(physical_displacement_ratio.detach().item()),
            "training_feasible": training_feasible,
            "physical_feasible": physical_feasible,
            "constraint_violation_value": float(constraint_violation.detach().item()),
            "fem_total_is_finite": self._scalar_tensor_is_finite(constraint_total),
            "fem_total_value": float(constraint_total.detach().item()),
        })

        fem_valid = debug["fem_solve_scalar_is_finite"] and debug["fem_total_is_finite"]
        debug["fem_valid"] = fem_valid
        debug["failure_reason"] = None if fem_valid else "Non-finite FEM constraint loss"

        self.trainer.last_fem_debug = debug
        if save_debug_history:
            self.trainer.fem_debug_history.append(debug.copy())

        return {
            "fem_total": constraint_total,
            "fem_valid": fem_valid,
            "failure_reason": debug["failure_reason"],
            "density_field": rho.detach(),
            "stiffness_factor_field": stiffness_factor.detach(),
            "stress_field": stress_field.detach() if isinstance(stress_field, torch.Tensor) else stress_field,
            "displacement_field": displacement_field.detach() if isinstance(displacement_field, torch.Tensor) else displacement_field,
            "loaded_boundary_displacement_field": loaded_boundary_displacement_field.detach() if isinstance(loaded_boundary_displacement_field, torch.Tensor) else loaded_boundary_displacement_field,
            "stress_constraint_loss": stress_constraint_loss,
            "displacement_constraint_loss": displacement_constraint_loss,
            "baseline_fem_loss": baseline_fem_loss,
            "violation_fem_loss": violation_fem_loss,
            "stress_constraint_excess": stress_excess,
            "displacement_constraint_excess": displacement_excess,
            "stress_ratio": physical_stress_ratio,
            "displacement_ratio": physical_displacement_ratio,
            "training_stress_ratio": training_stress_ratio,
            "training_displacement_ratio": training_displacement_ratio,
            "physical_stress_ratio": physical_stress_ratio,
            "physical_displacement_ratio": physical_displacement_ratio,
            "constraint_violation": constraint_violation,
            "training_feasible": training_feasible,
            "physical_feasible": physical_feasible,
            "stress_max": stress_max_tensor.detach(),
            "displacement_max": displacement_max_tensor.detach(),
        }
