import math
from dataclasses import dataclass, field


def _finite_nonnegative(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < 0.0:
        return float(default)
    return value


@dataclass
class AutomaticConstraintController:
    min_penalty: float = 1.0e-3
    max_penalty: float = 1.0e6
    growth: float = 1.25
    decay: float = 0.98
    gradient_ratio: float = 2.0
    eps: float = 1.0e-12
    penalties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "stress",
            "displacement",
            "seed_spacing",
            "length_under",
            "length_over",
        ):
            self.penalties.setdefault(name, 1.0)

    def update(
        self,
        *,
        violations: dict[str, float],
        objective_grad_norm: float | None = None,
        constraint_grad_norms: dict[str, float] | None = None,
    ) -> dict[str, float]:
        objective_grad = _finite_nonnegative(objective_grad_norm, 0.0)
        constraint_grad_norms = constraint_grad_norms or {}
        updated: dict[str, float] = {}
        for name, before in self.penalties.items():
            violation = _finite_nonnegative(violations.get(name, 0.0), 0.0)
            c_grad = _finite_nonnegative(
                constraint_grad_norms.get(name, 0.0),
                0.0,
            )
            target = before
            if violation > self.eps:
                if objective_grad > self.eps and c_grad > self.eps:
                    target = self.gradient_ratio * objective_grad / max(c_grad, self.eps)
                target = max(target, before * self.growth)
            else:
                target = before * self.decay
            bounded = min(max(target, self.min_penalty), self.max_penalty)
            updated[name] = bounded
        self.penalties.update(updated)
        return dict(self.penalties)

    def get(self, name: str, default: float = 1.0) -> float:
        return float(self.penalties.get(name, default))


def update_adaptive_fem_lambda(
    *,
    lambda_before: float,
    fem_is_valid: bool,
    constraint_violation: float,
    tolerance: float,
    growth: float,
    decay: float,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, str]:
    if not fem_is_valid:
        return min(lambda_before * growth, lambda_max), "grow_invalid_fem"
    if not math.isfinite(float(constraint_violation)):
        return min(lambda_before * growth, lambda_max), "grow_nonfinite_violation"
    if float(constraint_violation) > float(tolerance):
        return min(lambda_before * growth, lambda_max), "grow_constraint_violation"
    return max(lambda_before * decay, lambda_min), "decay_feasible"


def checkpoint_feasibility_key(
    *,
    physical_displacement_ratio: float,
    physical_stress_ratio: float,
    physical_feasible: bool | None = None,
    seed_spacing_feasible: bool,
    L_total: float | None = None,
    legacy_score: float | None = None,
    raw_total_fiber_length: float,
    mechanical_violation: float,
    total_loss_is_finite: bool,
    fem_is_valid: bool,
    global_step: int | float = 0,
    optimization_mode: str = "minimize_length",
    target_length_feasible: bool = True,
    **legacy_kwargs,
) -> tuple[bool, tuple[float, float, float]]:
    if L_total is None and legacy_score is None:
        legacy_score = legacy_kwargs.get("design" + "_score")
    displacement_ratio = float(physical_displacement_ratio)
    stress_ratio = float(physical_stress_ratio)
    if L_total is None:
        L_total = legacy_score
    score = float(L_total)
    fiber_length = float(raw_total_fiber_length)
    violation = float(mechanical_violation)
    step = float(global_step)

    if displacement_ratio < 0.0:
        displacement_ratio = float("inf")
    if stress_ratio < 0.0:
        stress_ratio = float("inf")
    if fiber_length < 0.0:
        fiber_length = float("nan")
    if violation < 0.0:
        violation = float("inf")
    if step < 0.0:
        step = float("inf")

    ratios_are_finite = (
        math.isfinite(displacement_ratio)
        and math.isfinite(stress_ratio)
    )
    score_is_finite = math.isfinite(score)
    fiber_length_is_finite = math.isfinite(fiber_length)
    violation_is_finite = math.isfinite(violation)
    step_is_finite = math.isfinite(step)

    physical_max_ratio = (
        max(displacement_ratio, stress_ratio)
        if ratios_are_finite
        else float("inf")
    )
    physical_ok = (
        bool(physical_feasible)
        if physical_feasible is not None
        else physical_max_ratio <= 1.0
    )
    mode_name = str(optimization_mode).strip().lower()
    feasible = (
        bool(fem_is_valid)
        and bool(total_loss_is_finite)
        and ratios_are_finite
        and score_is_finite
        and fiber_length_is_finite
        and violation_is_finite
        and step_is_finite
        and physical_ok
        and bool(seed_spacing_feasible)
        and (
            mode_name != "minimize_displacement_at_target_length"
            or bool(target_length_feasible)
        )
    )
    if feasible:
        if mode_name == "minimize_displacement_at_target_length":
            return True, (displacement_ratio, stress_ratio, fiber_length, step)
        return True, (fiber_length, physical_max_ratio, step)

    violation_score = (
        violation
        if violation_is_finite
        else float("inf")
    )

    # An invalid fibre length makes the geometry invalid, so it must
    # not be ranked using an otherwise finite design score.
    ranked_score = (
        score
        if score_is_finite and fiber_length_is_finite
        else float("inf")
    )

    ranked_step = (
        step
        if step_is_finite
        else float("inf")
    )

    return False, (
        violation_score,
        ranked_score,
        ranked_step,
    )


def checkpoint_raw_fiber_length(checkpoint) -> float:
    if checkpoint is None or not hasattr(checkpoint, "get"):
        return float("inf")
    row = checkpoint.get("row", {})
    if hasattr(row, "get"):
        return float(row.get("loss_total_fiber_length", float("inf")))
    return float("inf")


def checkpoint_L_total(checkpoint) -> float:
    if checkpoint is None or not hasattr(checkpoint, "get"):
        return float("inf")
    row = checkpoint.get("row", {})
    if hasattr(row, "get"):
        return float(
            row.get(
                "L_total",
                row.get(
                    "primary_objective",
                    row.get("design" + "_score", row.get("stage_monitor_raw", float("inf"))),
                ),
            )
        )
    return float("inf")


def checkpoint_stage_id(checkpoint) -> int | None:
    if checkpoint is None or not hasattr(checkpoint, "get"):
        return None
    row = checkpoint.get("row", {})
    if hasattr(row, "get") and row.get("stage", None) is not None:
        return int(row.get("stage"))
    if checkpoint.get("stage_id", None) is not None:
        return int(checkpoint.get("stage_id"))
    return None


def checkpoint_is_physical_candidate(
    checkpoint,
    *,
    first_physical_stage: int = 1,
) -> bool:
    stage_id = checkpoint_stage_id(checkpoint)
    return stage_id is None or int(stage_id) >= int(first_physical_stage)


def select_final_checkpoint(
    *,
    proposed_checkpoint=None,
    proposed_source: str = "unavailable",
    proposed_score: float = float("inf"),
    best_feasible_checkpoint=None,
    best_feasible_key=None,
    best_infeasible_checkpoint=None,
    best_infeasible_key=None,
    last_valid_checkpoint=None,
    first_physical_stage: int = 1,
) -> tuple[str, object, float]:
    selected_checkpoint = (
        proposed_checkpoint
        if checkpoint_is_physical_candidate(
            proposed_checkpoint,
            first_physical_stage=first_physical_stage,
        )
        else None
    )
    selected_source = proposed_source if selected_checkpoint is not None else "unavailable"
    selected_score = float(proposed_score) if selected_checkpoint is not None else float("inf")

    feasible_checkpoint = (
        best_feasible_checkpoint
        if checkpoint_is_physical_candidate(
            best_feasible_checkpoint,
            first_physical_stage=first_physical_stage,
        )
        else None
    )
    infeasible_checkpoint = (
        best_infeasible_checkpoint
        if checkpoint_is_physical_candidate(
            best_infeasible_checkpoint,
            first_physical_stage=first_physical_stage,
        )
        else None
    )
    valid_checkpoint = (
        last_valid_checkpoint
        if checkpoint_is_physical_candidate(
            last_valid_checkpoint,
            first_physical_stage=first_physical_stage,
        )
        else None
    )

    if feasible_checkpoint is not None:
        selected_checkpoint = feasible_checkpoint
        selected_source = "best_feasible"
        if best_feasible_key is not None:
            selected_score = float(best_feasible_key[0])
        elif hasattr(selected_checkpoint, "get"):
            selected_score = checkpoint_L_total(selected_checkpoint)
    elif infeasible_checkpoint is not None:
        selected_checkpoint = infeasible_checkpoint
        selected_source = "best_infeasible"
        if best_infeasible_key is not None:
            selected_score = float(best_infeasible_key[0])
    elif valid_checkpoint is not None:
        selected_checkpoint = valid_checkpoint
        selected_source = "last_valid_fallback"
        selected_score = (
            float(selected_checkpoint.get("stage_monitor_raw", float("inf")))
            if selected_checkpoint is not None and hasattr(selected_checkpoint, "get")
            else float("inf")
        )
    elif selected_checkpoint is not None:
        selected_source = proposed_source
        selected_score = float(proposed_score)

    return selected_source, selected_checkpoint, selected_score


def validate_optimizer_parameter_coverage(
    *,
    named_trainable_modules,
    optimizer,
) -> None:
    optimizer_parameter_list = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameter_list]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise RuntimeError(
            "A parameter appears in more than one optimizer parameter group."
        )

    optimizer_parameter_id_set = set(optimizer_parameter_ids)
    all_trainable_parameter_ids = {
        id(parameter)
        for _, module in named_trainable_modules
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    }

    missing_optimizer_parameter_ids = (
        all_trainable_parameter_ids - optimizer_parameter_id_set
    )
    unexpected_optimizer_parameter_ids = (
        optimizer_parameter_id_set - all_trainable_parameter_ids
    )

    if missing_optimizer_parameter_ids:
        missing_names = []
        for module_name, module in named_trainable_modules:
            if module is None:
                continue
            for parameter_name, parameter in module.named_parameters():
                if (
                    parameter.requires_grad
                    and id(parameter) in missing_optimizer_parameter_ids
                ):
                    missing_names.append(f"{module_name}.{parameter_name}")
        raise RuntimeError(
            "Trainable parameters are missing from the optimizer: "
            + ", ".join(missing_names)
        )

    if unexpected_optimizer_parameter_ids:
        raise RuntimeError(
            "Optimizer contains parameters that are not currently trainable "
            "in the live training modules."
        )
