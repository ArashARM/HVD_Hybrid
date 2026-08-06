import math


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
    hard_active_seed_count: float,
    min_active_seeds: int,
    design_score: float,
    raw_total_fiber_length: float,
    mechanical_violation: float,
    total_loss_is_finite: bool,
    fem_is_valid: bool,
    global_step: int | float = 0,
) -> tuple[bool, tuple[float, float, float]]:
    displacement_ratio = float(physical_displacement_ratio)
    stress_ratio = float(physical_stress_ratio)
    active_count = float(hard_active_seed_count)
    score = float(design_score)
    fiber_length = float(raw_total_fiber_length)
    violation = float(mechanical_violation)
    step = float(global_step)

    if displacement_ratio < 0.0:
        displacement_ratio = float("inf")
    if stress_ratio < 0.0:
        stress_ratio = float("inf")
    if active_count < 0.0:
        active_count = float("nan")
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
    active_count_is_finite = math.isfinite(active_count)
    score_is_finite = math.isfinite(score)
    fiber_length_is_finite = math.isfinite(fiber_length)
    violation_is_finite = math.isfinite(violation)
    step_is_finite = math.isfinite(step)

    physical_max_ratio = (
        max(displacement_ratio, stress_ratio)
        if ratios_are_finite
        else float("inf")
    )
    feasible = (
        bool(fem_is_valid)
        and bool(total_loss_is_finite)
        and ratios_are_finite
        and active_count_is_finite
        and score_is_finite
        and fiber_length_is_finite
        and violation_is_finite
        and step_is_finite
        and physical_max_ratio <= 1.0
        and active_count >= int(min_active_seeds)
    )
    if feasible:
        return True, (score, fiber_length, step)

    violation_score = violation if violation_is_finite else float("inf")
    ranked_score = score if score_is_finite else float("inf")
    ranked_fiber_length = fiber_length if fiber_length_is_finite else float("inf")
    ranked_step = step if step_is_finite else float("inf")
    return False, (violation_score, ranked_score, ranked_step)


def checkpoint_raw_fiber_length(checkpoint) -> float:
    if checkpoint is None or not hasattr(checkpoint, "get"):
        return float("inf")
    row = checkpoint.get("row", {})
    if hasattr(row, "get"):
        return float(row.get("loss_total_fiber_length", float("inf")))
    return float("inf")


def checkpoint_design_score(checkpoint) -> float:
    if checkpoint is None or not hasattr(checkpoint, "get"):
        return float("inf")
    row = checkpoint.get("row", {})
    if hasattr(row, "get"):
        return float(row.get("design_score", float("inf")))
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
            selected_score = checkpoint_design_score(selected_checkpoint)
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
