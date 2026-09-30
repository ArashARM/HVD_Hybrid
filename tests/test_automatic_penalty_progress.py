"""Controller tests require only unittest. Actual autograd-helper tests need torch.

Run from the repository root:
python -m unittest discover -s tests -p test_automatic_penalty_progress.py -v
"""
import ast
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from Training.FEMControl import AutomaticConstraintController

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "Training/MainTrain.py").read_text())
TRAINER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "NN_Trainer")


class ProgressControllerTests(unittest.TestCase):
    def controller(self, **kwargs):
        config = dict(update_interval=4, ema_alpha=1.0, growth=1.1,
                      decay=0.98, progress_rtol=0.05)
        config.update(kwargs)
        return AutomaticConstraintController(**config)

    def observe(self, c, violation, **kwargs):
        inputs = dict(violations={"stress": violation}, objective_grad_norm=1.0,
                      constraint_grad_norms={"stress": 2.0})
        inputs.update(kwargs)
        return c.update(**inputs)["stress"]

    def test_improving_violation_does_not_exponentially_grow(self):
        c = self.controller()
        weights = [self.observe(c, 0.98 ** i) for i in range(400)]
        self.assertEqual(max(weights), 1.0)

    def test_stalled_violation_grows_only_at_window_boundary(self):
        c = self.controller()
        self.assertEqual(self.observe(c, 1.0), 1.0)
        for _ in range(3):
            self.assertEqual(self.observe(c, 1.0), 1.0)
        self.assertAlmostEqual(self.observe(c, 1.0), 1.1)
        self.assertEqual(c.last_diagnostics["stress"]["reason"], "stalled_grow")
        self.assertAlmostEqual(self.observe(c, 1.0), 1.1)

    def test_worsening_violation_grows_gradually(self):
        c = self.controller()
        for i in range(5):
            value = self.observe(c, 1 + i)
        self.assertAlmostEqual(value, 1.1)

    def test_satisfied_constraint_waits_then_decays(self):
        c = self.controller()
        for _ in range(3):
            self.assertEqual(self.observe(c, 0), 1.0)
        self.assertAlmostEqual(self.observe(c, 0), .98)
        self.assertEqual(c.last_diagnostics["stress"]["reason"], "satisfied_decay")

    def test_improving_excessive_coefficient_can_rebalance_down(self):
        c = self.controller()
        c.penalties["stress"] = 1000.0
        self.assertAlmostEqual(self.observe(c, 1), 980.0)
        for v in (.9, .8, .7, .6):
            last = self.observe(c, v)
        self.assertAlmostEqual(last, 960.4)
        self.assertEqual(c.last_diagnostics["stress"]["reason"], "improving_rebalance_down")

    def test_initial_gradient_calibration_is_rate_limited(self):
        c = self.controller()
        self.assertAlmostEqual(self.observe(c, 1, constraint_grad_norms={"stress": 1e-10}), 1.1)

    def test_missing_zero_negative_and_nonfinite_gradients_hold(self):
        for grad in (None, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(grad=grad):
                c = self.controller()
                for _ in range(100):
                    value = self.observe(c, 1, constraint_grad_norms={"stress": grad})
                self.assertEqual(value, 1.0)
                self.assertFalse(c._progress)
                self.assertEqual(c.last_diagnostics["stress"]["reason"], "unusable_constraint_gradient")

    def test_explicit_gradient_failure_overrides_positive_norm(self):
        c = self.controller()
        for _ in range(10):
            value = self.observe(c, 1, constraint_grad_statuses={"stress": "autograd_error"})
        self.assertEqual(value, 1)

    def test_nonfinite_or_negative_violation_does_not_decay(self):
        for v in (float("nan"), float("inf"), -1.0, None):
            with self.subTest(v=v):
                c = self.controller()
                for _ in range(10):
                    self.observe(c, v)
                self.assertEqual(c.get("stress"), 1)
                self.assertEqual(c.last_diagnostics["stress"]["reason"], "invalid_violation")

    def test_inactive_and_absent_constraints_do_not_decay(self):
        c = self.controller()
        for _ in range(100):
            self.observe(c, 0, active_constraints=set())
        c.update(violations={})
        self.assertEqual(set(c.penalties.values()), {1.0})

    def test_skipped_steps_do_not_advance_progress_or_coefficients(self):
        c = self.controller()
        self.observe(c, 1)
        history, weights = copy.deepcopy(c._progress), dict(c.penalties)
        for _ in range(20):
            self.observe(c, 1, step_accepted=False)
        self.assertEqual(c._progress, history)
        self.assertEqual(c.penalties, weights)
        self.assertEqual(c.last_diagnostics["stress"]["reason"], "skipped_step")

    def test_constraints_are_independent(self):
        c = self.controller()
        for _ in range(5):
            c.update(violations={"stress": 1, "displacement": 0},
                     objective_grad_norm=1, constraint_grad_norms={"stress": 2})
        self.assertAlmostEqual(c.get("stress"), 1.1)
        self.assertAlmostEqual(c.get("displacement"), .98)
        self.assertEqual(c.get("length_under"), 1)

    def test_boundary_switching_does_not_recalibrate_every_crossing(self):
        c = self.controller()
        for _ in range(100):
            self.observe(c, .001, constraint_grad_norms={"stress": 1e-8})
            self.observe(c, 0)
        self.assertAlmostEqual(c.get("stress"), 1.1)

    def test_invalid_gradient_interrupt_does_not_recalibrate(self):
        c = self.controller()
        self.observe(c, 1, constraint_grad_norms={"stress": 1e-8})
        for _ in range(3):
            self.observe(c, .9, constraint_grad_norms={"stress": 0})
        self.assertAlmostEqual(self.observe(c, .8, constraint_grad_norms={"stress": 1e-8}), 1.1)

    def test_ema_does_not_treat_one_small_downward_spike_as_sustained_progress(self):
        c = self.controller(ema_alpha=.2, progress_rtol=.1)
        for v in (1, 1, 1, 1, .9):
            self.observe(c, v)
        self.assertEqual(c.last_diagnostics["stress"]["reason"], "stalled_grow")

    def test_zero_primary_gradient_allows_valid_constraint_recovery(self):
        c = self.controller()
        for _ in range(5):
            self.observe(c, 1, objective_grad_norm=0)
        self.assertAlmostEqual(c.get("stress"), 1.1)

    def test_bounds_are_enforced(self):
        c = self.controller(min_penalty=.5, max_penalty=1.2)
        for _ in range(100):
            self.observe(c, 1)
        self.assertEqual(c.get("stress"), 1.2)
        for _ in range(500):
            self.observe(c, 0)
        self.assertEqual(c.get("stress"), .5)

    def test_invalid_configuration_rejected(self):
        for config in ({"update_interval": 1}, {"update_interval": 2.5},
                       {"update_interval": True}, {"growth": .9}, {"decay": 1.1},
                       {"decay": 0}, {"ema_alpha": 0}, {"progress_rtol": 1},
                       {"progress_atol": -1}, {"eps": 0}, {"gradient_ratio": 0},
                       {"max_penalty": float("inf")}, {"min_penalty": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.controller(**config)

    def test_previously_failing_improvement_reproduction(self):
        c = self.controller(update_interval=25, ema_alpha=.2)
        weights = [self.observe(c, .1/(i+1)) for i in range(146)]
        self.assertEqual(max(weights), 1)


class TrainingIntegrationTests(unittest.TestCase):
    """Execute the actual trainer commit block without CAD/visualization imports."""
    def commit(self, controller, skipped):
        train = next(n for n in TRAINER.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        block = next(n for n in ast.walk(train) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "automatic_update_payload is not None")
        namespace = dict(automatic_constraints=controller, optimizer_step_skipped=skipped,
                         automatic_update_payload=dict(violations={"stress": 1.0},
                             objective_grad_norm=1, constraint_grad_norms={"stress": 2}))
        exec(compile(ast.Module(body=[copy.deepcopy(block)], type_ignores=[]), "<trainer commit>", "exec"), namespace)
        return namespace

    def test_actual_trainer_commit_on_success(self):
        c = AutomaticConstraintController(update_interval=2, ema_alpha=1, growth=1.1)
        for _ in range(3):
            ns = self.commit(c, False)
        self.assertAlmostEqual(c.get("stress"), 1.1)
        self.assertEqual(ns["automatic_update_reason"], "progress_window_update")

    def test_actual_trainer_commit_preserves_skipped_state(self):
        c = AutomaticConstraintController()
        ns = self.commit(c, True)
        self.assertEqual(c.get("stress"), 1)
        self.assertFalse(c._progress)
        self.assertEqual(ns["automatic_update_reason"], "skipped_optimizer_step")

    def test_controller_commit_is_after_optimizer_step_in_source(self):
        train = next(n for n in TRAINER.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)]
        updates = [n for n in calls if ast.unparse(n.func) == "automatic_constraints.update"]
        steps = [n for n in calls if ast.unparse(n.func) == "opt.step"]
        self.assertEqual(len(updates), 1)
        self.assertGreater(updates[0].lineno, max(n.lineno for n in steps))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GradientDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        methods = [copy.deepcopy(n) for n in TRAINER.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_autograd_grad_norm", "_named_trainable_params")]
        helper = ast.ClassDef(name="Helper", bases=[], keywords=[], body=methods, decorator_list=[])
        mod = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
        namespace = dict(torch=torch, math=math)
        exec(compile(mod, "<actual trainer gradient helper>", "exec"), namespace)
        cls.helper = namespace["Helper"]

    def model(self):
        model = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.fill_(1)
        return model

    def test_valid_gradient_and_repeated_backward(self):
        model = self.model(); loss = model.weight.square().sum(); info = {}
        for _ in range(3):
            self.assertAlmostEqual(self.helper._autograd_grad_norm(loss, [model], diagnostics=info), 2)
            self.assertEqual(info["status"], "ok")
        loss.backward()
        self.assertAlmostEqual(model.weight.grad.item(), 2)

    def test_zero_disconnected_and_unused_gradients_are_distinct(self):
        model = self.model(); info = {}
        self.helper._autograd_grad_norm((model.weight * 0).sum(), [model], diagnostics=info)
        self.assertEqual(info["status"], "zero_or_tiny_gradient")
        self.helper._autograd_grad_norm(torch.tensor(1.), [model], diagnostics=info)
        self.assertEqual(info["status"], "no_grad_path")
        self.helper._autograd_grad_norm(torch.tensor(1., requires_grad=True), [model], diagnostics=info)
        self.assertEqual(info["status"], "unused_parameters")

    def test_nonfinite_gradient_is_not_sanitized_into_valid_calibration(self):
        model = self.model(); info = {}
        with torch.no_grad():
            model.weight.zero_()
        self.assertEqual(self.helper._autograd_grad_norm(model.weight.sqrt().sum(), [model], diagnostics=info), 0)
        self.assertEqual(info["status"], "nonfinite_gradient")

    def test_runtime_error_is_reported(self):
        model = self.model(); info = {}; loss = model.weight.sum()
        with patch.object(torch.autograd, "grad", side_effect=RuntimeError("test solver failure")):
            self.assertEqual(self.helper._autograd_grad_norm(loss, [model], diagnostics=info), 0)
        self.assertEqual(info["status"], "autograd_error")
        self.assertIn("test solver failure", info["message"])

    def test_small_resolvable_gradient_is_retained(self):
        model = self.model(); info = {}
        norm = self.helper._autograd_grad_norm((model.weight * 1e-7).sum(), [model], diagnostics=info)
        self.assertAlmostEqual(norm, 1e-7, places=14)
        self.assertEqual(info["status"], "ok")


if __name__ == "__main__":
    unittest.main()
