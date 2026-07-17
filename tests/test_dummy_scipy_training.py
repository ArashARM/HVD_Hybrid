from __future__ import annotations

import torch
import torch.nn as nn

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder
import Training.MainTrain as main_train
from Training.MainTrain import (
    NN_Trainer,
    TrainingConfig,
    build_edge_type_mask,
    build_shared_curve_geometry,
    compute_all_edge_curve_lengths,
    needs_shared_curve_geometry,
    resolve_edge_types_in_losses,
)


def make_decoder(**kwargs):
    face_mesh = {
        "uv": torch.empty((0, 2)),
        "Xu": None,
        "Xv": None,
        "points_xyz": None,
    }
    return ContinuousVoronoiDecoder(None, face_mesh, **kwargs)


def test_training_config_numeric_defaults_are_scalars() -> None:
    cfg = TrainingConfig()

    assert isinstance(cfg.curve_length_worst_weight, float)
    assert isinstance(cfg.curve_length_outlier_weight, float)
    assert isinstance(cfg.cell_edge_uniform_eps, float)
    assert isinstance(cfg.cell_angle_eps, float)
    assert isinstance(cfg.cell_vertex_merge_tolerance, float)


def test_training_config_normalizes_single_item_numeric_tuples() -> None:
    cfg = TrainingConfig(cell_angle_eps=(1e-7,))

    assert cfg.cell_angle_eps == 1e-7


def test_resolve_edge_types_in_losses_modes() -> None:
    assert resolve_edge_types_in_losses("Interior") == (0,)
    assert resolve_edge_types_in_losses("interior") == (0,)
    assert resolve_edge_types_in_losses(" VDonly ") == (0, 1, 3)
    assert resolve_edge_types_in_losses("all") == (0, 1, 3, 4)

    for invalid in ("shell", "none", "0,1,3", "everything", None):
        try:
            resolve_edge_types_in_losses(invalid)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            assert "Edge_in_losses" in str(exc)
        else:
            raise AssertionError(f"Expected {invalid!r} to fail")


def test_edge_type_masks_exclude_reserved_type() -> None:
    edge_type = torch.tensor([0, 1, 2, 3, 4])

    assert torch.equal(
        build_edge_type_mask(edge_type, resolve_edge_types_in_losses("Interior")),
        torch.tensor([True, False, False, False, False]),
    )
    assert torch.equal(
        build_edge_type_mask(edge_type, resolve_edge_types_in_losses("VDonly")),
        torch.tensor([True, True, False, True, False]),
    )
    assert torch.equal(
        build_edge_type_mask(edge_type, resolve_edge_types_in_losses("all")),
        torch.tensor([True, True, False, True, True]),
    )


def test_edge_in_losses_modes_select_expected_lengths_and_skip_reserved() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    edge_curves_xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    geometry = build_shared_curve_geometry(
        {
            "edge_curves_xyz": edge_curves_xyz,
            "graph": {
                "edge_type": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            },
        }
    )

    trainer.cfg = TrainingConfig(Edge_in_losses="Interior")
    assert torch.allclose(trainer.curve_3d_edge_lengths(geometry), edge_curves_xyz.new_tensor([1.0]))

    trainer.cfg = TrainingConfig(Edge_in_losses="VDonly")
    assert torch.allclose(trainer.curve_3d_edge_lengths(geometry), edge_curves_xyz.new_tensor([1.0, 2.0, 3.0]))

    trainer.cfg = TrainingConfig(Edge_in_losses="all")
    assert torch.allclose(trainer.curve_3d_edge_lengths(geometry), edge_curves_xyz.new_tensor([1.0, 2.0, 3.0, 4.0]))


def test_compute_all_edge_curve_lengths_known_cases() -> None:
    curves = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(
        compute_all_edge_curve_lengths(curves),
        torch.tensor([5.0, 3.0], dtype=torch.float64),
    )
    assert compute_all_edge_curve_lengths(curves.new_empty((0, 3, 3))).shape == (0,)
    assert torch.equal(
        compute_all_edge_curve_lengths(curves.new_ones((2, 1, 3))),
        curves.new_zeros((2,)),
    )


def test_curve_length_loss_filters_to_edge_in_losses_mode() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(Edge_in_losses="Interior")
    edge_curves_xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": edge_curves_xyz,
        "graph": {
            "edge_type": torch.tensor([0, 0, 1, 4], dtype=torch.long),
        },
    }

    geometry = build_shared_curve_geometry(decoder_out)
    filtered = trainer.curve_3d_edge_lengths(geometry)
    all_lengths = trainer.curve_3d_edge_lengths(
        geometry,
        edge_types=resolve_edge_types_in_losses("all"),
    )
    loss = trainer.curve_length_similarity_loss(geometry)

    assert torch.allclose(filtered, edge_curves_xyz.new_tensor([1.0, 1.2]))
    assert torch.allclose(all_lengths, edge_curves_xyz.new_tensor([1.0, 1.2, 0.02, 8.0]))
    assert loss < edge_curves_xyz.new_tensor(100.0)
    loss.backward()
    assert edge_curves_xyz.grad is not None
    assert torch.isfinite(edge_curves_xyz.grad[:2]).all()


def test_shared_curve_geometry_gradients_feed_curve_losses() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        Edge_in_losses="VDonly",
        lam_cell_angle_uniform=0.0,
        lam_cell_radial_uniform=0.0,
    )
    curves = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": curves,
        "graph": {
            "edge_type": torch.tensor([0, 1, 3, 4], dtype=torch.long),
            "edge_seed_pair": torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=torch.long),
        },
    }

    geometry = build_shared_curve_geometry(decoder_out)
    total = (
        trainer.curve_length_similarity_loss(geometry)
        + trainer.cell_edge_uniformity_loss(geometry)
    )
    total.backward()

    assert curves.grad is not None
    assert torch.isfinite(curves.grad).all()
    assert torch.count_nonzero(curves.grad).item() > 0


def test_shared_curve_geometry_computes_lengths_once_for_losses_and_metrics(monkeypatch) -> None:
    calls = {"count": 0}
    original = main_train.compute_all_edge_curve_lengths

    def counted(curves):
        calls["count"] += 1
        return original(curves)

    monkeypatch.setattr(main_train, "compute_all_edge_curve_lengths", counted)

    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        Edge_in_losses="VDonly",
        lam_cell_angle_uniform=0.0,
        lam_cell_radial_uniform=0.0,
    )
    curves = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": curves,
        "graph": {
            "edge_type": torch.tensor([0, 1, 3], dtype=torch.long),
            "edge_seed_pair": torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long),
        },
    }

    geometry = main_train.build_shared_curve_geometry(decoder_out)
    trainer.curve_length_similarity_loss(geometry)
    trainer.cell_edge_uniformity_loss(geometry)
    selected_lengths = trainer.curve_3d_edge_lengths(geometry).detach()
    trainer.solution_topology_metrics(decoder_out, curve_lengths=selected_lengths)

    assert calls["count"] == 1


def test_shared_curve_geometry_not_built_when_losses_and_reporting_are_inactive(monkeypatch) -> None:
    calls = {"count": 0}

    def counted(curves):
        calls["count"] += 1
        return curves.new_zeros((curves.shape[0],))

    monkeypatch.setattr(main_train, "compute_all_edge_curve_lengths", counted)

    need_geometry = needs_shared_curve_geometry(
        compute_curve_length_loss=False,
        compute_cell_edge_uniform_loss=False,
        collect_curve_metrics=False,
        collect_topology_metrics=False,
    )
    decoder_out = {
        "edge_curves_xyz": torch.zeros((2, 2, 3)),
        "graph": {
            "edge_type": torch.tensor([0, 1]),
            "edge_seed_pair": torch.tensor([[0, 1], [1, 2]]),
        },
    }
    geometry = main_train.build_shared_curve_geometry(decoder_out) if need_geometry else None

    assert geometry is None
    assert calls["count"] == 0


def test_shared_curve_geometry_requires_matching_edge_type_metadata() -> None:
    curves = torch.zeros((2, 2, 3), dtype=torch.double)

    try:
        build_shared_curve_geometry({"edge_curves_xyz": curves, "graph": {}})
    except ValueError as exc:
        assert "edge_type" in str(exc)
    else:
        raise AssertionError("missing edge_type should fail")

    try:
        build_shared_curve_geometry(
            {
                "edge_curves_xyz": curves,
                "graph": {
                    "edge_type": torch.tensor([0]),
                    "edge_seed_pair": torch.tensor([[0, 1], [1, 2]]),
                },
            }
        )
    except ValueError as exc:
        assert "edge_type" in str(exc)
    else:
        raise AssertionError("mismatched edge_type should fail")


def test_shared_curve_geometry_validates_edge_seed_pair_shape() -> None:
    try:
        build_shared_curve_geometry(
            {
                "edge_curves_xyz": torch.zeros((2, 2, 3), dtype=torch.double),
                "graph": {
                    "edge_type": torch.tensor([0, 1]),
                    "edge_seed_pair": torch.tensor([[0, 1, 2]]),
                },
            }
        )
    except ValueError as exc:
        assert "edge_seed_pair" in str(exc)
    else:
        raise AssertionError("mismatched edge_seed_pair should fail")


def test_shared_curve_geometry_requires_edge_seed_pair_when_requested() -> None:
    try:
        build_shared_curve_geometry(
            {
                "edge_curves_xyz": torch.zeros((2, 2, 3), dtype=torch.double),
                "graph": {
                    "edge_type": torch.tensor([0, 1]),
                },
            },
            require_edge_seed_pair=True,
        )
    except ValueError as exc:
        assert "edge_seed_pair" in str(exc)
    else:
        raise AssertionError("missing edge_seed_pair should fail when required")


def test_cell_edge_uniformity_excludes_shell_edges_even_in_all_mode() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        Edge_in_losses="all",
        lam_cell_angle_uniform=0.0,
        lam_cell_radial_uniform=0.0,
    )
    edge_curves_xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": edge_curves_xyz,
        "graph": {
            "edge_type": torch.tensor([0, 1, 4], dtype=torch.long),
            "edge_seed_pair": torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.long),
        },
    }

    loss = trainer.cell_edge_uniformity_loss(build_shared_curve_geometry(decoder_out))

    assert torch.allclose(loss, edge_curves_xyz.new_zeros(()))


def test_compute_all_edge_curve_lengths_gradcheck() -> None:
    curves = torch.randn(3, 4, 3, dtype=torch.double, requires_grad=True)
    curves = curves + torch.arange(4, dtype=torch.double).view(1, 4, 1)

    assert torch.autograd.gradcheck(compute_all_edge_curve_lengths, (curves,))


class DummyUnitSquareCadDomain:
    def eval_uv_norm_batch(self, uv, return_inside_mask=False):
        xyz = torch.cat(
            (
                uv,
                torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device),
            ),
            dim=-1,
        )
        out = {"xyz": xyz}
        if return_inside_mask:
            out["inside_mask"] = torch.ones(uv.shape[:-1], dtype=torch.bool, device=uv.device)
        return out


def test_duplicate_suppression_active_seed_accounting_matches_training_helper() -> None:
    dtype = torch.float64
    face_mesh = {
        "uv": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=dtype,
        ),
        "Xu": torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=dtype),
        "Xv": torch.tensor([[0.0, 1.0, 0.0]] * 4, dtype=dtype),
        "points_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=dtype,
        ),
    }
    decoder = ContinuousVoronoiDecoder(
        DummyUnitSquareCadDomain(),
        face_mesh,
        use_seed_activation=True,
        use_trim_activity=False,
        duplicate_merge_sigma=0.08,
    )
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.12, 0.10],
            [0.80, 0.10],
            [0.82, 0.10],
            [0.50, 0.80],
        ],
        dtype=dtype,
    )
    w_raw = torch.ones((seeds.shape[0], seeds.shape[0]), dtype=dtype) * 0.02

    out = decoder(seeds_uv=seeds, w_raw=w_raw, generate_density_fiber=False)

    raw_count = int(out["seeds_uv"].shape[0])
    active_count = int(out["seed_active_mask"].sum().item())

    assert raw_count == 5
    assert active_count < raw_count
    assert int(out["active_seed_ids"].numel()) == active_count
    assert int(out["topology_seeds_uv"].shape[0]) == active_count

    counts = NN_Trainer._decoder_seed_activation_counts(out)
    assert counts["raw"] == raw_count
    assert counts["active"] == active_count
    assert counts["topology"] == active_count
    assert counts["inactive"] == raw_count - active_count


def test_prediction_clone_preserves_activation_metadata_for_timelapse() -> None:
    pred = {
        "face_id": 0,
        "seeds_raw": torch.zeros((5, 2), dtype=torch.float64),
        "w_raw": torch.zeros((5, 5), dtype=torch.float64),
        "seeds_uv": torch.zeros((5, 2), dtype=torch.float64),
        "seed_active_mask": torch.tensor([True, False, True, False, True]),
        "active_seed_ids": torch.tensor([0, 2, 4], dtype=torch.long),
        "seed_activity_weight": torch.tensor([1.0, 0.2, 1.0, 0.2, 1.0]),
        "topology_seeds_uv": torch.zeros((3, 2), dtype=torch.float64),
    }

    cached_pred = NN_Trainer._clone_pred_list([pred])[0]

    assert "seed_active_mask" in cached_pred
    assert "active_seed_ids" in cached_pred
    assert "topology_seeds_uv" in cached_pred
    assert int(cached_pred["seed_active_mask"].sum().item()) == int(cached_pred["topology_seeds_uv"].shape[0])


class DummyVoronoiSeedTrainer(nn.Module):
    """Optimize seed positions while rebuilding hard SciPy topology each step."""

    def __init__(
        self,
        initial_seeds: torch.Tensor,
        learning_rate: float = 1e-2,
        edge_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if initial_seeds.ndim != 2 or initial_seeds.shape[1] != 2:
            raise ValueError("initial_seeds must have shape [S, 2].")
        if initial_seeds.shape[0] < 3:
            raise ValueError("At least three seeds are required for Delaunay triangles.")

        self.seeds_uv = nn.Parameter(initial_seeds.detach().clone())
        self.decoder = ContinuousVoronoiDecoder(return_xyz=False)
        self.optimizer = torch.optim.Adam([self.seeds_uv], lr=learning_rate)
        self.edge_loss_weight = float(edge_loss_weight)

    def delaunay_equal_area_loss(self, out: dict) -> torch.Tensor:
        triangles_np = out["delaunay_triples_np"]
        if triangles_np.shape[0] == 0:
            raise RuntimeError("SciPy returned no Delaunay triangles.")

        triangles = torch.as_tensor(
            triangles_np,
            dtype=torch.long,
            device=self.seeds_uv.device,
        )
        points = self.seeds_uv[triangles]
        edge_01 = points[:, 1] - points[:, 0]
        edge_02 = points[:, 2] - points[:, 0]
        cross = edge_01[:, 0] * edge_02[:, 1] - edge_01[:, 1] * edge_02[:, 0]
        areas = 0.5 * cross.abs()
        return ((areas - areas.mean()) ** 2).mean()

    @staticmethod
    def voronoi_equal_edge_length_loss(out: dict) -> torch.Tensor:
        graph = out["graph"]
        nodes = graph["nodes_uv"]
        edges = graph["edge_index"]
        if edges.shape[0] == 0:
            raise RuntimeError("Generated Voronoi graph has no edges.")

        edge_vectors = nodes[edges[:, 0]] - nodes[edges[:, 1]]
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
        return ((edge_lengths - edge_lengths.mean()) ** 2).mean()

    def training_step(self, step: int) -> dict[str, float | int]:
        self.optimizer.zero_grad(set_to_none=True)

        # SciPy topology is deliberately rebuilt from the current seeds here.
        out = self.decoder(
            self.seeds_uv,
            topology_mode="scipy",
            return_xyz=False,
        )
        area_loss = self.delaunay_equal_area_loss(out)
        loss = area_loss
        if self.edge_loss_weight != 0.0:
            edge_loss = self.voronoi_equal_edge_length_loss(out)
            loss = loss + self.edge_loss_weight * edge_loss

        loss.backward()
        if self.seeds_uv.grad is None:
            raise AssertionError("seeds_uv.grad is None; gradient flow was broken.")
        grad_norm = float(torch.linalg.vector_norm(self.seeds_uv.grad).detach().cpu())
        if not torch.isfinite(self.seeds_uv.grad).all():
            raise AssertionError("seeds_uv.grad contains non-finite values.")
        if grad_norm <= 0.0:
            raise AssertionError("seeds_uv gradient norm must be greater than zero.")

        graph = out["graph"]
        diagnostics = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "num_nodes": int(graph["nodes_uv"].shape[0]),
            "num_edges": int(graph["edge_index"].shape[0]),
            "num_delaunay_triangles": int(out["delaunay_triples_np"].shape[0]),
        }
        print(
            f"step={diagnostics['step']:02d} "
            f"loss={diagnostics['loss']:.8e} "
            f"grad_norm={diagnostics['grad_norm']:.8e} "
            f"nodes={diagnostics['num_nodes']} "
            f"edges={diagnostics['num_edges']} "
            f"delaunay_triangles={diagnostics['num_delaunay_triangles']}"
        )

        self.optimizer.step()
        with torch.no_grad():
            self.seeds_uv.clamp_(0.0, 1.0)
        return diagnostics

    def _decode_current_structure(self) -> dict:
        return self.decoder(
            self.seeds_uv,
            topology_mode="scipy",
            return_xyz=False,
        )

    def _print_and_plot_structure(
        self,
        label: str,
        print_node_table: bool = False,
    ) -> dict:
        out = self._decode_current_structure()
        graph = out["graph"]
        print(f"\n{label} seeds_uv:")
        print(self.seeds_uv.detach().cpu())
        print(
            f"{label} structure: nodes={graph['nodes_uv'].shape[0]}, "
            f"edges={graph['edge_index'].shape[0]}, "
            f"delaunay_triangles={out['delaunay_triples_np'].shape[0]}"
        )
        self.decoder.plot_scipy_vs_generated_graph(
            self.seeds_uv.detach(),
            out=out,
            show_node_ids=True,
            print_node_table=print_node_table,
        )
        return out

    def fit(
        self,
        steps: int = 5,
        plot_first_last: bool = False,
        print_node_table: bool = False,
    ) -> list[dict[str, float | int]]:
        if plot_first_last:
            self._print_and_plot_structure(
                "Initial",
                print_node_table=print_node_table,
            )

        diagnostics = [self.training_step(step) for step in range(int(steps))]

        if plot_first_last:
            self._print_and_plot_structure(
                "Final",
                print_node_table=print_node_table,
            )
        return diagnostics


def irregular_test_seeds(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.12, 0.16],
            [0.42, 0.10],
            [0.81, 0.18],
            [0.20, 0.57],
            [0.55, 0.43],
            [0.87, 0.66],
            [0.36, 0.88],
            [0.72, 0.91],
        ],
        dtype=dtype,
    )


def test_scipy_topology_seed_gradients_with_area_loss() -> None:
    trainer = DummyVoronoiSeedTrainer(
        irregular_test_seeds(),
        learning_rate=5e-3,
    )
    diagnostics = trainer.fit(steps=3)

    assert all(item["grad_norm"] > 0.0 for item in diagnostics)
    assert all(item["num_nodes"] > 0 for item in diagnostics)
    assert all(item["num_edges"] > 0 for item in diagnostics)
    assert all(item["num_delaunay_triangles"] > 0 for item in diagnostics)
    assert torch.all((trainer.seeds_uv >= 0.0) & (trainer.seeds_uv <= 1.0))


def test_scipy_reconstructed_graph_geometry_has_seed_gradients() -> None:
    trainer = DummyVoronoiSeedTrainer(
        irregular_test_seeds(),
        learning_rate=2e-3,
        edge_loss_weight=0.25,
    )
    diagnostics = trainer.fit(steps=2)

    assert all(item["grad_norm"] > 0.0 for item in diagnostics)
    assert torch.all((trainer.seeds_uv >= 0.0) & (trainer.seeds_uv <= 1.0))


def test_smooth_edge_curves_are_differentiable_straight_segments() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = torch.tensor(
        [[0.2, 0.3], [0.8, 0.3]], dtype=torch.float64, requires_grad=True
    )
    # Make endpoint geometry depend on seeds, as it does in the decoder.
    vertices = torch.stack((seeds[0] + 0.1, seeds[1] - 0.1))
    edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    seed_pairs = torch.tensor([[0, 1], [-1, -1]], dtype=torch.long)

    curves = decoder.sample_smooth_edge_curves_uv(
        seeds, vertices, edges, seed_pairs, n_samples=11
    )

    s = torch.linspace(0.0, 1.0, 11, dtype=vertices.dtype).view(1, 11, 1)
    expected = (1.0 - s) * vertices[edges[:, 0], None, :] + s * vertices[edges[:, 1], None, :]

    assert curves.shape == (2, 11, 2)
    assert torch.allclose(curves, expected)

    curves.square().sum().backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad) > 0


def test_graph_edge_curves_are_straight_and_differentiable_for_euclidean_edges() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = irregular_test_seeds().requires_grad_(True)

    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    graph = out["graph"]
    curves = decoder.sample_graph_edge_curves_uv(
        seeds_uv=seeds,
        graph=graph,
        n_samples=13,
    )

    edges = graph["edge_index"]
    edge_type = graph["edge_type"]
    euclidean_edges = edge_type != 4
    assert bool(euclidean_edges.any())

    p0 = graph["nodes_uv"][edges[euclidean_edges, 0]]
    p1 = graph["nodes_uv"][edges[euclidean_edges, 1]]
    s = torch.linspace(0.0, 1.0, 13, dtype=p0.dtype, device=p0.device).view(1, 13, 1)
    expected = (1.0 - s) * p0[:, None, :] + s * p1[:, None, :]

    assert curves.shape == (edges.shape[0], 13, 2)
    assert torch.allclose(curves[euclidean_edges], expected, atol=1e-10, rtol=1e-10)

    curves[euclidean_edges].square().sum().backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad) > 0


def test_scipy_forward_includes_differentiable_smooth_edge_curves() -> None:
    seeds = irregular_test_seeds().requires_grad_(True)
    decoder = ContinuousVoronoiDecoder(return_xyz=False)

    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    curves = out["edge_curves_uv"]

    assert curves.shape[0] == out["graph"]["edge_index"].shape[0]
    assert curves.shape[1] >= decoder.tube_curve_samples
    assert curves.shape[2] == 2
    curves.square().mean().backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad) > 0


def test_smooth_edge_curves_xyz_uses_torch_cad_evaluator() -> None:
    class DummyTorchCadDomain:
        @staticmethod
        def eval_uv_norm_batch_torch(uv: torch.Tensor) -> torch.Tensor:
            return torch.cat((uv, (uv[:, :1] ** 2) + uv[:, 1:2]), dim=1)

    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    curves_uv = torch.rand((3, 7, 2), dtype=torch.float64, requires_grad=True)
    curves_xyz = decoder.sample_smooth_edge_curves_xyz(
        DummyTorchCadDomain(), curves_uv
    )

    assert curves_xyz.shape == (3, 7, 3)
    curves_xyz.sum().backward()
    assert curves_uv.grad is not None
    assert torch.isfinite(curves_uv.grad).all()


def test_square_boundary_edge_sampling_uses_boundary_support() -> None:
    decoder = make_decoder(return_xyz=False)
    dtype = torch.float64
    graph = {
        "boundary_curve_uv": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=dtype,
        ),
        "boundary_curve_offsets": torch.tensor([0, 2, 3, 4, 5], dtype=torch.long),
        "boundary_curve_loop_id": torch.zeros((4,), dtype=torch.long),
    }
    same_side = decoder.sample_cad_boundary_edge_uv(
        torch.tensor([0.0, 0.2], dtype=dtype),
        torch.tensor([0.0, 0.8], dtype=dtype),
        graph=graph,
        n_samples=17,
    )
    around_corner = decoder.sample_cad_boundary_edge_uv(
        torch.tensor([0.0, 0.4], dtype=dtype),
        torch.tensor([0.7, 0.0], dtype=dtype),
        graph=graph,
        n_samples=18,
    )

    for curve in (same_side, around_corner):
        assert curve is not None
        on_boundary = (
            torch.isclose(curve[:, 0], torch.zeros_like(curve[:, 0]), atol=1e-5)
            | torch.isclose(curve[:, 0], torch.ones_like(curve[:, 0]), atol=1e-5)
            | torch.isclose(curve[:, 1], torch.zeros_like(curve[:, 1]), atol=1e-5)
            | torch.isclose(curve[:, 1], torch.ones_like(curve[:, 1]), atol=1e-5)
        )
        assert on_boundary.all()


def test_graph_edge_curve_sampling_dispatches_only_shell_edges_to_boundary_support() -> None:
    decoder = make_decoder(return_xyz=False)
    seeds = torch.tensor(
        [[0.2, 0.3], [0.8, 0.3]], dtype=torch.float64, requires_grad=True
    )
    nodes = torch.tensor(
        [[0.0, 0.4], [0.7, 0.0], [0.5, 0.5]], dtype=torch.float64
    )
    graph = {
        "nodes_uv": nodes,
        "edge_index": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
        "edge_seed_pair": torch.tensor([[-1, -1], [0, 1]], dtype=torch.long),
        "edge_type": torch.tensor([4, 1], dtype=torch.long),
        "boundary_curve_uv": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        ),
        "boundary_curve_offsets": torch.tensor([0, 2, 3, 4, 5], dtype=torch.long),
        "boundary_curve_loop_id": torch.zeros((4,), dtype=torch.long),
    }

    curves = decoder.sample_graph_edge_curves_uv(seeds, graph, n_samples=32)
    shell = curves[0]
    shell_on_boundary = (
        torch.isclose(shell[:, 0], torch.zeros_like(shell[:, 0]), atol=1e-5)
        | torch.isclose(shell[:, 0], torch.ones_like(shell[:, 0]), atol=1e-5)
        | torch.isclose(shell[:, 1], torch.zeros_like(shell[:, 1]), atol=1e-5)
        | torch.isclose(shell[:, 1], torch.ones_like(shell[:, 1]), atol=1e-5)
    )
    assert shell_on_boundary.all()
    # Type 1 starts inside the box and must remain a straight Voronoi edge.
    first = curves[1, 0]
    first_on_boundary = (
        torch.isclose(first[0], first.new_tensor(0.0))
        | torch.isclose(first[0], first.new_tensor(1.0))
        | torch.isclose(first[1], first.new_tensor(0.0))
        | torch.isclose(first[1], first.new_tensor(1.0))
    )
    assert not bool(first_on_boundary)


def test_scipy_shell_curves_stay_on_uv_box() -> None:
    seeds = irregular_test_seeds().requires_grad_(True)
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    edge_type = out["graph"]["edge_type"]
    curves = decoder.sample_graph_edge_curves_uv(
        seeds, out["graph"], n_samples=64
    )

    print(torch.bincount(edge_type))
    print(curves.shape)
    shell_curves = curves[edge_type == 4]
    assert shell_curves.shape[0] > 0
    on_boundary = (
        torch.isclose(shell_curves[..., 0], torch.zeros_like(shell_curves[..., 0]), atol=1e-5)
        | torch.isclose(shell_curves[..., 0], torch.ones_like(shell_curves[..., 0]), atol=1e-5)
        | torch.isclose(shell_curves[..., 1], torch.zeros_like(shell_curves[..., 1]), atol=1e-5)
        | torch.isclose(shell_curves[..., 1], torch.ones_like(shell_curves[..., 1]), atol=1e-5)
    )
    assert on_boundary.all()


def test_soft_tube_field_has_curve_and_radius_gradients() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    query_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.get_default_dtype(),
    )
    curves_xyz = torch.tensor(
        [[[0.0, 0.1, 0.0], [0.5, 0.1, 0.0], [1.0, 0.1, 0.0]]],
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )
    log_radius = decoder.make_learnable_radius(0.02)
    radius = torch.nn.functional.softplus(log_radius)

    tube = decoder.soft_tube_occupancy(
        query_xyz=query_xyz,
        curves_xyz=curves_xyz,
        radius=radius,
        tau_distance=0.02,
        tau_occupancy=0.01,
    )
    larger = decoder.soft_tube_occupancy(
        query_xyz=query_xyz,
        curves_xyz=curves_xyz,
        radius=radius + 0.02,
        tau_distance=0.02,
        tau_occupancy=0.01,
    )

    assert tube["distance"].shape == (query_xyz.shape[0],)
    assert torch.all((tube["occupancy"] >= 0.0) & (tube["occupancy"] <= 1.0))
    assert torch.all(larger["occupancy"] >= tube["occupancy"])
    assert torch.allclose(radius.detach(), radius.new_tensor(0.02), atol=1e-6)

    loss = tube["occupancy"].mean() + 0.01 * radius
    loss.backward()
    assert curves_xyz.grad is not None
    assert torch.isfinite(curves_xyz.grad).all()
    assert log_radius.grad is not None
    assert torch.isfinite(log_radius.grad).all()


def test_width_raw_zero_means_core_curves_only_radius() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = irregular_test_seeds(dtype=torch.float64)
    w_zero = torch.zeros((seeds.shape[0], seeds.shape[0]), dtype=seeds.dtype)
    w_negative = -torch.ones_like(w_zero)
    w_large = torch.ones_like(w_zero)

    radius_zero = decoder.width(w_zero, seeds=seeds)
    radius_negative = decoder.width(w_negative, seeds=seeds)
    radius_large = decoder.width(w_large, seeds=seeds)

    assert torch.allclose(radius_zero, torch.zeros_like(radius_zero))
    assert torch.allclose(radius_negative, torch.zeros_like(radius_negative))
    assert torch.all(radius_large > radius_zero)


def test_return_xyz_false_keeps_lightweight_edge_curves_xyz_for_plotting() -> None:
    class PlaneCad:
        @staticmethod
        def eval_uv_norm_batch_torch(flat_uv: torch.Tensor) -> dict[str, torch.Tensor]:
            z = flat_uv.new_zeros((flat_uv.shape[0], 1))
            return {"xyz": torch.cat((flat_uv, z), dim=1)}

        @staticmethod
        def smooth_inside_activity(points_uv: torch.Tensor, tau: float=0.01) -> torch.Tensor:
            return points_uv.new_ones(points_uv.shape[:-1])

        @staticmethod
        def boundary_parameter(points_uv: torch.Tensor) -> torch.Tensor:
            u, v = points_uv[:, 0], points_uv[:, 1]
            bottom = torch.abs(v) <= 1e-4
            right = ~bottom & (torch.abs(u - 1.0) <= 1e-4)
            top = ~bottom & ~right & (torch.abs(v - 1.0) <= 1e-4)
            parameter = torch.empty_like(u)
            parameter[bottom] = u[bottom]
            parameter[right] = 1.0 + v[right]
            parameter[top] = 3.0 - u[top]
            parameter[~(bottom | right | top)] = 4.0 - v[~(bottom | right | top)]
            return parameter

    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = irregular_test_seeds(dtype=torch.float64).requires_grad_(True)
    w_raw = torch.zeros((seeds.shape[0], seeds.shape[0]), dtype=seeds.dtype)
    points_uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=seeds.dtype,
    )

    out = decoder(
        seeds_raw=seeds,
        w_raw=w_raw,
        points_uv=points_uv,
        cad_domain=PlaneCad(),
        return_xyz=False,
    )

    assert "edge_curves_xyz" in out
    assert "rho" not in out
    assert out["edge_curves_xyz"].shape[:2] == out["edge_curves_uv"].shape[:2]
    assert out["edge_curves_xyz"].shape[-1] == 3

    out["edge_curves_xyz"].square().mean().backward()
    assert seeds.grad is not None and torch.isfinite(seeds.grad).all()


def test_return_xyz_true_can_skip_density_fields_for_fast_struts() -> None:
    class PlaneCad:
        @staticmethod
        def eval_uv_norm_batch_torch(flat_uv: torch.Tensor) -> dict[str, torch.Tensor]:
            z = flat_uv.new_zeros((flat_uv.shape[0], 1))
            return {"xyz": torch.cat((flat_uv, z), dim=1)}

        @staticmethod
        def smooth_inside_activity(points_uv: torch.Tensor, tau: float=0.01) -> torch.Tensor:
            return points_uv.new_ones(points_uv.shape[:-1])

        @staticmethod
        def boundary_parameter(points_uv: torch.Tensor) -> torch.Tensor:
            u, v = points_uv[:, 0], points_uv[:, 1]
            bottom = torch.abs(v) <= 1e-4
            right = ~bottom & (torch.abs(u - 1.0) <= 1e-4)
            top = ~bottom & ~right & (torch.abs(v - 1.0) <= 1e-4)
            parameter = torch.empty_like(u)
            parameter[bottom] = u[bottom]
            parameter[right] = 1.0 + v[right]
            parameter[top] = 3.0 - u[top]
            parameter[~(bottom | right | top)] = 4.0 - v[~(bottom | right | top)]
            return parameter

    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = irregular_test_seeds(dtype=torch.float64).requires_grad_(True)
    w_raw = torch.zeros((seeds.shape[0], seeds.shape[0]), dtype=seeds.dtype)
    points_uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=seeds.dtype,
    )

    out = decoder(
        seeds_raw=seeds,
        w_raw=w_raw,
        points_uv=points_uv,
        cad_domain=PlaneCad(),
        return_xyz=True,
        compute_fields=False,
    )

    assert "edge_curves_xyz" in out
    assert "seeds_xyz" in out
    assert "rho" not in out
    assert "fiber3d" not in out
    assert out["edge_curves_xyz"].shape[:2] == out["edge_curves_uv"].shape[:2]
    assert out["edge_curves_xyz"].shape[-1] == 3

    (out["edge_curves_xyz"].square().mean() + out["seeds_xyz"].square().mean()).backward()
    assert seeds.grad is not None and torch.isfinite(seeds.grad).all()


def test_constructor_face_mesh_supplies_evaluation_tensors() -> None:
    dtype = torch.float64
    face_mesh = {
        "uv": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=dtype,
        ),
        "points_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
            dtype=dtype,
        ),
        "Xu": torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).expand(5, 3),
        "Xv": torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype).expand(5, 3),
        "faces_ijk": torch.tensor([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]], dtype=torch.long),
        "BBX": {"xmin": 0.0, "xmax": 1.0},
    }
    decoder = ContinuousVoronoiDecoder(face_mesh=face_mesh, return_xyz=False)
    seeds = irregular_test_seeds(dtype=dtype)
    w_raw = torch.zeros((seeds.shape[0], seeds.shape[0]), dtype=dtype)

    out = decoder(seeds_raw=seeds, w_raw=w_raw)

    assert decoder.point_Xyz.shape == (5, 3)
    assert decoder.point_UV.shape == (5, 2)
    assert decoder.Xu.shape == (5, 3)
    assert decoder.Xv.shape == (5, 3)
    assert decoder.XV.shape == (5, 3)
    assert decoder.faces_ijk.shape == (4, 3)
    assert decoder.mesh_info["BBX"]["xmax"] == 1.0
    assert out["rho"].shape == (5,)
    assert torch.allclose(out["centerline_radius"], torch.zeros_like(out["centerline_radius"]))


def test_curve_points_and_tangents_xyz_uses_finite_differences() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    points, tangents = decoder.curve_points_and_tangents_xyz(curves)

    assert points.shape == (3, 3)
    assert tangents.shape == (3, 3)
    expected = torch.tensor([1.0, 0.0, 0.0], dtype=curves.dtype).expand_as(tangents)
    assert torch.allclose(tangents, expected)


def test_soft_tube_fem_fields_have_valid_angles_and_gradients() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    elem_centers = torch.tensor(
        [[0.0, 0.1, 0.0], [0.5, 0.2, 0.0], [1.0, 0.1, 0.0]],
        dtype=torch.float64,
    )
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    log_radius = torch.nn.Parameter(
        decoder.make_learnable_radius(0.15).detach().to(dtype=torch.float64)
    )
    radius = torch.nn.functional.softplus(log_radius)
    fields = decoder.soft_tube_density_and_fiber_to_elements(
        elem_centers_xyz=elem_centers,
        curves_xyz=curves,
        radius=radius,
        tau_distance=0.02,
        tau_density=0.02,
        tau_fiber=0.02,
        rho_min=1e-3,
    )

    num_elements = elem_centers.shape[0]
    assert fields["density"].shape == (num_elements,)
    assert fields["fiber"].shape == (num_elements, 3)
    assert fields["phi"].shape == (num_elements,)
    assert fields["theta"].shape == (num_elements,)
    assert fields["distance"].shape == (num_elements,)
    assert torch.all(fields["density"] >= 1e-3)
    assert torch.all(fields["density"] <= 1.0)
    assert torch.isfinite(fields["phi"]).all()
    assert torch.isfinite(fields["theta"]).all()
    assert torch.allclose(
        torch.linalg.vector_norm(fields["fiber"], dim=1),
        torch.ones(num_elements, dtype=curves.dtype),
    )
    assert torch.allclose(fields["fiber"][:, 0], torch.ones(num_elements, dtype=curves.dtype))

    loss = fields["density"].mean() + fields["phi"].square().mean() + 0.01 * radius
    loss.backward()
    assert curves.grad is not None and torch.isfinite(curves.grad).all()
    assert log_radius.grad is not None and torch.isfinite(log_radius.grad).all()


def test_soft_tube_fem_fields_stream_cdist_chunks() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False, tube_cdist_max_values=6)
    elem_centers = torch.tensor(
        [[0.0, 0.2, 0.0], [0.5, 0.1, 0.0], [1.0, 0.2, 0.0], [1.5, 0.1, 0.0]],
        dtype=torch.float64,
    )
    curves = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    fields = decoder.soft_tube_density_and_fiber_to_elements(
        elem_centers_xyz=elem_centers,
        curves_xyz=curves,
        radius=0.15,
        tau_distance=0.02,
        tau_density=0.02,
        tau_fiber=0.02,
        rho_min=1e-3,
    )
    curve_points, _ = decoder.curve_points_and_tangents_xyz(curves)
    expected_distance = torch.cdist(elem_centers, curve_points).min(dim=1).values

    assert torch.allclose(fields["distance"], expected_distance)
    assert torch.isfinite(fields["density"]).all()
    assert torch.isfinite(fields["fiber"]).all()


def test_soft_lift_uv_to_xyz_streams_query_chunks() -> None:
    query_uv = torch.tensor(
        [
            [[0.05, 0.10], [0.95, 0.10], [0.25, 0.85]],
            [[0.75, 0.90], [0.50, 0.50], [0.10, 0.95]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    support_uv = torch.tensor(
        [[0.00, 0.00], [0.25, 0.50], [0.75, 0.50], [1.00, 1.00]],
        dtype=torch.float64,
        requires_grad=True,
    )
    support_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.4, 0.1], [0.8, 0.3, 0.2], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    chunked = ContinuousVoronoiDecoder(
        return_xyz=False,
        tube_lift_tau=0.2,
        tube_lift_max_values=5,
    )
    full = ContinuousVoronoiDecoder(
        return_xyz=False,
        tube_lift_tau=0.2,
        tube_lift_max_values=1_000_000,
    )

    actual = chunked.soft_lift_uv_to_xyz(
        query_uv,
        support_uv,
        support_xyz,
        u_periodic=True,
        v_periodic=True,
    )
    expected = full.soft_lift_uv_to_xyz(
        query_uv,
        support_uv,
        support_xyz,
        u_periodic=True,
        v_periodic=True,
    )

    assert torch.allclose(actual, expected)
    loss = actual.square().mean()
    loss.backward()
    assert query_uv.grad is not None and torch.isfinite(query_uv.grad).all()
    assert support_uv.grad is not None and torch.isfinite(support_uv.grad).all()
    assert support_xyz.grad is not None and torch.isfinite(support_xyz.grad).all()


def test_cell_edge_uniformity_loss_groups_edges_per_cell() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        lam_cell_angle_uniform=0.0,
        lam_cell_radial_uniform=0.0,
    )

    edge_curves_xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": edge_curves_xyz,
        "graph": {
            "edge_seed_pair": torch.tensor(
                [[0, 1], [0, 2], [0, 3], [1, 2]],
                dtype=torch.long,
            ),
            "edge_type": torch.tensor([0, 0, 0, 0], dtype=torch.long),
        },
    }

    loss = trainer.cell_edge_uniformity_loss(build_shared_curve_geometry(decoder_out))

    assert torch.allclose(loss, edge_curves_xyz.new_tensor(5.0 / 54.0))
    loss.backward()
    assert edge_curves_xyz.grad is not None
    assert torch.isfinite(edge_curves_xyz.grad).all()


def test_cell_edge_uniformity_loss_penalizes_per_cell_angles() -> None:
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        lam_cell_angle_uniform=1.0,
        lam_cell_radial_uniform=0.0,
    )

    h = 3.0 ** 0.5 / 2.0
    edge_curves_xyz = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, h, 0.0]],
            [[0.5, h, 0.0], [-0.5, h, 0.0]],
            [[-0.5, h, 0.0], [-1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [-0.5, -h, 0.0]],
            [[-0.5, -h, 0.0], [0.5, -h, 0.0]],
            [[0.5, -h, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    decoder_out = {
        "edge_curves_xyz": edge_curves_xyz,
        "graph": {
            "edge_seed_pair": torch.tensor(
                [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
                dtype=torch.long,
            ),
            "edge_type": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
        },
    }

    loss = trainer.cell_edge_uniformity_loss(build_shared_curve_geometry(decoder_out))

    assert loss > edge_curves_xyz.new_tensor(0.0)
    loss.backward()
    assert edge_curves_xyz.grad is not None
    assert torch.isfinite(edge_curves_xyz.grad).all()


def test_init_face_seed_uses_balanced_fps_by_default() -> None:
    class DummyGenerator:
        def __init__(self) -> None:
            self.called = False

        def fps_3d(self, points_xyz, n_samples, exclude_idx=None, seed=None):
            self.called = True
            assert int(n_samples) == 3
            assert int(seed) == 17
            assert torch.equal(exclude_idx, torch.tensor([0]))
            return torch.tensor([1, 3, 4], device=points_xyz.device)

    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(seed_number=3, seed_init_fps_seed=17)
    trainer.generator = DummyGenerator()
    trainer._true_open_boundary_idx = lambda face_tensor: torch.tensor([0])

    face_tensor = {
        "uv": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "points_xyz": torch.arange(15, dtype=torch.float32).reshape(5, 3),
    }

    seeds = trainer._init_face_seed(face_tensor)

    assert trainer.generator.called
    assert torch.equal(seeds, face_tensor["uv"][torch.tensor([1, 3, 4])])


def test_init_face_seed_can_use_seeded_random_points() -> None:
    class FpsShouldNotRun:
        def fps_3d(self, *args, **kwargs):
            raise AssertionError("fps_3d should not run for random seed init")

    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = TrainingConfig(
        seed_number=3,
        seed_init_fps_seed=11,
        use_balanced_seed_init=False,
    )
    trainer.generator = FpsShouldNotRun()
    trainer._true_open_boundary_idx = lambda face_tensor: torch.tensor([0, 4])

    face_tensor = {
        "uv": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "points_xyz": torch.arange(15, dtype=torch.float32).reshape(5, 3),
    }
    candidates = torch.tensor([1, 2, 3])
    expected_order = torch.randperm(candidates.numel(), generator=torch.Generator().manual_seed(11))
    expected_idx = candidates[expected_order[:3]]

    seeds = trainer._init_face_seed(face_tensor)

    assert torch.equal(seeds, face_tensor["uv"][expected_idx])


if __name__ == "__main__":
    test_scipy_topology_seed_gradients_with_area_loss()
    test_scipy_reconstructed_graph_geometry_has_seed_gradients()
