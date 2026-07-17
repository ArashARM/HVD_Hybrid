import torch
import matplotlib.pyplot as plt

from Decoder_CLasses.ContinuousVoronoiDecoder import (
    GUARD_SEED_EPS,
    ContinuousVoronoiDecoder,
    fixed_guard_seeds,
    remap_guard_seed_pairs,
)


def make_decoder(**kwargs):
    face_mesh = {
        "uv": torch.empty((0, 2)),
        "Xu": None,
        "Xv": None,
        "points_xyz": None,
    }
    return ContinuousVoronoiDecoder(None, face_mesh, **kwargs)


def test_scipy_topology_returns_unified_boundary_graph_fields():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
        ],
        dtype=torch.float32,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)

    assert "vertex_type" in out
    assert "edge_type" in out["edges"]
    assert "node_clip_source_vertices" in out
    assert "scipy_vertex_aug_seed_triples" in out
    assert "guard_seeds_uv" in out
    assert out["vertices_uv"].shape[0] == out["graph"]["nodes_uv"].shape[0]
    assert out["graph"]["node_type"].shape[0] == out["vertices_uv"].shape[0]
    assert out["edges"]["edge_index"].shape[1] == 2
    assert out["edges"]["edge_type"].shape[0] == out["edges"]["edge_index"].shape[0]


def test_scipy_topology_prunes_unreferenced_vertices_and_reports_final_counts():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.90, 0.90],
            [0.50, 0.45],
            [0.20, 0.55],
            [0.80, 0.60],
        ],
        dtype=torch.float64,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    diagnostics = out["diagnostics"]
    num_vertices = out["vertices_uv"].shape[0]
    edges = out["edges"]["edge_index"]

    assert edges.numel() == 0 or int(edges.max()) < num_vertices
    assert num_vertices == diagnostics["num_final_nodes"]
    assert edges.shape[0] == diagnostics["num_final_edges"]
    assert diagnostics["num_guard_ridges_skipped"] > 0
    assert diagnostics["num_real_real_infinite_ridges_skipped"] == 0
    assert torch.all(edges >= 0)


def test_scipy_topology_can_keep_isolated_vertices_for_debugging():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.90, 0.90],
            [0.50, 0.45],
            [0.20, 0.55],
            [0.80, 0.60],
        ],
        dtype=torch.float64,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    diagnostics = out["diagnostics"]

    assert out["vertices_uv"].shape[0] == diagnostics["num_final_nodes"]


def test_generated_graph_plot_labels_compact_node_and_edge_ids(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.90, 0.90],
            [0.50, 0.45], [0.20, 0.55], [0.80, 0.60],
        ],
        dtype=torch.float64,
    )
    out = decoder.forward_scipy_topology(seeds, return_xyz=False)

    fig, ax = decoder.plot_graph_output(
        seeds,
        out,
        show_node_ids=True,
        show_edge_ids=True,
        print_node_table=False,
    )
    labels = {text.get_text() for text in ax.texts}
    source_types = out["graph"]["boundary_source_type"]
    expected_node_labels = {
        (
            "I" if int(node_type) == 0
            else "C" if int(source_types[node_id]) == 6
            else "B"
        ) + str(node_id)
        for node_id, node_type in enumerate(out["graph"]["node_type"])
    }
    num_edges = out["graph"]["edge_index"].shape[0]

    assert expected_node_labels <= labels
    assert {f"e{edge_id}" for edge_id in range(num_edges)} <= labels
    assert f"nodes={out['graph']['nodes_uv'].shape[0]}" in ax.get_title()
    plt.close(fig)


def test_box_boundary_shell_is_closed_without_pair_bisector_completion():
    decoder = make_decoder()
    seeds = torch.tensor([[0.25, 0.50], [0.75, 0.50]], dtype=torch.float64)

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    graph = out["graph"]
    edges = graph["edge_index"]
    edge_type = graph["edge_type"]
    source_type = graph["boundary_source_type"]
    corner_ids = torch.nonzero(source_type == 6, as_tuple=False).flatten()

    assert corner_ids.numel() == 4
    assert int((edge_type == 4).sum()) == 4
    assert not torch.any(source_type == 3)
    for node_id in corner_ids.tolist():
        incident = (edges == node_id).any(dim=1)
        assert int((incident & (edge_type == 4)).sum()) == 2


def test_box_boundary_intersections_use_domain_boundary_source_type():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
            [0.50, 0.50],
        ],
        dtype=torch.float64,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    source_type = out["graph"]["boundary_source_type"]

    assert torch.any(source_type == 5)
    assert not torch.any(source_type == 2)


def test_packed_cad_boundary_piece_endpoints_are_fixed_shell_nodes():
    decoder = make_decoder(use_trim_activity=False)
    seeds = torch.tensor([[0.25, 0.50], [0.75, 0.50]], dtype=torch.float64)
    cad_domain = {
        "boundary_curve_uv": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        ),
        "boundary_curve_offsets": torch.tensor([0, 2, 3, 4, 5], dtype=torch.long),
    }

    out = decoder.forward_scipy_topology(seeds, cad_domain=cad_domain, return_xyz=False)
    graph = out["graph"]

    assert int((graph["boundary_source_type"] == 6).sum()) == 4
    assert int((graph["edge_type"] == 4).sum()) == 4


def test_dense_smooth_cad_boundary_samples_do_not_become_nodes():
    decoder = make_decoder(use_trim_activity=False)
    seeds = torch.tensor([[0.25, 0.50], [0.75, 0.50]], dtype=torch.float64)
    theta = torch.linspace(0.0, 2.0 * torch.pi, 65, dtype=torch.float64)
    boundary_uv = torch.stack(
        (
            0.5 + 0.35 * torch.cos(theta),
            0.5 + 0.35 * torch.sin(theta),
        ),
        dim=1,
    )
    cad_domain = {
        "boundary_curve_uv": boundary_uv,
        "boundary_curve_offsets": torch.tensor([0, boundary_uv.shape[0]], dtype=torch.long),
    }

    topo = decoder.build_scipy_voronoi_topology(seeds, cad_domain=cad_domain)

    assert int((topo["boundary_source_type"] == 6).sum()) == 1
    assert topo["vertices_uv"].shape[0] < boundary_uv.shape[0]


def test_guard_seed_topology_uses_only_finite_real_real_ridges():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.90, 0.90],
            [0.50, 0.45],
            [0.20, 0.55],
            [0.80, 0.60],
        ],
        dtype=torch.float64,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    diagnostics = out["diagnostics"]
    edge_pairs = out["edges"]["edge_seed_pair"]
    node_sources = out["node_clip_source_vertices"]

    assert diagnostics["num_guard_ridges_skipped"] > 0
    assert diagnostics["num_real_real_infinite_ridges_skipped"] == 0
    assert torch.all(edge_pairs < seeds.shape[0])
    assert torch.all((node_sources == -1) | (node_sources >= 0))


def test_finite_segment_nodes_stay_in_box_and_edges_reference_valid_nodes():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.12, 0.18],
            [0.86, 0.14],
            [0.20, 0.82],
            [0.78, 0.88],
            [0.50, 0.50],
            [0.35, 0.62],
        ],
        dtype=torch.float64,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    vertices = out["vertices_uv"]
    edges = out["edges"]["edge_index"]

    assert torch.isfinite(vertices).all()
    assert torch.all(vertices >= -1e-8)
    assert torch.all(vertices <= 1.0 + 1e-8)
    assert edges.numel() == 0 or int(edges.max()) < vertices.shape[0]


def test_finite_segment_vertices_are_differentiable_and_guards_are_constant():
    decoder = make_decoder()
    seeds = torch.tensor(
        [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
            [0.52, 0.48],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    out = decoder.forward_scipy_topology(seeds, return_xyz=False)
    loss = out["vertices_uv"].sum()
    loss.backward()

    moved = seeds.detach().clone()
    moved[4, 0] = moved[4, 0] + 0.05
    moved_out = decoder.forward_scipy_topology(moved, return_xyz=False)

    assert out["vertices_uv"].requires_grad
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert not out["guard_seeds_uv"].requires_grad
    same_shape = out["vertices_uv"].shape == moved_out["vertices_uv"].shape
    assert (not same_shape) or (not torch.allclose(out["vertices_uv"], moved_out["vertices_uv"]))


def test_fixed_guard_topology_rejects_periodic_domains():
    decoder = make_decoder(face_u_periodic=True)
    seeds = torch.tensor(
        [[0.15, 0.15], [0.85, 0.15], [0.50, 0.85], [0.50, 0.45]],
        dtype=torch.float64,
    )

    try:
        decoder.forward_scipy_topology(seeds, u_periodic=True)
    except NotImplementedError as error:
        assert "non-periodic [0,1]^2" in str(error)
    else:
        raise AssertionError("periodic fixed-guard topology should fail clearly.")


def test_fixed_guard_seeds_coordinates_and_gradient_state():
    guards = fixed_guard_seeds(device=torch.device("cpu"), dtype=torch.float64)
    d = float(torch.tensor(2.0, dtype=torch.float64).sqrt().item() + GUARD_SEED_EPS)
    expected = torch.tensor(
        [
            [0.5, -d],
            [1.0 + d, 0.5],
            [0.5, 1.0 + d],
            [-d, 0.5],
        ],
        dtype=torch.float64,
    )

    assert guards.shape == (4, 2)
    assert guards.requires_grad is False
    assert guards.grad_fn is None
    assert torch.allclose(guards, expected)


def test_fixed_guard_concat_preserves_real_seed_gradients_only():
    real = torch.tensor(
        [[0.3, 0.4], [0.7, 0.6]],
        dtype=torch.double,
        requires_grad=True,
    )
    guards = fixed_guard_seeds(device=real.device, dtype=real.dtype)
    combined = torch.cat([real, guards], dim=0)

    loss = combined[:2].square().sum()
    loss.backward()

    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert guards.grad is None


def test_fixed_guards_are_not_registered_parameters_or_state():
    decoder = make_decoder()
    guards = fixed_guard_seeds(device=torch.device("cpu"), dtype=torch.float64)

    assert list(decoder.parameters()) == []
    for value in decoder.state_dict().values():
        assert not (value.shape == guards.shape and torch.allclose(value.cpu(), guards))


def test_guard_seed_pair_remapping_filters_guard_guard_edges():
    pairs = torch.tensor(
        [
            [0, 1],
            [0, 4],
            [5, 1],
            [4, 5],
        ],
        dtype=torch.long,
    )

    keep, remapped = remap_guard_seed_pairs(pairs, real_seed_count=4)

    assert torch.equal(keep, torch.tensor([True, True, True, False]))
    assert torch.equal(
        remapped[keep],
        torch.tensor([[0, 1], [0, -1], [-1, 1]], dtype=torch.long),
    )


def test_fixed_guards_do_not_change_nearest_real_seed_inside_unit_square():
    real = torch.tensor(
        [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.20, 0.85],
            [0.80, 0.80],
            [0.50, 0.50],
        ],
        dtype=torch.double,
    )
    guards = fixed_guard_seeds(device=real.device, dtype=real.dtype)
    grid_1d = torch.linspace(0.0, 1.0, 31, dtype=real.dtype)
    uu, vv = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    query = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=1)

    nearest_real = torch.cdist(query, real).argmin(dim=1)
    nearest_augmented = torch.cdist(query, torch.cat([real, guards], dim=0)).argmin(dim=1)

    assert torch.equal(nearest_augmented, nearest_real)
