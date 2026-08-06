import torch

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder


def make_decoder(**kwargs):
    face_mesh = {
        "uv": torch.empty((0, 2)),
        "Xu": torch.empty((0, 3)),
        "Xv": torch.empty((0, 3)),
        "points_xyz": torch.empty((0, 3)),
    }
    return ContinuousVoronoiDecoder(Cad_domain=None, face_mesh=face_mesh, **kwargs)


def test_straight_segment_along_x_gives_x_fiber():
    decoder = make_decoder(use_spatial_pruning=False)
    curves = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.01, 0.0], [0.75, -0.02, 0.0]], dtype=torch.float64)

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=0.1,
        tau_density=0.02,
        tau_fiber=0.02,
    )

    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(out["fiber"], expected, atol=1e-6)


def test_density_decreases_with_distance_from_segment():
    decoder = make_decoder(use_spatial_pruning=False)
    curves = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    query = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.15, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=0.1,
        tau_density=0.03,
        tau_fiber=0.03,
    )

    assert out["density"][0] > out["density"][1] > out["density"][2]
    assert out["distance"][0] < out["distance"][1] < out["distance"][2]


def test_segment_outputs_match_sampled_output_shapes():
    query = torch.tensor(
        [[0.2, 0.0, 0.0], [0.5, 0.1, 0.0], [1.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    segment_decoder = make_decoder(use_segment_distance=True, use_spatial_pruning=False)
    sampled_decoder = make_decoder(use_segment_distance=False)

    segment_out = segment_decoder.soft_tube_density_and_fiber_to_elements(query, curves, radius=0.1)
    sampled_out = sampled_decoder.soft_tube_density_and_fiber_to_elements(query, curves, radius=0.1)

    for key in ("density", "fiber", "phi", "theta", "distance"):
        assert segment_out[key].shape == sampled_out[key].shape


def test_blocked_segment_topk_matches_full_distance_topk():
    decoder = make_decoder(use_segment_distance=True, use_spatial_pruning=False, nearest_segment_k=3)
    query = torch.tensor(
        [
            [0.1, 0.05, 0.0],
            [0.6, 0.02, 0.0],
            [1.3, -0.03, 0.0],
            [2.2, 0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    curves = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [2.5, 0.1, 0.0], [3.0, 0.2, 0.0]],
        ],
        dtype=torch.float64,
    )

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=0.1,
        tau_density=0.03,
        tau_fiber=0.04,
    )
    seg_a, seg_b, seg_tangents, _, _ = decoder.curve_segments_and_tangents_xyz(curves)
    distances = decoder.point_to_segments_distance(query, seg_a, seg_b)
    nearest_distances, nearest_ids = torch.topk(distances, k=3, dim=1, largest=False)
    fiber_weights = torch.softmax(-nearest_distances / query.new_tensor(0.04), dim=1)
    expected_fiber = decoder._safe_normalize_fiber(
        (fiber_weights.unsqueeze(-1) * seg_tangents[nearest_ids]).sum(dim=1)
    )

    assert torch.allclose(out["distance"], nearest_distances[:, 0], atol=1e-10)
    assert torch.allclose(out["fiber"], expected_fiber, atol=1e-10)


def test_empty_and_degenerate_curves_have_no_nans():
    decoder = make_decoder()
    query = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    fallback = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    empty_curves = query.new_empty((0, 2, 3))
    empty_out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        empty_curves,
        radius=0.1,
        fallback_fiber=fallback,
    )

    degenerate_curves = torch.zeros((1, 2, 3), dtype=torch.float64)
    degenerate_out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        degenerate_curves,
        radius=0.1,
        fallback_fiber=fallback,
    )

    for out in (empty_out, degenerate_out):
        for key in ("density", "fiber", "phi", "theta", "distance"):
            finite_or_inf = torch.isfinite(out[key]) | torch.isinf(out[key])
            assert finite_or_inf.all()
            assert not torch.isnan(out[key]).any()


def test_two_pass_segment_search_matches_full_matrix_and_has_finite_grads():
    torch.manual_seed(7)
    decoder = make_decoder(
        use_segment_distance=True,
        use_spatial_pruning=False,
        nearest_segment_k=4,
    )
    m = 20
    g = 15
    k = 4
    query = torch.randn((m, 3), dtype=torch.float32, requires_grad=True)
    curves = torch.randn((1, g + 1, 3), dtype=torch.float32, requires_grad=True)
    radius = torch.tensor(0.35, dtype=torch.float32, requires_grad=True)
    tau_density = 0.07
    tau_fiber = 0.09
    rho_min = 0.02

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=radius,
        tau_density=tau_density,
        tau_fiber=tau_fiber,
        rho_min=rho_min,
    )

    seg_a, seg_b, seg_tangents, _, _ = decoder.curve_segments_and_tangents_xyz(curves)
    full_distances = decoder.point_to_segments_distance(query, seg_a, seg_b)
    full_nearest_distances, full_nearest_ids = torch.topk(
        full_distances,
        k=k,
        dim=1,
        largest=False,
    )
    selected_distances = decoder.point_to_selected_segments_distance(
        query_xyz=query,
        seg_a_selected=seg_a[full_nearest_ids],
        seg_b_selected=seg_b[full_nearest_ids],
    )

    expected_density = rho_min + (1.0 - rho_min) * torch.sigmoid(
        (radius.clamp_min(decoder.eps) - full_nearest_distances[:, 0])
        / query.new_tensor(tau_density)
    )
    fiber_weights = torch.softmax(
        -full_nearest_distances / query.new_tensor(tau_fiber),
        dim=1,
    )
    expected_fiber = decoder._safe_normalize_fiber(
        (fiber_weights.unsqueeze(-1) * seg_tangents[full_nearest_ids]).sum(dim=1)
    )

    assert torch.allclose(selected_distances, full_nearest_distances, atol=1e-5)
    assert torch.equal(full_nearest_ids, torch.topk(full_distances, k=k, dim=1, largest=False).indices)
    assert torch.allclose(out["distance"], full_nearest_distances[:, 0], atol=1e-5)
    assert torch.allclose(out["density"], expected_density, atol=1e-5)
    assert torch.allclose(out["fiber"], expected_fiber, atol=1e-5)

    loss = out["density"].sum() + out["fiber"].sum() + selected_distances.sum()
    loss.backward()
    for grad in (query.grad, curves.grad, radius.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_selected_segment_distance_on_segment_has_finite_gradients():
    decoder = make_decoder()
    query = torch.tensor(
        [[0.5, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    seg_a = torch.tensor(
        [[[0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    seg_b = torch.tensor(
        [[[1.0, 0.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = decoder.point_to_selected_segments_distance(query, seg_a, seg_b)
    distance.sum().backward()

    assert not torch.isnan(distance).any()
    for grad in (query.grad, seg_a.grad, seg_b.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()

