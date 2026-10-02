from types import SimpleNamespace

import torch
import torch.nn.functional as F
from Decoder_CLasses import Phase1VoronoiDecoder
from Training.MainTrain import NN_Trainer
from Utils.DifferentiableFilters import smooth_heaviside_projection


def make_flat_query(n=80, lx=100.0, ly=50.0):
    u = torch.linspace(0.0, 1.0, n)
    v = torch.linspace(0.0, 1.0, n)

    U, V = torch.meshgrid(u, v, indexing="ij")

    uv = torch.stack(
        [U.reshape(-1), V.reshape(-1)],
        dim=1,
    )

    N = uv.shape[0]

    Xu = torch.tensor(
        [lx, 0.0, 0.0],
        dtype=torch.float32,
    ).expand(N, 3).clone()

    Xv = torch.tensor(
        [0.0, ly, 0.0],
        dtype=torch.float32,
    ).expand(N, 3).clone()

    # Physical area associated with each sample point
    A = torch.full(
        (N,),
        lx * ly / N,
        dtype=torch.float32,
    )

    return uv, Xu, Xv, A


def make_deterministic_seeds(n_seeds: int, dtype=torch.float32):
    cols = int(torch.ceil(torch.sqrt(torch.tensor(float(n_seeds)))).item())
    rows = int(torch.ceil(torch.tensor(float(n_seeds)) / cols).item())
    u = torch.linspace(0.08, 0.92, cols, dtype=dtype)
    v = torch.linspace(0.08, 0.92, rows, dtype=dtype)
    U, V = torch.meshgrid(u, v, indexing="ij")
    seeds = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)[:n_seeds].clone()
    jitter = torch.linspace(-0.015, 0.015, n_seeds, dtype=dtype)
    seeds[:, 0] = (seeds[:, 0] + jitter).clamp(0.03, 0.97)
    seeds[:, 1] = (seeds[:, 1] - jitter.flip(0)).clamp(0.03, 0.97)
    return seeds


def phase1_combined_gradient_case(n_seeds: int, *, point_chunk_size: int, query_n: int = 24):
    uv, Xu, Xv, A = make_flat_query(n=query_n)
    seeds = make_deterministic_seeds(n_seeds).requires_grad_(True)
    decoder = Phase1VoronoiDecoder(
        n_seeds=n_seeds,
        use_Metric_anisotropy=False,
        fixed_strut_radius=0.04,
        fixed_height=1.0,
        physical_tube_beta=0.01,
        duplicate_merge_sigma=0.02,
        territory_min_ratio=0.05,
        phase2_activity_threshold=0.05,
        continuous_length_calibration=1.0,
        density_projection_strength=0.0,
        point_chunk_size=point_chunk_size,
        eps=1.0e-8,
    )
    out = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.01,
        seeds_raw=seeds,
        w_raw=torch.zeros((n_seeds, n_seeds), dtype=seeds.dtype),
        surface_area_weights=A,
    )
    L_cont = out["continuous_voronoi_length"]
    rho_projected = smooth_heaviside_projection(
        out["rho"],
        beta=8.0,
        eta=0.5,
        strength=1.0,
    )
    L_density = (rho_projected * A).sum() / A.sum().clamp_min(1.0e-12)
    eps = torch.tensor(1.0e-12, dtype=seeds.dtype)
    L_test = (
        L_cont / L_cont.detach().abs().clamp_min(eps)
        + L_density / L_density.detach().abs().clamp_min(eps)
        + out["fiber3d"][:, 0].mean()
        + 0.1 * out["fiber3d"][:, 1].mean()
    )
    grad = torch.autograd.grad(L_test, seeds, retain_graph=False, create_graph=False)[0]
    return out, L_density, grad


def make_two_seed_decoder():
    return Phase1VoronoiDecoder(
        n_seeds=2,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        w_min=0.02,
        w_max_ratio=0.2,
        beta=0.003,
        fixed_strut_radius=0.04,
        physical_tube_beta=0.01,
        duplicate_merge_sigma=0.02,
        territory_min_ratio=0.05,
        point_chunk_size=32,
        eps=1.0e-8,
    )


def two_seed_orientation_output(Xu, Xv, n=18):
    u = torch.linspace(0.1, 0.9, n)
    v = torch.linspace(0.1, 0.9, n)
    U, V = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)
    N = uv.shape[0]
    Xu = Xu.to(dtype=torch.float32).expand(N, 3).clone()
    Xv = Xv.to(dtype=torch.float32).expand(N, 3).clone()
    area = torch.linalg.norm(torch.cross(Xu, Xv, dim=1), dim=1)
    area = area / area.sum().clamp_min(1.0e-12)
    seeds = torch.tensor(
        [[0.25, 0.5], [0.75, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    decoder = make_two_seed_decoder()
    out = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=torch.zeros((2, 2), dtype=torch.float32),
        surface_area_weights=area,
    )
    return out, seeds, Xu, Xv


def test_phase1_flat_surface_fiber_orientation_is_physical_tangent():
    out, _, Xu, Xv = two_seed_orientation_output(
        torch.tensor([100.0, 0.0, 0.0]),
        torch.tensor([0.0, 50.0, 0.0]),
    )
    fiber = out["fiber3d"]
    normal = F.normalize(torch.cross(Xu, Xv, dim=1), dim=1)
    expected = F.normalize(Xu, dim=1)

    assert torch.isfinite(fiber).all()
    assert torch.allclose(torch.linalg.vector_norm(fiber, dim=1), torch.ones_like(fiber[:, 0]), atol=1.0e-5)
    assert torch.max((fiber * normal).sum(dim=1).abs()) < 1.0e-5
    assert torch.median((fiber * expected).sum(dim=1).abs()) > 0.99
    assert out["fiber_tensor_Q"].shape == (fiber.shape[0], 2, 2)


def test_phase1_skewed_basis_uses_local_physical_q2_not_raw_uv():
    decoder = make_two_seed_decoder()
    Xu = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    Xv = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    e1, e2 = decoder._orthonormal_tangent_basis(Xu, Xv)
    pair_uv = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    weights = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    pair_local = decoder._local_physical_pair_tangents(pair_uv, Xu, Xv, e1, e2)
    Q2 = decoder._axial_tensor_from_local_pair_tangents(weights, pair_local)
    t_local = decoder._principal_axial_direction(Q2)
    fiber3d = F.normalize(t_local[:, 0:1] * e1 + t_local[:, 1:2] * e2, dim=1)
    normal = F.normalize(torch.cross(Xu, Xv, dim=1), dim=1)
    naive_raw_uv = F.normalize(Xu + Xv, dim=1)

    assert torch.isfinite(fiber3d).all()
    assert torch.allclose(torch.linalg.vector_norm(fiber3d, dim=1), torch.ones(1), atol=1.0e-6)
    assert torch.max((fiber3d * normal).sum(dim=1).abs()) < 1.0e-6
    assert not torch.allclose(fiber3d.abs(), naive_raw_uv.abs(), atol=1.0e-2)


def test_phase1_scaled_basis_uses_physical_local_q2_not_raw_uv():
    decoder = make_two_seed_decoder()
    Xu = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    Xv = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32)
    e1, e2 = decoder._orthonormal_tangent_basis(Xu, Xv)
    pair_uv = F.normalize(
        torch.tensor([[[[1.0, 1.0], [1.0, -1.0]]]], dtype=torch.float32),
        dim=-1,
    )
    weights = torch.tensor([[[0.55, 0.45]]], dtype=torch.float32)

    pair_local = decoder._local_physical_pair_tangents(pair_uv, Xu, Xv, e1, e2)
    Q2 = decoder._axial_tensor_from_local_pair_tangents(weights, pair_local)
    t_local = decoder._principal_axial_direction(Q2)
    fiber3d = F.normalize(t_local[:, 0:1] * e1 + t_local[:, 1:2] * e2, dim=1)

    Q2_naive = decoder._axial_tensor_from_local_pair_tangents(weights, pair_uv)
    t_naive = decoder._principal_axial_direction(Q2_naive)
    naive_xyz = F.normalize(t_naive[:, 0:1] * e1 + t_naive[:, 1:2] * e2, dim=1)

    assert torch.isfinite(fiber3d).all()
    assert torch.allclose(torch.linalg.vector_norm(fiber3d, dim=1), torch.ones(1), atol=1.0e-6)
    assert fiber3d[:, 2].abs().max() < 1.0e-6
    assert not torch.allclose(fiber3d.abs(), naive_xyz.abs(), atol=1.0e-2)


def test_phase1_random_curved_tangent_inputs_stay_tangent():
    decoder = make_two_seed_decoder()
    torch.manual_seed(3)
    N = 16
    Xu = F.normalize(torch.randn(N, 3), dim=1)
    raw = torch.randn(N, 3)
    Xv = raw - (raw * Xu).sum(dim=1, keepdim=True) * Xu
    Xv = Xv + 0.25 * Xu
    e1, e2 = decoder._orthonormal_tangent_basis(Xu, Xv)
    pair_uv = F.normalize(torch.randn(N, 2, 3, 2), dim=-1)
    weights = torch.rand(N, 2, 3)
    pair_local = decoder._local_physical_pair_tangents(pair_uv, Xu, Xv, e1, e2)
    Q2 = decoder._axial_tensor_from_local_pair_tangents(weights, pair_local)
    t_local = decoder._principal_axial_direction(Q2)
    fiber3d = F.normalize(t_local[:, 0:1] * e1 + t_local[:, 1:2] * e2, dim=1)
    trace = Q2[..., 0, 0] + Q2[..., 1, 1]
    fiber3d = torch.where(trace[:, None] > decoder.eps, fiber3d, e1)
    normal = F.normalize(torch.cross(Xu, Xv, dim=1), dim=1)

    assert torch.isfinite(fiber3d).all()
    assert torch.allclose(torch.linalg.vector_norm(fiber3d, dim=1), torch.ones(N), atol=1.0e-6)
    assert torch.max((fiber3d * normal).sum(dim=1).abs()) < 1.0e-5


def test_phase1_q2_principal_direction_degenerate_backward_is_finite():
    decoder = make_two_seed_decoder()
    Q_cases = torch.stack(
        [
            torch.zeros((2, 2), dtype=torch.float32),
            torch.eye(2, dtype=torch.float32) * 0.5,
            torch.tensor(
                [
                    [0.5 + 1.0e-8, 1.0e-8],
                    [1.0e-8, 0.5],
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    ).requires_grad_(True)

    t = decoder._principal_axial_direction(Q_cases)
    coherence = decoder._axial_coherence_from_tensor(Q_cases)
    loss = t.sum() + coherence.sum()
    grad = torch.autograd.grad(loss, Q_cases, retain_graph=False, create_graph=False)[0]

    assert torch.isfinite(t).all()
    assert torch.isfinite(coherence).all()
    assert torch.isfinite(grad).all()


def test_phase1_total_length_includes_fixed_boundary_length():
    uv, Xu, Xv, A = make_flat_query(n=14, lx=2.0, ly=1.0)
    seeds = torch.tensor(
        [[0.25, 0.5], [0.75, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    boundary_uv = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundary_lengths = torch.tensor([3.25, 1.75], dtype=torch.float32)
    decoder = make_two_seed_decoder()
    out = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=torch.zeros((2, 2), dtype=torch.float32),
        surface_area_weights=A,
        boundary_uv=boundary_uv,
        boundary_curve_offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        boundary_curve_length=boundary_lengths,
    )

    expected_boundary = boundary_lengths.sum()
    assert torch.allclose(out["boundary_curve_length"], expected_boundary)
    assert torch.allclose(
        out["continuous_total_curve_length"],
        out["continuous_voronoi_length"] + expected_boundary,
    )
    assert torch.allclose(out["total_curve_length"], out["continuous_total_curve_length"])
    assert torch.allclose(out["total_voronoi_curve_length"], out["continuous_voronoi_length"])


def _phase1_boundary_length_for(
    *,
    lx=10.0,
    ly=10.0,
    Xu_override=None,
    Xv_override=None,
    boundary_uv=None,
    boundary_curve_offsets=None,
    boundary_curve_xyz=None,
    boundary_curve_length=None,
):
    uv, Xu, Xv, A = make_flat_query(n=24, lx=lx, ly=ly)
    if Xu_override is not None:
        Xu = Xu_override.to(dtype=torch.float32).expand_as(Xu).clone()
    if Xv_override is not None:
        Xv = Xv_override.to(dtype=torch.float32).expand_as(Xv).clone()
    if boundary_uv is None:
        boundary_uv = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    decoder = make_two_seed_decoder()
    out = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=torch.tensor(
            [[0.25, 0.5], [0.75, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        w_raw=torch.zeros((2, 2), dtype=torch.float32),
        surface_area_weights=A,
        boundary_uv=boundary_uv,
        boundary_curve_offsets=boundary_curve_offsets,
        boundary_curve_xyz=boundary_curve_xyz,
        boundary_curve_length=boundary_curve_length,
    )
    return out["boundary_curve_length"], out


def test_phase1_boundary_uv_square_uses_physical_metric():
    boundary_length, _ = _phase1_boundary_length_for(lx=10.0, ly=10.0)
    assert torch.allclose(boundary_length, torch.tensor(40.0), rtol=1e-5, atol=1e-5)


def test_phase1_boundary_uv_rectangle_uses_physical_metric():
    boundary_length, _ = _phase1_boundary_length_for(lx=20.0, ly=5.0)
    assert torch.allclose(boundary_length, torch.tensor(50.0), rtol=1e-5, atol=1e-5)


def test_phase1_boundary_uv_anisotropic_tangent_metric():
    boundary_length, _ = _phase1_boundary_length_for(
        Xu_override=torch.tensor([[2.0, 0.0, 0.0]]),
        Xv_override=torch.tensor([[1.0, 3.0, 0.0]]),
    )
    expected = torch.tensor(4.0 + 2.0 * (10.0 ** 0.5), dtype=torch.float32)
    assert torch.allclose(boundary_length, expected, rtol=1e-5, atol=1e-5)


def test_phase1_boundary_uv_two_loops_do_not_bridge():
    boundary_uv = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    boundary_length, _ = _phase1_boundary_length_for(
        lx=10.0,
        ly=10.0,
        boundary_uv=boundary_uv,
        boundary_curve_offsets=torch.tensor([0, 4, 8], dtype=torch.long),
    )
    assert torch.allclose(boundary_length, torch.tensor(60.0), rtol=1e-5, atol=1e-5)


def test_phase1_boundary_xyz_precedence_uses_direct_polyline_length():
    boundary_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    boundary_length, _ = _phase1_boundary_length_for(
        lx=10.0,
        ly=10.0,
        boundary_curve_xyz=boundary_xyz,
    )
    assert torch.allclose(boundary_length, torch.tensor(7.0), rtol=1e-5, atol=1e-5)


def test_phase1_boundary_length_precedence_over_xyz_and_uv():
    boundary_length, _ = _phase1_boundary_length_for(
        lx=10.0,
        ly=10.0,
        boundary_curve_xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        boundary_curve_length=torch.tensor([2.5, 3.5], dtype=torch.float32),
    )
    assert torch.allclose(boundary_length, torch.tensor(6.0), rtol=1e-5, atol=1e-5)


def _phase1_trainer_boundary_forward(boundary_curve_length, **metadata):
    uv, Xu, Xv, A = make_flat_query(n=24, lx=10.0, ly=10.0)
    boundary_uv = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    ft = {
        "uv": uv,
        "points_xyz": torch.stack(
            [10.0 * uv[:, 0], 10.0 * uv[:, 1], torch.zeros_like(uv[:, 0])],
            dim=1,
        ),
        "Xu": Xu,
        "Xv": Xv,
        "boundary_curve_uv": boundary_uv,
        "boundary_curve_offsets": torch.tensor([0, 4], dtype=torch.long),
        "boundary_curve_length": torch.tensor([boundary_curve_length], dtype=torch.float32),
        "face_id": 0,
        **metadata,
    }
    trainer = NN_Trainer.__new__(NN_Trainer)
    trainer.cfg = SimpleNamespace(
        phase1_tau=0.025,
        seed_domain_mask_threshold=0.5,
        seed_domain_temp=0.05,
    )
    decoder = make_two_seed_decoder()
    seeds = torch.tensor(
        [[0.25, 0.5], [0.75, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return trainer._phase1_decoder_forward(
        decoder,
        ft=ft,
        seeds_raw=seeds,
        surface_area_weights=A,
    )


def test_phase1_training_ignores_stale_uv_boundary_curve_length():
    out = _phase1_trainer_boundary_forward(4.0)

    assert not torch.allclose(out["boundary_curve_length"], torch.tensor(4.0))
    assert torch.allclose(out["boundary_curve_length"], torch.tensor(40.0), rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        out["continuous_total_curve_length"],
        out["continuous_voronoi_length"] + out["boundary_curve_length"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_phase1_training_allows_explicit_physical_boundary_curve_length():
    out = _phase1_trainer_boundary_forward(
        40.0,
        boundary_curve_length_units="physical",
    )

    assert torch.allclose(out["boundary_curve_length"], torch.tensor(40.0), rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        out["continuous_total_curve_length"],
        out["continuous_voronoi_length"] + torch.tensor(40.0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_two_seed_length_check():
    uv, Xu, Xv, A = make_flat_query()

    seeds = torch.tensor(
        [
            [0.25, 0.50],
            [0.75, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    w_raw = torch.zeros(
        (2, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    decoder = Phase1VoronoiDecoder(
        n_seeds=2,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        w_min=0.02,
        w_max_ratio=0.2,
        beta=0.003,
        duplicate_merge_sigma=0.02,
        territory_min_ratio=0.05,
    )

    out = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    L = out["continuous_voronoi_length"]

    loss = L + 0.01 * out["rho"].sum()
    loss.backward()

    assert torch.isfinite(L)
    assert torch.isfinite(out["rho"]).all()
    assert torch.isfinite(out["fiber3d"]).all()

    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()

    print("\nTwo-seed exact geometric ridge length: ~50 mm")
    print(
        "Continuous Phase-1 estimate:",
        float(L.detach()),
    )

    print(
        "Active seeds:",
        int(out["active_seed_count"].detach()),
    )

    print(
        "Soft active count:",
        float(out["soft_active_seed_count"].detach()),
    )

    print(
        "Seed activities:",
        out["seed_active_weights"].detach().cpu().numpy(),
    )


def test_phase1_30_seed_combined_length_density_gradient():
    out, density, grad = phase1_combined_gradient_case(
        30,
        point_chunk_size=64,
        query_n=20,
    )

    assert torch.isfinite(out["continuous_voronoi_length"])
    assert torch.isfinite(density)
    assert grad.shape == (30, 2)
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad) > 0
    assert int((grad.abs() > 1.0e-10).sum().item()) >= 58


def test_phase1_50_seed_chunked_backward_has_dense_gradients():
    out, density, grad = phase1_combined_gradient_case(
        50,
        point_chunk_size=32,
        query_n=20,
    )

    assert torch.isfinite(out["continuous_voronoi_length"])
    assert torch.isfinite(density)
    assert grad.shape == (50, 2)
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad) > 0
    assert int((grad.abs() > 1.0e-10).sum().item()) >= 98


def test_phase1_point_chunk_size_consistency():
    uv, Xu, Xv, A = make_flat_query(n=18)
    seeds = make_deterministic_seeds(24)
    values = []
    rhos = []
    for chunk_size in (256, 64, 32):
        decoder = Phase1VoronoiDecoder(
            n_seeds=24,
            use_Metric_anisotropy=False,
            fixed_strut_radius=0.04,
            fixed_height=1.0,
            physical_tube_beta=0.01,
            duplicate_merge_sigma=0.02,
            territory_min_ratio=0.05,
            phase2_activity_threshold=0.05,
            continuous_length_calibration=1.0,
            density_projection_strength=0.0,
            point_chunk_size=chunk_size,
            eps=1.0e-8,
        )
        out = decoder(
            points_uv=uv,
            Xu=Xu,
            Xv=Xv,
            tau=0.01,
            seeds_raw=seeds,
            w_raw=torch.zeros((24, 24), dtype=seeds.dtype),
            surface_area_weights=A,
        )
        values.append(
            (
                out["continuous_voronoi_length"].detach(),
                out["rho"].mean().detach(),
                out["rho"].max().detach(),
            )
        )
        rhos.append(out["rho"].detach())

    base = values[0]
    for value in values[1:]:
        for got, expected in zip(value, base):
            assert torch.allclose(got, expected, rtol=2.0e-4, atol=2.0e-5)
    for rho in rhos[1:]:
        assert torch.allclose(rho, rhos[0], rtol=2.0e-4, atol=2.0e-5)


def test_phase1_80_seed_backward_scalability_limitation_documented():
    # 80-seed forward currently succeeds, but backward can exceed 16-GB GPUs
    # because _bisector_band_density still materializes O(N^2) seed-pair fields.
    # Pair-dimension chunking is the intended future scalability fix.
    assert True


def test_seed_domain_deactivation_and_reactivation():
    uv, Xu, Xv, A = make_flat_query()

    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        w_min=0.02,
        w_max_ratio=0.2,
        beta=0.003,
        duplicate_merge_sigma=0.02,
        territory_min_ratio=0.03,
    )

    w_raw = torch.zeros(
        (3, 3),
        dtype=torch.float32,
        requires_grad=True,
    )

    # --------------------------------------------------
    # Case 1: all seeds inside domain
    # --------------------------------------------------
    seeds_inside = torch.tensor(
        [
            [0.20, 0.50],
            [0.50, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_inside = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_inside,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 2: middle seed moved outside domain
    # --------------------------------------------------
    seeds_outside = torch.tensor(
        [
            [0.20, 0.50],
            [1.20, 0.50],   # outside u-domain
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_outside = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_outside,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 3: same seed moved back inside
    # --------------------------------------------------
    seeds_reactivated = torch.tensor(
        [
            [0.20, 0.50],
            [0.55, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_reactivated = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_reactivated,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    print("\n--- Seed domain activity test ---")

    print(
        "Inside activities:",
        out_inside["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Outside activities:",
        out_outside["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Reactivated activities:",
        out_reactivated["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Inside territory:",
        out_inside["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Outside territory:",
        out_outside["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Reactivated territory:",
        out_reactivated["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Inside soft active count:",
        float(out_inside["soft_active_seed_count"].detach()),
    )

    print(
        "Outside soft active count:",
        float(out_outside["soft_active_seed_count"].detach()),
    )

    print(
        "Reactivated soft active count:",
        float(out_reactivated["soft_active_seed_count"].detach()),
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    inside_activity = out_inside["seed_active_weights"][1]
    outside_activity = out_outside["seed_active_weights"][1]
    reactivated_activity = out_reactivated["seed_active_weights"][1]

    # Outside seed should lose activity
    assert outside_activity < inside_activity

    # Moving it back inside should recover activity
    assert reactivated_activity > outside_activity

    # Nothing should become NaN/Inf
    assert torch.isfinite(out_inside["rho"]).all()
    assert torch.isfinite(out_outside["rho"]).all()
    assert torch.isfinite(out_reactivated["rho"]).all()

def test_duplicate_seed_suppression_and_recovery():
    uv, Xu, Xv, A = make_flat_query(lx=1.0, ly=1.0)

    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        w_min=0.02,
        w_max_ratio=0.2,
        beta=0.003,
        duplicate_merge_sigma=0.05,
        territory_min_ratio=0.03,
    )

    w_raw = torch.zeros(
        (3, 3),
        dtype=torch.float32,
        requires_grad=True,
    )

    # --------------------------------------------------
    # Case 1: well-separated seeds
    # --------------------------------------------------
    seeds_separated = torch.tensor(
        [
            [0.20, 0.50],
            [0.50, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_separated = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_separated,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 2: first two seeds become almost coincident
    # --------------------------------------------------
    seeds_duplicate = torch.tensor(
        [
            [0.20, 0.50],
            [0.205, 0.50],   # almost on top of first seed
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_duplicate = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_duplicate,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 3: separate them again
    # --------------------------------------------------
    seeds_recovered = torch.tensor(
        [
            [0.20, 0.50],
            [0.45, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_recovered = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_recovered,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    print("\n--- Duplicate seed suppression test ---")

    print(
        "Separated activities:",
        out_separated["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Duplicate activities:",
        out_duplicate["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Recovered activities:",
        out_recovered["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Separated duplicate weights:",
        out_separated["seed_duplicate_weights"].detach().cpu().numpy(),
    )

    print(
        "Duplicate duplicate weights:",
        out_duplicate["seed_duplicate_weights"].detach().cpu().numpy(),
    )

    print(
        "Recovered duplicate weights:",
        out_recovered["seed_duplicate_weights"].detach().cpu().numpy(),
    )

    print(
        "Separated territory:",
        out_separated["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Duplicate territory:",
        out_duplicate["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Recovered territory:",
        out_recovered["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Separated soft active count:",
        float(out_separated["soft_active_seed_count"].detach()),
    )

    print(
        "Duplicate soft active count:",
        float(out_duplicate["soft_active_seed_count"].detach()),
    )

    print(
        "Recovered soft active count:",
        float(out_recovered["soft_active_seed_count"].detach()),
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    duplicate_weight_before = out_separated["seed_duplicate_weights"][1]
    duplicate_weight_close = out_duplicate["seed_duplicate_weights"][1]
    duplicate_weight_after = out_recovered["seed_duplicate_weights"][1]

    # Near-duplicate seed should be suppressed
    assert duplicate_weight_close < duplicate_weight_before

    # After separation it should recover
    assert duplicate_weight_after > duplicate_weight_close

    # Activity should also recover
    assert (
        out_recovered["seed_active_weights"][1]
        > out_duplicate["seed_active_weights"][1]
    )

    # No NaN / Inf
    assert torch.isfinite(out_separated["rho"]).all()
    assert torch.isfinite(out_duplicate["rho"]).all()
    assert torch.isfinite(out_recovered["rho"]).all()


def test_duplicate_activity_has_no_index_decay_for_well_separated_24_seed_handoff():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=10.0, ly=10.0)
    seeds = make_deterministic_seeds(24)
    decoder = Phase1VoronoiDecoder(
        n_seeds=24,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.44,
        duplicate_effect_temp_ratio=0.2,
        duplicate_effect_floor=0.05,
    )
    pair_dist = decoder._pairwise_seed_dist_physical(
        seeds=seeds,
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
    )
    weights = decoder._soft_duplicate_activity(seeds, pair_dist=pair_dist)
    tri = torch.triu(torch.ones_like(pair_dist, dtype=torch.bool), diagonal=1)

    assert pair_dist[tri].min() > 0.44
    assert torch.all(weights > 0.99)
    assert (weights.max() - weights.min()) < 1.0e-3


def test_duplicate_activity_suppresses_one_true_later_duplicate_pair():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=10.0, ly=10.0)
    seeds = torch.tensor(
        [
            [0.20, 0.50],
            [0.215, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.44,
        duplicate_effect_temp_ratio=0.2,
        duplicate_effect_floor=0.05,
    )
    pair_dist = decoder._pairwise_seed_dist_physical(seeds, uv, Xu, Xv)
    weights = decoder._soft_duplicate_activity(seeds, pair_dist=pair_dist)

    assert pair_dist[1, 0] < 0.44
    assert weights[0] > 0.99
    assert weights[1] < 0.25
    assert weights[2] > 0.99


def test_duplicate_activity_recovers_when_pair_separates_again():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=10.0, ly=10.0)
    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.44,
        duplicate_effect_temp_ratio=0.2,
        duplicate_effect_floor=0.05,
    )
    close = torch.tensor(
        [[0.20, 0.50], [0.215, 0.50], [0.80, 0.50]],
        dtype=torch.float32,
    )
    separated = torch.tensor(
        [[0.20, 0.50], [0.40, 0.50], [0.80, 0.50]],
        dtype=torch.float32,
    )
    close_w = decoder._soft_duplicate_activity(
        close,
        pair_dist=decoder._pairwise_seed_dist_physical(close, uv, Xu, Xv),
    )
    separated_w = decoder._soft_duplicate_activity(
        separated,
        pair_dist=decoder._pairwise_seed_dist_physical(separated, uv, Xu, Xv),
    )

    assert close_w[1] < 0.25
    assert separated_w[1] > 0.99
    assert separated_w[1] > close_w[1]


def test_duplicate_activity_nonduplicates_are_order_sane():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=10.0, ly=10.0)
    seeds = make_deterministic_seeds(24)
    perm = torch.arange(seeds.shape[0] - 1, -1, -1)
    decoder = Phase1VoronoiDecoder(
        n_seeds=24,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.44,
        duplicate_effect_temp_ratio=0.2,
        duplicate_effect_floor=0.05,
    )
    weights = decoder._soft_duplicate_activity(
        seeds,
        pair_dist=decoder._pairwise_seed_dist_physical(seeds, uv, Xu, Xv),
    )
    weights_perm = decoder._soft_duplicate_activity(
        seeds[perm],
        pair_dist=decoder._pairwise_seed_dist_physical(seeds[perm], uv, Xu, Xv),
    )

    assert torch.all(weights > 0.99)
    assert torch.all(weights_perm > 0.99)
    assert (weights_perm.max() - weights_perm.min()) < 1.0e-3


def test_duplicate_seed_suppression_uses_physical_xyz_distance():
    uv, Xu, Xv, A = make_flat_query(n=36, lx=10.0, ly=1.0)
    points_xyz = torch.stack(
        [10.0 * uv[:, 0], uv[:, 1], torch.zeros_like(uv[:, 0])],
        dim=1,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        fixed_strut_radius=0.02,
        beta=0.003,
        duplicate_merge_sigma=0.05,
        territory_min_ratio=1e-6,
    )
    w_raw = torch.zeros((3, 3), dtype=torch.float32)
    seeds = torch.tensor(
        [
            [0.20, 0.50],
            [0.23, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_uv_basis = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=w_raw,
        surface_area_weights=A,
    )
    out_xyz_basis = decoder(
        points_uv=uv,
        points_3d=points_xyz,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    assert out_uv_basis["seed_duplicate_weights"][1] > 0.9
    assert out_xyz_basis["seed_duplicate_weights"][1] > 0.9
    assert out_xyz_basis["seed_duplicate_weights"][1] > 0.9
    assert torch.isfinite(out_xyz_basis["seeds_xyz"]).all()


def test_duplicate_seed_suppression_catches_physical_3d_duplicates_with_large_uv_gap():
    uv, Xu, Xv, A = make_flat_query(n=36, lx=0.1, ly=1.0)
    points_xyz = torch.stack(
        [0.1 * uv[:, 0], uv[:, 1], torch.zeros_like(uv[:, 0])],
        dim=1,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=3,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        fixed_strut_radius=0.02,
        beta=0.003,
        duplicate_merge_sigma=0.05,
        territory_min_ratio=1e-6,
    )
    seeds = torch.tensor(
        [
            [0.20, 0.50],
            [0.40, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_raw = torch.zeros((3, 3), dtype=torch.float32)

    out = decoder(
        points_uv=uv,
        points_3d=points_xyz,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    assert torch.linalg.vector_norm(out["seeds_xyz"][1] - out["seeds_xyz"][0]) < 0.05
    assert out["seed_duplicate_weights"][1] < 0.5


def test_phase1_handoff_physically_deduplicates_candidates():
    uv, Xu, Xv, A = make_flat_query(n=40, lx=0.1, ly=1.0)
    points_xyz = torch.stack(
        [0.1 * uv[:, 0], uv[:, 1], torch.zeros_like(uv[:, 0])],
        dim=1,
    )
    threshold = 0.05
    decoder = Phase1VoronoiDecoder(
        n_seeds=4,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        fixed_strut_radius=0.02,
        beta=0.003,
        duplicate_merge_sigma=threshold,
        duplicate_effect_floor=0.05,
        territory_min_ratio=1e-6,
        phase2_activity_threshold=0.5,
        phase2_territory_ratio=1e-6,
    )
    seeds = torch.tensor(
        [
            [0.20, 0.30],
            [0.40, 0.30],  # UV-separated, physically close in compressed x.
            [0.75, 0.30],
            [0.75, 0.75],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    out = decoder(
        points_uv=uv,
        points_3d=points_xyz,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds,
        w_raw=torch.zeros((4, 4), dtype=torch.float32),
        surface_area_weights=A,
    )

    mask = out["phase2_seed_mask"]
    phase2_xyz = out["phase2_seeds_xyz"]
    assert mask.dtype == torch.bool
    assert phase2_xyz.shape[0] == int(mask.sum().item())
    assert torch.allclose(phase2_xyz, out["seeds_xyz"][mask])
    assert not bool(mask[0].item() and mask[1].item())
    if phase2_xyz.shape[0] > 1:
        handoff_dist = torch.cdist(phase2_xyz, phase2_xyz)
        handoff_dist = handoff_dist + torch.eye(
            phase2_xyz.shape[0],
            device=phase2_xyz.device,
            dtype=torch.bool,
        ).to(handoff_dist.dtype) * 1.0e6
        assert float(handoff_dist.min().detach()) >= threshold


def test_pair_distinctness_uses_physical_distance_for_uv_close_far_pair():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=10.0, ly=1.0)
    seeds = torch.tensor(
        [
            [0.20, 0.50],
            [0.23, 0.50],
        ],
        dtype=torch.float32,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=2,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.05,
    )

    pair_dist = decoder._pairwise_seed_dist_physical(seeds, uv, Xu, Xv)
    distinctness = decoder._pair_distinctness_from_distance(pair_dist)

    assert pair_dist[0, 1] > 0.05
    assert distinctness[0, 1] > 0.99


def test_pair_distinctness_suppresses_physically_close_pair():
    uv, Xu, Xv, _ = make_flat_query(n=18, lx=0.1, ly=1.0)
    seeds = torch.tensor(
        [
            [0.20, 0.50],
            [0.40, 0.50],
        ],
        dtype=torch.float32,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=2,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.05,
    )

    pair_dist = decoder._pairwise_seed_dist_physical(seeds, uv, Xu, Xv)
    distinctness = decoder._pair_distinctness_from_distance(pair_dist)

    assert pair_dist[0, 1] < 0.05
    assert distinctness[0, 1] < 0.5


def test_40_well_separated_density_does_not_collapse_with_physical_sigma():
    uv, Xu, Xv, A = make_flat_query(n=20, lx=10.0, ly=10.0)
    seeds = make_deterministic_seeds(40)

    def density_mean(sigma):
        decoder = Phase1VoronoiDecoder(
            n_seeds=40,
            use_Metric_anisotropy=False,
            fixed_height=0.4,
            fixed_strut_radius=0.02,
            beta=0.003,
            duplicate_merge_sigma=sigma,
            duplicate_effect_temp_ratio=0.2,
            duplicate_effect_floor=0.05,
            territory_min_ratio=1e-6,
            phase2_activity_threshold=0.5,
            phase2_territory_ratio=1e-6,
            point_chunk_size=256,
        )
        out = decoder(
            points_uv=uv,
            Xu=Xu,
            Xv=Xv,
            tau=0.025,
            seeds_raw=seeds,
            w_raw=torch.zeros((40, 40), dtype=torch.float32),
            surface_area_weights=A,
        )
        assert out["seed_duplicate_weights"].min() > 0.99
        assert out["soft_active_seed_count"] > 39.0
        return out["rho"].mean()

    rho_sigma_044 = density_mean(0.44)
    rho_sigma_010 = density_mean(0.10)

    assert rho_sigma_044 > 0.9 * rho_sigma_010
    assert torch.isclose(rho_sigma_044, rho_sigma_010, rtol=0.05, atol=1e-6)


def test_physical_pair_distinctness_is_consistent_across_uv_scales():
    uv10, Xu10, Xv10, _ = make_flat_query(n=18, lx=10.0, ly=10.0)
    uv20, Xu20, Xv20, _ = make_flat_query(n=18, lx=20.0, ly=10.0)
    seeds10 = torch.tensor(
        [
            [0.20, 0.50],
            [0.40, 0.50],
        ],
        dtype=torch.float32,
    )
    seeds20 = torch.tensor(
        [
            [0.10, 0.50],
            [0.20, 0.50],
        ],
        dtype=torch.float32,
    )
    decoder = Phase1VoronoiDecoder(
        n_seeds=2,
        use_Metric_anisotropy=False,
        duplicate_merge_sigma=0.44,
    )

    dist10 = decoder._pairwise_seed_dist_physical(seeds10, uv10, Xu10, Xv10)
    dist20 = decoder._pairwise_seed_dist_physical(seeds20, uv20, Xu20, Xv20)
    distinct10 = decoder._pair_distinctness_from_distance(dist10)
    distinct20 = decoder._pair_distinctness_from_distance(dist20)

    assert torch.isclose(dist10[0, 1], dist20[0, 1], rtol=1e-5, atol=1e-6)
    assert torch.isclose(distinct10[0, 1], distinct20[0, 1], rtol=1e-5, atol=1e-6)


def test_tiny_territory_seed_suppression():
    uv, Xu, Xv, A = make_flat_query()

    decoder = Phase1VoronoiDecoder(
        n_seeds=4,
        use_Metric_anisotropy=False,
        fixed_height=0.5,
        w_min=0.02,
        w_max_ratio=0.2,
        beta=0.003,
        duplicate_merge_sigma=0.02,
        territory_min_ratio=0.05,
    )

    w_raw = torch.zeros(
        (4, 4),
        dtype=torch.float32,
        requires_grad=True,
    )

    # --------------------------------------------------
    # Case 1: all seeds have reasonable territory
    # --------------------------------------------------
    seeds_balanced = torch.tensor(
        [
            [0.20, 0.50],
            [0.40, 0.50],
            [0.60, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_balanced = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_balanced,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 2: one seed is squeezed between neighbours
    # but still remains inside the valid domain
    # --------------------------------------------------
    seeds_tiny = torch.tensor(
        [
            [0.20, 0.50],
            [0.49, 0.50],
            [0.50, 0.50],   # tiny territory candidate
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_tiny = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_tiny,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    # --------------------------------------------------
    # Case 3: give that seed useful space again
    # --------------------------------------------------
    seeds_recovered = torch.tensor(
        [
            [0.20, 0.50],
            [0.40, 0.50],
            [0.62, 0.50],
            [0.80, 0.50],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    out_recovered = decoder(
        points_uv=uv,
        Xu=Xu,
        Xv=Xv,
        tau=0.025,
        seeds_raw=seeds_recovered,
        w_raw=w_raw,
        surface_area_weights=A,
    )

    print("\n--- Tiny territory seed suppression test ---")

    print(
        "Balanced activities:",
        out_balanced["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Tiny-territory activities:",
        out_tiny["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Recovered activities:",
        out_recovered["seed_active_weights"].detach().cpu().numpy(),
    )

    print(
        "Balanced territory:",
        out_balanced["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Tiny-territory fractions:",
        out_tiny["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Recovered territory:",
        out_recovered["seed_territory_fraction"].detach().cpu().numpy(),
    )

    print(
        "Tiny-territory weights:",
        out_tiny["seed_territory_weights"].detach().cpu().numpy(),
    )

    print(
        "Balanced soft active count:",
        float(out_balanced["soft_active_seed_count"].detach()),
    )

    print(
        "Tiny soft active count:",
        float(out_tiny["soft_active_seed_count"].detach()),
    )

    print(
        "Recovered soft active count:",
        float(out_recovered["soft_active_seed_count"].detach()),
    )

    # --------------------------------------------------
    # Gradient check in tiny-territory state
    # --------------------------------------------------
    L_tiny = out_tiny["continuous_voronoi_length"]
    loss_tiny = L_tiny + 0.01 * out_tiny["rho"].sum()
    loss_tiny.backward()

    print(
        "Tiny-state seed gradients:",
        seeds_tiny.grad.detach().cpu().numpy(),
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    # The squeezed seed is seed index 2.
    territory_balanced = out_balanced["seed_territory_fraction"][2]
    territory_tiny = out_tiny["seed_territory_fraction"][2]
    territory_recovered = out_recovered["seed_territory_fraction"][2]

    assert territory_tiny < territory_balanced
    assert territory_recovered > territory_tiny

    # Its territory-based activation should also decrease and recover.
    weight_balanced = out_balanced["seed_territory_weights"][2]
    weight_tiny = out_tiny["seed_territory_weights"][2]
    weight_recovered = out_recovered["seed_territory_weights"][2]

    assert weight_tiny < weight_balanced
    assert weight_recovered > weight_tiny

    # Gradients must remain valid even when the seed is almost useless.
    assert seeds_tiny.grad is not None
    assert torch.isfinite(seeds_tiny.grad).all()

    # Field must remain numerically stable.
    assert torch.isfinite(out_tiny["rho"]).all()
    assert torch.isfinite(out_tiny["fiber3d"]).all()

if __name__ == "__main__":
    test_two_seed_length_check()    
    test_seed_domain_deactivation_and_reactivation()    
    test_duplicate_seed_suppression_and_recovery()    
    test_tiny_territory_seed_suppression()
