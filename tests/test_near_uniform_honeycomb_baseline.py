from __future__ import annotations

import math

import pytest
import torch

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder
from Decoder_CLasses.NearUniformHoneycombBaseline import NearUniformHoneycombBaseline


class PlaneCad:
    def __init__(self, hole_radius: float | None = None):
        self.hole_radius = hole_radius

    def eval_uv_norm_batch_torch(self, uv: torch.Tensor) -> dict[str, torch.Tensor]:
        z = uv.new_zeros((uv.shape[0], 1))
        return {"xyz": torch.cat((uv, z), dim=1)}

    def eval_uv_norm_batch(self, uv: torch.Tensor, return_inside_mask: bool = False) -> dict[str, torch.Tensor]:
        out = self.eval_uv_norm_batch_torch(uv)
        if return_inside_mask:
            out["inside_mask"] = self.sample_trim_sdf(uv) >= 0.0
        return out

    def sample_trim_sdf(self, uv: torch.Tensor) -> torch.Tensor:
        outside = (
            (uv[:, 0] < 0.0)
            | (uv[:, 0] > 1.0)
            | (uv[:, 1] < 0.0)
            | (uv[:, 1] > 1.0)
        )
        box = torch.minimum(
            torch.minimum(uv[:, 0], 1.0 - uv[:, 0]),
            torch.minimum(uv[:, 1], 1.0 - uv[:, 1]),
        )
        if self.hole_radius is None:
            sdf = box
        else:
            dist_from_hole = torch.linalg.vector_norm(uv - uv.new_tensor([0.5, 0.5]), dim=1)
            sdf = torch.minimum(box, dist_from_hole - float(self.hole_radius))
        return torch.where(outside, uv.new_full(sdf.shape, -1.0), sdf)

    def smooth_inside_activity(self, uv: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
        return torch.sigmoid(self.sample_trim_sdf(uv) / uv.new_tensor(tau))

    def boundary_curve_tensors(self, as_torch: bool = True) -> dict[str, torch.Tensor]:
        dtype = torch.float64
        square = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=dtype,
        )
        offsets = [0, 2, 4, 6, 8]
        loop_ids = [0, 0, 0, 0]
        if self.hole_radius is not None:
            theta = torch.linspace(0.0, 2.0 * math.pi, 33, dtype=dtype)
            hole = torch.stack(
                (
                    0.5 + float(self.hole_radius) * torch.cos(theta),
                    0.5 - float(self.hole_radius) * torch.sin(theta),
                ),
                dim=1,
            )
            start = int(square.shape[0])
            square = torch.cat((square, hole), dim=0)
            offsets.append(start + int(hole.shape[0]))
            loop_ids.append(1)
        return {
            "boundary_curve_uv": square,
            "boundary_curve_offsets": torch.tensor(offsets, dtype=torch.long),
            "boundary_curve_loop_id": torch.tensor(loop_ids, dtype=torch.long),
        }


class NoSdfHoleCad:
    def __init__(self, hole_radius: float = 0.2):
        self.hole_radius = float(hole_radius)

    def eval_uv_norm_batch_torch(self, uv: torch.Tensor) -> dict[str, torch.Tensor]:
        z = uv.new_zeros((uv.shape[0], 1))
        return {"xyz": torch.cat((uv, z), dim=1)}

    def eval_uv_norm_batch(self, uv: torch.Tensor, return_inside_mask: bool = False) -> dict[str, torch.Tensor]:
        out = self.eval_uv_norm_batch_torch(uv)
        if return_inside_mask:
            out["inside_mask"] = self.smooth_inside_activity(uv) >= 0.5
        return out

    def smooth_inside_activity(self, uv: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
        in_box = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        outside_hole = torch.linalg.vector_norm(uv - uv.new_tensor([0.5, 0.5]), dim=1) >= self.hole_radius
        return (in_box & outside_hole).to(dtype=uv.dtype)


def make_face_mesh(n: int = 21, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    u = torch.linspace(0.0, 1.0, n, dtype=dtype)
    uu, vv = torch.meshgrid(u, u, indexing="xy")
    uv = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=1)
    points_xyz = torch.cat((uv, uv.new_zeros((uv.shape[0], 1))), dim=1)
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces_ijk = torch.tensor(faces, dtype=torch.long)
    face_areas = torch.full((faces_ijk.shape[0],), 0.5 / float((n - 1) * (n - 1)), dtype=dtype)
    return {
        "uv": uv,
        "points_xyz": points_xyz,
        "Xu": torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).expand(uv.shape[0], 3),
        "Xv": torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype).expand(uv.shape[0], 3),
        "faces_ijk": faces_ijk,
        "face_areas": face_areas,
    }


def make_baseline(**kwargs):
    dtype = kwargs.pop("dtype", torch.float64)
    cad = kwargs.pop("cad_domain", PlaneCad())
    face_mesh = kwargs.pop("face_mesh", make_face_mesh(dtype=dtype))
    target_seed_count = kwargs.pop("target_seed_count", None)
    spacing_uv = kwargs.pop("spacing_uv", 0.18)
    include_exterior_support_sites = kwargs.pop("include_exterior_support_sites", False)
    seed_domain_margin = kwargs.pop("seed_domain_margin", 0.0)
    decoder = ContinuousVoronoiDecoder(
        cad,
        face_mesh,
        use_trim_activity=True,
        strut_thickness=0.08,
        tube_curve_samples=8,
        edge_trim_samples=8,
        use_spatial_pruning=False,
        seed_domain_margin=seed_domain_margin,
    )
    return NearUniformHoneycombBaseline(
        decoder=decoder,
        cad_domain=cad,
        face_mesh=face_mesh,
        target_seed_count=target_seed_count,
        spacing_uv=spacing_uv,
        include_exterior_support_sites=include_exterior_support_sites,
        **kwargs,
    )


def test_deterministic_triangular_lattice_generation() -> None:
    baseline = make_baseline()
    a, mask_a = baseline._retained_lattice_np(0.18)
    b, mask_b = baseline._retained_lattice_np(0.18)

    assert torch.allclose(torch.as_tensor(a), torch.as_tensor(b))
    assert torch.equal(torch.as_tensor(mask_a), torch.as_tensor(mask_b))


def test_staggered_row_spacing() -> None:
    baseline = make_baseline()
    h = 0.2
    lattice = baseline._triangular_lattice_np(h)
    rows = sorted(set(round(float(y), 12) for y in lattice[:, 1]))
    row0 = lattice[abs(lattice[:, 1] - rows[len(rows) // 2]) < 1e-10]
    row1 = lattice[abs(lattice[:, 1] - rows[len(rows) // 2 + 1]) < 1e-10]
    row0 = row0[row0[:, 0].argsort()]
    row1 = row1[row1[:, 0].argsort()]

    assert torch.allclose(torch.diff(torch.as_tensor(row0[:, 0]))[:3], torch.full((3,), h, dtype=torch.float64))
    assert rows[len(rows) // 2 + 1] - rows[len(rows) // 2] == pytest.approx(math.sqrt(3.0) * h / 2.0)
    assert min(abs((x1 - x0) - 0.5 * h) for x0 in row0[:, 0] for x1 in row1[:, 0]) < 1e-10


def test_dtype_and_device_are_preserved() -> None:
    baseline = make_baseline(dtype=torch.float32)
    out = baseline.generate(device=torch.device("cpu"), dtype=torch.float32, generate_density_fiber=False)

    assert out["seeds_uv"].dtype == torch.float32
    assert out["rho"].dtype == torch.float32
    assert out["seeds_uv"].device.type == "cpu"


def test_returned_dictionary_matches_decoder_schema_and_fields_are_finite() -> None:
    baseline = make_baseline()
    out = baseline.generate(dtype=torch.float64, generate_density_fiber=True)

    for key in (
        "seeds",
        "seeds_uv",
        "seeds_xyz",
        "edge_curves_uv",
        "edge_curves_xyz",
        "edge_curve_lengths_xyz",
        "total_curve_length",
        "total_voronoi_curve_length",
        "rho",
        "density",
        "fiber3d",
        "tube_distance",
        "strut_thickness",
        "centerline_radius",
        "graph",
        "edges",
        "mode",
        "baseline_mode",
        "decoder_mode",
        "interior_seed_mask",
        "support_seed_mask",
        "diagnostics",
    ):
        assert key in out
    assert out["mode"] == "near_uniform_honeycomb"
    assert out["decoder_mode"] == "scipy_topology"
    assert out["baseline_mode"] == "uv_hex"
    assert out["interior_seed_mask"].dtype == torch.bool
    assert out["support_seed_mask"].dtype == torch.bool
    assert out["interior_seed_mask"].shape == (out["seeds_uv"].shape[0],)
    assert torch.equal(out["support_seed_mask"], ~out["interior_seed_mask"])
    assert torch.isfinite(out["rho"]).all()
    assert torch.isfinite(out["fiber3d"]).all()
    assert torch.isfinite(out["tube_distance"]).all()
    expected_lengths = torch.linalg.vector_norm(
        out["edge_curves_xyz"][:, 1:, :] - out["edge_curves_xyz"][:, :-1, :],
        dim=-1,
    ).sum(dim=1)
    assert torch.allclose(out["edge_curve_lengths_xyz"], expected_lengths)
    assert torch.allclose(out["total_curve_length"], expected_lengths.sum())
    edge_type = out["graph"]["edge_type"]
    vd_mask = (edge_type == 0) | (edge_type == 1) | (edge_type == 3)
    assert torch.allclose(out["total_voronoi_curve_length"], expected_lengths[vd_mask].sum())
    assert torch.allclose(out["graph"]["edge_curve_lengths_xyz"], out["edge_curve_lengths_xyz"])
    assert torch.allclose(out["graph"]["total_curve_length"], out["total_curve_length"])
    assert out["diagnostics"]["total_curve_length"] == pytest.approx(
        float(out["total_curve_length"].detach().cpu().item())
    )


def test_fiber_vectors_are_unit_or_safely_normalized() -> None:
    out = make_baseline().generate(generate_density_fiber=True)
    norms = torch.linalg.vector_norm(out["fiber3d"], dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_decoder_guard_ridges_are_excluded_from_reinforcement_edges() -> None:
    out = make_baseline(
        include_exterior_support_sites=True,
        support_band_cells=1.0,
        seed_domain_margin=0.12,
    ).generate(
        generate_density_fiber=False
    )
    graph = out["graph"]
    pairs = graph["edge_seed_pair"]
    edge_type = graph["edge_type"]
    real_count = int(out["seeds_uv"].shape[0])
    reinforcement = (edge_type == 0) | (edge_type == 1) | (edge_type == 3)

    assert ((pairs[reinforcement] < real_count) | (pairs[reinforcement] == -1)).all()
    assert int(out["diagnostics"]["num_guard_ridges_skipped"]) >= 0


def test_exterior_support_sites_are_retained_near_boundary_within_margin() -> None:
    baseline = make_baseline(
        include_exterior_support_sites=True,
        support_band_cells=2.0,
        seed_domain_margin=0.08,
        spacing_uv=0.07,
    )
    seeds, interior = baseline._retained_lattice_np(0.07)
    support = seeds[~interior]

    assert support.shape[0] > 0
    assert ((seeds >= -0.08 - 1e-12) & (seeds <= 1.08 + 1e-12)).all()
    assert (((support < 0.0) | (support > 1.0)).any(axis=1)).any()


def test_distance_to_valid_domain_ignores_outside_box_sdf_sentinel() -> None:
    baseline = make_baseline(seed_domain_margin=0.2)
    uv = torch.tensor([[-0.05, 0.5], [0.5, 0.5]], dtype=torch.float64)
    distances = baseline._distance_to_valid_domain_np(uv.numpy())

    assert distances[0] == pytest.approx(0.05)
    assert distances[1] == pytest.approx(0.0)


def test_internal_hole_excludes_physical_seeds() -> None:
    cad = PlaneCad(hole_radius=0.2)
    face_mesh = make_face_mesh()
    baseline = make_baseline(cad_domain=cad, face_mesh=face_mesh, spacing_uv=0.08)
    seeds, interior = baseline._retained_lattice_np(0.08)
    physical = torch.as_tensor(seeds[interior], dtype=torch.float64)

    assert (torch.linalg.vector_norm(physical - torch.tensor([0.5, 0.5]), dim=1) >= 0.2 - 1e-10).all()


def test_distance_to_hole_without_sdf_uses_nearest_valid_auxiliary_sample() -> None:
    cad = NoSdfHoleCad(hole_radius=0.2)
    face_mesh = make_face_mesh(n=21)
    baseline = make_baseline(cad_domain=cad, face_mesh=face_mesh, spacing_uv=0.08)
    distances = baseline._distance_to_valid_domain_np(
        torch.tensor([[0.5, 0.5], [0.1, 0.1]], dtype=torch.float64).numpy()
    )

    assert distances[0] > 0.0
    assert distances[1] == pytest.approx(0.0)


def test_physical_relax_does_not_increase_mean_sample_distance() -> None:
    base = make_baseline(mode="uv_hex", spacing_uv=0.22)
    seeds, interior = base._retained_lattice_np(0.22)
    before = base._coverage_diagnostics(seeds, interior)["mean_nearest_sample_distance"]

    relaxed = make_baseline(mode="physical_relax", spacing_uv=0.22, relaxation_steps=2)
    out = relaxed.generate(generate_density_fiber=False)
    after = out["diagnostics"]["mean_nearest_sample_distance"]

    assert after <= before + 1e-8


def test_physical_relax_energy_does_not_increase_and_support_sites_do_not_assign() -> None:
    baseline = make_baseline(
        mode="physical_relax",
        spacing_uv=0.12,
        relaxation_steps=1,
        include_exterior_support_sites=True,
        support_band_cells=1.0,
        seed_domain_margin=0.08,
    )
    seeds, interior = baseline._retained_lattice_np(0.12)
    samples_uv, samples_xyz, _ = baseline._valid_sample_data_np()
    _, assigned = baseline._physical_assignment_np(
        seeds,
        interior,
        samples_xyz,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert set(assigned.tolist()).issubset(set(torch.nonzero(torch.as_tensor(interior)).flatten().tolist()))

    out = baseline.generate(generate_density_fiber=False)
    assert out["diagnostics"]["relaxation_final_energy"] <= out["diagnostics"]["relaxation_initial_energy"] + 1e-12
    assert out["support_seed_mask"].any()


def test_collision_prevention_raises_when_no_separated_samples_exist() -> None:
    baseline = make_baseline(mode="physical_relax", spacing_uv=0.1, relaxation_steps=1)
    seeds = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float64).numpy()
    interior = torch.tensor([True, True, True]).numpy()
    samples = torch.tensor([[0.5, 0.5]], dtype=torch.float64).numpy()

    with pytest.raises(ValueError, match="separated valid configuration"):
        baseline._resolve_seed_collisions_np(seeds, interior, samples, spacing=0.1)


def test_collision_relocation_is_valid_separated_and_near_original() -> None:
    baseline = make_baseline(mode="physical_relax", spacing_uv=0.1, relaxation_steps=1)
    seeds = torch.tensor(
        [[0.5, 0.5], [0.5001, 0.5], [0.8, 0.2]],
        dtype=torch.float64,
    ).numpy()
    interior = torch.tensor([True, True, True]).numpy()
    samples = torch.tensor(
        [[0.5002, 0.5], [0.5012, 0.5], [0.51, 0.5], [0.7, 0.7]],
        dtype=torch.float64,
    ).numpy()

    resolved = baseline._resolve_seed_collisions_np(seeds, interior, samples, spacing=0.1)
    tol = baseline._collision_tolerance(0.1)
    distances = torch.cdist(torch.as_tensor(resolved), torch.as_tensor(resolved))
    distances.fill_diagonal_(float("inf"))

    assert baseline._inside_domain_np(resolved).all()
    assert float(distances.min()) >= tol
    assert torch.allclose(torch.as_tensor(resolved[0]), torch.tensor([0.5012, 0.5], dtype=torch.float64))


def test_unresolved_duplicate_support_sites_fail_final_validation() -> None:
    baseline = make_baseline(spacing_uv=0.1)
    seeds = torch.tensor(
        [[0.1, 0.1], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2]],
        dtype=torch.float64,
    ).numpy()

    with pytest.raises(ValueError, match="excessively close"):
        baseline._validate_sites_for_decoder(seeds, spacing=0.1)


def test_deterministic_phase_and_rotation_change_lattice_repeatably() -> None:
    a = make_baseline(phase_uv=(0.03, -0.02), rotation_degrees=15.0)
    b = make_baseline(phase_uv=(0.03, -0.02), rotation_degrees=15.0)
    c = make_baseline(phase_uv=(0.00, 0.00), rotation_degrees=15.0)

    seeds_a, _ = a._retained_lattice_np(0.18)
    seeds_b, _ = b._retained_lattice_np(0.18)
    seeds_c, _ = c._retained_lattice_np(0.18)

    assert torch.allclose(torch.as_tensor(seeds_a), torch.as_tensor(seeds_b))
    assert not torch.allclose(torch.as_tensor(seeds_a[: min(len(seeds_a), len(seeds_c))]), torch.as_tensor(seeds_c[: min(len(seeds_a), len(seeds_c))]))


def test_flat_rectangular_interior_cells_are_predominantly_six_sided() -> None:
    out = make_baseline(spacing_uv=0.11).generate(generate_density_fiber=False)

    assert out["diagnostics"]["hexagonal_cell_fraction"] >= 0.6


def test_invalid_parameter_combinations_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        make_baseline(target_seed_count=10, spacing_uv=0.1)
    with pytest.raises(ValueError, match="spacing_uv"):
        make_baseline(spacing_uv=-0.1)
    with pytest.raises(ValueError, match="mode"):
        make_baseline(mode="geodesic_cvt")
    with pytest.raises(ValueError, match="relaxation_steps"):
        make_baseline(mode="physical_relax", relaxation_steps=0)
