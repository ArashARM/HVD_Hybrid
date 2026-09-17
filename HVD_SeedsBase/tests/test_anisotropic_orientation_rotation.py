import math

import pytest
import torch

from neuraltomo_fem.anisotropicFE_new import (
    H8_anisotropic_K,
    orientation_frame_from_fiber,
    rotate_engineering_stiffness,
)


ORTHOTROPIC_MATERIAL = dict(
    material_E1=120.0,
    material_E2=18.0,
    material_E3=11.0,
    material_nu12=0.23,
    material_nu23=0.31,
    material_nu13=0.19,
    material_G12=7.0,
    material_G23=4.0,
    material_G13=5.0,
)


def _h8(material=None):
    return H8_anisotropic_K(
        device=torch.device("cpu"),
        element_size=(1.0, 1.0, 1.0),
        **(ORTHOTROPIC_MATERIAL if material is None else material),
    )


def _angles_from_fiber(fiber):
    fiber = torch.nn.functional.normalize(fiber, dim=-1)
    phi = torch.atan2(fiber[..., 1], fiber[..., 0])
    theta = torch.acos(torch.clamp(fiber[..., 2], -1.0, 1.0))
    return phi, theta


def test_authoritative_frames_are_right_handed_and_use_requested_a1_float64():
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    directions = torch.nn.functional.normalize(directions, dim=1)
    surface_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(directions)
    Q = orientation_frame_from_fiber(directions, surface_normal)

    assert torch.allclose(Q[:, :, 0], directions, atol=1.0e-12, rtol=1.0e-12)
    identity = torch.eye(3, dtype=torch.float64).expand(directions.shape[0], 3, 3)
    assert torch.allclose(Q.transpose(1, 2) @ Q, identity, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(torch.linalg.det(Q), torch.ones(directions.shape[0], dtype=torch.float64), atol=1.0e-12)


def test_rotated_stiffness_is_symmetric_positive_definite_and_used_by_angle2ke():
    h8 = _h8()
    h8.C_inv = h8.C_inv.double()
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    directions = torch.nn.functional.normalize(directions, dim=1)
    phi, theta = _angles_from_fiber(directions)

    h8.angle2Ke(phi, theta, torch.ones(directions.shape[0], dtype=torch.float64))
    C = h8.temp_C
    expected = rotate_engineering_stiffness(
        h8.C_inv.expand(directions.shape[0], 6, 6),
        h8.orientation_matrix,
    )

    assert torch.allclose(C, C.transpose(1, 2), atol=1.0e-10, rtol=1.0e-10)
    assert torch.linalg.eigvalsh(C).amin().item() > 0.0
    assert torch.allclose(C, expected, atol=1.0e-10, rtol=1.0e-10)


def test_isotropic_stiffness_is_invariant_for_random_rotations_float64():
    E = 42.0
    nu = 0.29
    G = E / (2.0 * (1.0 + nu))
    h8 = _h8(
        dict(
            material_E1=E,
            material_E2=E,
            material_E3=E,
            material_nu12=nu,
            material_nu23=nu,
            material_nu13=nu,
            material_G12=G,
            material_G23=G,
            material_G13=G,
        )
    )
    C0 = h8.C_inv.double()
    generator = torch.Generator(device="cpu").manual_seed(7)
    fibers = torch.randn((16, 3), dtype=torch.float64, generator=generator)
    normals = torch.randn((16, 3), dtype=torch.float64, generator=generator)
    Q = orientation_frame_from_fiber(fibers, normals)
    C = rotate_engineering_stiffness(C0.expand(16, 6, 6), Q)

    assert torch.allclose(C, C0.expand_as(C), atol=1.0e-10, rtol=1.0e-10)


def test_xy_plus_and_minus_45_have_opposite_coupling_shear_signs():
    h8 = _h8()
    h8.C_inv = h8.C_inv.double()
    directions = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    ) / math.sqrt(2.0)
    Q = orientation_frame_from_fiber(
        directions,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(2, 3),
    )
    C = rotate_engineering_stiffness(h8.C_inv.double().expand(2, 6, 6), Q)

    assert float(C[0, 0, 5]) == pytest.approx(float(-C[1, 0, 5]), rel=1.0e-10, abs=1.0e-10)
    assert float(C[0, 1, 5]) == pytest.approx(float(-C[1, 1, 5]), rel=1.0e-10, abs=1.0e-10)
    assert abs(float(C[0, 0, 5])) > 1.0e-8


def test_orientation_rotation_remains_differentiable():
    h8 = _h8()
    h8.C_inv = h8.C_inv.double()
    fiber = torch.tensor([[1.0, 0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    Q = orientation_frame_from_fiber(fiber, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    C = rotate_engineering_stiffness(h8.C_inv.double().expand(1, 6, 6), Q)
    loss = C[0].square().sum()
    loss.backward()

    assert fiber.grad is not None
    assert torch.isfinite(fiber.grad).all()
