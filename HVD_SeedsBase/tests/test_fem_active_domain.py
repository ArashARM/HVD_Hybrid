from types import SimpleNamespace

import numpy as np
import torch

from neuraltomo_fem.run_fem_loss import NeuralTOMOFEM


def _node_id(i, j, k, nelx=2, nely=1):
    return k * (nely + 1) * (nelx + 1) + i * (nely + 1) + j


def _tiny_active_domain_problem():
    nelx, nely, nelz = 2, 1, 1
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    force = np.zeros((ndof, 1), dtype=float)
    fixed_nodes = np.array([_node_id(0, j, k) for k in range(nelz + 1) for j in range(nely + 1)], dtype=np.int64)
    loaded_nodes = np.array([_node_id(1, j, k) for k in range(nelz + 1) for j in range(nely + 1)], dtype=np.int64)
    force[3 * loaded_nodes + 2, 0] = -1.0 / loaded_nodes.size
    fixed = np.concatenate([3 * fixed_nodes + c for c in range(3)]).astype(np.int64)

    problem = SimpleNamespace(
        mesh={"type": "grid", "nelx": nelx, "nely": nely, "nelz": nelz, "elemSize": (1.0, 1.0, 1.0)},
        boundaryCondition={"force": force, "fixed": fixed, "numDOFPerNode": 3},
        materialProperty={
            "length_unit": "mm",
            "force_unit": "N",
            "stress_unit": "MPa",

            # Required by GridMesh.initK()
            "E": 20.0,
            "nu": 0.25,
            "penal": 3.0,

            "material_E1": 100.0,
            "material_E2": 20.0,
            "material_E3": 20.0,
            "material_nu12": 0.25,
            "material_nu23": 0.25,
            "material_nu13": 0.25,
            "material_G12": 8.0,
            "material_G23": 7.0,
            "material_G13": 8.0,
        },
                elem_occupancy=np.array([True, False]),
    )
    return problem, fixed_nodes, loaded_nodes


def _solve_disp(stiffness):
    problem, _, _ = _tiny_active_domain_problem()
    fem = NeuralTOMOFEM(problem, device="cpu")
    stiffness = torch.as_tensor(stiffness, dtype=torch.float32)
    phi = torch.zeros(2)
    theta = torch.zeros(2)
    fem(stiffness, phi, theta, penal=1.0)
    return fem.fe.displacement_mag_elem.max(), fem


def test_active_domain_excludes_outside_element_and_orphan_dofs():
    problem, fixed_nodes, loaded_nodes = _tiny_active_domain_problem()
    fem = NeuralTOMOFEM(problem, device="cpu")
    mesh = fem.fe.mesh

    assert mesh.active_element_ids.tolist() == [0]
    assert mesh.nullElem.tolist() == [1]
    assert mesh.active_iK.size == 24 * 24

    inactive_only_nodes = np.array([_node_id(2, j, k) for k in range(2) for j in range(2)], dtype=np.int64)
    assert not np.intersect1d(mesh.active_node_ids, inactive_only_nodes).size
    assert not np.intersect1d(mesh.free, np.concatenate([3 * inactive_only_nodes + c for c in range(3)])).size
    assert np.intersect1d(mesh.active_node_ids, fixed_nodes).size == fixed_nodes.size
    assert np.intersect1d(mesh.active_node_ids, loaded_nodes).size == loaded_nodes.size


def test_outside_floor_invariance_and_repeatability():
    vals = []
    for _ in range(3):
        disp, _fem = _solve_disp([1.0, 1.0e-6])
        vals.append(disp.detach())
    assert torch.allclose(vals[0], vals[1], rtol=0.0, atol=1.0e-7)
    assert torch.allclose(vals[1], vals[2], rtol=0.0, atol=1.0e-7)

    low_outside, _ = _solve_disp([1.0, 1.0e-6])
    high_outside, _ = _solve_disp([1.0, 1.0])
    assert torch.allclose(low_outside, high_outside, rtol=0.0, atol=1.0e-7)


def test_uniform_active_stiffness_scaling_and_gradient_direction():
    base, _ = _solve_disp([1.0, 0.0])
    stiffer, _ = _solve_disp([1.1, 0.0])
    softer, _ = _solve_disp([0.9, 0.0])

    assert stiffer < base
    assert softer > base
    assert torch.allclose(stiffer / base, stiffer.new_tensor(1.0 / 1.1), rtol=7.5e-2, atol=7.5e-2)
    assert torch.allclose(softer / base, softer.new_tensor(1.0 / 0.9), rtol=7.5e-2, atol=7.5e-2)

    problem, _, _ = _tiny_active_domain_problem()
    fem = NeuralTOMOFEM(problem, device="cpu")
    stiffness = torch.tensor([1.0, 0.0], dtype=torch.float32, requires_grad=True)
    phi = torch.zeros(2)
    theta = torch.zeros(2)
    fem(stiffness, phi, theta, penal=1.0)
    disp = fem.fe.displacement_mag_elem.max()
    disp.backward()
    grad_dir = stiffness.grad[0].detach()

    eps = 1.0e-3
    plus, _ = _solve_disp([1.0 + eps, 0.0])
    minus, _ = _solve_disp([1.0 - eps, 0.0])
    fd = (plus - minus) / (2.0 * eps)
    assert grad_dir < 0.0
    assert fd < 0.0
    assert torch.allclose(grad_dir, fd, rtol=2.5e-1, atol=2.5e-1)
