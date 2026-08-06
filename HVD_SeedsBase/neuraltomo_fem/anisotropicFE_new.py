import math
import warnings

import numpy as np
import torch

def _unit_checked_material(kwargs):
    length_unit = kwargs.get("length_unit", "mm")
    force_unit = kwargs.get("force_unit", "N")
    stress_unit = kwargs.get("stress_unit", "MPa")
    if (length_unit, force_unit, stress_unit) != ("mm", "N", "MPa"):
        raise ValueError(
            "FEM material units must be length='mm', force='N', stress='MPa'. "
            f"Got {(length_unit, force_unit, stress_unit)}."
        )

class H8_isotropic_K:
    pass

class H8_anisotropic_K:
    def __init__(self, device=torch.device('cuda'), **kwargs):
        _unit_checked_material(kwargs)
        # for const number
        _sqrt_3_5 = math.sqrt(3 / 5)
        if any(k in kwargs for k in ("Ef", "Et", "nuf", "nut")):
            warnings.warn(
                "Legacy material keys Ef/Et/nuf/nut are mapped to explicit "
                "orthotropic CCF fields. Prefer material_E1/E2/E3, "
                "material_nu12/nu23/nu13, and material_G12/G23/G13.",
                RuntimeWarning,
                stacklevel=2,
            )
        _E_f = kwargs.get('material_E1', kwargs.get('material_E_longitudinal', kwargs.get('Ef', None)))
        _E_t = kwargs.get('material_E2', kwargs.get('material_E_transverse', kwargs.get('Et', None)))
        _E_3 = kwargs.get('material_E3', _E_t)
        _nu_f = kwargs.get('material_nu12', kwargs.get('material_nu_longitudinal', kwargs.get('nuf', 0.3)))
        _nu_t = kwargs.get('material_nu23', kwargs.get('material_nu_transverse', kwargs.get('nut', 0.3)))
        _nu_13 = kwargs.get('material_nu13', _nu_f)
        _G_12 = kwargs.get('material_G12', kwargs.get('material_shear_modulus', kwargs.get('Gf', None)))
        _G_23 = kwargs.get('material_G23', None)
        _G_13 = kwargs.get('material_G13', _G_12)
        if _E_f is None or _E_t is None or _E_3 is None:
            raise ValueError(
                "Anisotropic FEM requires material_E1, material_E2, and "
                "material_E3, in MPa."
            )
        if any(k in kwargs for k in ("Ef", "Et", "nuf", "nut")):
            if _G_12 is None:
                _G_12 = float(_E_f) / (2.0 * (1.0 + float(_nu_f)))
            if _G_13 is None:
                _G_13 = _G_12
            if _G_23 is None:
                _G_23 = float(_E_t) / (2.0 * (1.0 + float(_nu_t)))
        if _G_12 is None or _G_13 is None or _G_23 is None:
            raise ValueError(
                "CCF orthotropic shear moduli must be explicit: "
                "material_G12, material_G23, and material_G13."
            )

        if 'P' not in kwargs:
            self.P = torch.tensor([[ 0.00126103,  0.00017645, -0.00143748,  0.        ,  0.        , 0.        ],
                                    [ 0.00017645,  0.00265957, -0.00283602,  0.        ,  0.        ,0.        ],
                                    [-0.00143748, -0.00283602,  0.0042735 ,  0.        ,  0.        ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.03125   ,  0.        ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.03125   ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,0.03125   ]],
                                  dtype=torch.float32, device=device)
            self.Q = torch.tensor([0.00283733, 0.0099734 , 0.03632479, 0.        , 0.        ,0.        ],
                                  dtype=torch.float32, device=device)

        else:
            self.P = torch.tensor(kwargs['P'], dtype=torch.float32, device=device)
            self.Q = torch.tensor(kwargs['Q'], dtype=torch.float32, device=device)

        self.Ef = _E_f
        self.Et = _E_t
        self.nuf = _nu_f
        self.nut = _nu_t

        element_size = kwargs.get("element_size", (1.0, 1.0, 1.0))
        if len(element_size) != 3:
            raise ValueError(f"element_size must have three entries, got {element_size}")
        self.element_size = tuple(float(v) for v in element_size)
        if any((not math.isfinite(v) or v <= 0.0) for v in self.element_size):
            raise ValueError(f"element_size entries must be positive finite, got {self.element_size}")

        E1, E2, E3 = float(_E_f), float(_E_t), float(_E_3)
        nu12, nu23, nu13 = float(_nu_f), float(_nu_t), float(_nu_13)
        G12, G23, G13 = float(_G_12), float(_G_23), float(_G_13)
        for name, value in {
            "material_E1": E1,
            "material_E2": E2,
            "material_E3": E3,
            "material_G12": G12,
            "material_G23": G23,
            "material_G13": G13,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite, got {value}.")
        nu21 = nu12 * E2 / E1
        nu31 = nu13 * E3 / E1
        nu32 = nu23 * E3 / E2
        C = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                      [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                      [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                      [0, 0, 0, 1 / G23, 0, 0],
                      [0, 0, 0, 0, 1 / G13, 0],
                      [0, 0, 0, 0, 0, 1 / G12]])
        if not np.all(np.isfinite(C)) or not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
            raise ValueError("Orthotropic compliance matrix is non-finite or not symmetric.")
        self.C_inv_np = np.linalg.inv(C)
        if (
            not np.all(np.isfinite(self.C_inv_np))
            or not np.allclose(self.C_inv_np, self.C_inv_np.T, rtol=1e-5, atol=1e-5)
            or np.linalg.eigvalsh(self.C_inv_np).min() <= 0.0
        ):
            raise ValueError("Orthotropic constitutive matrix is not mechanically admissible.")
        self.C_inv = torch.tensor(self.C_inv_np, dtype=torch.float32, device=device)

        # 3 - point Gauss integration
        integration_point = torch.tensor([-_sqrt_3_5, 0, _sqrt_3_5],
                                         dtype=torch.float32) / 2
        integration_weight = torch.tensor([5 / 9, 8 / 9, 5 / 9],
                                          dtype=torch.float32) / 2

        all_intergration_points = np.vstack(
            np.meshgrid(integration_point, integration_point, integration_point)).reshape(3, -1).T

        all_intergration_weight_temp = np.vstack(
            np.meshgrid(integration_weight, integration_weight, integration_weight)).reshape(3, -1).T

        int_weight = all_intergration_weight_temp[:, 0] * \
                     all_intergration_weight_temp[:, 1] * \
                     all_intergration_weight_temp[:, 2]
        self.int_weight = torch.tensor(int_weight, device=device)

        # B = np.einsum('i,ijk->ijk', all_intergration_weight, self.matrixB(all_intergration_points))
        # [x,y,z,zy,zx,yx]
        self.B = self.physical_B(all_intergration_points, device=device)
        self.NodeB = self.physical_B(np.array([[0.0, 0.0, 0.0]]), device=device)[0]

    def physical_B(self, xyz, device=None):
        device = self.C_inv.device if device is None else device
        pts = torch.as_tensor(xyz, dtype=torch.float32, device=device).reshape(-1, 3)
        xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
        signs = torch.tensor(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=torch.float32,
            device=device,
        )
        sx, sy, sz = signs[:, 0], signs[:, 1], signs[:, 2]
        dN_dxi = 0.125 * sx[None, :] * (1.0 + sy[None, :] * (2.0 * eta[:, None])) * (1.0 + sz[None, :] * (2.0 * zeta[:, None])) * 2.0
        dN_deta = 0.125 * sy[None, :] * (1.0 + sx[None, :] * (2.0 * xi[:, None])) * (1.0 + sz[None, :] * (2.0 * zeta[:, None])) * 2.0
        dN_dzeta = 0.125 * sz[None, :] * (1.0 + sx[None, :] * (2.0 * xi[:, None])) * (1.0 + sy[None, :] * (2.0 * eta[:, None])) * 2.0
        hx, hy, hz = self.element_size
        dN_dx = dN_dxi / hx
        dN_dy = dN_deta / hy
        dN_dz = dN_dzeta / hz
        B = torch.zeros((pts.shape[0], 6, 24), dtype=torch.float32, device=device)
        for a in range(8):
            c = 3 * a
            B[:, 0, c + 0] = dN_dx[:, a]
            B[:, 1, c + 1] = dN_dy[:, a]
            B[:, 2, c + 2] = dN_dz[:, a]
            B[:, 3, c + 1] = dN_dz[:, a]
            B[:, 3, c + 2] = dN_dy[:, a]
            B[:, 4, c + 0] = dN_dz[:, a]
            B[:, 4, c + 2] = dN_dx[:, a]
            B[:, 5, c + 0] = dN_dy[:, a]
            B[:, 5, c + 1] = dN_dx[:, a]
        return B

    def matrixB(self, xyz):
        # [x,y,z,zy,zx,yx]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # x, y, z = torch.unbind(xyz, -1)
        o = np.zeros_like(x)
        b = (-(0.5 - y) * (0.5 - z), o, o, (0.5 - y) * (0.5 - z), o, o,
             (0.5 - z) * (y + 0.5), o, o, -(0.5 - z) * (y + 0.5), o, o,
             -(0.5 - y) * (z + 0.5), o, o, (0.5 - y) * (z + 0.5), o, o,
             (y + 0.5) * (z + 0.5), o, o, -(y + 0.5) * (z + 0.5), o, o,
             o, -(0.5 - x) * (0.5 - z), o, o, -(0.5 - z) * (x + 0.5), o,
             o, (0.5 - z) * (x + 0.5), o, o, (0.5 - x) * (0.5 - z), o,
             o, -(0.5 - x) * (z + 0.5), o, o, -(x + 0.5) * (z + 0.5), o,
             o, (x + 0.5) * (z + 0.5), o, o, (0.5 - x) * (z + 0.5), o,
             o, o, -(0.5 - x) * (0.5 - y), o, o, -(0.5 - y) * (x + 0.5),
             o, o, -(x + 0.5) * (y + 0.5), o, o, -(0.5 - x) * (y + 0.5),
             o, o, (0.5 - x) * (0.5 - y), o, o, (0.5 - y) * (x + 0.5),
             o, o, (x + 0.5) * (y + 0.5), o, o, (0.5 - x) * (y + 0.5),
             o, -(0.5 - x) * (0.5 - y), -(0.5 - x) * (0.5 - z), o, -(0.5 - y) * (x + 0.5), -(0.5 - z) * (x + 0.5),
             o, -(x + 0.5) * (y + 0.5), (0.5 - z) * (x + 0.5), o, -(0.5 - x) * (y + 0.5), (0.5 - x) * (0.5 - z),
             o, (0.5 - x) * (0.5 - y), -(0.5 - x) * (z + 0.5), o, (0.5 - y) * (x + 0.5), -(x + 0.5) * (z + 0.5),
             o, (x + 0.5) * (y + 0.5), (x + 0.5) * (z + 0.5), o, (0.5 - x) * (y + 0.5), (0.5 - x) * (z + 0.5),
             -(0.5 - x) * (0.5 - y), o, -(0.5 - y) * (0.5 - z), -(0.5 - y) * (x + 0.5), o, (0.5 - y) * (0.5 - z),
             -(x + 0.5) * (y + 0.5), o, (0.5 - z) * (y + 0.5), -(0.5 - x) * (y + 0.5), o, -(0.5 - z) * (y + 0.5),
             (0.5 - x) * (0.5 - y), o, -(0.5 - y) * (z + 0.5), (0.5 - y) * (x + 0.5), o, (0.5 - y) * (z + 0.5),
             (x + 0.5) * (y + 0.5), o, (y + 0.5) * (z + 0.5), (0.5 - x) * (y + 0.5), o, -(y + 0.5) * (z + 0.5),
             -(0.5 - x) * (0.5 - z), -(0.5 - y) * (0.5 - z), o, -(0.5 - z) * (x + 0.5), (0.5 - y) * (0.5 - z), o,
             (0.5 - z) * (x + 0.5), (0.5 - z) * (y + 0.5), o, (0.5 - x) * (0.5 - z), -(0.5 - z) * (y + 0.5), o,
             -(0.5 - x) * (z + 0.5), -(0.5 - y) * (z + 0.5), o, -(x + 0.5) * (z + 0.5), (0.5 - y) * (z + 0.5), o,
             (x + 0.5) * (z + 0.5), (y + 0.5) * (z + 0.5), o, (0.5 - x) * (z + 0.5), -(y + 0.5) * (z + 0.5), o)
        return np.stack(b, -1).reshape((xyz.shape[0], 6, 24))

    def angle2Ke(self, phi, theta, stiffness_factor, density_penal=1.0):
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        cosT2, sinT2 = cosT * cosT, sinT * sinT

        cosP, sinP = torch.cos(phi), torch.sin(phi)
        cosP2, sinP2 = cosP * cosP, sinP * sinP

        o = torch.zeros_like(phi) # 0-vector

        R = torch.stack((
            cosP2 * cosT2, cosP2 * sinT2, sinP2, 2 * cosP * sinP * sinT, -2 * cosP * cosT * sinP, -2 * cosP2 * cosT * sinT,
            sinT2, cosT2, o, o, o, 2 * cosT * sinT,
            cosT2 * sinP2, sinP2 * sinT2, cosP2, -2 * cosP * sinP * sinT, 2 * cosP * cosT * sinP,
            -2 * cosT * sinP2 * sinT,
            cosT * sinP * sinT, -cosT * sinP * sinT, o, cosP * cosT, cosP * sinT, sinP * (cosT2 - sinT2),
            cosP * cosT2 * sinP, cosP * sinP * sinT2, -cosP * sinP, -sinT * (cosP2 - sinP2),
            cosT * (cosP2 - sinP2), -2 * cosP * cosT * sinP * sinT,
            cosP * cosT * sinT, -cosP * cosT * sinT, o, -cosT * sinP, -sinP * sinT, cosP * (cosT2 - sinT2)
        ), -1).reshape(phi.shape + (6, 6))

        # C_inv 6x6
        # C = R @ self.C_inv.unsqueeze(0).expand(batch_size, -1, -1) @ R.transpose(1, 2)

        C = torch.einsum('bij,jk,blk->bil', R, self.C_inv, R)
        C_new = torch.einsum('bji,bjk->bik', R, C)

        # self.B 27x6x24, C nx6x6
        B = self.B
        weight = self.int_weight
        # BT = B.transpose(1, 2)

        detJ = self.element_size[0] * self.element_size[1] * self.element_size[2]
        BT_C_B = torch.einsum('d,dji,bjk,dkl->bil', weight * detJ, B, C, B)
        # `stiffness_factor` already contains the SIMP penalization
        # and the minimum stiffness ratio. Do not apply either again.
        dK = torch.einsum('i,ijk->ijk', stiffness_factor, BT_C_B)
        self.temp_C = C_new
        self.T = R
        return dK


# via rotation Matrix
class H8_anisotropic_K_R:
    def __init__(self, device=torch.device('cuda'), **kwargs):
        raise RuntimeError(
            "H8_anisotropic_K_R is deprecated. Use H8_anisotropic_K, which "
            "uses explicit orthotropic CCF shear moduli, physical B-matrix "
            "scaling, and direct stiffness_factor assembly."
        )
        # for const number
        _sqrt_3_5 = math.sqrt(3 / 5)
        s, t, r = 0.5, 0.5, 0.5
        _E_f = 5 if 'Ef' not in kwargs else kwargs['Ef']
        _E_t = 1 if 'Et' not in kwargs else kwargs['Et']
        _nu_f = 0.3 if 'nuf' not in kwargs else kwargs['nuf']
        _nu_t = 0.32 if 'nut' not in kwargs else kwargs['nut']

        self.Ef = _E_f
        self.Et = _E_t
        self.nuf = _nu_f
        self.nut = _nu_t

        _G_t = _E_t / (2 * (1 + _nu_t))
        _G_f = _E_f / (2 * (1 + _nu_f))

        C = np.array([[1 / _E_f, -_nu_f / _E_f, -_nu_f / _E_f, 0, 0, 0],
                      [-_nu_f / _E_f, 1 / _E_t, -_nu_t / _E_t, 0, 0, 0],
                      [-_nu_f / _E_f, -_nu_t / _E_t, 1 / _E_t, 0, 0, 0],
                      [0, 0, 0, 1 / _G_t, 0, 0],
                      [0, 0, 0, 0, 1 / _G_f, 0],
                      [0, 0, 0, 0, 0, 1 / _G_f]])
        self.C_inv_np = np.linalg.inv(C)
        self.C_inv = torch.tensor(self.C_inv_np, dtype=torch.float32, device=device)

        # 3 - point Gauss integration
        integration_point = torch.tensor([-_sqrt_3_5, 0, _sqrt_3_5],
                                         dtype=torch.float32) / 2
        integration_weight = torch.tensor([5 / 9, 8 / 9, 5 / 9],
                                          dtype=torch.float32) / 2

        all_intergration_points = np.vstack(
            np.meshgrid(integration_point, integration_point, integration_point)).reshape(3, -1).T
        all_intergration_weight_temp = np.vstack(
            np.meshgrid(integration_weight, integration_weight, integration_weight)).reshape(3, -1).T
        int_weight = all_intergration_weight_temp[:, 0] * \
                     all_intergration_weight_temp[:, 1] * \
                     all_intergration_weight_temp[:, 2]
        self.int_weight = torch.tensor(int_weight, device=device)
        # B = np.einsum('i,ijk->ijk', all_intergration_weight, self.matrixB(all_intergration_points))
        self.B = torch.tensor(self.matrixB(all_intergration_points), dtype=torch.float32, device=device)

    def matrixB(self, xyz):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # x, y, z = torch.unbind(xyz, -1)
        o = np.zeros_like(x)
        b = (-(0.5 - y) * (0.5 - z), o, o, (0.5 - y) * (0.5 - z), o, o,
             (0.5 - z) * (y + 0.5), o, o, -(0.5 - z) * (y + 0.5), o, o,
             -(0.5 - y) * (z + 0.5), o, o, (0.5 - y) * (z + 0.5), o, o,
             (y + 0.5) * (z + 0.5), o, o, -(y + 0.5) * (z + 0.5), o, o,
             o, -(0.5 - x) * (0.5 - z), o, o, -(0.5 - z) * (x + 0.5), o,
             o, (0.5 - z) * (x + 0.5), o, o, (0.5 - x) * (0.5 - z), o,
             o, -(0.5 - x) * (z + 0.5), o, o, -(x + 0.5) * (z + 0.5), o,
             o, (x + 0.5) * (z + 0.5), o, o, (0.5 - x) * (z + 0.5), o,
             o, o, -(0.5 - x) * (0.5 - y), o, o, -(0.5 - y) * (x + 0.5),
             o, o, -(x + 0.5) * (y + 0.5), o, o, -(0.5 - x) * (y + 0.5),
             o, o, (0.5 - x) * (0.5 - y), o, o, (0.5 - y) * (x + 0.5),
             o, o, (x + 0.5) * (y + 0.5), o, o, (0.5 - x) * (y + 0.5),
             o, -(0.5 - x) * (0.5 - y), -(0.5 - x) * (0.5 - z), o, -(0.5 - y) * (x + 0.5), -(0.5 - z) * (x + 0.5),
             o, -(x + 0.5) * (y + 0.5), (0.5 - z) * (x + 0.5), o, -(0.5 - x) * (y + 0.5), (0.5 - x) * (0.5 - z),
             o, (0.5 - x) * (0.5 - y), -(0.5 - x) * (z + 0.5), o, (0.5 - y) * (x + 0.5), -(x + 0.5) * (z + 0.5),
             o, (x + 0.5) * (y + 0.5), (x + 0.5) * (z + 0.5), o, (0.5 - x) * (y + 0.5), (0.5 - x) * (z + 0.5),
             -(0.5 - x) * (0.5 - y), o, -(0.5 - y) * (0.5 - z), -(0.5 - y) * (x + 0.5), o, (0.5 - y) * (0.5 - z),
             -(x + 0.5) * (y + 0.5), o, (0.5 - z) * (y + 0.5), -(0.5 - x) * (y + 0.5), o, -(0.5 - z) * (y + 0.5),
             (0.5 - x) * (0.5 - y), o, -(0.5 - y) * (z + 0.5), (0.5 - y) * (x + 0.5), o, (0.5 - y) * (z + 0.5),
             (x + 0.5) * (y + 0.5), o, (y + 0.5) * (z + 0.5), (0.5 - x) * (y + 0.5), o, -(y + 0.5) * (z + 0.5),
             -(0.5 - x) * (0.5 - z), -(0.5 - y) * (0.5 - z), o, -(0.5 - z) * (x + 0.5), (0.5 - y) * (0.5 - z), o,
             (0.5 - z) * (x + 0.5), (0.5 - z) * (y + 0.5), o, (0.5 - x) * (0.5 - z), -(0.5 - z) * (y + 0.5), o,
             -(0.5 - x) * (z + 0.5), -(0.5 - y) * (z + 0.5), o, -(x + 0.5) * (z + 0.5), (0.5 - y) * (z + 0.5), o,
             (x + 0.5) * (z + 0.5), (y + 0.5) * (z + 0.5), o, (0.5 - x) * (z + 0.5), -(y + 0.5) * (z + 0.5), o)
        return np.stack(b, -1).reshape((xyz.shape[0], 6, 24))

    def angle2KeExt(self, vol_ratio, rotationMatrix, stiffness_factor, density_penal=1.0):
        EF = vol_ratio * self.Ef + (1 - vol_ratio) * self.Et
        ET = 1. / (vol_ratio / self.Ef + (1 - vol_ratio) / self.Et)
        _nu_f, _nu_t = self.nuf, self.nut

        G_f = EF / (2 * (1 + _nu_f))
        G_t = ET / (2 * (1 + _nu_t))
        o = torch.zeros_like(vol_ratio)

        C = torch.stack((1 / EF, -_nu_f / EF, -_nu_f / EF, o, o, o,
                         -_nu_f / EF, 1 / ET, -_nu_t / ET, o, o, o,
                         -_nu_f / EF, -_nu_t / ET, 1 / ET, o, o, o,
                         o, o, o, 1 / G_t, o, o,
                         o, o, o, o, 1 / G_f, o,
                         o, o, o, o, o, 1 / G_f), -1).reshape(vol_ratio.shape + (6, 6))
        C_inv = torch.inverse(C)
        R = rotationMatrix
        T = torch.stack((
            R[0, 0] ** 2, R[0, 1] ** 2, R[0, 2] ** 2, 2 * R[0, 1] * R[0, 2], 2 * R[0, 0] * R[0, 2], 2 * R[0, 0] * R[0, 1],
            R[1, 0] ** 2, R[1, 1] ** 2, R[1, 2] ** 2, 2 * R[1, 1] * R[1, 2], 2 * R[1, 0] * R[1, 2], 2 * R[1, 0] * R[1, 1],
            R[2, 0] ** 2, R[2, 1] ** 2, R[2, 2] ** 2, 2 * R[2, 1] * R[2, 2], 2 * R[2, 0] * R[2, 2], 2 * R[2, 0] * R[2, 1],
            R[0, 0] * R[1, 0], R[0, 1] * R[1, 1], R[0, 2] * R[1, 2], R[0, 1] * R[1, 2] + R[0, 2] * R[1, 1], R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0], R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0],
            R[1, 0] * R[2, 0], R[1, 1] * R[2, 1], R[1, 2] * R[2, 2], R[1, 1] * R[2, 2] + R[1, 2] * R[2, 1], R[1, 0] * R[2, 2] + R[1, 2] * R[2, 0], R[1, 0] * R[2, 1] + R[1, 1] * R[2, 0],
            R[0, 0] * R[2, 0], R[0, 1] * R[2, 1], R[0, 2] * R[2, 2], R[0, 1] * R[2, 2] + R[0, 2] * R[2, 1], R[0, 0] * R[2, 2] + R[0, 2] * R[2, 0], R[0, 0] * R[2, 1] + R[0, 1] * R[2, 0],
        ), -1).reshape(6, 6)

        RC = torch.einsum('bij,jk,blk->bil', T, C_inv, T)

        weight, B = self.int_weight, self.B
        BT_C_B = torch.einsum('d,dji,bjk,dkl->bil', weight, B, RC, B)
        # `stiffness_factor` already contains the SIMP penalization
        # and the minimum stiffness ratio. Do not apply either again.
        dK = torch.einsum('i,ijk->ijk', stiffness_factor, BT_C_B)
        return dK

if __name__ == '__main__':
    K = H8_anisotropic_K()

    phi = torch.linspace(0, torch.pi * 2, 100).cuda()
    theta = torch.zeros(100).cuda()
    density = torch.zeros(100, dtype=torch.float32).cuda() + 1
    L = K.angle2Ke(phi, theta, density)
    L9 = L[9].detach().cpu().numpy()
    # map = [18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    # LMap = L[:, map, :][:, :, map]
    # L9 = LMap[9].detach().cpu().numpy()
    print('sinT:{}, cosT:{}, sinP:{}, cosP:{}'.format(math.sin(theta[9]), math.cos(theta[9]),
                                                      math.sin(phi[9]), math.cos(phi[9])))
    print(L)
    # need to check L[99] == matlab code result
