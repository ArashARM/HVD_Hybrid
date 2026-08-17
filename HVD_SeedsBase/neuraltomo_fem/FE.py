import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# from sksparse.cholmod import cholesky

import torch
from neuraltomo_fem.gridMesher import GridMesh


from neuraltomo_fem.Km2Compliance import *
#from anisotropicFE import angle2Ke
from neuraltomo_fem.anisotropicFE_new import H8_anisotropic_K
# from quadMesher import QuadMesh
# -----------------------#

class FE:
    # -----------------------#
    def __init__(self, problem, device='cuda'):
        if problem.mesh['type'] == 'grid':


            self.mesh = GridMesh(problem)
        # elif(mesh['type'] == 'quad'):
        #    self.mesh = QuadMesh(mesh, matProp, bc)
        self.init_Matrix_idx(device)


        if type(problem.materialProperty) is dict:
            self.H8 = H8_anisotropic_K(device, element_size=self.mesh.elemSize, **problem.materialProperty)
        else:
            self.H8 = H8_anisotropic_K(device, element_size=self.mesh.elemSize, **problem.materialProperty.__dict__)

    # This function precomputes the indices for assembling the global stiffness matrix K from the element stiffness matrices. 
    # It also prepares the force vector f for the free degrees of freedom. 
    # The indices are filtered to exclude fixed degrees of freedom, and a mapping is created to convert global DOF indices to the reduced system indices used in the linear solve.
    def init_Matrix_idx(self, device):
        """
        Precompute matrix assembly indices for a structured 3D hexahedral mesh.

        Creates:
        self.edofMat : (nele, 24) tensor
            Global DOF indices for each element (8 nodes * 3 dof/node)
        self.iK : (nele*24*24,) tensor
            Row indices for sparse stiffness assembly
        self.jK : (nele*24*24,) tensor
            Column indices for sparse stiffness assembly
        """
        self.Ksize = self.mesh.free.flatten().shape[0]
        row_indices_np = self.mesh.active_iK  # NumPy array of active row indices
        col_indices_np = self.mesh.active_jK  # NumPy array of active column indices'

        # keep_index = np.delete(np.arange(0, self.mesh.ndof, dtype=int), self.mesh.fixed)
        keep_index = self.mesh.free
        mask = np.isin(row_indices_np, keep_index) & np.isin(col_indices_np, keep_index)

        filtered_row_indices = row_indices_np[mask].astype(int)
        filtered_col_indices = col_indices_np[mask].astype(int)
        self.valid_mask = torch.from_numpy(mask).bool().to(device)

        ## ** Need to make sure Array not out of bounds ** ##
        indexMap = np.full(self.mesh.ndof, -1, dtype=int)
        indexMap[keep_index] = np.arange(0, self.Ksize, dtype=int)

        # Map filtered indices to new indices in keep_index
        self.new_row_indices = indexMap[filtered_row_indices]
        self.new_col_indices = indexMap[filtered_col_indices]

        self.f = torch.tensor(self.mesh.f[keep_index, 0], dtype=torch.float32, device=device)


    def solve_c_new(self, phi, theta, stiffness_factor, penal=1.0, isotropic=False):
        # self.u = torch.zeros((self.mesh.ndof, 1), device=density.device)
        active_ids = torch.as_tensor(self.mesh.active_element_ids, dtype=torch.long, device=stiffness_factor.device)
        if isotropic:
            ## isotropic
            E = self.mesh.Emax * stiffness_factor[active_ids]
            KE = torch.tensor(self.mesh.KE[self.mesh.active_element_ids], dtype=torch.float32, device=stiffness_factor.device)
            sK = torch.einsum('i,ijk->ijk', E, KE).flatten()
        else:
            ## anisotropic
            sK = self.H8.angle2Ke(phi[active_ids], theta[active_ids], stiffness_factor[active_ids], penal).flatten()

        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        c = sk2c(d, (self.new_row_indices, self.new_col_indices), f, self.f, self.Ksize)
        return c

    def solve_stress_new(self, phi, theta, stiffness_factor, penal=1.0, isotropic=False):
        self.u = torch.zeros((self.mesh.ndof, 1), dtype=torch.float32, device=stiffness_factor.device)
        active_ids = torch.as_tensor(self.mesh.active_element_ids, dtype=torch.long, device=stiffness_factor.device)
        if isotropic:
            ## isotropic
            E = self.mesh.Emax * stiffness_factor[active_ids]
            KE = torch.tensor(self.mesh.KE[self.mesh.active_element_ids], dtype=torch.float32, device=stiffness_factor.device)
            sK = torch.einsum('i,ijk->ijk', E, KE).flatten()
            B = torch.tensor(self.mesh.B.T, dtype=torch.float32, device=stiffness_factor.device).T
            C = torch.tensor(self.mesh.C, dtype=torch.float32, device=stiffness_factor.device).expand(self.mesh.numElems,-1,-1)
        else:
            ## anisotropic
            sK = self.H8.angle2Ke(phi[active_ids], theta[active_ids], stiffness_factor[active_ids], penal).flatten()

            B = self.H8.NodeB
            C_active = self.H8.temp_C
            T_active = self.H8.T

        #i = self.sparseKIdx
        # selects only the valid entries of sK that correspond to the free DOFs, effectively removing contributions from fixed DOFs.
        #  This is crucial for correctly assembling the reduced stiffness matrix used in the linear solve.
        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        u = sk2u(d, (self.new_row_indices, self.new_col_indices), f, self.f, self.Ksize)
        self.u[self.mesh.free, 0] = u
        c = (self.f*u).sum()
        uElem = self.u[self.mesh.edofMat].reshape(self.mesh.numElems, self.mesh.numDOFPerElem)
        uElem_active = uElem[active_ids]
        uElemNodes = uElem.reshape(self.mesh.numElems, 8, 3)
        uElemNodes_active = uElemNodes[active_ids]
        disp_mag_active = torch.linalg.norm(uElemNodes_active, dim=2).max(dim=1).values
        disp_mag_elem = torch.zeros((self.mesh.numElems,), dtype=torch.float32, device=stiffness_factor.device)
        disp_mag_elem[active_ids] = disp_mag_active
        force_vec = torch.as_tensor(self.mesh.f[:, 0], dtype=torch.float32, device=stiffness_factor.device).reshape(self.mesh.numNodes, 3)
        uNodes = self.u.reshape(self.mesh.numNodes, 3)
        node_disp_mag = torch.linalg.norm(uNodes, dim=1)
        force_norm = torch.linalg.norm(force_vec, dim=1)
        loaded_node_mask = force_norm > 1e-12
        if torch.any(loaded_node_mask):
            loaded_force_dir = force_vec[loaded_node_mask] / force_norm[loaded_node_mask, None]
            loaded_disp_load_dir = torch.abs(torch.sum(uNodes[loaded_node_mask] * loaded_force_dir, dim=1))
            loaded_disp_mag = node_disp_mag[loaded_node_mask]
        else:
            loaded_disp_load_dir = torch.empty(0, dtype=torch.float32, device=stiffness_factor.device)
            loaded_disp_mag = torch.empty(0, dtype=torch.float32, device=stiffness_factor.device)
        load_dir = force_vec.sum(dim=0)
        load_dir_norm = torch.linalg.norm(load_dir)
        if load_dir_norm <= 1e-12:
            load_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=stiffness_factor.device)
        else:
            load_dir = load_dir / load_dir_norm
        disp_load_dir_active = torch.abs(torch.einsum('eij,j->ei', uElemNodes_active, load_dir)).mean(dim=1)
        disp_load_dir_elem = torch.zeros((self.mesh.numElems,), dtype=torch.float32, device=stiffness_factor.device)
        disp_load_dir_elem[active_ids] = disp_load_dir_active
        if isotropic:
            C_active = C[active_ids]
            T_active = None
        sigma_active = torch.einsum('bij,jk,bk -> bi', C_active, B, uElem_active)
        sigmaElem = torch.zeros((self.mesh.numElems, 6), dtype=torch.float32, device=stiffness_factor.device)
        sigmaElem[active_ids] = sigma_active
        sigma_for_vm = sigmaElem
        sxx, syy, szz = sigma_for_vm[:, 0], sigma_for_vm[:, 1], sigma_for_vm[:, 2]
        syz, sxz, sxy = sigma_for_vm[:, 3], sigma_for_vm[:, 4], sigma_for_vm[:, 5]
        stress_vm = torch.sqrt(
            torch.clamp(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2),
                min=0.0,
            )
        )
        self.displacement_mag_elem = disp_mag_elem
        self.displacement_mag_loaded_boundary = loaded_disp_mag
        self.displacement_load_dir_elem = disp_load_dir_elem
        self.displacement_load_dir_loaded_boundary = loaded_disp_mag
        self.displacement_load_dir_loaded_boundary_projection = loaded_disp_load_dir
        self.sigmaElem = sigmaElem
        self.stress_vm = stress_vm
        
        # sigmaElem = torch.einsum('bij,jk,bk,bim -> bm', C, B, uElem, T)

        P, Q = self.H8.P, self.H8.Q
        _A_active = 0.5 * torch.einsum('ij, jk, ik ->i', sigma_active, P, sigma_active)
        _B_active = torch.einsum('i, ji ->j', Q, sigma_active)
        _A = torch.zeros((self.mesh.numElems,), dtype=torch.float32, device=stiffness_factor.device)
        _B = torch.zeros((self.mesh.numElems,), dtype=torch.float32, device=stiffness_factor.device)
        _A[active_ids] = _A_active
        _B[active_ids] = _B_active
        _C = -1

        root_active = torch.zeros_like(_A_active)
        is_linear = _A_active < 1e-5

        # the max Force that element can retain
        BL = _B_active[is_linear]
        BNL = _B_active[~is_linear]
        ANL = _A_active[~is_linear]
        root_active[is_linear] = 1 / (BL.abs() + 1e-5)
        root_active[~is_linear] = (-BNL + (BNL * BNL - 4 * ANL * _C).sqrt()) / (2 * ANL)
        root = torch.zeros_like(_A)
        root[active_ids] = root_active


        Fmin = torch.linalg.vector_norm(root_active.abs() + 1e-5, ord=-6) # + 1e-5*root.mean()
        # Fmin = torch.linalg.vector_norm(root.abs(), ord=-10)
        FRealMin = root_active.min()
        Criterion = _A * FRealMin * FRealMin + _B * FRealMin + _C

        self.FRealMin = FRealMin
        self.Criterion = Criterion
        self.stress_root = root

        if torch.isnan(Fmin):
            print('Fmin is NAN!')
        return Fmin, c

    def solve(self, density):
        self.u=np.zeros((self.mesh.ndof,1))
        active_ids = self.mesh.active_element_ids
        E = self.mesh.material['E'] * np.asarray(density).reshape(-1)[active_ids]
        sK = np.einsum('i,ijk->ijk', E, self.mesh.KE[active_ids]).flatten()

        K = coo_matrix((sK, (self.mesh.active_iK, self.mesh.active_jK)), shape=(self.mesh.ndof, self.mesh.ndof)).tocsc()
        K = K[self.mesh.free, :][:, self.mesh.free].tocsc()

        B = self.mesh.f[self.mesh.free,0]
        B = scipy.sparse.linalg.spsolve(K, B)
        self.u[self.mesh.free,0]=np.array(B)
        uElem = self.u[self.mesh.edofMat].reshape(self.mesh.numElems,self.mesh.numDOFPerElem)
        self.Jelem = np.zeros((self.mesh.numElems,), dtype=float)
        self.Jelem[active_ids] = np.einsum(
            'ik,ik->i',
            np.einsum('ij,ijk->ik', uElem[active_ids], self.mesh.KE[active_ids]),
            uElem[active_ids],
        )
        return self.u, self.Jelem
