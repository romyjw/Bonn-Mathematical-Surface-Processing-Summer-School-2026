import torch
from torch import autograd as Grad
from torch.nn import functional as F
from torch.nn import Module
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class DifferentialModule(Module):

    def gradient(self, out, wrt, allow_unused=False):

        B = 1 if len(out.size()) < 3 else out.size(0)
        N = out.size(0) if len(out.size()) < 3 else out.size(1)
        R = out.size(-1)
        C = wrt.size(-1)

        gradients = []
        for dim in range(R):
            out_p = out[..., dim].flatten()
            select = torch.ones(out_p.size(), dtype=torch.float32).to(out.device)
            gradient = Grad.grad(outputs=out_p, inputs=wrt, grad_outputs=select, create_graph=True, allow_unused=allow_unused)[0]
            gradients.append(gradient)

        if len(out.size()) < 3:
            J_f_uv = torch.cat(gradients, dim=1).view(N, R, C)
        else:
            J_f_uv = torch.cat(gradients, dim=2).view(B, N, R, C)

        return J_f_uv

    def backprop(self, out, wrt):
        select = torch.ones(out.size(), dtype=torch.float32).to(out.device)
        J = Grad.grad(outputs=out, inputs=wrt, grad_outputs=select, create_graph=True)[0]
        J = J.view(wrt.size())
        return J

    def parametrise(self, wrt):
        '''Compute an orthonormal basis (with consistent orientation) for the tangent planes on the sphere.'''

        X, Y, Z = wrt[:, 0], wrt[:, 1], wrt[:, 2]
        # Relabel coordinates near the poles so we never hit N/S pole singularities
        mask = abs(Z) < 0.5
        x, y, z = X*mask + Y*(~mask), Y*mask + Z*(~mask), Z*mask + X*(~mask)

        # sphere parametrisation: ( sin u cos v, sin u sin v, cos u ), u in [0,2pi], v in [0,pi]
        cos_u = z
        sin_u = torch.sqrt(x**2 + y**2)
        sin_v = y / sin_u
        cos_v = x / sin_u

        # (dg_du, dg_dv) is the orthonormal R matrix in the SNS paper notation
        dg_du = torch.stack((cos_u*cos_v, cos_u*sin_v, -1.0*sin_u)).transpose(0, 1)
        dg_dv = torch.stack((-1.0*sin_v, cos_v, 0.0*cos_v)).transpose(0, 1)

        dg_du = dg_du * mask.unsqueeze(-1) + dg_du[:, (2, 0, 1)] * (~mask.unsqueeze(-1))
        dg_dv = dg_dv * mask.unsqueeze(-1) + dg_dv[:, (2, 0, 1)] * (~mask.unsqueeze(-1))

        return dg_du, dg_dv

    def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):

        if jacobian is None:
            jacobian = self.gradient(out=out, wrt=wrt)

        dg_du, dg_dv = self.parametrise(wrt)

        dx_du = (jacobian @ dg_du.unsqueeze(-1)).squeeze(-1)
        dx_dv = (jacobian @ dg_dv.unsqueeze(-1)).squeeze(-1)

        cross_prod = torch.cross(dx_du, dx_dv, dim=-1)

        # normalize, except when cross products are too small to be reliable
        idx_small = cross_prod.pow(2).sum(-1) < 10.0**-7
        normals = F.normalize(cross_prod, p=2, dim=-1)
        normals[idx_small] = cross_prod[idx_small]

        geo_jacobian = torch.stack((dx_du, dx_dv)).permute(1, 2, 0)

        if return_grad:
            return normals, geo_jacobian, jacobian, dg_du, dg_dv

        return normals

    def compute_FFF(self, geo_jacobian=None, out=None, wrt=None, return_grad=False):

        if geo_jacobian is None:
            normals, geo_jacobian, _, _, _ = self.compute_normals(out=out, wrt=wrt, return_grad=True)

        FFF = geo_jacobian.transpose(1, 2) @ geo_jacobian

        I_E = FFF[:, 0, 0]
        I_F = FFF[:, 0, 1]
        I_G = FFF[:, 1, 1]

        if return_grad:
            return I_E, I_F, I_G, geo_jacobian, normals
        return I_E, I_F, I_G

    def compute_SFF(self, out=None, wrt=None, return_grad=False, return_normals=False):
        normals, geo_jacobian, _, dg_da, dg_db = self.compute_normals(out=out, wrt=wrt, return_grad=True)

        # ra, rb are dr_da and dr_db; raa, rab, rbb are their second derivatives
        ra = geo_jacobian[..., 0]
        rb = geo_jacobian[..., 1]

        ra_deriv = self.gradient(out=ra, wrt=wrt)
        rb_deriv = self.gradient(out=rb, wrt=wrt)

        raa = (ra_deriv @ (dg_da.unsqueeze(1).transpose(1, 2))).squeeze()
        rab = (ra_deriv @ (dg_db.unsqueeze(1).transpose(1, 2))).squeeze()
        rbb = (rb_deriv @ (dg_db.unsqueeze(1).transpose(1, 2))).squeeze()

        L = (raa * normals).sum(-1)
        M = (rab * normals).sum(-1)
        N = (rbb * normals).sum(-1)

        if return_grad and return_normals:
            return L, M, N, geo_jacobian, normals
        if return_grad:
            return L, M, N, geo_jacobian
        return L, M, N

    def compute_area_distortion(self, out=None, wrt=None):
        '''Compute local area distortion from the sphere to the surface, using FFF.'''
        E, F, G = self.compute_FFF(out=out, wrt=wrt)
        distortion = torch.sqrt(E*G - F.pow(2))
        return distortion

    def compute_curvature(self, out=None, wrt=None, compute_principal_directions=False, prevent_nans=True):
        I_E, I_F, I_G, geo_jacobian, normals = self.compute_FFF(geo_jacobian=None, out=out, wrt=wrt, return_grad=True)
        L, M, N = self.compute_SFF(out=out, wrt=wrt, return_grad=False, return_normals=False)

        if prevent_nans:
            big = 10000000000 * torch.ones_like(L, dtype=L.dtype)
            L = L * (torch.abs(L) < big)
            M = M * (torch.abs(M) < big)
            N = N * (torch.abs(N) < big)

        # Principal curvatures are eigenvalues of (FFFinv)(SFF).
        # Coefficients of the characteristic equation det(SFF - k*FFF) = 0:
        A = (I_E * I_G - I_F.pow(2))
        B = 2 * M * I_F - (I_E * N + I_G * L)
        C = (L * N - M.pow(2))

        H = -B / (2.0 * A)
        K = C / A
        H *= -1  # flip sign to match graphics convention

        if compute_principal_directions:
            k1 = (-B - torch.sqrt(B**2 - 4*A*C)) / (2.0*A)
            k2 = (-B + torch.sqrt(B**2 - 4*A*C)) / (2.0*A)

            e1 = geo_jacobian[:, :, 0]
            e2 = geo_jacobian[:, :, 1]

            x1 = (k1*I_E - L) / (M - k1*I_F)
            if prevent_nans:
                big = 10000000 * torch.ones_like(x1)
                x1 = torch.where(torch.abs(x1) < big, x1, 0.0)

            dir1 = torch.stack((torch.ones_like(x1), x1)).T
            dir1 = (dir1[:, 0].T*e1.T + dir1[:, 1].T*e2.T).T
            dir1 = F.normalize(dir1, p=2, dim=1)

            x2 = (k2*I_E - L) / (M - k2*I_F)
            if prevent_nans:
                big = 10000000 * torch.ones_like(x1)
                x2 = torch.where(torch.abs(x2) < big, x2, 0.0)

            dir2 = torch.stack((torch.ones_like(x2), x2)).T
            dir2 = (dir2[:, 0].T*e1.T + dir2[:, 1].T*e2.T).T
            dir2 = F.normalize(dir2, p=2, dim=1)

            frame = torch.stack([dir1, dir2, normals]).transpose(0, 1)
            signs = torch.linalg.det(frame)
            dir1 = (dir1.T * signs.T).T

            abs_k1 = abs(k1)
            abs_k2 = abs(k2)
            MAC  = abs_k1 * (abs_k1 >= abs_k2) + abs_k2 * (abs_k1 < abs_k2)
            SMAC = k1    * (abs_k1 >= abs_k2) + k2    * (abs_k1 < abs_k2)

            return H, K, MAC, SMAC, (dir1, dir2), (k1, k2), normals

        return H, K, None, None, None, None, None

    def laplace_beltrami_divgrad(self, out=None, wrt=None, f=None, f_defined_on_sphere=False):
        '''Computes LBO on a function f defined on the surface, as the surface divergence of the surface gradient of f.'''

        normals, _, jacobian3D, _, _ = self.compute_normals(out=out, wrt=wrt, return_grad=True)
        inv_jacobian3D = torch.linalg.inv(jacobian3D)

        if not f_defined_on_sphere:
            df = self.gradient(out=f.unsqueeze(-1), wrt=out).squeeze()
        else:
            df = (self.gradient(out=f.unsqueeze(-1), wrt=wrt).squeeze().unsqueeze(1) @ inv_jacobian3D).squeeze()

        # Surface gradient: euclidean gradient minus its normal component
        surf_grad = df - torch.sum(df * normals, axis=1).unsqueeze(-1) * normals

        d_surf_grad = self.gradient(out=surf_grad, wrt=wrt) @ inv_jacobian3D
        div_surf_grad = d_surf_grad[:, 0, 0] + d_surf_grad[:, 1, 1] + d_surf_grad[:, 2, 2]

        # Remove the component of the divergence due to variation in the normal direction
        normals_term = (normals.unsqueeze(1) @ d_surf_grad @ normals.unsqueeze(2)).squeeze()
        LB_f = div_surf_grad - normals_term

        return LB_f

    def laplace_beltrami_MC(self, normals, meancurv, f, grad_f=None, hessian_f=None):
        '''Computes LB(f) = Delta(f) - 2H grad(f).n - nT Hess(f) n.'''

        divgrad = hessian_f[:, 0, 0] + hessian_f[:, 1, 1] + hessian_f[:, 2, 2]
        meancurv_term = -2 * meancurv * ((grad_f * normals).sum(-1))
        hessian_term = -1 * (normals.unsqueeze(1) @ hessian_f @ normals.unsqueeze(2)).squeeze()
        LB_f = divgrad + meancurv_term + hessian_term

        return LB_f
