import torch
from torch import autograd as Grad
from torch.nn import functional as F
from torch.nn import Module
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class DifferentialModule(Module):

    def differentiate(self, out, wrt, allow_unused=False):

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

    

    def sphere_tangent_basis(self, wrt):
        '''Compute an orthonormal basis (with consistent orientation) for the tangent planes on the sphere.
        Input: wrt is a point/points (X,Y,Z) on the sphere.
        Output: (dg_du, dg_dv) is an orthonormal tangent frame at that point on the sphere.

        (dg_du, dg_dv) can also be thought of as a rotation matrix. It is the R matrix, in the notation of our paper
        [Neural Geometry Processing via Spherical Neural Surfaces, Williamson and Mitra 2025]'''

        X, Y, Z = wrt[:, 0], wrt[:, 1], wrt[:, 2]

        # Trick: relabel coordinates (X,Y,Z -> Y,Z,X) within a radius of the poles, so we never hit the N/S pole singularities.
        # i.e. in Atlas terminology, we use different local charts near the poles.
        mask = abs(Z) < 0.5
        x, y, z = X*mask + Y*(~mask), Y*mask + Z*(~mask), Z*mask + X*(~mask)
        

        # From the standard sphere parametrisation: ( sin u cos v, sin u sin v, cos u )
        # u (latitude) is in [0,pi], v (longitude) in [0,2pi]
        cos_u = z
        sin_u = torch.sqrt(x**2 + y**2)
        sin_v = y / sin_u
        cos_v = x / sin_u

        #Formula for the differential of the standard sphere parametrisation
        dg_du = torch.stack((cos_u*cos_v, cos_u*sin_v, -1.0*sin_u)).transpose(0, 1)
        dg_dv = torch.stack((-1.0*sin_v, cos_v, 0.0*cos_v)).transpose(0, 1)

        #Undo the co-ordinate relabelling trick
        dg_du = dg_du * mask.unsqueeze(-1) + dg_du[:, (2, 0, 1)] * (~mask.unsqueeze(-1))
        dg_dv = dg_dv * mask.unsqueeze(-1) + dg_dv[:, (2, 0, 1)] * (~mask.unsqueeze(-1))

        return dg_du, dg_dv

    def _surface_jacobian(self, out, wrt):
        '''Project the SNS Jacobian onto the sphere tangent frame.

        Returns geo_jacobian (N, 3, 2): columns are dx/du and dx/dv, the partial
        derivatives of the SNS map along the two sphere tangent directions.
        Also returns the raw sphere-space Jacobian and tangent basis vectors, which
        compute_normals and compute_SFF need for second-derivative calculations.
        '''
        jacobian = self.differentiate(out=out, wrt=wrt)
        dg_du, dg_dv = self.sphere_tangent_basis(wrt)
        dx_du = (jacobian @ dg_du.unsqueeze(-1)).squeeze(-1)
        dx_dv = (jacobian @ dg_dv.unsqueeze(-1)).squeeze(-1)
        geo_jacobian = torch.stack((dx_du, dx_dv)).permute(1, 2, 0)
        return geo_jacobian, jacobian, dg_du, dg_dv

    def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):

        if jacobian is None:
            geo_jacobian, jacobian, dg_du, dg_dv = self._surface_jacobian(out, wrt)
        else:
            dg_du, dg_dv = self.sphere_tangent_basis(wrt)
            dx_du = (jacobian @ dg_du.unsqueeze(-1)).squeeze(-1)
            dx_dv = (jacobian @ dg_dv.unsqueeze(-1)).squeeze(-1)
            geo_jacobian = torch.stack((dx_du, dx_dv)).permute(1, 2, 0)

        dx_du = geo_jacobian[..., 0]
        dx_dv = geo_jacobian[..., 1]
        cross_prod = torch.cross(dx_du, dx_dv, dim=-1)

        # normalize, except when cross products are too small to be reliable
        idx_small = cross_prod.pow(2).sum(-1) < 10.0**-7
        normals = F.normalize(cross_prod, p=2, dim=-1)
        normals[idx_small] = cross_prod[idx_small]

        if return_grad:
            return normals, geo_jacobian, jacobian, dg_du, dg_dv

        return normals

    def compute_FFF(self, geo_jacobian=None, out=None, wrt=None, return_grad=False):
        '''First Fundamental Form: E, F, G coefficients of the metric tensor.
        Does not require normals — uses _surface_jacobian directly.
        '''
        if geo_jacobian is None:
            geo_jacobian, _, _, _ = self._surface_jacobian(out, wrt)

        FFF = geo_jacobian.transpose(1, 2) @ geo_jacobian

        I_E = FFF[:, 0, 0]
        I_F = FFF[:, 0, 1]
        I_G = FFF[:, 1, 1]

        if return_grad:
            return I_E, I_F, I_G, geo_jacobian
        return I_E, I_F, I_G

    def compute_SFF(self, out=None, wrt=None, return_grad=False, return_normals=False):
        normals, geo_jacobian, _, dg_da, dg_db = self.compute_normals(out=out, wrt=wrt, return_grad=True)

        #Think of a and b as being co-ordinates on the tangent plane of the sphere.
        # ra, rb are dr_da and dr_db; raa, rab, rbb are their second derivatives
        ra = geo_jacobian[..., 0]
        rb = geo_jacobian[..., 1]

        ra_deriv = self.differentiate(out=ra, wrt=wrt)
        rb_deriv = self.differentiate(out=rb, wrt=wrt)

        raa = (ra_deriv @ (dg_da.unsqueeze(1).transpose(1, 2))).squeeze()
        rab = (ra_deriv @ (dg_db.unsqueeze(1).transpose(1, 2))).squeeze()
        rbb = (rb_deriv @ (dg_db.unsqueeze(1).transpose(1, 2))).squeeze()

        L = (raa * normals).sum(-1)
        M = (rab * normals).sum(-1)
        N = (rbb * normals).sum(-1)

        if return_normals and not return_grad:
            return L, M, N, normals
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
        I_E, I_F, I_G, geo_jacobian = self.compute_FFF(out=out, wrt=wrt, return_grad=True)
        L, M, N, normals = self.compute_SFF(out=out, wrt=wrt, return_normals=True)

        if prevent_nans:
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)

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
                x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)

            dir1 = torch.stack((torch.ones_like(x1), x1)).T
            dir1 = (dir1[:, 0].T*e1.T + dir1[:, 1].T*e2.T).T
            dir1 = F.normalize(dir1, p=2, dim=1)

            x2 = (k2*I_E - L) / (M - k2*I_F)
            if prevent_nans:
                x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)

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


    def laplace_beltrami_MC(self, normals, meancurv, f, grad_f=None, hessian_f=None):
        '''Mean-curvature form of Laplace Beltrami operator. See the paper
        [Neural Geometry Processing via Spherical Neural Surfaces, Williamson and Mitra 2025].
        Input: a scalarfield f on R3, its gradient and its hessian,
                normals and mean curvature of the surface.

        Output: Laplace Beltrami operator of the surface, applied to the scalarfield f restricted to the surface.
        LB(f) = Delta(f) - 2H grad(f).n - nT Hess(f) n.'''

        divgrad = hessian_f[:, 0, 0] + hessian_f[:, 1, 1] + hessian_f[:, 2, 2]
        meancurv_term = -2 * meancurv * ((grad_f * normals).sum(-1))
        hessian_term = -1 * (normals.unsqueeze(1) @ hessian_f @ normals.unsqueeze(2)).squeeze()
        LB_f = divgrad + meancurv_term + hessian_term

        return LB_f
    

    def laplace_beltrami_divgrad(self, out=None, wrt=None, f=None, f_defined_on_sphere=False):
        '''Another way to compute the Laplace Beltrami operator.
        Computes LBO on a function f defined on the surface, as the surface divergence of the surface gradient of f.'''

        normals, _, jacobian3D, _, _ = self.compute_normals(out=out, wrt=wrt, return_grad=True)
        inv_jacobian3D = torch.linalg.inv(jacobian3D)

        if not f_defined_on_sphere:
            df = self.differentiate(out=f.unsqueeze(-1), wrt=out).squeeze()
        else:
            df = (self.differentiate(out=f.unsqueeze(-1), wrt=wrt).squeeze().unsqueeze(1) @ inv_jacobian3D).squeeze()

        # Surface gradient: euclidean gradient minus its normal component
        surf_grad = df - torch.sum(df * normals, axis=1).unsqueeze(-1) * normals

        d_surf_grad = self.differentiate(out=surf_grad, wrt=wrt) @ inv_jacobian3D
        div_surf_grad = d_surf_grad[:, 0, 0] + d_surf_grad[:, 1, 1] + d_surf_grad[:, 2, 2]

        # Remove the component of the divergence due to variation in the normal direction
        normals_term = (normals.unsqueeze(1) @ d_surf_grad @ normals.unsqueeze(2)).squeeze()
        LB_f = div_surf_grad - normals_term

        return LB_f
