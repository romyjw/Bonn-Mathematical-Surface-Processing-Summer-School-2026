import logging
import numpy as np
import torch
from math import ceil

logger = logging.getLogger(__name__)


class BatchDiffQuant:
    def __init__(self, vertices, model, diffmod, batch_size, scaling=1.0):
        n = vertices.shape[0]

        self.vertices = vertices
        self.model = model
        self.diffmod = diffmod
        self.batch_size = batch_size

        self.num_batches = ceil(n / batch_size)
        self.all_output_vertices = np.zeros_like(vertices)
        self.all_normals = np.zeros_like(vertices)
        self.all_H = np.zeros(n)
        self.all_K = np.zeros(n)
        self.all_directions = [np.zeros((n, 3)), np.zeros((n, 3))]
        self.all_MAC = np.zeros(n)
        self.all_k1 = np.zeros(n)
        self.all_k2 = np.zeros(n)
        self.all_distortions = np.zeros(n)
        self.all_principals = [np.zeros(n), np.zeros(n)]
        self.all_beltrami_H = np.zeros(n)
        self.all_area_distortions = np.zeros(n)
        self.all_beltrami_on_X = np.zeros((n, 3))
        self.all_beltrami_on_scalar = np.zeros(n)
        self.n = n
        self.scaling = scaling

    @staticmethod
    def _to_np(t):
        return t.detach().numpy().copy()

    def _get_batch(self, i, requires_grad=True):
        logger.info('Computing batch %d of %d (%d vertices total)', i + 1, self.num_batches, self.n)
        start = self.batch_size * i
        end = min(self.batch_size * (i + 1), self.n)
        tensorvertices = torch.Tensor(self.vertices[start:end, :])
        if requires_grad:
            tensorvertices.requires_grad = True
        output_vertices = self.scaling * self.model.forward(tensorvertices)
        return tensorvertices, output_vertices, start, end

    def _compute_beltrami_vector(self, output_vertices, tensorvertices):
        bx = self.diffmod.laplace_beltrami_divgrad(out=output_vertices, wrt=tensorvertices, f=output_vertices[:, 0])
        by = self.diffmod.laplace_beltrami_divgrad(out=output_vertices, wrt=tensorvertices, f=output_vertices[:, 1])
        bz = self.diffmod.laplace_beltrami_divgrad(out=output_vertices, wrt=tensorvertices, f=output_vertices[:, 2])
        return torch.stack((bx, by, bz)).transpose(0, 1)

    def compute_SNS(self):
        for i in range(self.num_batches):
            _, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)

    def compute_normals(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            normals = self.diffmod.compute_normals(out=output_vertices, wrt=tensorvertices)
            self.all_normals[start:end, :] = self._to_np(normals)

    def compute_area_distortions(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            area_distortion = self.diffmod.compute_area_distortion(out=output_vertices, wrt=tensorvertices)
            self.all_area_distortions[start:end] = self._to_np(area_distortion)

    def compute_curvature(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            H, K, _, _, _, _, _ = self.diffmod.compute_curvature(
                out=output_vertices, wrt=tensorvertices,
                compute_principal_directions=False, prevent_nans=False,
            )
            self.all_H[start:end] = self._to_np(H)
            self.all_K[start:end] = self._to_np(K)

    def compute_directions(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            H, K, MAC, _, (dir1, dir2), (k1, k2), normals = self.diffmod.compute_curvature(
                out=output_vertices, wrt=tensorvertices,
                compute_principal_directions=True, prevent_nans=False,
            )
            self.all_H[start:end] = self._to_np(H)
            self.all_K[start:end] = self._to_np(K)
            self.all_MAC[start:end] = self._to_np(MAC)
            self.all_k1[start:end] = self._to_np(k1)
            self.all_k2[start:end] = self._to_np(k2)
            self.all_normals[start:end, :] = self._to_np(normals)
            self.all_directions[0][start:end] = self._to_np(dir1)
            self.all_directions[1][start:end] = self._to_np(dir2)

    def compute_beltrami_on_X(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            beltrami_result = self._compute_beltrami_vector(output_vertices, tensorvertices)
            self.all_beltrami_on_X[start:end] = self._to_np(beltrami_result)

    def compute_beltrami_on_scalar(self, f):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            beltrami_result = self.diffmod.laplace_beltrami_divgrad(
                out=output_vertices, wrt=tensorvertices, f=f(output_vertices),
            )
            self.all_beltrami_on_scalar[start:end] = self._to_np(beltrami_result)

    def compute_beltrami_H(self):
        for i in range(self.num_batches):
            tensorvertices, output_vertices, start, end = self._get_batch(i)
            self.all_output_vertices[start:end, :] = self._to_np(output_vertices)
            normals = self.diffmod.compute_normals(out=output_vertices, wrt=tensorvertices)
            beltrami_result = self._compute_beltrami_vector(output_vertices, tensorvertices)
            sign = -1 * torch.sign((beltrami_result * normals).sum(-1))
            self.all_beltrami_H[start:end] = self._to_np(0.5 * sign * torch.linalg.norm(beltrami_result, dim=1))
