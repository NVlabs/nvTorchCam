# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
from nvtorchcam import utils


class TestCoords(unittest.TestCase):
    def setUp(self):
        pass

    # @unittest.skip('Uncomment to skip this test')
    def test_in_image(self):
        pix = torch.tensor([[-1.0,-1.0],
                     [-1.0,1.0],
                     [1.0,-1.0],
                     [1.0,1.0],
                     [0.0,0.0],
                     [1.0,0.0],
                     [0.0,-1.0]])
        

        valid = utils.in_image(pix)
        expected = torch.tensor([False, False, False, False,  True, False, False])
        torch.testing.assert_close(valid, expected)

    # @unittest.skip('Uncomment to skip this test')
    def test_get_normalized_grid(self):
        device = 'cpu'
        grid = utils.get_normalized_grid((3, 6), device=device)
        gt_grid_x = torch.tensor([[-0.8333, -0.5000, -0.1667,  0.1667,  0.5000,  0.8333],
                                  [-0.8333, -0.5000, -0.1667,
                                      0.1667,  0.5000,  0.8333],
                                  [-0.8333, -0.5000, -0.1667,  0.1667,  0.5000,  0.8333]])
        gt_grid_y = torch. tensor([[-0.6667, -0.6667, -0.6667, -0.6667, -0.6667, -0.6667],
                                   [0.0000,  0.0000,  0.0000,
                                       0.0000,  0.0000,  0.0000],
                                   [0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667]])

        gt_grid = torch.stack((gt_grid_x, gt_grid_y), dim=-1)
        torch.testing.assert_close(grid, gt_grid, atol=1e-4, rtol=1e-3)

    # @unittest.skip('Uncomment to skip this test')

    def test_samples_from_image(self):
        # Have both integer location (and make sure it's the value of the pixel) and locations to be interpolated.
        device = 'cpu'
        c, h, w = 3, 10, 20
        image = torch.linspace(0, 1, c*h*w).reshape(c, h, w)
        grid = utils.get_normalized_grid(image.shape[1:3], device=device)
        recon_image = utils.samples_from_image(image, grid)
        torch.testing.assert_close(image, recon_image)

    # @unittest.skip('Uncomment to skip this test')
    def test_samples_from_cubemap(self):
        device = 'cuda'
        b, c, w = 2, 3, 10
        cubemap = torch.linspace(
            0, 1, b*c*6*w*w, device=device).reshape(b, c, 6*w, w)
        grid = utils.get_normalized_grid_cubemap(w, device=device)
        recon_cubemap = utils.samples_from_cubemap(
            cubemap, grid.unsqueeze(0).expand(b, -1, -1, -1))
        torch.testing.assert_close(cubemap, recon_cubemap)

    # @unittest.skip('Uncomment to skip this test')
    def test_mat_multiply(self):
        # single matrix vector
        A = torch.randn(3, 3)
        x = torch.randn(3)
        Ax = utils.apply_matrix(A, x)
        Ax_torch = torch.mm(A, x.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(Ax, Ax_torch)

        # many vectors times a single matrix
        A = torch.randn(3, 3)
        x = torch.randn(10, 3)
        Ax = utils.apply_matrix(A, x)
        Ax_torch = torch.mm(A, x.transpose(0, 1)).transpose(0, 1)
        torch.testing.assert_close(Ax, Ax_torch)

        # batch of matrices times batch of single vectors
        A = torch.randn(2, 4, 4)
        x = torch.randn(2, 4)
        Ax = utils.apply_matrix(A, x)
        Ax_torch = torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        torch.testing.assert_close(Ax, Ax_torch)

        # double batch times double group
        A = torch.randn(2, 5, 3, 3)  # *b = (2,5)
        x = torch.randn(2, 5, 10, 12, 3)  # *g = (10,12)
        Ax = utils.apply_matrix(A, x)
        Ax_torch = torch.bmm(A.reshape(2*5, 3, 3), x.reshape(2*5, 10*12,
                             3).transpose(1, 2)).transpose(1, 2).reshape(2, 5, 10, 12, 3)
        torch.testing.assert_close(Ax, Ax_torch)

    # @unittest.skip('Uncomment to skip this test')
    def test_apply_affine(self):
        A = torch.arange(0, 16).reshape(4, 4).float()
        A[3, :] = torch.tensor([0, 0, 0, 1.0])
        x = torch.tensor([1, 2, 1.0])
        Ax = utils.apply_affine(A, x)
        Ax_expected = torch.tensor([7., 27., 47.])
        torch.testing.assert_close(Ax, Ax_expected)

    # @unittest.skip('Uncomment to skip this test')
    def test_normalized_unnormalize_pts(self):
        device = 'cpu'
        gt_unnorm_grid = torch.tensor([[[0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000],
                                        [0.5000, 1.5000, 2.5000,
                                            3.5000, 4.5000, 5.5000],
                                        [0.5000, 1.5000, 2.5000,
                                            3.5000, 4.5000, 5.5000],
                                        [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000]],

                                       [[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                                        [1.5000, 1.5000, 1.5000,
                                            1.5000, 1.5000, 1.5000],
                                        [2.5000, 2.5000, 2.5000,
                                            2.5000, 2.5000, 2.5000],
                                        [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000]]]).permute(1, 2, 0)
        hw = (4, 6)
        grid = utils.get_normalized_grid(hw, device=device)
        unnorm_grid = utils.pixel_pts_from_normalized_pts(grid, hw)
        torch.testing.assert_close(unnorm_grid, gt_unnorm_grid)
        recon_grid = utils.normalized_pts_from_pixel_pts(unnorm_grid, hw)
        torch.testing.assert_close(grid, recon_grid)

    # @unittest.skip('Uncomment to skip this test')
    def test_normalized_unnormalize_intrinsics(self):
        device = 'cpu'
        hw = (4, 6)
        pts = torch.randn(10, 2)
        n_intrinsics = torch.tensor([1.2, 1.1, 0.3, 0.2])
        intrinsics = utils.pixel_intrinsics_from_normalized_intrinsics(
            n_intrinsics, hw)

        pix = pts[:, :2]*intrinsics[None, :2] + intrinsics[None, 2:]
        n_pix = pts[:, :2]*n_intrinsics[None, :2] + n_intrinsics[None, 2:]
        other_n_pix = utils.normalized_pts_from_pixel_pts(pix, hw)
        torch.testing.assert_close(n_pix, other_n_pix)

        recon_n_intrinsics = utils.normalized_intrinsics_from_pixel_intrinsics(
            intrinsics, hw)
        torch.testing.assert_close(n_intrinsics, recon_n_intrinsics)

    # @unittest.skip('Uncomment to skip this test')
    def test_flat_intrinsics_from_matrix(self):
        gt_intrinsics_matrix = torch.tensor([[[1.2000, 0.0000, 0.3000],
                                              [0.0000, 1.1000, 0.2000],
                                              [0.0000, 0.0000, 1.0000]],

                                             [[2.1000, 0.0000, 0.3000],
                                              [0.0000, 1.5000, 0.4000],
                                              [0.0000, 0.0000, 1.0000]]])
        gt_flat_intrinsics = torch.tensor(
            [[1.2, 1.1, 0.3, 0.2], [2.1, 1.5, 0.3, 0.4]])
        intrinsics_matrix = utils.intrinsics_matrix_from_flat_intrinsics(
            gt_flat_intrinsics)
        torch.testing.assert_close(gt_intrinsics_matrix, intrinsics_matrix)

        flat_intrinsics = utils.flat_intrinsics_from_intrinsics_matrix(
            intrinsics_matrix)
        torch.testing.assert_close(gt_flat_intrinsics, flat_intrinsics)

    # @unittest.skip('Uncomment to skip this test')
    def test_spherical_and_cart(self):
        in_cart = torch.randn(100, 3)
        phi_theta, r = utils.spherical_from_cart(in_cart)
        out_cart = utils.cart_from_spherical(phi_theta, r)

        torch.testing.assert_close(in_cart, out_cart)

    # @unittest.skip('Uncomment to skip this test')
    def test_compute_jacobian(self):
        b = 2
        n = 10
        dim = 3
        A = torch.randn(b, dim, dim)
        # (b, n, dim) -> (b, n, dim)
        def f(x): return torch.einsum('bij,bnj->bni', A, x)
        x = torch.randn(b, n, dim)
        f_of_x, Df = utils.compute_jacobian(f, x)
        torch.testing.assert_close(A.unsqueeze(1).expand(-1, n, -1, -1), Df)

    # @unittest.skip('Uncomment to skip this test')
    def test_fit_polynomial(self):
        x = torch.linspace(0, 1, 100)
        x2 = x**2
        x3 = x**3
        y = torch.stack([x2, 2*x2+x3], dim=0)
        expected = torch.tensor([[0, 0, 1.0, 0], [0, 0, 2.0, 1.0]])
        pred = utils.fit_polynomial(x.unsqueeze(0).expand(2, -1), y, 3)
        torch.testing.assert_close(expected, pred)


if __name__ == '__main__':
    unittest.main()
    from io import StringIO
    with unittest.mock.patch('sys.stdout', new=StringIO()) as std_out:
        unittest.main()
