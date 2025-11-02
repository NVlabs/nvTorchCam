# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import numpy as np
from nvtorchcam import cameras
from nvtorchcam import utils
import cv2


# TODO: Since tests provide usage guidance, I think that here we should name all outputs, even those we don't use (i.e., don't use '_')


class TestCameras(unittest.TestCase):
    def setUp(self):
        self.pts = torch.tensor(
            [
                [
                    [-1.0841, -1.1539, 0.5728],
                    [2.0005, 1.3709, 0.2397],
                    [-0.0466, -1.2245, -0.8749],
                    [-1.0040, -0.7289, -0.1083],
                ],
                [
                    [0.0613, -0.8232, 1.0593],
                    [0.4924, 1.6029, 0.5246],
                    [-0.8611, -0.9941, -1.4192],
                    [-0.4211, 1.4871, 1.8990],
                ],
            ]
        )

        self.pix = torch.tensor(
            [
                [[1.0993, 0.9824], [-0.5771, 0.6910], [-1.7918, -0.1632], [-0.4430, 0.4839]],
                [[0.6547, 0.3039], [-0.9362, -0.6653], [0.8035, 0.3613], [0.7079, -2.2261]],
            ]
        )

        self.pos_depths = torch.tensor(
            [[0.5890, 0.2369, 0.8817, 0.8845], [0.6597, 0.5008, 0.7557, 0.8208]]
        )

        self.neg_depths = torch.tensor(
            [[-1.8059, -0.0886, 1.0758, 0.1906], [-0.8546, 0.2547, 1.0518, 0.0417]]
        )

    def project_to_pixel_test(self, camera, pts, unit_vec, rtol=None, atol=None):
        # should hold for all cameras under all circumstances
        pix, depth, valid_proj = camera.project_to_pixel(pts, unit_vec)
        origin, dir, valid_unproj = camera.pixel_to_ray(pix, unit_vec)
        recon_pts = origin + dir * depth.unsqueeze(-1)
        # assert valid_proj implies valid_unproj
        assert torch.all(valid_unproj[valid_proj])
        torch.testing.assert_close(
            pts[valid_proj, :], recon_pts[valid_proj, :], rtol=rtol, atol=atol
        )

    def pixel_to_ray_test(self, camera, pix, depth, unit_vec, rtol=None, atol=None):
        # only holds if depth is in valid range for pixel and camera unproject is injective
        origin, dir, valid_unproj = camera.pixel_to_ray(pix, unit_vec)
        pts = origin + dir * depth.unsqueeze(-1)
        recon_pix, recon_depth, valid_proj = camera.project_to_pixel(pts, unit_vec)
        valid = valid_proj & valid_unproj
        torch.testing.assert_close(pix[valid, :], recon_pix[valid, :], rtol=rtol, atol=atol)
        torch.testing.assert_close(depth[valid], recon_depth[valid], rtol=rtol, atol=atol)

    # @unittest.skip('Uncomment to skip this test')
    def test_pinhole_camera(self):
        flat_intrinsics = torch.tensor([[1, 1, 0.4, 0.5], [2, 2, -0.4, 0.5]])
        # even for negative z_min pinhole camera unproject is injective
        z_min = torch.ones(2) * (-10)
        camera = cameras.PinholeCamera.make(intrinsics=flat_intrinsics, z_min=z_min)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.neg_depths, unit_vec)

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.neg_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_orthographic_camera(self):
        flat_intrinsics = torch.tensor([[1, 1, 0.4, 0.5], [2, 2, -0.4, 0.5]])
        # even for negative z_min pinhole camera unproject is injective
        z_min = torch.ones(2) * (-10)
        camera = cameras.OrthographicCamera.make(intrinsics=flat_intrinsics, z_min=z_min)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.neg_depths, unit_vec)

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.neg_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_equirectangular_camera(self):
        b = 2
        phi_range = torch.tensor([-np.pi, np.pi]).unsqueeze(0).expand(b, -1)
        theta_range = torch.tensor([0, np.pi]).unsqueeze(0).expand(b, -1)
        min_distance = torch.ones(b) * 0.01
        camera = cameras.EquirectangularCamera.make(
            phi_range=phi_range, theta_range=theta_range, min_distance=min_distance
        )

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_opencv_fisheye(self):
        b = 2
        flat_intrinsics = torch.tensor([[1, 1, 0.4, 0.5], [2, 2, -0.4, 0.5]])
        k1 = 0.1 * torch.ones(b)
        k2 = 0.1 * torch.ones(b)
        k3 = torch.zeros(b)
        k4 = torch.zeros(b)
        theta_max = torch.ones(b) * (3 / 4) * np.pi
        distance_min = torch.ones(b) * (0.0)
        distortion_coeffs = torch.stack((k1, k2, k3, k4), dim=-1)
        camera = cameras.OpenCVFisheyeCamera.make(
            flat_intrinsics, distortion_coeffs, theta_max, distance_min
        )

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_opencv_fisheye_compared_to_opencv(self):
        # Note: this only holds when max_theta < pi/2. OpenCV projects x and -x to the same point, so it can't handle FoVs
        # larger than 180. Our implementation is more general
        max_theta = torch.tensor(np.pi / 2).float()
        pts = self.pts[0]
        distortion_coeffs = torch.tensor([0.1, 0.2, 0.1, 0.1])
        flat_intrinsics = torch.tensor([2, 2, -0.4, 0.5])
        torch_cam = cameras.OpenCVFisheyeCamera.make(
            flat_intrinsics, distortion_coeffs, max_theta, torch.tensor(0).float()
        )
        pix, depth, valid = torch_cam.project_to_pixel(pts, False)

        intrinsics = utils.intrinsics_matrix_from_flat_intrinsics(flat_intrinsics)
        pix_cv, _ = cv2.fisheye.projectPoints(
            pts.unsqueeze(1).numpy(),
            torch.zeros(3).numpy(),
            torch.zeros(3).numpy(),
            intrinsics.numpy(),
            distortion_coeffs.numpy(),
        )
        pix_cv = torch.from_numpy(pix_cv).squeeze(1)
        torch.testing.assert_close(pix_cv[valid, :], pix[valid, :])

    # @unittest.skip('Uncomment to skip this test')

    def test_opencv(self):
        b = 2
        flat_intrinsics = torch.tensor([[1, 1, 0.4, 0.5], [2, 2, -0.4, 0.5]])
        ks = torch.tensor(
            [[0.1, 0.05, 0.01, 0.02, 0.01, 0.005], [0.08, 0.07, 0.01, 0.03, 0.02, 0.003]]
        )
        ps = torch.tensor([[0.05, 0.02], [0.08, 0.03]])
        camera = cameras.OpenCVCamera.make(flat_intrinsics, ks, ps)

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_opencv_compared_to_opencv(self):
        pts = self.pts[0]
        ks = torch.tensor([0.1, 0.05, 0.01, 0.02, 0.01, 0.005])
        ps = torch.tensor([0.05, 0.02])
        flat_intrinsics = torch.tensor([2, 2, -0.4, 0.5])
        torch_cam = cameras.OpenCVCamera.make(flat_intrinsics, ks, ps)
        pix, depth, valid = torch_cam.project_to_pixel(pts, False)

        intrinsics = utils.intrinsics_matrix_from_flat_intrinsics(flat_intrinsics)
        opencv_params = torch.cat((ks[:2], ps, ks[2:]), dim=0).numpy()
        pix_cv, _ = cv2.projectPoints(
            pts.unsqueeze(1).numpy(),
            torch.zeros(3).numpy(),
            torch.zeros(3).numpy(),
            intrinsics.numpy(),
            opencv_params,
        )
        pix_cv = torch.from_numpy(pix_cv).squeeze(1)
        torch.testing.assert_close(pix_cv[valid, :], pix[valid, :])

    # @unittest.skip('Uncomment to skip this test')
    def test_forward_backward_fisheye(self):
        unit_vec = False
        b = 2
        fwpoly = (
            torch.tensor([0.0000, 412.9887, -54.4291, 111.8926, -32.7999])
            .unsqueeze(0)
            .expand(b, -1)
        )
        bwpoly = (
            torch.tensor([0.0000e00, 2.4903e-03, -4.3055e-08, -1.3850e-09, 9.8613e-13])
            .unsqueeze(0)
            .expand(b, -1)
        )
        cx_cy = torch.tensor([969.006165, 599.42144]).unsqueeze(0).expand(b, -1)
        fx_fy = torch.ones(b, 2)
        flat_intrinsics = torch.cat((fx_fy, cx_cy), dim=-1)
        theta_max = torch.ones(b) * (3 / 4) * np.pi
        distance_min = torch.ones(b) * (0.0)
        camera = cameras.BackwardForwardPolynomialFisheyeCamera.make(
            flat_intrinsics, fwpoly, bwpoly, theta_max, distance_min
        )

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec, atol=0.01, rtol=0.1)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec, atol=0.01, rtol=0.1)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_forward_backward_fisheye_concatenate_different_degrees(self):
        # TODO this just tests that degree promotion runs without errors. Should need to add test to ensure correct functionality
        fwpoly_4 = torch.tensor([0.0000, 412.9887, -54.4291, 111.8926, -32.7999]).unsqueeze(0)
        bwpoly_4 = torch.tensor(
            [0.0000e00, 2.4903e-03, -4.3055e-08, -1.3850e-09, 9.8613e-13]
        ).unsqueeze(0)
        fwpoly_5 = torch.tensor([0.0000, 412.9887, -54.4291, 111.8926, -32.7999, 1.2]).unsqueeze(0)
        bwpoly_6 = torch.tensor(
            [0.0000e00, 2.4903e-03, -4.3055e-08, -1.3850e-09, 9.8613e-13, 1.3]
        ).unsqueeze(0)
        cx_cy = torch.tensor([969.006165, 599.42144]).unsqueeze(0)
        fx_fy = torch.ones(1, 2)
        flat_intrinsics = torch.cat((fx_fy, cx_cy), dim=-1)
        theta_max = torch.ones(1) * (3 / 4) * np.pi
        distance_min = torch.ones(1) * (0.0)
        camera_a = cameras.BackwardForwardPolynomialFisheyeCamera.make(
            flat_intrinsics, fwpoly_4, bwpoly_4, theta_max, distance_min
        )
        camera_b = cameras.BackwardForwardPolynomialFisheyeCamera.make(
            flat_intrinsics, fwpoly_5, bwpoly_6, theta_max, distance_min
        )
        cat_cam = torch.cat([camera_a, camera_b])

    # @unittest.skip('Uncomment to skip this test')

    def test_kitti360_fisheye(self):
        b = 2
        fx = 1.336e3 * torch.ones(b)
        fy = 1.336e3 * torch.ones(b)
        cx = 7.169e2 * torch.ones(b)
        cy = 7.057e2 * torch.ones(b)
        k1 = 1.6798e-2 * torch.ones(b)
        k2 = 1.6548 * torch.ones(b)
        xi = 2.213 * torch.ones(b)
        theta_max = (np.pi / 2) * torch.ones(b)
        distance_min = torch.zeros(b)

        flat_intrinsics = torch.stack((fx, fy, cx, cy), dim=-1)
        camera = cameras.Kitti360FisheyeCamera.make(
            flat_intrinsics, k1, k2, xi, theta_max, distance_min
        )

        unit_vec = True
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

        unit_vec = False
        self.project_to_pixel_test(camera, self.pts, unit_vec)
        self.pixel_to_ray_test(camera, self.pix, self.pos_depths, unit_vec)

    # @unittest.skip('Uncomment to skip this test')
    def test_tensor_like_functions(self):
        flat_intrinsics = torch.tensor([[1, 1, 0.4, 0.5], [2, 2, -0.4, 0.5]])
        # even for negative z_min pinhole camera unproject is injective
        z_min = torch.ones(2) * (-10)
        camera = cameras.PinholeCamera.make(intrinsics=flat_intrinsics, z_min=z_min)
        assert camera.shape == (2,)

        # unsqueeze
        assert camera.unsqueeze(0).shape == (1, 2)
        assert camera.unsqueeze(-1).shape == (2, 1)
        assert isinstance(camera.unsqueeze(0), cameras.PinholeCamera)

        # expand
        camera_32 = camera.unsqueeze(0).expand(3, 2)
        assert camera_32.shape == (3, 2)
        assert isinstance(camera_32, cameras.PinholeCamera)
        camera_32 = camera.unsqueeze(0).expand(3, -1)
        assert camera_32.shape == (3, 2)
        camera_32 = camera.unsqueeze(0).expand((3, -1))
        assert camera_32.shape == (3, 2)

        # reshape
        camera_32 = camera.unsqueeze(0).expand(3, -1)
        assert camera_32.reshape(2, 3).shape == (2, 3)
        assert camera_32.reshape(-1, 3).shape == (2, 3)

        camera_234 = camera.unsqueeze(-1).unsqueeze(-1).expand(2, 3, 4)
        assert camera_234.permute(2, 1, 0).shape == (4, 3, 2)
        assert camera_234.permute(2, 0, 1).shape == (4, 2, 3)

        # transpose
        camera_234 = camera.unsqueeze(-1).unsqueeze(-1).expand(2, 3, 4)
        assert camera_234.transpose(0, 1).shape == (3, 2, 4)
        assert camera_234.transpose(1, 2).shape == (2, 4, 3)

        # squeeze
        camera_121 = camera.unsqueeze(0).unsqueeze(-1)
        assert camera_121.shape == (1, 2, 1)
        assert camera_121.squeeze().shape == (2,)
        assert camera_121.squeeze(0).shape == (2, 1)
        assert camera_121.squeeze(-1).shape == (1, 2)
        assert camera_121.squeeze(2).shape == (1, 2)

    # @unittest.skip('Uncomment to skip this test')
    def test_heterogeneous_batch_get_rays_and_project(self):
        pin_camera = cameras.PinholeCamera.make(
            intrinsics=torch.tensor([[1, 1, 0.4, 0.5], [2, 2.5, -0.4, 0.2]])
        )
        ortho_camera = cameras.OrthographicCamera.make(
            intrinsics=torch.tensor([[1, 1.2, -0.2, 0.1]])
        )
        hetero_camera = torch.cat((pin_camera, ortho_camera))

        pts_pin = 10 * torch.randn(2, 5, 3)
        pts_ortho = 10 * torch.randn(1, 5, 3)
        pts_hetero = torch.cat((pts_pin, pts_ortho))

        proj_pts_pin, depth_pts_pin, valid_pin = pin_camera.project_to_pixel(pts_pin)
        proj_pts_ortho, depth_pts_ortho, valid_ortho = ortho_camera.project_to_pixel(pts_ortho)
        proj_pts_hetero, depth_pts_hetero, valid_hetero = hetero_camera.project_to_pixel(pts_hetero)

        proj_pts_hetero_expected = torch.cat((proj_pts_pin, proj_pts_ortho))
        depth_pts_hetero_expected = torch.cat((depth_pts_pin, depth_pts_ortho))
        valid_hetero_expected = torch.cat((valid_pin, valid_ortho))

        torch.testing.assert_close(proj_pts_hetero, proj_pts_hetero_expected)
        torch.testing.assert_close(depth_pts_hetero, depth_pts_hetero_expected)
        torch.testing.assert_close(valid_hetero, valid_hetero_expected)

        origin_pin, dirs_pin, valid_pin = pin_camera.pixel_to_ray(proj_pts_pin, False)
        origin_ortho, dirs_ortho, valid_ortho = ortho_camera.pixel_to_ray(proj_pts_ortho, False)
        origin_hetero, dirs_hetero, valid_hetero = hetero_camera.pixel_to_ray(
            torch.cat((proj_pts_pin, proj_pts_ortho)), False
        )

        origin_hetero_expected = torch.cat((origin_pin, origin_ortho))
        dirs_hetero_expected = torch.cat((dirs_pin, dirs_ortho))
        valid_hetero_expected = torch.cat((valid_pin, valid_ortho))

        torch.testing.assert_close(origin_hetero, origin_hetero_expected)
        torch.testing.assert_close(dirs_hetero, dirs_hetero_expected)
        torch.testing.assert_close(valid_hetero, valid_hetero_expected)

    # @unittest.skip('Uncomment to skip this test')

    def test_heterogeneous_batch_tensor_like_functions(self):
        def heterogeneous_ortho_pin_to_tensor(test_cam):
            """Utility function to convert heterogenous batches of Orthographic and Pinhole cameras a flat tensor of their parameters.
            This is possible because Orthographic and Pinhole cameras have the same memory layout.
            """
            out = torch.zeros(*test_cam.shape, 5)
            out_write = out.reshape(-1, 5)
            for k, v in test_cam.my_dict.items():
                out_write[v[0], :4] = v[1]._values["affine"]
                out_write[v[0], 4:5] = v[1]._values["z_min"]
            return out

        pin_values = torch.rand(3, 4, 5)
        pin_camera = cameras.PinholeCamera.make(
            intrinsics=pin_values[..., :4], z_min=pin_values[..., 4]
        )
        ortho_values = torch.rand(3, 2, 5)
        ortho_camera = cameras.OrthographicCamera.make(
            intrinsics=ortho_values[..., :4], z_min=ortho_values[..., 4]
        )

        # test homo-homo cat
        expected = torch.cat((pin_values, ortho_values), dim=1)
        hetero_camera = torch.cat((pin_camera, ortho_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values, ortho_values), dim=1)
        hetero_camera = torch.cat((pin_camera, ortho_camera, ortho_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # hetero-homo cat
        expected = torch.cat((pin_values, ortho_values), dim=1)
        expected = torch.cat((expected, ortho_values), dim=1)
        hetero_camera = torch.cat((pin_camera, ortho_camera), dim=1)
        hetero_camera = torch.cat((hetero_camera, ortho_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # hetero-hetero cat
        expected = torch.cat((pin_values, ortho_values), dim=1)
        expected = torch.cat((expected, expected), dim=1)
        hetero_camera = torch.cat((pin_camera, ortho_camera), dim=1)
        hetero_camera = torch.cat((hetero_camera, hetero_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # test transpose
        expected = torch.cat((pin_values, ortho_values), dim=1).transpose(0, 1)
        hetero_camera = torch.cat((pin_camera, ortho_camera), dim=1).transpose(0, 1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # test_reshape
        expected = torch.cat((pin_values, ortho_values), dim=1).reshape(-1, 5)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).reshape(-1)
        )
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values), dim=1).reshape(9, 2, 5)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).reshape(9, 2)
        )
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values), dim=1).reshape(-1, 2, 5)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).reshape(-1, 2)
        )
        torch.testing.assert_close(expected, actual)

        # unsqueeze
        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(1)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(1)
        )
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(-2)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(-1)
        )
        torch.testing.assert_close(expected, actual)

        # expand
        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(1).expand(-1, 4, -1, -1)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(1).expand(-1, 4, -1)
        )
        torch.testing.assert_close(expected, actual)

        # squeeze
        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(1).squeeze(2)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(1).squeeze(2)
        )
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(1).squeeze(1)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(1).squeeze(1)
        )
        torch.testing.assert_close(expected, actual)

        expected = torch.cat((pin_values, ortho_values), dim=1).unsqueeze(1).squeeze()
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).unsqueeze(1).squeeze()
        )
        torch.testing.assert_close(expected, actual)

        # indexing
        values = torch.cat((pin_values, ortho_values), dim=1)  # (3,6,5)
        hetero_cam = torch.cat((pin_camera, ortho_camera), dim=1)  # (3,6)
        torch.testing.assert_close(values[0:2], heterogeneous_ortho_pin_to_tensor(hetero_cam[0:2]))

        values = torch.cat((pin_values, ortho_values), dim=1).transpose(0, 1)  # (6,3,5)
        hetero_cam = torch.cat((pin_camera, ortho_camera), dim=1).transpose(0, 1)  # (6,3)
        torch.testing.assert_close(
            values[:, 0:2], heterogeneous_ortho_pin_to_tensor(hetero_cam[:, 0:2])
        )

        # test stack
        pin_values = torch.rand(3, 4, 5)
        pin_camera = cameras.PinholeCamera.make(
            intrinsics=pin_values[..., :4], z_min=pin_values[..., 4]
        )
        ortho_values = torch.rand(3, 4, 5)
        ortho_camera = cameras.OrthographicCamera.make(
            intrinsics=ortho_values[..., :4], z_min=ortho_values[..., 4]
        )

        expected = torch.stack((pin_values, ortho_values), dim=1)
        hetero_camera = torch.stack((pin_camera, ortho_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        expected = torch.stack((pin_values, ortho_values, ortho_values), dim=1)
        hetero_camera = torch.stack((pin_camera, ortho_camera, ortho_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # hetero-homo stack
        pin_ortho = torch.cat((pin_values, ortho_values), dim=1)
        pin_pin = torch.cat((pin_values, pin_values), dim=1)
        pin_ortho_camera = torch.cat((pin_camera, ortho_camera), dim=1)
        pin_pin_camera = torch.cat((pin_camera, pin_camera), dim=1)
        expected = torch.stack((pin_ortho, pin_pin), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.stack((pin_ortho_camera, pin_pin_camera), dim=1)
        )
        torch.testing.assert_close(expected, actual)

        # hetero-hetero stack
        expected = torch.stack((pin_values, ortho_values), dim=1)
        expected = torch.stack((expected, expected), dim=1)
        hetero_camera = torch.stack((pin_camera, ortho_camera), dim=1)
        hetero_camera = torch.stack((hetero_camera, hetero_camera), dim=1)
        actual = heterogeneous_ortho_pin_to_tensor(hetero_camera)
        torch.testing.assert_close(expected, actual)

        # flip
        expected = torch.cat((pin_values, ortho_values), dim=1).flip(1).flip(0)
        actual = heterogeneous_ortho_pin_to_tensor(
            torch.cat((pin_camera, ortho_camera), dim=1).flip(1).flip(0)
        )
        torch.testing.assert_close(expected, actual)

    # @unittest.skip('Uncomment to skip this test')
    def test_heterogeneous_batch_devolves_to_homogeneous(self):
        pin_values = torch.rand(3, 4, 5)
        pin_camera = cameras.PinholeCamera.make(
            intrinsics=pin_values[..., :4], z_min=pin_values[..., 4]
        )
        ortho_values = torch.rand(3, 2, 5)
        ortho_camera = cameras.OrthographicCamera.make(
            intrinsics=ortho_values[..., :4], z_min=ortho_values[..., 4]
        )
        hetero_camera = torch.cat((pin_camera, ortho_camera), dim=1)

        devolved_pin_camera = hetero_camera[:, 0:4]
        assert isinstance(devolved_pin_camera, cameras.PinholeCamera)

        for k in devolved_pin_camera._values.keys():
            torch.testing.assert_close(devolved_pin_camera._values[k], pin_camera._values[k])

    # @unittest.skip('Uncomment to skip this test')
    def test_crop(self):
        flat_intrinsics = torch.tensor([1, 1, 0.4, 0.5])
        # flat_intrinsics = torch.tensor([1,1,0.0,0.0])
        # even for negative z_min pinhole camera unproject is injective
        z_min = torch.tensor(1e-6)

        camera_pin = cameras.PinholeCamera.make(intrinsics=flat_intrinsics, z_min=z_min)
        camera_ortho = cameras.OrthographicCamera.make(intrinsics=flat_intrinsics, z_min=z_min)

        theta_max = torch.tensor((3 / 4) * np.pi).float()
        distance_min = torch.tensor(0.0)
        distortion_coeffs = torch.tensor([0.1, 0.1, 0, 0])
        camera_fish = cameras.OpenCVFisheyeCamera.make(
            flat_intrinsics, distortion_coeffs, theta_max, distance_min
        )

        for camera in [camera_pin, camera_ortho, camera_fish]:
            origin, dir, valid = camera.get_camera_rays((4, 6), False)
            cropped_camera = camera.crop(
                torch.tensor([1, 4, 1, 3]), normalized=False, image_shape=dir.shape[-3:-1]
            )
            cropped_origin, cropped_dir, cropped_valid = cropped_camera.get_camera_rays(
                (2, 3), False
            )
            torch.testing.assert_close(cropped_dir, dir[1:3, 1:4])
            torch.testing.assert_close(cropped_origin, origin[1:3, 1:4])
            assert torch.all(cropped_valid == valid[1:3, 1:4])

    # @unittest.skip('Uncomment to skip this test')
    def test_apply_affine(self):
        flat_intrinsics = torch.rand(2, 4)
        z_min = torch.tensor([1e-6, 1e-6])
        intrinsics = utils.intrinsics_matrix_from_flat_intrinsics(flat_intrinsics)
        camera_pin = cameras.PinholeCamera.make(intrinsics=flat_intrinsics, z_min=z_min)

        flat_A = torch.randn(2, 4)
        A = utils.intrinsics_matrix_from_flat_intrinsics(flat_A)

        camera_mult_left = camera_pin.affine_transform(flat_A)
        mult_left_intrinsics = utils.intrinsics_matrix_from_flat_intrinsics(
            camera_mult_left._values["affine"]
        )
        torch.testing.assert_close(mult_left_intrinsics, torch.bmm(A, intrinsics))

        camera_mult_right = camera_pin.affine_transform(flat_A, multiply_on_right=True)
        mult_right_intrinsics = utils.intrinsics_matrix_from_flat_intrinsics(
            camera_mult_right._values["affine"]
        )
        torch.testing.assert_close(mult_right_intrinsics, torch.bmm(intrinsics, A))

    # @unittest.skip('Uncomment to skip this test')
    def test_is_central(self):
        pin_values = torch.rand(3, 4, 5)
        pin_camera = cameras.PinholeCamera.make(
            intrinsics=pin_values[..., :4], z_min=pin_values[..., 4]
        )
        ortho_values = torch.rand(3, 2, 5)
        ortho_camera = cameras.OrthographicCamera.make(
            intrinsics=ortho_values[..., :4], z_min=ortho_values[..., 4]
        )

        max_theta = torch.tensor(np.pi / 2).float()
        distortion_coeffs = torch.tensor([0.1, 0.2, 0.1, 0.1])
        flat_intrinsics = torch.tensor([2, 2, -0.4, 0.5])
        opencv_fisheye = cameras.OpenCVFisheyeCamera.make(
            flat_intrinsics, distortion_coeffs, max_theta, torch.tensor(0).float()
        )
        opencv_fisheye = opencv_fisheye.unsqueeze(0).unsqueeze(0).expand(3, 1)

        assert opencv_fisheye.is_central()
        assert pin_camera.is_central()
        assert not ortho_camera.is_central()

        hetero_central = torch.cat([opencv_fisheye, pin_camera], dim=1)
        hetero_non_central = torch.cat([pin_camera, ortho_camera], dim=1)

        assert hetero_central.is_central()
        assert not hetero_non_central.is_central()


if __name__ == "__main__":
    from io import StringIO

    with unittest.mock.patch("sys.stdout", new=StringIO()) as std_out:
        unittest.main()
