# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Functional implementation of cameras project_to_pixel and pixel_to_ray functions."""

from typing import Optional, Dict, Tuple, Union, Any, List
import torch
import torch.nn.functional as F
from torch import Tensor
from nvtorchcam import utils
from nvtorchcam.diff_newton_inverse import DifferentiableNewtonInverse

__all__ = ["orthographic_camera_project_to_pixel",
           "orthographic_camera_pixel_to_ray",
           "pinhole_camera_project_to_pixel",
           "pinhole_camera_pixel_to_ray",
           "equirectangular_camera_project_to_pixel",
           "equirectangular_camera_pixel_to_ray",
           "opencv_fisheye_camera_project_to_pixel",
           "opencv_fisheye_camera_pixel_to_ray",
           "opencv_camera_project_to_pixel",
           "opencv_camera_pixel_to_ray",
           "backward_forward_polynomial_fisheye_camera_project_to_pixel",
           "backward_forward_polynomial_fisheye_camera_pixel_to_ray",
           "kitti360_fisheye_camera_project_to_pixel",
           "kitti360_fisheye_camera_pixel_to_ray",
           "cube_camera_project_to_pixel",
           "cube_camera_pixel_to_ray"
           ]

def camera_infer_batch_group(func):
    """
    Decorator to transform a function taking args of the form:
    (b, k_1) x ... x (b, k_n) x (b, g, d) and returning (b, g, d_1) ... (b, g, d_n)
    to a function taking args of the form:
    (*b, k_1) x ... x (*b, k_n) x (*b, *g, d) and returning (b*, g*, d_1) ... (b*, g*, d_n).
    Note that if any of the d_i == 1 dimension will be squeezed.
    """

    def flatten_unflatten(*args, **kwargs):
        """args[:-1] all have shape (*batch_shape, d_i)"""
        batch_shape = args[0].shape[:-1]
        batch_dims = len(batch_shape)
        group_shape = args[-1].shape[batch_dims:-1]
        batch_numel = batch_shape.numel()
        group_numel = group_shape.numel()
        reshaped_args = []
        for i in range(len(args) - 1):
            assert args[i].shape[:-1] == batch_shape
            reshaped_args.append(args[i].reshape(batch_numel, -1))
        assert args[-1].shape[:batch_dims] == batch_shape
        reshaped_args.append(args[-1].reshape(batch_numel, group_numel, -1))
        result = func(*reshaped_args, **kwargs)
        reshaped_result = []
        for x in result:
            x = x.reshape(*batch_shape, *group_shape, -1)
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            reshaped_result.append(x)
        return reshaped_result

    return flatten_unflatten


@camera_infer_batch_group
def orthographic_camera_project_to_pixel(
    flat_intrinsics: Tensor, z_min: Tensor, pts: Tensor, depth_is_along_ray: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Orthographic camera projection function. Points are marked valid if it has depth
    value > z_min.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        z_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    valid = pts[:, :, 2] > z_min
    z_min = z_min.unsqueeze(-1)
    depth = pts[:, :, 2]

    pix = pts[:, :, :2]

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def orthographic_camera_pixel_to_ray(
    flat_intrinsics: Tensor, pix: Tensor, unit_vec: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Orthographic camera pixel_to_ray function. All pixels are marked valid.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """
    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    dirs = torch.cat((torch.zeros_like(pix), torch.ones_like(pix[:, :, 0:1])), dim=-1)
    origin = torch.cat((pix, torch.zeros_like(pix[:, :, 0:1])), dim=-1)
    valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)
    return origin, dirs, valid


@camera_infer_batch_group
def pinhole_camera_project_to_pixel(
    flat_intrinsics: Tensor, z_min: Tensor, pts: Tensor, depth_is_along_ray: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pinhole camera projection function. Points are marked valid if it has depth value > z_min

    Args:
        flat_intrinsics: (*batch_shape, 4)
        z_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    valid = pts[:, :, 2] > z_min
    z_min = z_min.unsqueeze(-1)

    denom = torch.max(pts[:, :, 2:3], z_min)
    pix = pts[:, :, 0:2] / denom
    if depth_is_along_ray:
        depth = torch.norm(pts, dim=-1) * torch.sign(pts[:, :, 2])
    else:
        depth = pts[:, :, 2]

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def pinhole_camera_pixel_to_ray(
    flat_intrinsics: Tensor, pix: Tensor, unit_vec: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pinhole camera pixel_to_ray function. All pixels are marked valid.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """
    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    dirs = torch.cat((pix, torch.ones_like(pix[:, :, 0:1])), dim=-1)
    if unit_vec:
        dirs = F.normalize(dirs, dim=-1)
    valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)
    origin = torch.zeros_like(dirs)

    return origin, dirs, valid


@camera_infer_batch_group
def equirectangular_camera_project_to_pixel(
    flat_intrinsics: Tensor, distance_min: Tensor, pts: Tensor, depth_is_along_ray: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Equirectangular camera projection function. Points are marked valid if 
    point has distance value > distance_min.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        distance_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    pix, r = utils.spherical_from_cart(pts)
    valid = r > distance_min
    if depth_is_along_ray:
        depth = r
    else:
        depth = torch.abs(pts[:, :, 2])

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def equirectangular_camera_pixel_to_ray(
    flat_intrinsics: Tensor, pix: Tensor, restrict_valid_rays=True, unit_vec: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Equirectangular camera pixel_to_ray function. If restrict_valid_rays=False all pixels are 
    marked valid, otherwise only pixels with angles in [-pi,pi] x [0, pi] are valid.

    If restrict_valid_rays=False multiple pixels may correspond to the same ray.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        pix: (*batch_shape, *group_shape, 2)
        restriect_valid_rays: bool
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """
    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    dirs = utils.cart_from_spherical(pix)
    if not unit_vec:
        dirs = dirs / torch.abs(dirs[:, :, 2:3])

    origin = torch.zeros_like(dirs)
    if restrict_valid_rays:
        valid = (
            (pix[:, :, 0] < torch.pi)
            & (pix[:, :, 0] >= -torch.pi)
            & (pix[:, :, 1] <= torch.pi)
            & (pix[:, :, 1] >= 0)
        )
    else:
        valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)

    return origin, dirs, valid


@camera_infer_batch_group
def opencv_fisheye_camera_project_to_pixel(
    flat_intrinsics: Tensor,
    distortion_coeffs: Tensor,
    theta_max: Tensor,
    distance_min: Tensor,
    pts: Tensor,
    depth_is_along_ray: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """OpenCV fisheye camera projection function. Points are marked valid if they have 
    distance > distance_min and they make an angle with the camera axis of less than theta_max.
    
    Args:
        flat_intrinsics: (*batch_shape, 4)
        distortion_coeffs: (*batch_shape, 4)
        theta_max: (*batch_shape, 1)
        distance_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    rays = F.normalize(pts, dim=-1)  # (b,n,3)
    theta = torch.acos(rays[:, :, 2:3])

    theta_d = opencv_fisheye_distortion(distortion_coeffs, theta)

    normalized_pix = F.normalize(rays[:, :, :2], dim=-1)
    pix = normalized_pix * theta_d

    distance = torch.norm(pts, dim=-1)

    valid = (theta.squeeze(-1) < theta_max) & (distance > distance_min)

    if depth_is_along_ray:
        depth = distance
    else:
        depth = torch.abs(pts[..., 2])

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def opencv_fisheye_camera_pixel_to_ray(
    flat_intrinsics: Tensor,
    distortion_coeffs: Tensor,
    theta_max: Tensor,
    pix: Tensor,
    unit_vec: bool = False,
    num_iters: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """OpenCV fisheye camera pixel to ray function. Valid if pixels are in the disk corresponding to
    rays with angles < theta_max.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        distortion_coeffs: (*batch_shape, 4)
        theta_max: (*batch_shape, 1)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """
    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    theta_d = torch.norm(pix, dim=-1, keepdim=True)  # (b,n,1)
    theta = opencv_fisheye_undistortion(distortion_coeffs, theta_d, num_iters=num_iters)

    dirs_xy = torch.sin(theta) * F.normalize(pix, dim=-1)
    dirs_z = torch.cos(theta)
    dirs = torch.cat((dirs_xy, dirs_z), dim=-1)

    theta_d_max = opencv_fisheye_distortion(distortion_coeffs, theta_max.reshape(-1, 1, 1))
    valid = theta_d < theta_d_max

    if not unit_vec:
        dirs = dirs / torch.abs(dirs[:, :, 2:3])

    origin = torch.zeros_like(dirs)
    return origin, dirs, valid


def opencv_fisheye_distortion(ks: Tensor, theta: Tensor) -> Tensor:
    """distortion function for opencv fisheye camera. (b,4) (b,n,1) -> (b,n,1)"""
    theta = theta.squeeze(-1)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (
        1 + ks[:, 0:1] * theta2 + ks[:, 1:2] * theta4 + ks[:, 2:3] * theta6 + ks[:, 3:4] * theta8
    )
    return theta_d.unsqueeze(-1)


class OpencvFisheyeUndistortion(DifferentiableNewtonInverse):
    @staticmethod
    def my_function(theta, ks):
        return opencv_fisheye_distortion(ks, theta)


def opencv_fisheye_undistortion(ks: Tensor, theta_d: Tensor, num_iters) -> Tensor:
    """undistortion function for opencv fisheye camera. (b,4) x (b,n,1) -> (b,n,1)"""
    return OpencvFisheyeUndistortion.apply(theta_d, theta_d, num_iters, ks)


@camera_infer_batch_group
def opencv_camera_project_to_pixel(
    flat_intrinsics: Tensor,
    ks: Tensor,
    ps: Tensor,
    z_min: Tensor,
    pts: Tensor,
    depth_is_along_ray: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """OpenCV camera projection function. Points are marked valid if it has depth value > z_min.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        ks: (*batch_shape, 6)
        ps: (*batch_shape, 2)
        z_min: (*batch_shape, 1)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    valid = pts[:, :, 2] > z_min
    z_min = z_min.unsqueeze(-1)
    denom = torch.max(pts[:, :, 2:3], z_min)
    pix_undist = pts[:, :, 0:2] / denom

    pix = opencv_distortion(ks, ps, pix_undist)

    if depth_is_along_ray:
        depth = torch.norm(pts, dim=-1) * torch.sign(pts[:, :, 2])
    else:
        depth = pts[..., 2]

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def opencv_camera_pixel_to_ray(
    flat_intrinsics: Tensor,
    ks: Tensor,
    ps: Tensor,
    pix: Tensor,
    unit_vec: bool = False,
    num_iters: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """OpenCV camera pixel to ray function. All pixels are marked valid. 

    Args:
        flat_intrinsics: (*batch_shape, 4)
        ks: (*batch_shape, 6)
        ps: (*batch_shape, 2)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """
    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    # pix_undist = utils.newton_inverse(lambda x: opencv_distortion(
    #    ks, ps, x), pix, pix, iters=num_iters)
    pix_undist = opencv_undistortion(ks, ps, pix, num_iters)

    dirs = torch.cat((pix_undist, torch.ones_like(pix_undist[:, :, 0:1])), dim=-1)

    if unit_vec:
        dirs = F.normalize(dirs, dim=-1)
    valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)
    origin = torch.zeros_like(dirs)
    return origin, dirs, valid


def opencv_distortion(ks: Tensor, ps: Tensor, pix: Tensor) -> Tensor:
    """distortion function for opencv fisheye camera. (b,4) x (b,2) x (b,n,2) -> (b,n,2)"""

    u2_v2 = pix**2
    uv = torch.prod(pix, dim=2)
    r2 = torch.sum(u2_v2, dim=2)
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + ks[:, 0:1] * r2 + ks[:, 1:2] * r4 + ks[:, 2:3] * r6) / (
        1 + ks[:, 3:4] * r2 + ks[:, 4:5] * r4 + ks[:, 5:6] * r6
    )

    pix = pix * radial.unsqueeze(-1)
    pix = pix + 2 * ps.unsqueeze(1) * uv.unsqueeze(-1)
    p2_p1 = ps.flip(-1)
    pix = pix + p2_p1.unsqueeze(1) * (r2.unsqueeze(-1) + 2 * u2_v2)
    return pix


class OpencvUndistortion(DifferentiableNewtonInverse):
    @staticmethod
    def my_function(pix, ks, ps):
        return opencv_distortion(ks, ps, pix)


def opencv_undistortion(ks: Tensor, ps: Tensor, pix: Tensor, num_iters: int) -> Tensor:
    return OpencvUndistortion.apply(pix, pix, num_iters, ks, ps)


@camera_infer_batch_group
def backward_forward_polynomial_fisheye_camera_project_to_pixel(
    flat_intrinsics: Tensor,
    proj_poly: Tensor,
    unproj_poly: Tensor,
    theta_max: Tensor,
    distance_min: Tensor,
    pts: Tensor,
    depth_is_along_ray: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """BackwardForwardPolynomial fisheye camera projection function. Points are marked valid if 
    they have distance > distance_min and they make an angle with the camera axis of less than 
    theta_max. Model is similar to opencv fisheye except it accepts arbitrary degree polynomials 
    and requires a user specified backward polynomial for pixel to ray.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        proj_poly: (*batch_shape, k)
        unproj_poly: (*batch_shape, l) not actually used
        theta_max: (*batch_shape, 1)
        distance_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """
    rays = F.normalize(pts, dim=-1)  # (b,n,3)
    theta = torch.acos(rays[:, :, 2])

    theta_d = utils.apply_poly(proj_poly, theta)

    normalized_pix = F.normalize(rays[:, :, :2], dim=-1)
    pix = normalized_pix * theta_d.unsqueeze(-1)

    distance = torch.norm(pts, dim=-1)
    valid = (theta < theta_max) & (distance > distance_min)

    if depth_is_along_ray:
        depth = distance
    else:
        depth = torch.abs(pts[..., 2])

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def backward_forward_polynomial_fisheye_camera_pixel_to_ray(
    flat_intrinsics: Tensor,
    proj_poly: Tensor,
    unproj_poly: Tensor,
    theta_max: Tensor,
    pix: Tensor,
    unit_vec: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """BackwardForwardPolynomial fisheye camera pix to ray function. Valid if pixels are in the
    disk corresponding to rays with angles < theta_max.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        proj_poly: (*batch_shape, k)
        unproj_poly: (*batch_shape, l)
        theta_max: (*batch_shape, 1)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    theta_d = torch.norm(pix, dim=-1)
    theta = utils.apply_poly(unproj_poly, theta_d)
    theta = theta.unsqueeze(-1)
    dirs_xy = torch.sin(theta) * F.normalize(pix, dim=-1)
    dirs_z = torch.cos(theta)
    dirs = torch.cat((dirs_xy, dirs_z), dim=-1)

    theta_d_max = utils.apply_poly(proj_poly, theta_max)
    valid = theta_d < theta_d_max

    if not unit_vec:
        dirs = dirs / torch.abs(dirs[:, :, 2:3])

    origin = torch.zeros_like(dirs)
    return origin, dirs, valid


@camera_infer_batch_group
def kitti360_fisheye_camera_project_to_pixel(
    flat_intrinsics: Tensor,
    ks: Tensor,
    xi: Tensor,
    theta_max: Tensor,
    distance_min: Tensor,
    pts: Tensor,
    depth_is_along_ray: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """KITTI360 fisheye camera projection function. Points are marked valid if they have 
    distance > distance_min and they make an angle with the camera axis of less than theta_max.
    
    First projects point onto a unit sphere at the camera origin, then projects point on the sphere
    to a two parameter distorted pinhole camera with optical center located at (0,0,-xi). Assertion
    error if theta_max > arccos(-1/xi), since points making an angle greater than arccos(-1/xi) are
    on the side of sphere further along the -z direction.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        ks: (*batch_shape, 2)
        xi: (*batch_shape, 1)
        theta_max: (*batch_shape, 1)
        distance_min: (*batch_shape, 1)
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 2)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    cos_theta_max = torch.cos(theta_max)
    assert torch.all(cos_theta_max > -1/xi)

    pts_n = F.normalize(pts, dim=-1)
    distance = torch.norm(pts, dim=-1)
    valid = (pts_n[:, :, 2] >= cos_theta_max) & (distance > distance_min)

    u_v = pts_n[:, :, :2] / (pts_n[:, :, 2:3] + xi.unsqueeze(-1))
    r = torch.norm(u_v, dim=-1, keepdim=True)

    r_d = kitti360_fisheye_distortion(ks, r)
    pix = r_d * F.normalize(u_v, dim=-1)

    if depth_is_along_ray:
        depth = distance
    else:
        depth = torch.abs(pts[:, :, 2])

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = f1_f2 * pix + c1_c2

    return pix, depth, valid


@camera_infer_batch_group
def kitti360_fisheye_camera_pixel_to_ray(
    flat_intrinsics: Tensor,
    ks: Tensor,
    xi: Tensor,
    theta_max: Tensor,
    pix: Tensor,
    unit_vec: bool = False,
    num_iters: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """KITTI360 fisheye camera pixel_to_ray function. Valid if pixels are in the disk corresponding 
    to rays with angles < theta_max.

    Shoot a ray from a two parameter distorted pinhole camera located at (0,0,-xi) toward the unit
    sphere, and take the intersection that is further along the ray to get the ray direction.
    Assertion error if theta_max > arccos(-1/xi), since rays beyond this angle do not hit the unit
    sphere.

    Args:
        flat_intrinsics: (*batch_shape, 4)
        ks: (*batch_shape, 2)
        xi: (*batch_shape, 1)
        theta_max: (*batch_shape, 1)
        pix: (*batch_shape, *group_shape, 2)
        unit_vec: bool

    Returns:
        origin: (*self.shape, *group_shape, 3)
        dirs: (*self.shape, *group_shape, 3)
        valid: (*self.shape, *group_shape)
    """

    cos_theta_max = torch.cos(theta_max)
    assert torch.all(cos_theta_max > -1/xi)
    sin_theta_max = torch.sin(theta_max)
    r_max = sin_theta_max/ (xi + cos_theta_max)
    r_d_max = kitti360_fisheye_distortion(ks, r_max.unsqueeze(1)).squeeze(1)
    

    f1_f2 = flat_intrinsics[:, None, 0:2]
    c1_c2 = flat_intrinsics[:, None, 2:4]
    pix = (pix - c1_c2) / f1_f2

    r_d = torch.norm(pix, dim=-1)  # (b,n)
    valid = r_d < r_d_max  # (b,n)

    r = kitti360_fisheye_undistortion(ks, r_d.unsqueeze(-1), num_iters)
    r = r.squeeze(-1)

    r2 = r**2
    discrim = (1 - xi**2) * r2 + 1
    discrim = discrim.clamp(min=0)

    alpha = (xi + torch.sqrt(discrim)) / (r2 + 1)  # (b,n)
    dirs_z = alpha - xi
    dirs_xy = (alpha * r).unsqueeze(-1) * F.normalize(pix, dim=-1)

    dirs = torch.cat((dirs_xy, dirs_z.unsqueeze(-1)), dim=-1)

    if not unit_vec:
        dirs = dirs / torch.abs(dirs[:, :, 2:3])

    origin = torch.zeros_like(dirs)
    return origin, dirs, valid


def kitti360_fisheye_distortion(ks: Tensor, r: Tensor) -> Tensor:
    """(b,2) x (b,n,1), -> (b,n,1)"""
    r2 = r**2
    scale = 1 + ks[:, 0:1, None] * r2 + ks[:, 1:2, None] * r2 * r2
    return r * scale


class Kitti360FisheyeUndistortion(DifferentiableNewtonInverse):
    @staticmethod
    def my_function(r, ks):
        return kitti360_fisheye_distortion(ks, r)


def kitti360_fisheye_undistortion(ks: Tensor, r_d: Tensor, num_iters) -> Tensor:
    """undistortion function for opencv fisheye camera. (b,4) x (b,n,1) -> (b,n,1)"""
    return Kitti360FisheyeUndistortion.apply(r_d, r_d, num_iters, ks)


def cube_camera_project_to_pixel(
    pts: Tensor, depth_is_along_ray: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Cube Camera project to pixel function. All points are valid.

    Args:
        pts: (*batch_shape, *group_shape, 3)
        depth_is_along_ray: bool

    Returns:
        pix: (*batch_shape, *group_shape, 3)
        depth: (*batch_shape, *group_shape)
        valid: (*batch_shape, *group_shape)
    """

    if depth_is_along_ray:
        depth = torch.norm(pts, dim=-1)
    else:
        depth = torch.max(torch.abs(pts), dim=-1)[0]

    pix = pts / depth.unsqueeze(-1).clamp_min(1e-12)
    valid = torch.ones_like(pix[..., 0], dtype=torch.bool)
    return pix, depth, valid


def cube_camera_pixel_to_ray(pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Cube camera pixel to ray function. All pixel are marked valid

    Args:
        pix: (*batch_shape, *group_shape, 3)
        unit_vec: bool

    Returns:
        origin: (*batch_shape, *group_shape, 3)
        dirs: (*batch_shape, *group_shape, 3)
        valid: (*batch_shape, *group_shape)
    """

    if unit_vec:
        dirs = F.normalize(pix, dim=-1)
    else:
        dirs = pix / torch.max(torch.abs(pix), dim=-1, keepdim=True).clamp_min(1e-12)

    origin = torch.zeros_like(dirs)
    valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)
    return origin, dirs, valid
