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

from __future__ import annotations
import torch
from typing import Tuple, Union, Optional
from torch import Tensor

__all__ = [
    "in_image",
    "get_normalized_grid",
    "get_normalized_grid_cubemap",
    "samples_from_image",
    "samples_from_cubemap",
    "apply_matrix",
    "apply_affine",
    "normalized_intrinsics_from_pixel_intrinsics",
    "pixel_intrinsics_from_normalized_intrinsics",
    "normalized_pts_from_pixel_pts",
    "pixel_pts_from_normalized_pts",
    "flat_intrinsics_from_intrinsics_matrix",
    "intrinsics_matrix_from_flat_intrinsics",
    "cart_from_spherical",
    "spherical_from_cart",
    "compute_jacobian",
    "fit_polynomial",
    "apply_poly",
    "crop_to_affine",
    "flatten_cubemap_for_visual",
]


def transform_infer_batch_group(func):
    """Transform function taking (b, i, j) x (b, g, k) -> (b, g, k) to a function taking
    (b*, i, j) x (b*, g*, k) -> (b*, g*, k)
    """

    def flatten_unflatten(A, x, **kwargs):
        batch_shape = A.shape[:-2]
        assert x.shape[: len(batch_shape)] == batch_shape
        group_shape = x.shape[len(batch_shape) : -1]
        batch_numel = batch_shape.numel()
        group_numel = group_shape.numel()
        result = func(
            A.reshape(batch_numel, *A.shape[-2:]), x.reshape(batch_numel, group_numel, -1), **kwargs
        )
        reshaped_result = result.reshape(*batch_shape, *group_shape, -1)
        return reshaped_result

    return flatten_unflatten


def in_image(pix: Tensor) -> Tensor:
    """Test if point is in the unit square [-1,1]x[-1,1]

    Args:
        pix:  (*, 2)

    Returns:
        :       (*,)
    """
    return torch.all(pix < 1, dim=-1) & torch.all(pix > -1, dim=-1)


def get_normalized_grid(
    res: Tuple[int, int], device: torch.device, pad: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Tensor:
    """Get normalized pixel locations. Use pad to include grid points beyond [-1,1]x[-1,1].

    Args:
        res: (h,w)
        device: torch.device
        pad: Tuple[int, int, int, int] = (0, 0, 0, 0)

    Returns:
        pix: (h,w,2)
    """
    start_x = -1 + 1 / res[1] - (2 / res[1]) * pad[0]
    end_x = 1 - 1 / res[1] + (2 / res[1]) * pad[1]
    start_y = -1 + 1 / res[0] - (2 / res[0]) * pad[2]
    end_y = 1 - 1 / res[0] + (2 / res[0]) * pad[3]
    x_pix = torch.linspace(
        start_x, end_x, res[1] + pad[0] + pad[1], dtype=torch.float, device=device
    )
    y_pix = torch.linspace(
        start_y, end_y, res[0] + pad[2] + pad[3], dtype=torch.float, device=device
    )
    x_y = torch.meshgrid([x_pix, y_pix], indexing="xy")
    pix = torch.stack(x_y, dim=2)
    return pix


def get_normalized_grid_cubemap(res: int, device: torch.device, pad: int = 0) -> Tensor:
    """Get normalized cubemap locations locations. Use pad to include grid points beyond face.

    Args:
        res: w

    Returns:
        pix: (6*w,w,3)
        device: torch.device
        pad: int = 0
    """
    grid = get_normalized_grid((res, res), device=device, pad=(pad,) * 4)  # (w,w,2)
    o = torch.ones_like(grid[:, :, 0])
    face0 = torch.stack((o, -grid[:, :, 1], -grid[:, :, 0]), dim=-1)
    face1 = torch.stack((-o, -grid[:, :, 1], grid[:, :, 0]), dim=-1)
    face2 = torch.stack((grid[:, :, 0], o, grid[:, :, 1]), dim=-1)
    face3 = torch.stack((grid[:, :, 0], -o, -grid[:, :, 1]), dim=-1)
    face4 = torch.stack((grid[:, :, 0], -grid[:, :, 1], o), dim=-1)
    face5 = torch.stack((-grid[:, :, 0], -grid[:, :, 1], -o), dim=-1)
    pix = torch.cat((face0, face1, face2, face3, face4, face5), dim=0)  # (*res, 3)
    return pix


def samples_from_image(
    image: Tensor,
    pts: Tensor,
    mode: str = "bilinear",
    align_corners: Optional[bool] = False,
    return_in_image_mask: bool = False,
    padding_mode: str = "zeros",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Interpolate a batch of images at point locations pts. If return_in_image_mask = True also
    return a mask indicated which pixels are within the image bounds

    Args:
            image (*batch_shape, channels, height, width)
            pts: (*batch_shape, *group_shape, 2)
        Returns:
            values: (*batch_shape, channels, *group_shape)
            mask: (*batch_shape, *group_shape)
    """
    batch_shape = image.shape[:-3]
    group_shape, batch_numel = _get_group_shape(batch_shape, pts.shape[:-1])

    # support evaluating bool image
    if image.dtype == torch.bool:
        is_bool = True
        image = image.half()
    else:
        is_bool = False
    image = image.reshape(batch_numel, *image.shape[-3:])
    pts_flat = pts.reshape(batch_numel, -1, 1, 2)
    values = torch.nn.functional.grid_sample(
        image, pts_flat, align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )  # (batch_numel, c, group_numel, 1)
    values = values.reshape(*batch_shape, -1, *group_shape)

    if is_bool:
        values = values > 0.5

    if return_in_image_mask:
        mask = in_image(pts)
        return values, mask
    else:
        return values


def samples_from_cubemap(cubemap: Tensor, pts: Tensor, mode: str = "bilinear") -> Tensor:
    """Interpolate a batch of cubes at directions pts.

    Args:
            image (*batch_shape, channels, 6*width, width)
            pts: (*batch_shape, *group_shape, 3)
        Returns:
            values: (*batch_shape, channels, *group_shape)
    """
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise ImportError(
            "The 'nvdiffrast' package is required for this functionality. "
            "Please install it using: pip install .[cubemap]"
        ) from exc
    if mode == "bilinear":
        mode = "linear"
    batch_shape = cubemap.shape[:-3]
    group_shape, batch_numel = _get_group_shape(batch_shape, pts.shape[:-1])

    cubemap = cubemap.reshape(batch_numel, *cubemap.shape[-3:]).unflatten(
        2, (6, -1)
    )  # (b, c, 6, w, w)
    cubemap = cubemap.permute(0, 2, 3, 4, 1)  # (b, 6, w, w, c)
    out_dtype = cubemap.dtype
    pts_flat = pts.reshape(batch_numel, -1, 1, 3)
    values = dr.texture(
        cubemap.float().contiguous(),
        pts_flat.float().contiguous(),
        boundary_mode="cube",
        filter_mode=mode,
    )  # (b,-1,1,c)
    values = values.type(out_dtype)
    values = values.permute(0, 3, 1, 2)
    values = values.reshape(*batch_shape, -1, *group_shape)

    return values


@transform_infer_batch_group
def apply_matrix(A: Tensor, pts: Tensor) -> Tensor:
    """Transform batches of groups of points by batches of matrices

    Args:
        A: (*batch_shape, d, d)
        pts: (*batch_shape, *group_shape, d)
    Returns:
        : (*batch_shape, *group_shape, d)

    """
    assert A.size(2) == pts.size(2)
    return torch.einsum("bij,bnj->bni", A, pts)


@transform_infer_batch_group
def apply_affine(T: Tensor, pts: Tensor) -> Tensor:
    """Transform batches of groups of points by batches of affine transformations

    Args:
        A: (*batch_shape, d+1, d+1)
        pts: (*batch_shape, *group_shape, d)
    Returns:
        : (*batch_shape, *group_shape, d)
    """
    k = pts.size(-1)
    assert T.shape[1:3] == (k + 1, k + 1)
    R = T[:, :k, :k]
    t = T[:, :k, k]
    trans_points = torch.einsum("bij,bnj->bni", R, pts) + t.unsqueeze(1)
    return trans_points


def _get_group_shape(
    batch_shape: torch.Size, batch_group_shape: torch.Size
) -> Tuple[torch.Size, int]:
    n = len(batch_shape)
    assert batch_shape == batch_group_shape[:n]
    group_shape = batch_group_shape[n:]
    batch_numel = batch_shape.numel()
    return group_shape, batch_numel


def normalized_intrinsics_from_pixel_intrinsics(
    intrinsics: Tensor, image_shape: Tuple[int, int]
) -> Tensor:
    """Transform intrinsics matrix in pixel coordinates to one in normalized coordinates.

    Args:
        intrinsics: (*, 3, 3) or (*, 4)
        image_shape: (height, width)
    Returns:
        intrinsics_n: (*, 3, 3) or (*, 4)
    """
    if intrinsics.shape[-2:] == (3, 3):
        intrinsics = flat_intrinsics_from_intrinsics_matrix(intrinsics)
        got_matrix_format = True
    elif intrinsics.shape[-1] == 4:
        got_matrix_format = False
    else:
        raise RuntimeError("intrinsics must have shape (*, 3, 3) or (*, 4)")

    image_shape_t = torch.tensor((image_shape[1], image_shape[0]), device=intrinsics.device)
    image_shape_t = image_shape_t.reshape(*([1] * (intrinsics.dim() - 1)), 2)
    image_shape_half_inv = 2.0 / image_shape_t
    new_scale = intrinsics[..., :2] * image_shape_half_inv
    new_shift = intrinsics[..., 2:] * image_shape_half_inv - 1
    intrinsics_n = torch.cat((new_scale, new_shift), dim=-1)

    if got_matrix_format:
        intrinsics_n = intrinsics_matrix_from_flat_intrinsics(intrinsics_n)
    return intrinsics_n


def pixel_intrinsics_from_normalized_intrinsics(
    intrinsics_n: Tensor, image_shape: Tuple[int, int]
) -> Tensor:
    """Transform normalized intrinsics matrix to intrinsics matrix in pixel coordinates.

    Args:
        intrinsics_n: (*, 3, 3) or (*, 4)
        image_shape: (height, width)
    Returns:
        intrinsics: (*, 3, 3) or (*, 4)
    """
    if intrinsics_n.shape[-2:] == (3, 3):
        intrinsics_n = flat_intrinsics_from_intrinsics_matrix(intrinsics_n)
        got_matrix_format = True
    elif intrinsics_n.shape[-1] == 4:
        got_matrix_format = False
    else:
        raise RuntimeError("intrinsics must have shape (*, 3, 3) or (*, 4)")

    image_shape_t = torch.tensor((image_shape[1], image_shape[0]), device=intrinsics_n.device)
    image_shape_t = image_shape_t.reshape(*([1] * (intrinsics_n.dim() - 1)), 2)
    image_shape_half = image_shape_t / 2.0
    new_scale = intrinsics_n[..., :2] * image_shape_half
    new_shift = intrinsics_n[..., 2:] * image_shape_half + image_shape_half
    intrinsics = torch.cat((new_scale, new_shift), dim=-1)
    if got_matrix_format:
        intrinsics = intrinsics_matrix_from_flat_intrinsics(intrinsics)
    return intrinsics


def normalized_pts_from_pixel_pts(n_pts: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """Transform pts in pixel coordinates to points in normalized coordinates.

    Args:
        n_pts: (*, 2)
        image_shape: (height, width)
    Returns:
        pts: (*, 2)
    """
    image_shape_t = torch.tensor((image_shape[1], image_shape[0]), device=n_pts.device)
    image_shape_t = image_shape_t.reshape(*([1] * (n_pts.dim() - 1)), 2)
    image_shape_half_inv = 2.0 / image_shape_t
    pts = image_shape_half_inv * n_pts - 1
    return pts


def pixel_pts_from_normalized_pts(pts: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """Transform points in normalized coordinates to points in pixel coordinates

    Args:
        pts: (*, 2)
        image_shape: (height, width)
    Returns:
        n_pts: (*, 2)
    """
    image_shape_t = torch.tensor((image_shape[1], image_shape[0]), device=pts.device)
    image_shape_t = image_shape_t.reshape(*([1] * (pts.dim() - 1)), 2)
    image_shape_half = image_shape_t / 2.0
    n_pts = image_shape_half * pts + image_shape_half
    return n_pts


def flat_intrinsics_from_intrinsics_matrix(K: Tensor) -> Tensor:
    """Converts 3x3 intrinsics matrix to flat representation see
    intrinsics_matrix_from_flat_intrinsics

    Args:
        K: (*, 3, 3)
    Returns:
        flat_intrinsics: (*, 4)
    """
    return torch.stack((K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]), dim=-1)


def intrinsics_matrix_from_flat_intrinsics(flat_intrinsics: Tensor) -> Tensor:
    """Converts flat_intrinsics of format (f1, f2, p1, p2) to 3x3 intrinsics matrix
        [[f1, 0, p1]
        [0 ,f2, p2]
        [0,  0,  1]]

    Args:
        flat_intrinsics: (*, 4)
    Returns:
        intrinsics_matrix: (*, 3, 3)
    """
    intrinsics_matrix = torch.zeros(
        *flat_intrinsics.shape[:-1],
        3,
        3,
        device=flat_intrinsics.device,
        dtype=flat_intrinsics.dtype,
    )
    intrinsics_matrix[..., 0, 0] = flat_intrinsics[..., 0]
    intrinsics_matrix[..., 1, 1] = flat_intrinsics[..., 1]
    intrinsics_matrix[..., 0, 2] = flat_intrinsics[..., 2]
    intrinsics_matrix[..., 1, 2] = flat_intrinsics[..., 3]
    intrinsics_matrix[..., 2, 2] = 1
    return intrinsics_matrix


def cart_from_spherical(phi_theta: Tensor, r: Union[Tensor, float] = 1.0) -> Tensor:
    """Cartesian coordinates from spherical coordinates

    Args:
        phi_theta: (*, 2)
        r:         (*)

    Returns:
        out:         (*, 3)
    """
    theta = phi_theta[..., 1]
    phi = phi_theta[..., 0]
    s = torch.sin(theta)
    z = r * s * torch.cos(phi)
    x = r * s * torch.sin(phi)
    y = -r * torch.cos(theta)
    out = torch.stack((x, y, z), dim=-1)

    return out


def spherical_from_cart(vec: Tensor, clamp_value: float = 1):
    """Spherical coordinates from cartesian
       -pi < phi < pi, 0 < theta < pi

    Args:
        vec:           (*, 3)

    Returns:
        phi_theta:     (*, 2)
        r:             (*,)
    """
    r = torch.norm(vec, p=2, dim=-1).clamp(min=1e-6)
    theta = torch.acos((-vec[..., 1] / r).clamp(-clamp_value, clamp_value))
    phi = torch.atan2(vec[..., 0], vec[..., 2])
    phi_theta = torch.stack((phi, theta), dim=-1)

    return phi_theta, r


def compute_jacobian(f, x, *theta):
    """Compute the jacobian of f with respect to x where f is a function parametrized by theta_i

    Args:
        f: function taking (b, n, d) x (b, t_1) x ... x (b, t_k) -> (b, n, d)
           s.t. out[b, i, j] = f_b(theta_1[b,:], ..., theta_k[b,:], x[b, i, j])
        theta: Tensors each of shapes (b, t_i)
        x: Tensor (b, n, d)
    Returns:
        y = f(x, *theta): Tensor (b,n,d)
        Df = partial_f/partial_x at x: Tensor (b,n,d,d)
    """
    b, n, dim = x.shape
    id = torch.eye(dim, device=x.device).reshape(1, 1, dim, dim).expand(b, n, dim, dim)
    Df_columns = []
    x = x.detach()
    theta = tuple(theta_i.detach() for theta_i in theta)
    x.requires_grad = True
    y = f(x, *theta)
    for i in range(0, dim):
        torch.autograd.backward(y, id[:, :, i], retain_graph=True)
        Df_columns.append(x.grad.detach().clone())
        x.grad.zero_()

    Df = torch.stack(Df_columns, dim=2)
    return y.detach(), Df


def fit_polynomial(x: Tensor, y: Tensor, degree: int) -> Tensor:
    """Fit a polynomial given a number of (x,y) pairs

    Args:
        x:        (b, num_points)
        y:        (b, num_points)
        degree:   degree of polynomial to fit
    Returns:
        out:           (b, degree+1)
    """
    cur_col = torch.ones_like(x)
    cols = [cur_col]
    for _ in range(degree):
        cur_col = cur_col * x
        cols.append(cur_col)
    mat = torch.stack(cols, dim=-1)
    coeffs = torch.linalg.lstsq(mat, y)[0]
    return coeffs


def apply_poly(coeffs: Tensor, vals: Tensor) -> Tensor:
    """Apply a batch of polynomials to batches of groups of numbers

    formally:
    out[b,i] = coeffs[b,0] * vals[b,:]**0 + coeffs[b,1] * vals[b,:]**1 + ...
                   + coeffs[b,k-1] * vals[b,:]**(k-1)
    Args:
        coeffs:        (b, k)
        val:           (b, n)
    Returns:
        out:           (b, n)
    """
    k = coeffs.size(1)
    out = torch.zeros_like(vals)
    for i in range(k - 1, -1, -1):
        out = coeffs[:, i : i + 1] + out * vals
    return out


def crop_to_affine(
    lrtb: Tensor, normalized: bool = True, image_shape: Optional[Tuple[int, int]] = None
):
    """Given the 4 corners of a box as a tensor lrtb = left, right, top, bottom return the
    corresponding affine transform

    Args:
        lrtb: (b*, 4)
        normalized: bool, whether 4 corners are in normalized image coordinates
        image_shape: needed when not using normalized coordinates
    Returns:
        out:           (b, n)
    """
    lrtb = lrtb.unflatten(-1, (2, 2))
    lt = lrtb[..., 0]
    rb = lrtb[..., 1]
    if not normalized:
        if image_shape is None:
            raise RuntimeError("If using unnormalized pixel positions must specify image_shape")
        lt = normalized_pts_from_pixel_pts(lt, image_shape)
        rb = normalized_pts_from_pixel_pts(rb, image_shape)

    det = lt - rb
    scale = -2 / det
    shift = (lt + rb) / det
    flat_affine = torch.cat((scale, shift), dim=-1)
    return flat_affine


def flatten_cubemap_for_visual(cubemap: Tensor, mode: int = 0):
    """Flatten cubemap for visualization. Mode 0 is t shape. Mode 1
        less empty space

    Args:
        cubemap: (*, c, 6*w, w)

    Returns:
        flattened:  (*, c, 3w, 4w) if mode == 0 else (*, c, 2w, 4w)
    """
    batch_shape = cubemap.shape[:-3]
    # (b, c, 6, w, w)
    cubemap = cubemap.reshape(-1, *cubemap.shape[-3:]).unflatten(-2, (6, -1))
    faces = torch.unbind(cubemap, dim=2)
    middle = torch.cat([faces[1], faces[4], faces[0], faces[5]], dim=-1)  # (b, c, w, 4w)
    if mode == 0:
        z = torch.full_like(faces[0], torch.nan)
        top = torch.cat([z, faces[2], z, z], dim=-1)
        bottom = torch.cat([z, faces[3], z, z], dim=-1)
        flattened = torch.cat([top, middle, bottom], dim=2)
    else:
        w = cubemap.size(-1)
        wo2 = int(w / 2)

        top_left = faces[2][:, :, wo2:, :]
        top_right = torch.flip(faces[2][:, :, :wo2, :], (2, 3))
        z = torch.full_like(top_left, torch.nan)
        top = torch.cat([z, top_left, z, top_right], dim=-1)

        bottom_left = faces[3][:, :, :wo2, :]
        bottom_right = torch.flip(faces[3][:, :, wo2:, :], (2, 3))
        bottom = torch.cat([z, bottom_left, z, bottom_right], dim=-1)
        flattened = torch.cat([top, middle, bottom], dim=2)
    flattened = flattened.flip(2)
    return flattened.reshape(batch_shape + flattened.shape[1:])
