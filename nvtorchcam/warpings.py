import torch
import torch.nn as nn
from nvtorchcam import utils
from nvtorchcam import cameras
from typing import Optional, Dict, Tuple, Union, Any, Callable, List
from torch import Tensor
import torchvision.transforms as TVT

__all__ = [
    "backwarp_warp_pts",
    "backward_warp",
    "resample_by_intrinsics",
    "get_consistancy_check_data",
    "fuse_depths_mvsnet",
    "stereo_rectify",
    "ray_sphere_intersection",
    "render_sphere_image",
    "affine_transform_image",
    "crop_resize_image",
]


TensorOrTensorList = Union[Tensor, List[Tensor]]


def backwarp_warp_pts(
    trg_cam: cameras.CameraBase,
    trg_depth: Tensor,
    src_cam: cameras.CameraBase,
    trg_cam_to_src_cam: Tensor,
    depth_is_along_ray: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute interpolation points to warp a source image or *group_shape source images to
    a target camera given a target camera depth.

    Args:
        trg_cam: (*batch_shape)
        trg_depth: (*batch_shape, *depthmaps_per_camera, h, w)
        src_cam: (*batch_shape, *group_shape) if we want to warp *group_shape src images to target
        trg_cam_to_src_cam (*batch_shape, *group_shape, 4, 4)
        depth_is_along_ray: True = using distance, False = using depth

    Returns:
        src_pts: (*batch_shape, *group_shape, *depthmaps_per_camera, h, w, pixel_dim) points to
                 interpolate src for warping
        src_pts_depth (*batch_shape, *group_shape, *depthmaps_per_camera, h, w)
        valid_mask:  (*batch_shape, *group_shape, *depthmaps_per_camera, h, w)
    """
    batch_shape = trg_cam.shape
    assert (
        trg_depth.shape[: len(batch_shape)] == batch_shape
    ), "invalid depth shape in backward_warp_pts"
    assert src_cam.shape + (4, 4) == trg_cam_to_src_cam.shape
    hw = trg_depth.shape[-2:]
    group_shape, batch_numel = utils._get_group_shape(batch_shape, src_cam.shape)
    depthmaps_per_cam_shape, _ = utils._get_group_shape(batch_shape, trg_depth.shape[:-2])
    group_numel = group_shape.numel()

    trg_cam = trg_cam.reshape(batch_numel)  # (b)
    src_cam = src_cam.reshape(batch_numel, group_numel)  # (b,g)
    trg_cam_to_src_cam = trg_cam_to_src_cam.reshape(batch_numel, group_numel, 4, 4)
    trg_depth = trg_depth.reshape(batch_numel, -1, *hw)  # (b,d,h,w)

    trg_pc, valid_unproj = trg_cam.unproject_depth(
        trg_depth, depth_is_along_ray=depth_is_along_ray
    )  # (b,d,h,w,3) (b,d,h,w)
    trg_pc = trg_pc.unsqueeze(1).expand(-1, group_numel, -1, -1, -1, -1)  # (b,g,d,h,w,3)
    src_pc = utils.apply_affine(trg_cam_to_src_cam, trg_pc)  # (b,g,d,h,w,3)

    src_pts, src_pts_depth, valid_proj = src_cam.project_to_pixel(
        src_pc, depth_is_along_ray
    )  # (b,g,d,h,w,pixel_dim)

    valid_mask = valid_unproj.unsqueeze(1) & valid_proj
    if src_pts.size(-1) == 2:
        valid_mask = valid_mask & utils.in_image(src_pts)

    src_pts = src_pts.reshape(*batch_shape, *group_shape, *depthmaps_per_cam_shape, *hw, -1)
    valid_mask = valid_mask.reshape(*batch_shape, *group_shape, *depthmaps_per_cam_shape, *hw)
    src_pts_depth = src_pts_depth.reshape(*batch_shape, *group_shape, *depthmaps_per_cam_shape, *hw)

    return src_pts, src_pts_depth, valid_mask


def backward_warp(
    trg_cam: cameras.CameraBase,
    trg_depth: Tensor,
    src_cam: cameras.CameraBase,
    src_image: TensorOrTensorList,
    trg_cam_to_src_cam: Tensor,
    interp_mode: Union[str, List[str]] = "bilinear",
    padding_mode: Union[str, List[str]] = "zeros",
    depth_is_along_ray: bool = False,
    set_invalid_pix_to_nan: bool = True,
) -> Tuple[TensorOrTensorList, Tensor, Tensor]:
    """Warp a source image or N source images to a target camera given target camera depth

    Args:
        trg_cam: (*batch_shape)
        trg_depth: (*batch_shape, *depthmaps_per_camera, h, w)
        src_cam: (*batch_shape, *group_shape)
        src_image: (*batch_shape, *group_shape, c, h', w') or list of these shapes
        trg_cam_to_src_cam: (*batch_shape, *group_shape, 4, 4)
        depth_is_along_ray: True = using distance, False = using depth
        interp_mode: method of interpolation if src_image is list should be a list of the same length


    Returns:
        src_image_warped: (*batch_shape, *group_shape, c, *depthmaps_per_camera, h, w) or list of these
        pts_src: (*batch_shape, *group_shape, *depthmaps_per_camera, h, w, pixel_dim)
        valid_mask: (*batch_shape, *group_shape, *depthmaps_per_camera, h, w)
    """
    src_image, interp_mode, padding_mode, unwrap_list = _parse_image_interp_mode_padding_mode(
        src_image, interp_mode, padding_mode
    )

    src_pts, _, valid_mask = backwarp_warp_pts(
        trg_cam, trg_depth, src_cam, trg_cam_to_src_cam, depth_is_along_ray
    )

    src_image_warped = []
    for src_image_i, mode_i, padding_mode_i in zip(src_image, interp_mode, padding_mode):
        if mode_i == "nearest":
            align_corners = None
        else:
            align_corners = False
        if isinstance(src_cam, cameras.CubeCamera):
            src_image_warped_i = utils.samples_from_cubemap(src_image_i, src_pts, mode=mode_i)
        else:
            src_image_warped_i = utils.samples_from_image(
                src_image_i,
                src_pts,
                mode=mode_i,
                align_corners=align_corners,
                padding_mode=padding_mode_i,
            )
        if set_invalid_pix_to_nan:
            src_image_warped_i = torch.where(
                valid_mask.unsqueeze(len(src_cam.shape)), src_image_warped_i, torch.nan
            )
        src_image_warped.append(src_image_warped_i)

    if unwrap_list:
        src_image_warped = src_image_warped[0]

    return src_image_warped, src_pts, valid_mask


def resample_by_intrinsics(
    src_image: TensorOrTensorList,
    src_cam: cameras.CameraBase,
    trg_cam: cameras.CameraBase,
    trg_size: Tuple[int, int],
    rotation_trg_to_src: Optional[Tensor] = None,
    interp_mode: Union[str, List[str]] = "nearest",
    padding_mode: Union[str, List[str]] = "zeros",
    depth_is_along_ray: bool = False,
    set_invalid_pix_to_nan: bool = True,
) -> Tuple[TensorOrTensorList, Tensor]:
    """Warp an image from a src_camera model to a
    Args:
        src_image: (*batch_shape, c, h, w) or list of (*batch_shape, c, h, w)
        src_camera: (*batch_shape)
        trg_camera: (*batch_shape)
        trg_size: (h', w') = height_width TODO better notation
        rotation_trg_to_src: (*batch_shape, 4, 4)
        mode: interpolation mode
        unit_vec: should have no effect except some corner cases to be explained ...
        set_invalid_pix_to_nan: Whether to set invalid pixels in output image to nan
    Returns:
        trg_image: (*batch_shape, c, h', w') or list of (*batch_shape, c, h', w')
        valid_mask: (*batch_shape, h', w')
    """

    if not (src_cam.is_central() and trg_cam.is_central()):
        raise RuntimeError("got non-central camera in resample_by_intrinsics.")
    batch_shape = src_cam.shape

    trg_to_src = (
        torch.eye(4, device=src_cam.device)
        .reshape(*([1] * len(batch_shape)), 4, 4)
        .repeat(*batch_shape, 1, 1)
    )
    if rotation_trg_to_src is not None:
        trg_to_src[..., :3, :3] = rotation_trg_to_src

    trg_depth = (
        torch.ones(trg_size, device=src_cam.device)
        .reshape(*([1] * (len(batch_shape))), *trg_size)
        .expand(*batch_shape, *trg_size)
    )
    src_image_warped, _, valid_mask = backward_warp(
        trg_cam,
        trg_depth,
        src_cam,
        src_image,
        trg_to_src,
        interp_mode,
        padding_mode,
        depth_is_along_ray,
        set_invalid_pix_to_nan,
    )

    return src_image_warped, valid_mask


def get_consistancy_check_data(
    trg_cam: cameras.CameraBase,
    trg_depth: Tensor,
    src_cam: cameras.CameraBase,
    src_depth: Tensor,
    trg_cam_to_src_cam: Tensor,
    depth_is_along_ray: bool = False,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        trg_cam: (*batch_shape)
        trg_depth: (*batch_shape, h, w)
        src_cam: (*batch_shape, *group_shape)
        src_depth: (*batch_shape, *group_shape, h', w')
        trg_cam_to_src_cam: (*batch_shape, *group_shape, 4, 4)
        depth_is_along_ray: True = using distance, False = using depth

    Returns:
        recon_pixel_grid: (*batch_shape, *group_shape, h, w, 2)
        recon_trg_depth: (*batch_shape, *group_shape, h, w)
        valid_mask: (*batch_shape, *group_shape, h, w)
    """

    # (*batch_shape, *group_shape, 1, h, w) (*batch_shape, *group_shape, h, w, 2)   (*batch_shape, *group_shape,  h, w)
    src_depth_warped, src_pts, valid_bw_warp = backward_warp(
        trg_cam,
        trg_depth,
        src_cam,
        src_depth.unsqueeze(-3),
        trg_cam_to_src_cam,
        depth_is_along_ray=depth_is_along_ray,
        interp_mode=interp_mode,
        padding_mode=padding_mode,
    )

    # (*batch_shape, *group_shape,  h, w, 3) (*batch_shape, *group_shape,  h, w, 3) #(*batch_shape, *group_shape,  h, w)
    origin, dirs, valid_unproj = src_cam.pixel_to_ray(src_pts, depth_is_along_ray)
    # (*batch_shape, *group_shape,  h, w, 3)
    pc_src = origin + dirs * src_depth_warped.squeeze(-3).unsqueeze(-1)
    # (*batch_shape, *group_shape,  h, w, 3)
    pc_trg = utils.apply_affine(torch.inverse(trg_cam_to_src_cam), pc_src)

    # (*batch_shape, *group_shape,  h, w, 2) (*batch_shape, *group_shape,  h, w) (*batch_shape, *group_shape,  h, w)
    recon_pixel_grid, recon_trg_depth, valid_proj = trg_cam.project_to_pixel(
        pc_trg, depth_is_along_ray
    )

    valid_mask = valid_bw_warp & valid_unproj & valid_proj

    return recon_pixel_grid, recon_trg_depth, valid_mask


def fuse_depths_mvsnet(
    trg_cam: cameras.CameraBase,
    trg_depth: Tensor,
    src_cam: cameras.CameraBase,
    src_depth: Tensor,
    trg_cam_to_src_cam: Tensor,
    pixel_error_threshold: float = 1,
    relative_depth_error_threshold: float = 0.01,
    num_image_threshold=3,
    depth_is_along_ray: bool = False,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Logic borrowed from https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/test.py,
       adapted for any camera model with batching and GPU support

    Args:
        trg_cam: (*batch_shape)
        trg_depth: (*batch_shape, h, w)
        src_cam:  (*batch_shape, N)
        src_depth: (*batch_shape, N, h', w')
        trg_cam_to_src_cam: (*batch_shape, N, 4, 4)
        depth_is_along_ray: True = using distance, False = using depth

    Return:
        fused_depth (*batch_size, h, w)
        fused_valid (*batch_size, h, w)
    """

    res = trg_depth.shape[-2:]
    # (*batch_shape, N, h, w, 2) (*batch_shape, N, h, w) (*batch_shape, N, h, w)
    n_recon_pixel_grid, recon_trg_depth, valid_mask = get_consistancy_check_data(
        trg_cam,
        trg_depth,
        src_cam,
        src_depth,
        trg_cam_to_src_cam,
        depth_is_along_ray,
        interp_mode,
        padding_mode,
    )
    recon_pixel_grid = utils.pixel_pts_from_normalized_pts(
        n_recon_pixel_grid, res
    )  # (*batch_shape, N, h, w, 2)

    n_pixel_grid = trg_cam.get_normalized_grid(res).unsqueeze(-4)  # (*batch_shape, 1, h, w, 2)
    pixel_grid = utils.pixel_pts_from_normalized_pts(
        n_pixel_grid, res
    )  # (*batch_shape, 1, h, w, 2)

    # (*batch_shape, N, h, w)
    pixel_error = torch.norm(recon_pixel_grid - pixel_grid, dim=-1)
    # (*batch_shape, N, h, w)
    pixel_error_mask = pixel_error < pixel_error_threshold

    trg_depth = trg_depth.unsqueeze(-3)  # (*batch_shape, 1, h, w)
    rel_depth_err = torch.abs(trg_depth - recon_trg_depth) / trg_depth  # (*batch_size, N, h, w)
    # (*batch_size, N, h, w)
    rel_depth_err_mask = rel_depth_err < relative_depth_error_threshold

    # (*batch_size, N, h, w)
    src_masks = valid_mask & pixel_error_mask & rel_depth_err_mask
    _, _, trg_mask = trg_cam.get_camera_rays(res, depth_is_along_ray)  # (*batch_size, h, w)
    trg_mask = trg_mask.unsqueeze(-3)  # (*batch_size, 1, h, w)

    # (*batch_size, N+1, h, w)
    depths_to_fuse = torch.cat([trg_depth, recon_trg_depth], dim=-3)
    # (*batch_size, N+1, h, w)
    fusion_masks = torch.cat([trg_mask, src_masks], dim=-3)

    fusion_masks_sum = torch.sum(fusion_masks, dim=-3)  # (*batch_size, h, w)

    fused_depth = (
        torch.sum(depths_to_fuse * fusion_masks, dim=-3) / fusion_masks_sum
    )  # (*batch_size, h, w)
    # (*batch_size, h, w)
    fused_valid = fusion_masks_sum >= num_image_threshold

    return fused_depth, fused_valid


def stereo_rectify(
    images: TensorOrTensorList,
    cams: cameras.CameraBase,
    to_world: Tensor,
    height_width: Tuple[int, int],
    top_bottom: bool = True,
    trg_cams: Optional[cameras.CameraBase] = None,
    interp_mode: Union[str, List[str]] = "nearest",
    padding_mode: Union[str, List[str]] = "zeros",
    depth_is_along_ray: bool = False,
    set_invalid_pix_to_nan: bool = True,
) -> Tuple[TensorOrTensorList, Tensor, Tensor]:
    """Warping images from central cameras such that they have the same extrinsic rotation matrix.

    Args:
        image: (*b , 2, c, h, w) where image[:,0] is top (left) image and image[:,1] is bottom (right) image
        cams: (*b, 2)
        to_world: (*b, 2, 4, 4)
        height_width: (h', w') = height_width of rectified images TODO better notation
        top_bottom: whether simulated cameras are on top of each other (generally for ERPs) or side-by-side (generally for pinholes)
        trg_cams: (*b, 2) camera model images are warped on to. Default is ERP
    Returns:
        new_images: (*b, 2, c, h', w')
        new_to_world: (*b, 4, 4)
        valid_mask: (*b, 2, h', w')
    """

    assert cams.shape[-1] == 2
    assert cams.shape == images.shape[:-3]
    assert to_world.shape == cams.shape + (4, 4)

    if not cams.is_central():
        raise RuntimeError("got non-central camera in rectify_onto_two_erps.")

    if trg_cams is None:
        trg_cams = cameras.EquirectangularCamera.make(batch_shape=cams.shape)
    else:
        assert cams.shape == trg_cams.shape

    trg_cams = trg_cams.to(to_world.device)

    # get new to world
    displacement_cam0_to_cam1 = torch.nn.functional.normalize(
        to_world[..., 1, :3, 3] - to_world[..., 0, :3, 3], dim=-1
    )  # (*b, 3)
    new_z = torch.sum(to_world[..., :3, 2], dim=-2)  # (*b, 3)
    new_z = torch.nn.functional.normalize(
        _make_orthogonal(displacement_cam0_to_cam1, new_z), dim=-1
    )
    if top_bottom:
        new_y = displacement_cam0_to_cam1
        new_x = torch.cross(new_y, new_z, dim=-1)
    else:
        new_x = displacement_cam0_to_cam1
        new_y = torch.cross(new_z, new_x, dim=-1)

    R_new_to_world = torch.stack([new_x, new_y, new_z], dim=-1)  # (*b, 3, 3)

    R_new_to_world = R_new_to_world.reshape(-1, 3, 3).unsqueeze(1).expand(-1, 2, -1, -1)
    R_old_to_world_inv = to_world[..., :3, :3].transpose(-1, -2).reshape(-1, 2, 3, 3)
    R = torch.bmm(R_old_to_world_inv.flatten(0, 1), R_new_to_world.flatten(0, 1)).reshape(
        *cams.shape, 3, 3
    )

    new_images, valid_mask = resample_by_intrinsics(
        images,
        cams,
        trg_cams,
        height_width,
        rotation_trg_to_src=R,
        interp_mode=interp_mode,
        padding_mode=padding_mode,
        depth_is_along_ray=depth_is_along_ray,
        set_invalid_pix_to_nan=set_invalid_pix_to_nan,
    )
    new_to_world = to_world.clone()
    new_to_world[..., :3, :3] = R_new_to_world
    return new_images, new_to_world, valid_mask


def _make_orthogonal(u, v):
    """make v orthogonal to u by Gram-Schmitt

      Args:
        u: (*b , n)
        v: (*b, n)

    Returns:
        new_v: (b*,n)
    """
    u_norm = torch.nn.functional.normalize(u, dim=-1)
    dot = torch.sum(u_norm * v, dim=-1, keepdim=True)
    new_v = v - dot * u_norm
    return new_v


def ray_sphere_intersection(
    origins: Tensor, dirs: Tensor, center: Tensor = torch.zeros(3), radius: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Intersect batch of rays with a sphere. Used for testing warping functions

    Args:
        origins: (*batch_shape,3)
        dirs: (*batch_shape, 3)
        center: (3,)
        radius: float

    Return:
        intersection: (*batch_size, 3)
        distance: (*batch_size)
        valid: (*batch_size)
    """
    batch_shape = origins.shape[:-1]
    center = center.to(origins.device)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)

    oc = origins - center.reshape(*([1] * len(batch_shape)), 3)

    dot = torch.sum(oc * dirs, dim=-1)
    c = torch.sum(oc**2, dim=-1) - radius**2

    discriminant = dot**2 - c

    # Initialize masks and intersection points
    sqrt_discrim = torch.sqrt(discriminant.clamp(min=0))

    t0 = -dot - sqrt_discrim
    t1 = -dot + sqrt_discrim

    valid = (discriminant >= 0) & (t1 >= 0)
    distance = torch.where(t0 > 0, t0, t1)

    intersection_points = origins + distance.unsqueeze(-1) * dirs

    return intersection_points, distance, valid


def render_sphere_image(
    cam: cameras.CameraBase,
    to_world: Tensor,
    res: Tuple[int, int],
    radius: float = 1.0,
    invalid_value: float = float("nan"),
) -> Tuple[Tensor, Tensor]:
    """Render images of sphere where the sphere is textured using its coordinate locations also return distance maps.
        This is used to test warping functions

    Args:
        cam: (*batch_shape,)
        to_world: (*batch_shape, 4, 4)
        center: Tuple(h, w)
        radius: float

    Return:
        image: (*batch_size, 3, h, w)
        distance: (*batch_size, 1, h, w)
    """
    origin, dirs, valid_rays = cam.get_camera_rays(res, True)
    origin = utils.apply_affine(to_world, origin)
    dirs = utils.apply_matrix(to_world[..., :3, :3], dirs)
    intersection_points, distance, valid_intersect = ray_sphere_intersection(
        origin, dirs, radius=radius
    )
    valid = valid_rays & valid_intersect
    intersection_points[~valid, :] = invalid_value
    distance[~valid] = torch.nan

    image = intersection_points.transpose(-1, -2).transpose(-2, -3)
    distance = distance.unsqueeze(-3)
    depth = distance / dirs[..., 2].unsqueeze(1)

    return image, distance, depth


def affine_transform_image(
    image: TensorOrTensorList,
    affine: Tensor,
    interp_mode: Union[str, List[str]] = "bilinear",
    padding_mode: Union[str, List[str]] = "zeros",
    out_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[TensorOrTensorList, Tensor]:
    """Apply an affine transform to an image. Is consistent with TensorDictionaryAffineCamera.affine_transform
       in the sense that if image0 comes from camera0 and
       $ image1 = affine_transform_image(image, affine)
       $ camera1 = camera.affine_transform(affine)
       Then image1 comes from camera1

    Args:
        image: (*batch_shape, c, h, w)  or list of (*batch_shape, c, h, w)
        affine: (*batch_shape, 4)
        interp_mode: method of interpolation if image is list should be a list of the same length
        padding_mode:
        out_shape: height and width of output tensor, if not specified equal to input height and width

    Returns:
        transformed_image: (*batch_shape, c, *out_shape) or list of (*batch_shape, c,  *out_shape)
        valid_mask: (*batch_shape,  *out_shape)
    """

    affine = cameras._parse_intrinsics(affine)
    batch_shape = affine.shape[:-1]
    image, interp_mode, padding_mode, unwrap_list = _parse_image_interp_mode_padding_mode(
        image, interp_mode, padding_mode
    )

    if out_shape is None:
        if isinstance(image, list):
            out_shape = image[0].shape[-2:]
        else:
            out_shape = image.shape[-2:]

    grid = utils.get_normalized_grid(out_shape, device=image[0].device)  # (h, w, 2)
    grid = grid.reshape(*([1] * len(batch_shape)), *grid.shape)
    affine = affine.unsqueeze(-2).unsqueeze(-2)
    grid = (grid - affine[..., 2:4]) / affine[..., 0:2]

    valid_mask = utils.in_image(grid)
    transformed_image = []
    for image_i, padding_mode_i, interp_mode_i in zip(image, padding_mode, interp_mode):
        transformed_image.append(
            utils.samples_from_image(image_i, grid, padding_mode=padding_mode_i, mode=interp_mode_i)
        )

    if unwrap_list:
        transformed_image = transformed_image[0]

    return transformed_image, valid_mask


def crop_resize_image(
    image: TensorOrTensorList,
    lrtb: Tensor,
    normalized=True,
    interp_mode: Union[str, List[str]] = "bilinear",
    padding_mode: Union[str, List[str]] = "zeros",
    out_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[TensorOrTensorList, Tensor]:
    """Crop and resize an image. Is consistent with TensorDictionaryAffineCamera.crop

    Args:
        image: (*batch_shape, c, h, w)  or list of (*batch_shape, c, h, w)
        lrtb: (*batch_shape, 4) left, right, top, bottom
        normalized: Whether lrtb are in normalized coordinates or pixel coordinates
        interp_mode: method of interpolation if image is list should be a list of the same length
        padding_mode:
        out_shape: height and width of output tensor, if not specified equal to input height and width

    Returns:
        transformed_image: (*batch_shape, c, *out_shape) or list of (*batch_shape, c,  *out_shape)
        valid_mask: (*batch_shape,  *out_shape)
    """
    if isinstance(image, list):
        image_shape = image[0].shape[-2:]
    else:
        image_shape = image.shape[-2:]

    affine = utils.crop_to_affine(lrtb, normalized=normalized, image_shape=image_shape)
    return affine_transform_image(
        image,
        affine,
        interp_mode=interp_mode,
        padding_mode=padding_mode,
        out_shape=out_shape,
    )


def _parse_image_interp_mode_padding_mode(
    image: TensorOrTensorList,
    interp_mode: Union[str, List[str]],
    padding_mode: Union[str, List[str]],
) -> Tuple[List[Tensor], List[str], List[str], bool]:
    if isinstance(image, list):  # if images are not a list make them a list of length 1
        if not isinstance(interp_mode, list):
            interp_mode = [interp_mode] * len(image)
        else:
            assert len(interp_mode) == len(image)
        if not isinstance(padding_mode, list):
            padding_mode = [padding_mode] * len(image)
        else:
            assert len(padding_mode) == len(image)
        is_list = False
    else:
        image = [image]
        interp_mode = [interp_mode]
        padding_mode = [padding_mode]
        is_list = True
    return image, interp_mode, padding_mode, is_list


class RandomResizedCropFlip(nn.Module):
    def __init__(
        self,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        flip_probability: float = 0.0,
        mode: str = "torchvision",
        interp_mode: Union[str, List[str]] = "bilinear",
        padding_mode: Union[str, List[str]] = "zeros",
        share_crop_across_views: bool = False,
        world_flip: bool = False,
        out_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Randomly crop (and possibly flip) a batch of scene views, adjust camera intrinsics, and rescale 
        to out_shape.

        If mode == 'torchvision', the crop is based on torchvision's RandomResizedCrop algorithm. See 
        torchvision's documentation for scale and ratio definitions.

        If mode == 'width_aspect', the crop is selected by choosing a uniform number w in scale, which 
        is the fraction of the original image width for the crop. Then, a uniform number r in ratio is 
        chosen, and w/r is the fraction of the original height in the crop. Finally, a random position 
        for this box location is chosen.

        Args:
            scale: When mode == 'torchvision', as defined in torchvision's RandomResizedCrop algorithm.
                   When mode == 'width_aspect', select random width fraction in scale.
            ratio: When mode == 'torchvision', as defined in torchvision's RandomResizedCrop algorithm.
                   When mode == 'width_aspect', then as above.
            flip_probability: Probability images will be horizontally flipped.
            interp_mode: Method of interpolation.
            padding_mode: Padding mode for "uncrops".
            share_crop_across_views: Whether to use the same crop for each view of the same scene.
            world_flip: Whether to flip to_world matrices during horizontal flip so the world is 
                        flipped and focal lengths stay positive.
            out_shape: Height and width of output tensor. If None, equal to input height and width.
        """
        super().__init__()

        self.scale = scale
        self.ratio = ratio
        self.flip_probability = flip_probability
        self.interp_mode = interp_mode
        self.padding_mode = padding_mode
        self.share_crop_across_frames = share_crop_across_views
        self.world_flip = world_flip
        self.out_shape = out_shape
        if mode == "torchvision":
            self.get_crop_matrix = self.get_crop_matrix_torchvision
        elif mode == "width_aspect":
            self.get_crop_matrix = self.get_crop_matrix_width_aspect
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'torchvision' or 'width_aspect'.")

    def get_crop_matrix_width_aspect(
        self, N: int, device: torch.device, image: TensorOrTensorList
    ) -> Tensor:
        new_half_width = torch.empty(N, device=device).uniform_(*self.scale)
        new_half_height = new_half_width / torch.empty(N, device=device).uniform_(*self.scale)

        new_center_hori = (1 - new_half_width) * torch.rand(N, device=device)
        new_center_vert = (1 - new_half_height) * torch.rand(N, device=device)

        crop = torch.zeros(N, 3, 3, device=device)
        crop[:, 0, 0] = 1 / new_half_width
        crop[:, 1, 1] = 1 / new_half_height
        crop[:, 0, 2] = new_center_hori
        crop[:, 1, 2] = new_center_vert
        crop[:, 2, 2] = 1
        return crop

    def get_crop_matrix_torchvision(
        self, N: int, device: torch.device, image: TensorOrTensorList
    ) -> Tensor:
        if isinstance(image, List):
            image = image[0]
        lrtbs = []

        for _ in range(N):
            i, j, h, w = TVT.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            lrtbs.append((j, j + w, i, i + h))

        lrtb = torch.tensor(lrtbs, device=device)

        flat_affine = utils.crop_to_affine(lrtb, normalized=False, image_shape=image.shape[-2:])
        crop = utils.intrinsics_matrix_from_flat_intrinsics(flat_affine)
        return crop

    def forward(
        self,
        image: TensorOrTensorList,
        camera: cameras.TensorDictionaryAffineCamera,
        to_world: Optional[Tensor] = None,
    ):
        """
        Args:
            image: (batch, views, c, h, w)  or list of (batch, views, c, h, w)
            camera: (batch, views)
            to_world: (batch, views, 4, 4) needed when using world flipping

        Returns:
            new_image: (batch, views, c, h, w)  or list of (batch, views, c, h, w)
            new_camera: (batch, views)
            new_to_world: (batch, views, 4, 4)
            valid_mask: (batch, views, h, w)
        """
        if to_world is None and self.world_flip:
            raise RuntimeError("Must include a world matrix")

        if to_world is not None:
            new_to_world = to_world.clone()
        else:
            new_to_world = None

        device = camera.device
        b, views = camera.shape

        new_camera = camera

        if self.flip_probability > 0.0:
            flip = 2 * (torch.rand(b, device=device) > self.flip_probability).float() - 1

            flip3 = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(b, views, 1, 1)
            flip3[:, :, 0, 0] = flip.reshape(-1, 1)

            new_camera = new_camera.affine_transform(flip3)
            if self.world_flip:
                new_camera = new_camera.affine_transform(flip3, multiply_on_right=True)
                # conjugate to_world by flip4
                new_to_world[:, :, 0, :] *= flip.reshape(-1, 1, 1)
                new_to_world[:, :, :, 0] *= flip.reshape(-1, 1, 1)
        else:
            flip3 = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(b, views, 1, 1)

        if self.share_crop_across_frames:
            crop_matrix = self.get_crop_matrix(b, device, image).unsqueeze(1).expand(b, views, 3, 3)
        else:
            crop_matrix = self.get_crop_matrix(b * views, device, image).reshape(b, views, 3, 3)

        new_camera = new_camera.affine_transform(crop_matrix)

        crop_matrix = torch.bmm(crop_matrix.flatten(0, 1), flip3.flatten(0, 1)).unflatten(
            0, (b, views)
        )

        new_image, valid = affine_transform_image(
            image,
            crop_matrix,
            interp_mode=self.interp_mode,
            out_shape=self.out_shape,
            padding_mode=self.padding_mode,
        )

        return new_image, new_camera, new_to_world, valid
