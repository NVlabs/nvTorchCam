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
import math
import warnings
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Dict,
    Tuple,
    Union,
    Any,
    List,
    Iterator,
    Sequence,
    cast,
    Type,
    DefaultDict,
)
from typing_extensions import Self, TypeAlias
import re

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map
from nvtorchcam import utils
import nvtorchcam.cameras_functional as CF

__all__ = [
    "CameraBase",
    "TensorDictionaryAffineCamera",
    "PinholeCamera",
    "OrthographicCamera",
    "EquirectangularCamera",
    "OpenCVCamera",
    "OpenCVFisheyeCamera",
    "BackwardForwardPolynomialFisheyeCamera",
    "Kitti360FisheyeCamera",
    "CubeCamera",
]

ShapeLike: TypeAlias = Union[Tuple[int, ...], List[int]]


def adjust_shape_message(msg: str) -> str:
    lists = re.findall(r"\[([^\]]+)\]", msg)

    last_removed = None
    new_msg = msg

    for lst in lists:
        parts = [p.strip() for p in lst.split(",")]
        if len(parts) > 1:
            last_removed = int(parts[-1])
            new_inside = ", ".join(parts[:-1])
        else:
            new_inside = ""

        new_msg = new_msg.replace(f"[{lst}]", f"[{new_inside}]", 1)

    if last_removed is None:
        return new_msg

    def shrink(match):
        size = int(match.group(1))
        return f"input of size {size // last_removed}"

    new_msg = re.sub(r"input of size (\d+)", shrink, new_msg)

    return new_msg


class CameraBase(ABC):
    """Abstract base class for all cameras.
    Implements __torch_function__ for stack and cat which will create a heterogeneous camera batch
    if trying to concatenate derived classes of different types.
    """

    @abstractmethod
    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the ray corresponding to a pixel in camera coordinates. Also returns whether ray is
        considered valid.

        Args:
            pix: (*self.shape, *group_shape, pixel_dim)
            unit_vec: bool

        Returns:
            origin: (*self.shape, *group_shape, 3)
            dirs: (*self.shape, *group_shape, 3)
            valid: (*self.shape, *group_shape)
        """
        raise NotImplementedError("")

    @abstractmethod
    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project a 3D point in camera coordinates to a pixel. Also return depth and whether valid.

        Args:
            pts: (*self.shape, *group_shape, 3)
            depth_is_along_ray: bool

        Returns:
            pix: (*self.shape, *group_shape, pixel_dim)
            depth: (*self.shape, *group_shape)
            valid: (*self.shape, *group_shape)
        """
        raise NotImplementedError("")

    @property
    @abstractmethod
    def shape(self) -> ShapeLike:
        raise NotImplementedError("")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError("")

    @abstractmethod
    def is_central(self) -> bool:
        """Return whether camera model is central"""
        raise NotImplementedError("")

    @abstractmethod
    def __getitem__(self, index: slice) -> CameraBase:
        raise NotImplementedError("")

    @abstractmethod
    def _cat(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        """Concatenate a homogeneous sequence of cameras."""
        raise NotImplementedError("")

    @abstractmethod
    def _stack(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        """Stack a homogeneous sequence of cameras."""
        raise NotImplementedError("")

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def reshape(self, shape: Union[int, ShapeLike], *extra: int) -> Self:
        raise NotImplementedError("classes derived from CameraBase should implement reshape")

    @abstractmethod
    def permute(self, dims: Union[int, ShapeLike], *extra: int) -> Self:
        raise NotImplementedError("classes derived from CameraBase should implement permute")

    @abstractmethod
    def transpose(self, dim0: int, dim1: int) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def squeeze(self, dim: Optional[int] = None) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def unsqueeze(self, dim: int) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def expand(self, size: Union[int, ShapeLike], *extra: int) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def flip(self, dims: ShapeLike) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def clone(self) -> Self:
        raise NotImplementedError("")

    @abstractmethod
    def detach(self) -> Self:
        raise NotImplementedError("")

    def get_camera_rays(
        self, res: Tuple[int, int], unit_vec: bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get rays for camera with sensor size res

        Args:
            res: (h,w)
            unit_vec: where dirs is normalized to be a unit vector

        Returns:
            origin: (*self.shape, h, w, 3)
            dirs: (*self.shape, h, w, 3)
            valid: (*self.shape, h, w)
        """
        grid = self.get_normalized_grid(res)
        return self.pixel_to_ray(grid, unit_vec)

    def get_normalized_grid(self, res: Tuple[int, int]) -> Tensor:
        """Get normalized grid with shape res and batch_shape like self

        Args:
            res: (h,w)

        Returns:
            grid: (*self.shape, h, w, pixel_dim)
        """
        grid = utils.get_normalized_grid(res, self.device)
        shape = self.shape
        for _ in range(len(shape)):
            grid = grid.unsqueeze(0)
        grid = grid.expand(*shape, -1, -1, -1)
        return grid

    def unproject_depth(
        self, depth: Tensor, to_world: Optional[Tensor] = None, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Unproject *depthmaps_per_camera to pointcloud images. Optionally transform pointcloud
        images to world coordinates.

        Args:
            depth: (*self.shape, *depthmaps_per_camera, h, w)
            to_world (*self.shape, 4, 4)

        Returns:
            point_cloud: (*self.shape, *depth_maps_per_camera, 3)
            valid: (*self.shape, *depth_maps_per_camera)
        """
        group_shape, batch_numel = utils._get_group_shape(self.shape, depth.shape[:-2])

        origin, dirs, valid = self.get_camera_rays(depth.shape[-2:], unit_vec=depth_is_along_ray)
        for _ in group_shape:
            origin = origin.unsqueeze(-4)
            dirs = dirs.unsqueeze(-4)
            valid = valid.unsqueeze(-3)

        point_cloud = origin + dirs * depth.unsqueeze(-1)
        valid = valid.expand(point_cloud.shape[:-1])

        if to_world is not None:
            point_cloud = utils.apply_affine(to_world, point_cloud)
        return point_cloud, valid

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == torch.cat:
            dim = kwargs.get("dim", 0)
            obj_list = args[0]
            # homogeneous case
            if len(types) == 1 and _HeterogeneousCamera not in types:
                return obj_list[0]._cat(obj_list, dim=dim)
            else:
                if CubeCamera in types:
                    raise RuntimeError("CubeCamera is not supported in heterogeneous batch")
                obj_list_as_hetero1: List[CameraBase] = []
                for obj in obj_list:
                    if not isinstance(obj, _HeterogeneousCamera):
                        obj = _HeterogeneousCamera._homogeneous_to_heterogeneous(obj)
                    obj_list_as_hetero1.append(obj)
                return obj_list_as_hetero1[0]._cat(obj_list_as_hetero1, dim=dim)
        if func == torch.stack:
            dim = kwargs.get("dim", 0)
            obj_list = args[0]
            if len(types) == 1 and _HeterogeneousCamera not in types:
                return obj_list[0]._stack(obj_list, dim=dim)
            else:
                if CubeCamera in types:
                    raise RuntimeError("CubeCamera is not supported in heterogeneous batch")
                obj_list_as_hetero2: List[CameraBase] = []
                for obj in obj_list:
                    if not isinstance(obj, _HeterogeneousCamera):
                        obj = _HeterogeneousCamera._homogeneous_to_heterogeneous(obj)
                    obj_list_as_hetero2.append(obj)
                return obj_list_as_hetero2[0]._stack(obj_list_as_hetero2, dim=dim)

        raise NotImplementedError("")


class TensorDictionaryCamera(CameraBase):
    """Abstract base to implement tensor-like operations for cameras. Holds a dictionary of
    parameters called _values. Derived classes should not have any additional attributes.
    Any attributes should be stored in = _shared_attributes. Note that when concatenating (stacking)
    objects with _shared_attributes, the shared attributes will derive from the first object
    in the list being concatenated (stacked).
    """

    def __init__(
        self, _values: Dict[str, Tensor], _shared_attributes: Optional[Dict[str, Any]] = None
    ):
        self._values = _values
        self._shared_attributes = _shared_attributes
        key = next(iter(_values.keys()))
        self._shape = _values[key].shape[:-1]
        self._device = _values[key].device
        assert all(v.shape[:-1] == self._shape for k, v in self._values.items())
        assert all(v.device == self._device for k, v in self._values.items())

    @property
    def shape(self) -> ShapeLike:
        return self._shape

    @property
    def device(self) -> torch.device:
        return self._device

    def __str__(self) -> str:
        return str(type(self)) + str(self._values)

    def __getitem__(self, index: slice) -> Self:
        if isinstance(index, tuple) and (len(index) < len(self.shape)):
            raise IndexError("bad slice index")
        return type(self)({k: v[index] for k, v in self._values.items()}, self._shared_attributes)

    def _cat(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        new_values: Dict[str, Tensor] = {}
        for k in self._values.keys():
            new_values[k] = torch.cat([obj._values[k] for obj in obj_list], dim=dim)
        return type(self)(new_values, self._shared_attributes)

    def _stack(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        new_values: Dict[str, Tensor] = {}
        for k in self._values.keys():
            new_values[k] = torch.stack([obj._values[k] for obj in obj_list], dim=dim)
        return type(self)(new_values, self._shared_attributes)

    def to(self, device: torch.device) -> Self:
        return type(self)(
            {k: v.to(device) for k, v in self._values.items()}, self._shared_attributes
        )

    def reshape(self, shape: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(shape, (tuple, list)):
            shape_t = shape
        else:
            shape_t = (shape, *extra)

        try:
            new_values = {k: v.reshape(*shape_t, v.shape[-1]) for k, v in self._values.items()}
        except Exception as e:
            s = str(e)
            s2 = adjust_shape_message(s)
            raise type(e)(s2) from None

        return type(self)(new_values, self._shared_attributes)

    def permute(self, dims: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(dims, (tuple, list)):
            dims_t = dims
        else:
            dims_t = (dims, *extra)

        length = len(self.shape)
        if not all(isinstance(x, int) for x in dims_t):
            raise TypeError("permute(): argument 'dims' must be tuple of ints")

        if len(dims_t) != length:
            raise RuntimeError(
                f"permute(dims): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = {length} is not equal to len(dims) = {len(dims_t)}"
            )

        dims_t2 = []
        for x in dims_t:
            if x >= length or x < -length:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [{-length}, {length-1}], but got {x})"
                )
            if x < 0:
                dims_t2.append(x - 1)
            else:
                dims_t2.append(x)

        try:
            new_values = {k: v.permute(*dims_t2, len(self.shape)) for k, v in self._values.items()}
        except Exception as e:
            s = str(e)
            s2 = adjust_shape_message(s)
            raise type(e)(s2) from None

        return type(self)(new_values, self._shared_attributes)

    def transpose(self, dim0: int, dim1: int) -> Self:
        if (dim0 >= len(self.shape)) or (dim1 >= len(self.shape)):
            raise IndexError()
        new_values = {k: v.transpose(dim0, dim1) for k, v in self._values.items()}
        return type(self)(new_values, self._shared_attributes)

    def squeeze(self, dim: Optional[int] = None) -> Self:
        if dim is None:
            new_values: Dict[str, Tensor] = {}
            for k, v in self._values.items():
                new_v = v.squeeze()
                if v.shape[-1] == 1:
                    new_v = new_v.unsqueeze(-1)
                new_values[k] = new_v
        else:
            if dim >= len(self.shape):
                raise IndexError()
            if dim == -1:
                dim = len(self.shape) - 1
            new_values = {k: v.squeeze(dim) for k, v in self._values.items()}
        return type(self)(new_values, self._shared_attributes)

    def unsqueeze(self, dim: int) -> Self:
        if (dim > len(self.shape)) or (dim < -1):
            raise IndexError()
        if dim == -1:
            dim = len(self.shape)
        new_values = {k: v.unsqueeze(dim) for k, v in self._values.items()}
        return type(self)(new_values, self._shared_attributes)

    def expand(self, size: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(size, (tuple, list)):
            sizes_t = size
        else:
            sizes_t = (size, *extra)

        try:
            new_values = {k: v.expand(*sizes_t, -1) for k, v in self._values.items()}
        except Exception as e:
            s = str(e)
            s2 = adjust_shape_message(s)
            raise type(e)(s2) from None

        return type(self)(new_values, self._shared_attributes)

    def flip(self, dims: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(dims, (tuple, list)):
            dims_t = dims
        else:
            dims_t = (dims, *extra)
        if not all(isinstance(x, int) for x in dims_t):
            raise TypeError("flip(): argument 'dims' must be tuple of ints")

        length = len(self.shape)
        if len(dims_t) == 0:
            return self

        for d in dims_t:
            if d >= length or d < -length:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [{-length}, {length-1}], but got {d})"
                )

        new_values = {k: v.flip(dims_t) for k, v in self._values.items()}
        return type(self)(new_values, self._shared_attributes)

    def clone(self) -> Self:
        new_values = {k: v.clone() for k, v in self._values.items()}
        return type(self)(
            new_values,
            self._shared_attributes.copy() if self._shared_attributes is not None else None,
        )

    def __getattr__(self, name: str) -> Tensor:
        """Allow users to access self._values and self._shared_attributes with . operator."""
        if name == "_values":
            raise AttributeError("necessary for loading with multiple workers?")
        if name in self._values:
            return self._values[name]
        if (self._shared_attributes is not None) and name in self._shared_attributes:
            return self._shared_attributes[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def named_tensors(self) -> Iterator[Tuple[str, Tensor]]:
        """Get an iterator work self._values. Useful for setting .requires_grad etc."""
        return iter(self._values.items())

    def detach(self) -> Self:
        new_values = {k: v.detach() for k, v in self._values.items()}
        return type(self)(
            new_values,
            self._shared_attributes.copy() if self._shared_attributes is not None else None,
        )


class TensorDictionaryAffineCamera(TensorDictionaryCamera):
    """Base class for all cameras that project via a projection function followed by an affine
    intrinsics matrix. Also implements cropping and transforming intrinsics."""

    def affine_transform(self, affine: Tensor, multiply_on_right: bool = False) -> Self:
        """Multiply intrinsics on the left by an affine transformation. Supports multiplying on the
        right with flag multiply_on_right which is needed for mirroring images and the entire world
        as an augmentation.

        Args:
            affine: (*self.shape, 4) or (*self.shape, 3, 3)
            multiply_on_right: bool

        Returns:
            new camera with multiplied intrinsics.
        """
        affine = _parse_intrinsics(affine)
        assert affine.shape[:-1] == self.shape

        new_scale = self._values["affine"][..., 0:2] * affine[..., 0:2]
        if multiply_on_right:
            new_shift = (
                self._values["affine"][..., 0:2] * affine[..., 2:4]
                + self._values["affine"][..., 2:4]
            )
        else:
            new_shift = self._values["affine"][..., 2:4] * affine[..., 0:2] + affine[..., 2:4]
        new_affine = torch.cat((new_scale, new_shift), dim=-1)
        new_values = self._values.copy()
        new_values["affine"] = new_affine
        return type(self)(_values=new_values, _shared_attributes=self._shared_attributes)

    def crop(
        self, lrtb: Tensor, normalized: bool = True, image_shape: Optional[Tuple[int, int]] = None
    ) -> Self:
        """Transform the intrinsics so camera will be consistent with a cropped version of the
        image.

        Args:
            lrtb: (*self.shape, 4) left, right, top, bottom. The crop box coordinates
            normalized: Whether lrtb are in normalized coordinates or pixel coordinates
            image_shape:  needed when not using normalized coordinates

         Returns:
            new camera with modified intrinsics
        """
        flat_affine = utils.crop_to_affine(lrtb, normalized, image_shape)
        return self.affine_transform(flat_affine)


class PinholeCamera(TensorDictionaryAffineCamera):
    """Standard pinhole camera model."""

    @classmethod
    def make(cls, intrinsics: Tensor, z_min: Union[Tensor, float] = 1e-6) -> Self:
        """Make a pinhole camera from intrinsics and set a z_min for marking validity.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            z_min:  (*) or float
        """
        intrinsics = _parse_intrinsics(intrinsics)
        z_min = _check_shape_and_convert_scalar_float(
            z_min, intrinsics.shape[:-1], "z_min", intrinsics.device
        )

        _values = {"affine": intrinsics, "z_min": z_min.unsqueeze(-1)}
        return cls(_values)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.pinhole_camera_project_to_pixel(
            self.affine, self.z_min, pts, depth_is_along_ray=depth_is_along_ray
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.pinhole_camera_pixel_to_ray(self.affine, pix, unit_vec=unit_vec)


class OrthographicCamera(TensorDictionaryAffineCamera):
    """Standard orthographic camera model."""

    @classmethod
    def make(cls, intrinsics: Tensor, z_min: Union[Tensor, float] = 1e-6) -> Self:
        """Make a orthographic camera from intrinsics and set a z_min for marking validity.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            z_min:  (*) or float
        """
        intrinsics = _parse_intrinsics(intrinsics)
        z_min = _check_shape_and_convert_scalar_float(
            z_min, intrinsics.shape[:-1], "z_min", intrinsics.device
        )

        _values = {"affine": intrinsics, "z_min": z_min.unsqueeze(-1)}
        return cls(_values)

    def is_central(self) -> bool:
        return False

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.orthographic_camera_project_to_pixel(
            self.affine, self.z_min, pts, depth_is_along_ray=depth_is_along_ray
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.orthographic_camera_pixel_to_ray(self.affine, pix, unit_vec=unit_vec)


class EquirectangularCamera(TensorDictionaryAffineCamera):
    """Equirectangular camera. Possibly cropped to capture limited azimuth and elevation."""

    @classmethod
    def make(
        cls,
        phi_range: Union[Tensor, Tuple[float, float]] = (-torch.pi, torch.pi),
        theta_range: Union[Tensor, Tuple[float, float]] = (0, torch.pi),
        intrinsics: Tensor = None,
        min_distance: Union[Tensor, float] = 1e-6,
        batch_shape: Optional[ShapeLike] = None,
        restrict_valid_rays: bool = True,
    ) -> Self:
        """Make an equirectangular camera.

        Creation options:
            0. Set intrinsics (a.k.a) affine transform directly and infer batch from it
            1. phi_range and theta_range are pairs and batch_shape is a tuple
            2. Set phi_range and theta_range as tensors of shape (*b, 2)

        Args:
            phi_range: (*, 2) or Pair
            theta_range: (*, 2) or Pair
            intrinsics: (*, 4) or (*, 3, 3)
            min_distance: (*) or float
        """

        if intrinsics is not None:
            # case 0: user specified intrinsics
            affine = _parse_intrinsics(intrinsics)
        else:
            if batch_shape is None:
                # case 1: user specified batch shape and phi and theta range as pairs
                assert isinstance(phi_range, Tensor) and isinstance(theta_range, Tensor)
                assert phi_range.shape[:-1] == theta_range.shape[:-1]
                batch_shape = phi_range.shape[:-1]
            else:
                # case 2: user specified phi and theta range as tensors
                assert isinstance(phi_range, tuple) and isinstance(theta_range, tuple)
                phi_range = (
                    torch.tensor(phi_range)
                    .reshape(*([1] * len(batch_shape)), 2)
                    .expand(*batch_shape, 2)
                )
                theta_range = (
                    torch.tensor(theta_range)
                    .reshape(*([1] * len(batch_shape)), 2)
                    .expand(*batch_shape, 2)
                )

            # compute affine based on phi_range and theta_range
            batch_shape = phi_range.shape[:-1]
            phi_diff = phi_range[..., 1] - phi_range[..., 0]
            theta_diff = theta_range[..., 1] - theta_range[..., 0]
            fx = 2 / phi_diff
            fy = 2 / theta_diff
            px = (phi_range[..., 0] + phi_range[..., 1]) / phi_diff
            py = -(theta_range[..., 0] + theta_range[..., 1]) / theta_diff
            affine = torch.stack((fx, fy, px, py), dim=-1)

        min_distance = _check_shape_and_convert_scalar_float(
            min_distance, affine.shape[:-1], "min_distance", affine.device
        )

        if not restrict_valid_rays:
            warnings.warn(
                "Creating EquirectangularCamera with restrict_valid_rays=False."
                "This means unprojection is not injective."
            )
        if torch.any(min_distance <= 0):
            warnings.warn(
                "Creating EquirectangularCamera with min_distance <= 0."
                "This means unprojection is not injective."
            )

        _values = {"affine": affine, "min_distance": min_distance.unsqueeze(-1)}
        _shared_attributes = {"restrict_valid_rays": restrict_valid_rays}
        return cls(_values, _shared_attributes)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.equirectangular_camera_project_to_pixel(
            self.affine, self.min_distance, pts, depth_is_along_ray=depth_is_along_ray
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.equirectangular_camera_pixel_to_ray(
            self.affine, pix, restrict_valid_rays=self.restrict_valid_rays, unit_vec=unit_vec
        )


class OpenCVFisheyeCamera(TensorDictionaryAffineCamera):
    @classmethod
    def make(
        cls,
        intrinsics: Tensor,
        distortion_coeffs: Tensor,
        theta_max: Union[Tensor, float],
        distance_min: Union[Tensor, float] = 1e-6,
        num_undistort_iters: int = 100,
    ) -> Self:
        """Make an OpenCVFisheyeCamera.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            distortion_coeffs: (*, 4)
            theta_max: (*) or float
            distance_min: (*) or float
        """
        intrinsics = _parse_intrinsics(intrinsics)
        batch_shape = intrinsics.shape[:-1]
        distortion_coeffs = _check_shape_and_convert_scalar_float(
            distortion_coeffs, batch_shape + (4,), "distortion_coeffs", intrinsics.device
        )
        theta_max = _check_shape_and_convert_scalar_float(
            theta_max, batch_shape, "theta_max", intrinsics.device
        )
        distance_min = _check_shape_and_convert_scalar_float(
            distance_min, batch_shape, "distance_min", intrinsics.device
        )

        if torch.any(theta_max > torch.pi):
            raise RuntimeError(
                "theta max must be less than pi i.e. FoV must be less than 2pi = 360 degrees"
            )
        if torch.any((theta_max > torch.pi / 2) & (distance_min < 0)):
            warnings.warn(
                "Creating OpenCVFisheyeCamera with theta_max > pi/2 and distance_min < 0."
                "This means unprojection is not injective."
            )

        _values = {
            "affine": intrinsics,
            "distortion_coeffs": distortion_coeffs,
            "theta_max": theta_max.unsqueeze(-1),
            "distance_min": distance_min.unsqueeze(-1),
        }
        _shared_attributes = {"num_iters": num_undistort_iters}
        return cls(_values, _shared_attributes)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.opencv_fisheye_camera_project_to_pixel(
            self.affine,
            self.distortion_coeffs,
            self.theta_max,
            self.distance_min,
            pts,
            depth_is_along_ray=depth_is_along_ray,
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.opencv_fisheye_camera_pixel_to_ray(
            self.affine,
            self.distortion_coeffs,
            self.theta_max,
            pix,
            unit_vec=unit_vec,
            num_iters=self.num_iters,
        )


class OpenCVCamera(TensorDictionaryAffineCamera):
    @classmethod
    def make(
        cls,
        intrinsics: Tensor,
        ks: Tensor,
        ps: Union[Tensor, float],
        z_min: Union[Tensor, float] = 1e-6,
        num_undistort_iters: int = 100,
    ) -> Self:
        """Make an OpenCVCamera. Doesn't support thin prism model.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            ks: (*, 6)
            ps: (*, 2)
            z_min: (*) or float
        """
        intrinsics = _parse_intrinsics(intrinsics)
        batch_shape = intrinsics.shape[:-1]
        ks = _check_shape_and_convert_scalar_float(ks, batch_shape + (6,), "ks", intrinsics.device)
        ps = _check_shape_and_convert_scalar_float(ps, batch_shape + (2,), "ps", intrinsics.device)
        z_min = _check_shape_and_convert_scalar_float(
            z_min, batch_shape, "z_min", intrinsics.device
        )

        _values = {"affine": intrinsics, "ks": ks, "ps": ps, "z_min": z_min.unsqueeze(-1)}
        _shared_attributes = {"num_iters": num_undistort_iters}
        return cls(_values, _shared_attributes)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.opencv_camera_project_to_pixel(
            self.affine, self.ks, self.ps, self.z_min, pts, depth_is_along_ray=depth_is_along_ray
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.opencv_camera_pixel_to_ray(
            self.affine, self.ks, self.ps, pix, unit_vec=unit_vec, num_iters=self.num_iters
        )


class BackwardForwardPolynomialFisheyeCamera(TensorDictionaryAffineCamera):
    @classmethod
    def make(
        cls,
        intrinsics: Tensor,
        proj_poly: Tensor,
        unproj_poly: Tensor,
        theta_max: Union[Tensor, float],
        distance_min: Union[Tensor, float] = 1e-6,
    ) -> Self:
        """Make a BackwardForwardPolynomialFisheyeCamera.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            proj_poly: (*, d1)
            unproj_poly: (*, d2)
            theta_max: (*) or float
            distance_min: (*) or float
        """
        intrinsics = _parse_intrinsics(intrinsics)
        batch_shape = intrinsics.shape[:-1]
        proj_poly = _check_shape_and_convert_scalar_float(
            proj_poly, batch_shape + (-1,), "proj_poly", intrinsics.device
        )
        unproj_poly = _check_shape_and_convert_scalar_float(
            unproj_poly, batch_shape + (-1,), "proj_poly", intrinsics.device
        )
        theta_max = _check_shape_and_convert_scalar_float(
            theta_max, batch_shape, "theta_max", intrinsics.device
        )
        distance_min = _check_shape_and_convert_scalar_float(
            distance_min, batch_shape, "distance_min", intrinsics.device
        )

        if torch.any(theta_max > torch.pi):
            raise RuntimeError(
                "theta max must be less than pi i.e. FoV must be less than 2pi = 360 degrees"
            )
        if torch.any((theta_max > torch.pi / 2) & (distance_min < 0)):
            warnings.warn(
                "Creating OpenCVFisheyeCamera with distance_min < 0 and theta_max > pi/2."
                "This means unprojection is not injective"
            )

        _values = {
            "affine": intrinsics,
            "proj_poly": proj_poly,
            "unproj_poly": unproj_poly,
            "theta_max": theta_max.unsqueeze(-1),
            "distance_min": distance_min.unsqueeze(-1),
        }
        return cls(_values)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.backward_forward_polynomial_fisheye_camera_project_to_pixel(
            self.affine,
            self.proj_poly,
            self.unproj_poly,
            self.theta_max,
            self.distance_min,
            pts,
            depth_is_along_ray=depth_is_along_ray,
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.backward_forward_polynomial_fisheye_camera_pixel_to_ray(
            self.affine, self.proj_poly, self.unproj_poly, self.theta_max, pix, unit_vec=unit_vec
        )

    def promote_degree(self, new_unproj_deg: int, new_proj_deg: int) -> Self:
        """Create a new version of the camera with higher degree, coefficients set to zero."""
        new_values = self._values.copy()
        new_values["unproj_poly"] = F.pad(
            new_values["unproj_poly"], (0, new_unproj_deg - new_values["unproj_poly"].size(-1))
        )
        new_values["proj_poly"] = F.pad(
            new_values["proj_poly"], (0, new_proj_deg - new_values["proj_poly"].size(-1))
        )
        return type(self)(new_values, self._shared_attributes)

    def _cat(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        """Promote degree of lower degree polynomial if concatenating different degrees."""
        new_proj_deg = max(obj._values["proj_poly"].size(-1) for obj in obj_list)
        new_unproj_deg = max(obj._values["unproj_poly"].size(-1) for obj in obj_list)
        new_obj_list = [obj.promote_degree(new_unproj_deg, new_proj_deg) for obj in obj_list]
        return super()._cat(new_obj_list, dim=dim)

    def _stack(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        """Promote degree of lower degree polynomial if stacking different degrees."""
        new_proj_deg = max(obj._values["proj_poly"].size(-1) for obj in obj_list)
        new_unproj_deg = max(obj._values["unproj_poly"].size(-1) for obj in obj_list)
        new_obj_list = [obj.promote_degree(new_unproj_deg, new_proj_deg) for obj in obj_list]
        return super()._stack(new_obj_list, dim=dim)


class Kitti360FisheyeCamera(TensorDictionaryAffineCamera):
    @classmethod
    def make(
        cls,
        intrinsics: Tensor,
        k1: Union[Tensor, float],
        k2: Union[Tensor, float],
        xi: Union[Tensor, float],
        theta_max: Union[Tensor, float],
        distance_min: Union[Tensor, float] = 1e-6,
        num_undistort_iters: int = 100,
    ) -> Self:
        """Create a Kitti360FisheyeCamera.

        Args:
            intrinsics: (*, 4) or (*, 3, 3)
            k1: (*) or float
            k2: (*) or float
            xi: (*) or float
            theta_max: (*) or float
            distance_min: (*) or float
        """

        intrinsics = _parse_intrinsics(intrinsics)
        batch_shape = intrinsics.shape[:-1]
        k1 = _check_shape_and_convert_scalar_float(k1, batch_shape, "k1", intrinsics.device)
        k2 = _check_shape_and_convert_scalar_float(k2, batch_shape, "k2", intrinsics.device)
        xi = _check_shape_and_convert_scalar_float(xi, batch_shape, "xi", intrinsics.device)
        theta_max = _check_shape_and_convert_scalar_float(
            theta_max, batch_shape, "theta_max", intrinsics.device
        )
        distance_min = _check_shape_and_convert_scalar_float(
            distance_min, batch_shape, "distance_min", intrinsics.device
        )

        if torch.any(xi < 1):
            raise RuntimeError("Kitti360FisheyeCamera only implemented for x_i greater than 1")
        if torch.any(theta_max > torch.pi):
            raise RuntimeError(
                "theta max must be less than pi i.e. FoV must be less than 2pi = 360 degrees."
            )
        max_theta_max = torch.acos(-1 / xi)
        if torch.any(theta_max > max_theta_max):
            raise RuntimeError(
                "theta_max too large. Largest possible theta_max is {} got theta_max {}".format(
                    max_theta_max, theta_max
                )
            )
        if torch.any((theta_max > torch.pi / 2) & (distance_min < 0)):
            warnings.warn(
                "Creating Kitti360 Camera with theta_max > pi/2 and distance_min <0."
                "This means unprojection is not injective"
            )

        ks = torch.stack((k1, k2), dim=-1)
        _values = {
            "affine": intrinsics,
            "ks": ks,
            "xi": xi.unsqueeze(-1),
            "theta_max": theta_max.unsqueeze(-1),
            "distance_min": distance_min.unsqueeze(-1),
        }
        _shared_attributes = {"num_iters": num_undistort_iters}
        return cls(_values, _shared_attributes)

    def is_central(self) -> bool:
        return True

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.kitti360_fisheye_camera_project_to_pixel(
            self.affine,
            self.ks,
            self.xi,
            self.theta_max,
            self.distance_min,
            pts,
            depth_is_along_ray=depth_is_along_ray,
        )

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        return CF.kitti360_fisheye_camera_pixel_to_ray(
            self.affine,
            self.ks,
            self.xi,
            self.theta_max,
            pix,
            unit_vec=unit_vec,
            num_iters=self.num_iters,
        )


class _HeterogeneousCamera(CameraBase):
    def __init__(
        self, my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]], shape: ShapeLike
    ):
        # keys are camera type, values are tuples of
        # (tensor representing linear index, and array of camera of k type)
        self.my_dict = my_dict
        self._shape = torch.Size(shape)
        key = next(iter(my_dict.keys()))
        self._device = my_dict[key][1].device

    @property
    def shape(self) -> ShapeLike:
        return self._shape

    @property
    def device(self) -> torch.device:
        return self._device

    def is_central(self) -> bool:
        return all(x[1].is_central() for x in self.my_dict.values())

    def __str__(self) -> str:
        out = "shape: {}".format(self.shape)
        for k, v in self.my_dict.items():
            out += "{}: (ptrs={},values={}) \n".format(str(k), str(v[0]), str(v[1]))
        return out

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        group_shape, batch_numel = utils._get_group_shape(self.shape, pix.shape[:-1])
        pix = pix.reshape(batch_numel, -1, 2)

        origin = torch.zeros(*pix.shape[:2], 3, device=pix.device)
        dirs = torch.zeros(*pix.shape[:2], 3, device=pix.device)
        valid = torch.zeros(*pix.shape[:2], device=pix.device, dtype=torch.bool)

        for k, (idx, cam) in self.my_dict.items():
            origin[idx, :, :], dirs[idx, :, :], valid[idx, :] = cam.pixel_to_ray(
                pix[idx, :, :], unit_vec=unit_vec
            )

        origin = origin.reshape(*self.shape, *group_shape, 3)
        dirs = dirs.reshape(*self.shape, *group_shape, 3)
        valid = valid.reshape(self.shape + group_shape)
        return origin, dirs, valid

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        group_shape, batch_numel = utils._get_group_shape(self.shape, pts.shape[:-1])
        pts = pts.reshape(batch_numel, -1, 3)
        pix = torch.zeros(*pts.shape[:2], 2, device=pts.device)
        depth = torch.zeros(*pts.shape[:2], device=pts.device)
        valid = torch.zeros(*pts.shape[:2], device=pts.device, dtype=torch.bool)

        for k, (idx, cam) in self.my_dict.items():
            pix[idx, :, :], depth[idx, :], valid[idx, :] = cam.project_to_pixel(
                pts[idx, :, :], depth_is_along_ray=depth_is_along_ray
            )

        pix = pix.reshape(*self.shape, *group_shape, 2)
        depth = depth.reshape(self.shape + group_shape)
        valid = valid.reshape(self.shape + group_shape)
        return pix, depth, valid

    @staticmethod
    def _homogeneous_to_heterogeneous(cam: CameraBase) -> _HeterogeneousCamera:
        ptrs = torch.arange(math.prod(cam.shape))
        obj = cam.reshape(-1)
        my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {type(cam): (ptrs, obj)}
        return _HeterogeneousCamera(my_dict, cam.shape)

    def _to_homogeneous(self) -> CameraBase:
        keys = self.my_dict.keys()
        if len(keys) != 1:
            raise RuntimeError(
                "only heterogeneous camera with 1 type can be converted to homogeneous"
            )
        linear_indices, objs = self.my_dict[list(keys)[0]]
        objs = objs[_invert_permutation(linear_indices)]
        objs = objs.reshape(self.shape)
        return objs

    def __getitem__(self, index: slice) -> CameraBase:
        temp = torch.arange(math.prod(self.shape)).reshape(self.shape)
        old_to_keep = temp[index]
        new_shape = old_to_keep.shape
        old_to_keep = old_to_keep.reshape(-1)
        old_to_new = -torch.ones(math.prod(self.shape), dtype=temp.dtype)
        old_to_new[old_to_keep] = torch.arange(old_to_keep.size(0))

        new_my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for type_k, (old_ptr_type_k, old_obj_type_k) in self.my_dict.items():
            new_ptr_type_k = old_to_new[old_ptr_type_k]
            idx = new_ptr_type_k > -1
            new_ptr_type_k = new_ptr_type_k[idx]
            if len(new_ptr_type_k) > 0:
                new_objects_type_k = old_obj_type_k[idx]
                new_my_dict[type_k] = (new_ptr_type_k, new_objects_type_k)

        out = type(self)(new_my_dict, new_shape)
        if len(out.my_dict.keys()) == 1:
            return out._to_homogeneous()
        return out

    @classmethod
    def _catfirst(cls, obj_list: Sequence[Self]) -> Self:
        shape0 = obj_list[0].shape
        assert all(x.shape[1:] == shape0[1:] for x in obj_list)

        offset = 0
        new_my_dict: DefaultDict[Type[CameraBase], Tuple[List[Tensor], List[CameraBase]]] = (
            defaultdict(lambda: ([], []))
        )
        new_first_dim = sum([x.shape[0] for x in obj_list])

        for obj in obj_list:
            for type_k, (homo_obj_ptrs, homo_obj) in obj.my_dict.items():
                new_my_dict[type_k][0].append(homo_obj_ptrs + offset)
                new_my_dict[type_k][1].append(homo_obj)
            offset += math.prod(obj.shape)

        new_my_dict2: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for k, v in new_my_dict.items():
            new_my_dict2[k] = (torch.cat(v[0]), torch.cat(v[1]))
        return cls(new_my_dict2, (new_first_dim, *shape0[1:]))

    def _cat(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        obj_list = [x.transpose(0, dim) for x in obj_list]
        cat_first = self._catfirst(obj_list)
        out = cat_first.transpose(0, dim)
        return out

    def _stack(self, obj_list: Sequence[Self], dim: int = 0) -> Self:
        obj_list = [x.unsqueeze(dim) for x in obj_list]
        return self._cat(obj_list, dim=dim)

    def transpose(self, dim0: int, dim1: int) -> Self:
        temp = torch.arange(math.prod(self.shape)).reshape(self.shape)
        new_to_old = temp.transpose(dim0, dim1)
        new_shape = new_to_old.shape
        new_to_old = new_to_old.reshape(-1)
        old_to_new = _invert_permutation(new_to_old)
        new_my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for k, v in self.my_dict.items():
            new_my_dict[k] = (old_to_new[v[0]], v[1])

        return type(self)(new_my_dict, new_shape)

    def permute(self, dims: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(dims, (tuple, list)):
            perm = dims
        else:
            perm = (dims, *extra)
        temp = torch.arange(math.prod(self.shape)).reshape(self.shape)
        new_to_old = temp.permute(*perm)
        new_shape = new_to_old.shape
        new_to_old = new_to_old.reshape(-1)
        old_to_new = _invert_permutation(new_to_old)
        new_my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for k, v in self.my_dict.items():
            new_my_dict[k] = (old_to_new[v[0]], v[1])

        return type(self)(new_my_dict, new_shape)

    def to(self, device: torch.device) -> Self:
        new_my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for k, v in self.my_dict.items():
            new_my_dict[k] = (v[0].to(device), v[1].to(device))
        return type(self)(new_my_dict, self.shape)

    def reshape(self, shape: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(shape, (tuple, list)):
            new_shape = shape
        else:
            new_shape = (shape, *extra)
        if not all(isinstance(x, int) for x in new_shape):
            raise TypeError("reshape(): argument 'shape' must be tuple of ints")
        new_shape = cast(Tuple[int], new_shape)
        indices_of_neg1 = [index for index, value in enumerate(new_shape) if value == -1]
        if len(indices_of_neg1) > 1:
            raise RuntimeError("only one dimension can be inferred")
        elif len(indices_of_neg1) == 1:
            idx = indices_of_neg1[0]
            infered_dim = int(-math.prod(self.shape) / math.prod(new_shape))
            new_shape_inf = (*new_shape[:idx], infered_dim, *new_shape[idx + 1 :])
        else:
            new_shape_inf = new_shape
        if math.prod(new_shape_inf) != math.prod(self.shape):
            raise RuntimeError(
                "shape '{}' is invalid for input of size {}".format(
                    list(new_shape), math.prod(self.shape)
                )
            )

        return type(self)(self.my_dict, new_shape_inf)

    def squeeze(self, dim: Optional[int] = None) -> Self:
        if dim is None:
            new_shape = torch.Size([x for x in self.shape if x != 1])
            return type(self)(self.my_dict, new_shape)
        elif self.shape[dim] == 1:
            return type(self)(self.my_dict, (*self.shape[:dim], *self.shape[dim + 1 :]))
        else:
            return self

    def unsqueeze(self, dim: int) -> Self:
        if (dim > len(self.shape)) or (dim < -(len(self.shape) + 1)):
            raise IndexError()
        if dim < 0:
            dim = len(self.shape) + 1 + dim
        new_shape = torch.Size((*self.shape[:dim], 1, *self.shape[dim:]))
        return type(self)(self.my_dict, new_shape)

    def expand(self, size: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(size, (tuple, list)):
            expand_shape = size
        else:
            expand_shape = (size, *extra)

        out = self
        if len(expand_shape) != len(out.shape):
            raise RuntimeError("The expanded size must be the same length as the original Tensor")
        for dim in range(len(expand_shape)):
            if (expand_shape[dim] != -1) and (expand_shape[dim] != out.shape[dim]):
                if out.shape[dim] != 1:
                    raise RuntimeError(
                        f"The expanded size of the tensor ({expand_shape[dim]}) must match the existing size ({out.shape[dim]}) at non-singleton dimension {dim}.  Target sizes: {list(expand_shape)}.  Tensor sizes: {list(out.shape)}"
                    )
                out = torch.cat([out] * expand_shape[dim], dim=dim)
        return out

    def flip(self, dims: Union[int, ShapeLike], *extra: int) -> Self:
        if isinstance(dims, (tuple, list)):
            dims_t = dims
        else:
            dims_t = (dims, *extra)
        temp = torch.arange(math.prod(self.shape)).reshape(self.shape)
        new_to_old = temp.flip(dims_t)
        new_shape = new_to_old.shape
        new_to_old = new_to_old.reshape(-1)
        old_to_new = _invert_permutation(new_to_old)
        new_my_dict: Dict[Type[CameraBase], Tuple[Tensor, CameraBase]] = {}
        for k, v in self.my_dict.items():
            new_my_dict[k] = (old_to_new[v[0]], v[1])

        return type(self)(new_my_dict, new_shape)

    def clone(self) -> Self:
        new_my_dict = {k: (v[0].clone(), v[1].clone()) for k, v in self.my_dict.items()}
        return type(self)(new_my_dict, self.shape)

    def detach(self) -> Self:
        new_my_dict = {k: (v[0].detach(), v[1].detach()) for k, v in self.my_dict.items()}
        return type(self)(new_my_dict, self.shape)


class CubeCamera(TensorDictionaryCamera):
    @classmethod
    def make(cls, batch_shape: ShapeLike, device: torch.device = "cpu") -> Self:
        """Make a CubeCamera."""
        # simple way to hold shape and device and implement tensor-like operations
        return cls({"tensor": torch.zeros(batch_shape, device=device).unsqueeze(-1)})

    def pixel_to_ray(self, pix: Tensor, unit_vec: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Pixels can be any 3d point. Just normalizes with 2-norm of inf-norm.

        Args:
            pix: (*self.shape, *group_shape, 3)
            unit_vec: bool

        Returns:
            origin: (*self.shape, *group_shape, 3)
            dirs: (*self.shape, *group_shape, 3)
            valid: (*self.shape, *group_shape)
        """
        return CF.cube_camera_pixel_to_ray(pix, unit_vec=unit_vec)

    def project_to_pixel(
        self, pts: Tensor, depth_is_along_ray: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Just normalizes the pts with 2-norm or inf-norm

        Args:
            pts: (*self.shape, *group_shape, 3)
            depth_is_along_ray: bool

        Returns:
            pix: (*self.shape, *group_shape, 3)
            depth: (*self.shape, *group_shape)
            valid: (*self.shape, *group_shape)
        """
        return CF.cube_camera_project_to_pixel(pts, depth_is_along_ray=depth_is_along_ray)

    def is_central(self) -> bool:
        return True

    def get_camera_rays(
        self, res: Tuple[int, int], unit_vec: bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if res[0] != 6 * res[1]:
            raise RuntimeError("invalid cubemap shape")
        dirs = utils.get_normalized_grid_cubemap(res[1], device=self.device)  # (*res, 3)
        if unit_vec:
            dirs = F.normalize(dirs, dim=-1)
        dirs = dirs.reshape(tuple([1] * len(self.shape)) + dirs.shape).expand(
            *self.shape, -1, -1, -1
        )
        origin = torch.zeros_like(dirs)
        valid = torch.ones_like(dirs[..., 0], dtype=torch.bool)
        return origin, dirs, valid

    def get_normalized_grid(self, res: Tuple[int, int]) -> Tensor:
        raise RuntimeError("CubeCamera does not support get_normalized_grid")


def _invert_permutation(perm: Tensor) -> Tensor:
    """invert a permutation."""
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
    return inv_perm


def _parse_intrinsics(intrinsics: Tensor) -> Tensor:
    """if given matrix intrinsics convert to flat intrinsics and raise error if
    intrinsics is not matrix or flat intrinsics"""
    if intrinsics.shape[-2:] == (3, 3):
        intrinsics = utils.flat_intrinsics_from_intrinsics_matrix(intrinsics)
    if intrinsics.shape[-1] != 4:
        raise RuntimeError(
            "intrinsics matrix must be of shape (*,4) or (*, 3, 3) but got shape {}".format(
                intrinsics.shape
            )
        )
    return intrinsics


def _check_shape_and_convert_scalar_float(
    value: Union[Tensor, float], expected_shape: Tuple, name: str, device: torch.device
) -> Tensor:
    if isinstance(value, float):
        value = torch.full(expected_shape, value, device=device)
    elif any((v != e) and (e != -1) for v, e in zip(value.shape, expected_shape)):
        raise RuntimeError(
            "Inconsistent batch shape. Expected {} to be shape {} but got shape {}".format(
                name, value.shape, expected_shape
            )
        )
    return value


# maybe move these to __init__.py? is that allowed


def collate_camera_fn(batch, *, collate_fn_map=None):
    """make camera object compatible with automatic dataset batching"""
    return torch.stack(batch)


default_collate_fn_map.update([(CameraBase, collate_camera_fn)])
