import torch
from typing import Optional, Dict, Tuple, Union, Any, Callable
from torch import Tensor


def in_image(pix: Tensor) -> Tensor:
    """ Test if point is in the unit square [-1,1]x[-1,1]
    Args:
        pix:  (*, 2) 

    Returns:
        :       (*,)
    """
    return torch.all(pix < 1, dim=-1) & torch.all(pix > -1, dim=-1)

def get_normalized_grid(res: Tuple[int, int], device: torch.device) -> Tensor:
    """ Get normalized pixel locations
    Args:
        res: (h,w)

    Returns:
        pix: (h,w,2)
    """
    x_pix = torch.linspace(-1+1/res[1],1-1/res[1], res[1], dtype=torch.float, device=device)
    y_pix = torch.linspace(-1+1/res[0],1-1/res[0], res[0], dtype=torch.float, device=device)
    x_y = torch.meshgrid([x_pix,y_pix],indexing='xy') 
    pix = torch.stack(x_y,dim=2)
    return pix

def samples_from_image(image: Tensor, 
                       pts: Tensor, 
                       mode: str = 'bilinear', 
                       align_corners: bool = False, 
                       return_in_image_mask: bool = False,
                       padding_mode: str = 'zeros') -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """ Interpolate a batch of images at point locations pts. If return_in_image_mask = True also return
        a mask indicated which pixels are within the image bounds
    Args:
            image (*batch_shape, channels, height, width)
            pts: (*batch_shape, *group_shape, 2) 
        Returns:
            values: (*batch_shape, channels, *group_shape)
            mask: (*batch_shape, *group_shape)
    """
    batch_shape = image.shape[:-3]
    group_shape, batch_numel = _get_group_shape(batch_shape, pts.shape[:-1])

    #suport evaluating bool image
    if image.dtype == torch.bool:
        is_bool = True
        image = image.half()
    else:
        is_bool = False
    image = image.reshape(batch_numel, *image.shape[-3:])
    pts_flat = pts.reshape(batch_numel,-1,1,2)
    values = torch.nn.functional.grid_sample(image, pts_flat, align_corners = align_corners, mode = mode, padding_mode = padding_mode) #(batch_numel, c, group_numel, 1)
    values = values.reshape(*batch_shape, -1, *group_shape)

    if is_bool:
        values = values > 0.5

    if return_in_image_mask:
        mask = in_image(pts)
        return values, mask
    else:
        return values

def _mat_multiply_base(A: Tensor, pts: Tensor) -> Tensor:
    #(b,i,j) (b,n,j)
    #return (b,n,i)
    assert A.size(2) == pts.size(2)
    return torch.einsum('bij,bnj->bni',A,pts)

#(*batch_shape,i,j) (*batch_shape,*group_shape,j)
#return (*batch_shape,*group_shape,i)
def mat_multiply(A: Tensor, pts: Tensor) -> Tensor:
    """ Transform batches of groups of points by batches of matrices
    Args:
        A: (*batch_shape, d, d)
        pts: (*batch_shape, *group_shape, d) 
    Returns:
        : (*batch_shape, *group_shape, d)
           
    """
    return _transform_flatten_wrapper(A,pts,_mat_multiply_base)


def _apply_affine_base(T: Tensor, pts: Tensor) -> Tensor:
    #(b,k+1,k+1) (b,n,k)
    #return (b,n,k)
    k = pts.size(-1)
    assert T.shape[1:3] == (k+1,k+1)
    R = T[:,:k,:k]
    t = T[:,:k,k]
    trans_points = _mat_multiply_base(R,pts) + t.unsqueeze(1)
    return trans_points



def apply_affine(T: Tensor, pts: Tensor) -> Tensor:
    """ Transform batches of groups of points by batches of affine transformations
    Args:
        A: (*batch_shape, d+1, d+1)
        pts: (*batch_shape, *group_shape, d) 
    Returns:
        : (*batch_shape, *group_shape, d)
    """
    return _transform_flatten_wrapper(T,pts,_apply_affine_base)


def _apply_homography_base(H: Tensor, pts: Tensor) -> Tensor:
    #(b,k+1,k+1) (b,n,k)
    #return (b,n,k)
    k = pts.size(-1)
    assert H.shape[1:3] == (k+1,k+1)
    y = _mat_multiply_base(H[:,:,:k], pts) + H[:,:,k].unsqueeze(1)
    y = y[:,:,:k]/y[:,:,k:]
    return y

def apply_homography(H: Tensor, pts: Tensor) -> Tensor:
    """ Transform batches of groups of points by batches of homography transforms
    Args:
        H: (*batch_shape, d+1, d+1)
        pts: (*batch_shape, *group_shape, d) 
    Returns:
        : (*batch_shape, *group_shape, d)
    """
    return _transform_flatten_wrapper(H,pts,_apply_homography_base)


def _transform_flatten_wrapper(A: Tensor, pts: Tensor, base_fun: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    #(*batch_shape, i, j)  (*batch_shape,*group_shape,k)
    batch_shape = A.shape[:-2]
    group_shape, batch_numel = _get_group_shape(batch_shape, pts.shape[:-1])
    A = A.reshape(batch_numel,*A.shape[-2:])
    pts = pts.reshape(batch_numel, -1, pts.shape[-1])
    y = base_fun(A,pts)
    y = y.reshape(*batch_shape,*group_shape,-1)
    return y

def _get_group_shape(batch_shape: torch.Size, batch_group_shape: torch.Size) -> Tuple[torch.Size, int]:
    n = len(batch_shape)
    assert batch_shape == batch_group_shape[:n]
    group_shape = batch_group_shape[n:]
    batch_numel = batch_shape.numel()
    return group_shape, batch_numel

def normalized_intrinsics_from_pixel_intrinsics(intrinsics: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    if intrinsics.shape[-2:] == (3,3):
        intrinsics = flat_intrinsics_from_intrinsics_matrix(intrinsics)
        got_matrix_format = True
    elif intrinsics.shape[-1] == 4:
        got_matrix_format = False
    else:
        raise RuntimeError('intrinsics must have shape (*, 3, 3) or (*, 4)')

    image_shape = torch.tensor(
        (image_shape[1], image_shape[0]), device=intrinsics.device)
    image_shape = image_shape.reshape(*([1]*(intrinsics.dim()-1)), 2)
    image_shape_half_inv = 2/image_shape
    new_scale = intrinsics[..., :2]*image_shape_half_inv
    new_shift = intrinsics[..., 2:]*image_shape_half_inv - 1
    intrinsics_n = torch.cat((new_scale, new_shift), dim=-1)

    if got_matrix_format:
        intrinsics_n = intrinsics_matrix_from_flat_intrinsics(intrinsics_n)
    return intrinsics_n


def pixel_intrinsics_from_normalized_intrinsics(intrinsics_n: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    if intrinsics_n.shape[-2:] == (3,3):
        intrinsics = flat_intrinsics_from_intrinsics_matrix(intrinsics)
        got_matrix_format = True
    elif intrinsics_n.shape[-1] == 4:
        got_matrix_format = False
    else:
        raise RuntimeError('intrinsics must have shape (*, 3, 3) or (*, 4)')

    image_shape = torch.tensor(
        (image_shape[1], image_shape[0]), device=intrinsics_n.device)
    image_shape = image_shape.reshape(*([1]*(intrinsics_n.dim()-1)), 2)
    image_shape_half = image_shape/2
    new_scale = intrinsics_n[..., :2]*image_shape_half
    new_shift = intrinsics_n[..., 2:]*image_shape_half + image_shape_half
    intrinsics = torch.cat((new_scale, new_shift), dim=-1)
    if got_matrix_format:
        intrinsics = intrinsics_matrix_from_flat_intrinsics(intrinsics)
    return intrinsics


def normalized_pts_from_pixel_pts(n_pts: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    image_shape = torch.tensor(
        (image_shape[1], image_shape[0]), device=n_pts.device)
    image_shape = image_shape.reshape(*([1]*(n_pts.dim()-1)), 2)
    image_shape_half_inv = 2/image_shape
    pts = image_shape_half_inv*n_pts - 1
    return pts


def pixel_pts_from_normalized_pts(pts: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    image_shape = torch.tensor(
        (image_shape[1], image_shape[0]), device=pts.device)
    image_shape = image_shape.reshape(*([1]*(pts.dim()-1)), 2)
    image_shape_half = image_shape/2
    n_pts = image_shape_half*pts + image_shape_half
    return n_pts


def flat_intrinsics_from_intrinsics_matrix(K: Tensor) -> Tensor:
    return torch.stack((K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]), dim=-1)


def intrinsics_matrix_from_flat_intrinsics(flat_intrinsics: Tensor) -> Tensor:
    intrinsics_matrix = torch.zeros(
        *flat_intrinsics.shape[:-1], 3, 3, device=flat_intrinsics.device, dtype=flat_intrinsics.dtype)
    intrinsics_matrix[..., 0, 0] = flat_intrinsics[..., 0]
    intrinsics_matrix[..., 1, 1] = flat_intrinsics[..., 1]
    intrinsics_matrix[..., 0, 2] = flat_intrinsics[..., 2]
    intrinsics_matrix[..., 1, 2] = flat_intrinsics[..., 3]
    intrinsics_matrix[..., 2, 2] = 1
    return intrinsics_matrix


def cart_from_spherical(phi_theta: Tensor, r: Union[Tensor, float] = 1.0) -> Tensor:
    """
    Args:
        phi_theta: (*, 2) 
        r:         (*)

    Returns:
        out:         (*, 3)
    """
    theta = phi_theta[..., 1]
    phi = phi_theta[..., 0]
    s = torch.sin(theta)
    z = r*s*torch.cos(phi)
    x = r*s*torch.sin(phi)
    y = -r*torch.cos(theta)
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


def newton_inverse(f, y: Tensor, initial_x: Tensor, iters: int = 10) -> Tensor:
    x = initial_x
    for i in range(0, iters):
        with torch.enable_grad():
            f_of_x, Df = compute_jacobian(f, x)
        inv_Df = torch.inverse(Df)
        diff = y-f_of_x
        delta_x = torch.einsum('bnij,bnj->bni', inv_Df, diff)
        x = x + delta_x
        x = x.detach()
    return x


def compute_jacobian(f, x: Tensor) -> Tuple[Tensor, Tensor]:
    # function taking (b,n,d) to (b,n,d) and (b,n,d)
    # return (b,n,d) (b,n,d,d)
    b, n, dim = x.shape
    id = torch.eye(dim, device=x.device).reshape(
        1, 1, dim, dim).expand(b, n, dim, dim)
    Df_columns = []
    x.requires_grad = True
    y = f(x)
    for i in range(0, dim):
        torch.autograd.backward(y, id[:, :, i], retain_graph=True)
        Df_columns.append(x.grad.clone().detach())
        x.grad.zero_()

    Df = torch.stack(Df_columns, dim=2)
    return y.detach(), Df


def fit_polynomial(x: Tensor, y: Tensor, degree: int) -> Tensor:
    # args: (b,n), (b,n)  int
    # degree=n means has n+1 coefficients
    cur_col = torch.ones_like(x)
    cols = [cur_col]
    for _ in range(degree):
        cur_col = cur_col*x
        cols.append(cur_col)
    mat = torch.stack(cols, dim=-1)
    coeffs = torch.linalg.lstsq(mat, y)[0]
    return coeffs

def apply_poly(coeffs: Tensor, vals: Tensor) -> Tensor:
    # coeffs[:,i] = the coefficient on x^i
    # (b,k) (b,n)
    # return (b,n)
    k = coeffs.size(1)
    out = torch.zeros_like(vals)
    for i in range(k-1, -1, -1):
        out = coeffs[:, i:i+1] + out*vals
    return out

def crop_to_affine(lrtb: Tensor, normalized: bool = True, image_shape: Tuple[int, int] = None):
    """lrtb (*b, 4) of left, right, top, bottom"""
    lrtb = lrtb.unflatten(-1, (2, 2))
    lt = lrtb[..., 0]
    rb = lrtb[..., 1]
    if not normalized:
        if image_shape is None:
            raise RuntimeError(
                'If using unnormalized pixel positions must specify image_shape')
        lt = normalized_pts_from_pixel_pts(lt, image_shape)
        rb = normalized_pts_from_pixel_pts(rb, image_shape)

    det = lt - rb
    scale = -2/det
    shift = (lt+rb)/det
    flat_affine = torch.cat((scale, shift), dim=-1)
    return flat_affine