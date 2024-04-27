from typing import Tuple
from torch import Tensor
import torch
from nvtorchcam.utils import compute_jacobian


class DifferentiableNewtonInverse(torch.autograd.Function):
    """Extend this class with a concrete implementation of my_function and the class with compute
    the inverse function using Newton's method. Also calculates the derivative of the inverse
    function backward using implicit and inverse function theorems.

    See cameras_functional.opencv_fisheye_undistortion for example
    """

    @staticmethod
    def my_function(x: Tensor, *theta: Tensor):
        """Implement this function in subclasses and this class will compute its inverse.

        Args:
           x: (b,n,d)
           theta: (b,t_1) ..., (b,t_k)

        Return
           y: (b,n,d)
        """
        raise NotImplementedError()

    @staticmethod
    def forward(ctx, y: Tensor, initial_x: Tensor, iters: int, *theta: Tensor) -> Tensor:
        """Calculate y s.t. y = my_function(x, theta)

        Args:
            y: (b,n,d)
            theta: Tuple of (b,t_1) ... (b, t_k)
            initial_x: (b,n,d)
            iters: int number of Newton iterations

        Returns
            x: (b,n,d)
        """
        with torch.no_grad():
            x = initial_x
            for i in range(0, iters):
                with torch.enable_grad():
                    f_of_x, Df = compute_jacobian(ctx._forward_cls.my_function, x, *theta)

                diff = y - f_of_x
                delta_x = torch.linalg.solve(Df, diff)
                # inv_Df = torch.inverse(Df)
                # delta_x = torch.einsum('bnij,bnj->bni', inv_Df, diff)
                x = x + delta_x

            ctx.theta = tuple(theta_i.detach() for theta_i in theta)
            ctx.x = x.detach()

        return x

    @staticmethod
    def backward(ctx, dL_dx: Tensor) -> Tuple[Tensor, None, None, Tensor]:
        """Calculate gradients using the inverse and implicit function theorem.

        Args:
            dL_dx: (b,n,d)

        Returns
            dL_dy: (b,n,d)
            dL_dtheta: Tuple of (b,t_1) ... (b, t_k)
        """

        x = ctx.x
        theta = ctx.theta
        with torch.enable_grad():
            _, df_dx = compute_jacobian(ctx._forward_cls.my_function, x, *theta)  # (b,n,d,d)

        dL_dy = torch.linalg.solve(df_dx.transpose(-1, -2), dL_dx)
        # dg_dy = torch.inverse(df_dx)
        # dL_dy = torch.einsum('bnij,bnj->bnj', dg_dy, dL_dx) #(b,n,d)

        with torch.enable_grad():
            theta = tuple(theta_i.detach().clone() for theta_i in theta)
            for theta_i in theta:
                theta_i.requires_grad = True
            y = ctx._forward_cls.my_function(x.detach(), *theta)
            torch.autograd.backward(y, dL_dy)

        dL_dtheta = tuple(-theta_i.grad.detach().clone() for theta_i in theta)

        return dL_dy, None, None, *dL_dtheta
