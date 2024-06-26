# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
from nvtorchcam import diff_newton_inverse


class DifferentiableNewtonInverse(unittest.TestCase):
    def setUp(self):
        pass

    # @unittest.skip('Uncomment to skip this test')
    def test_basic2d_fun(self):
        def f(x, theta1, theta2):
            """(b,g,2) (b,1) (b,1) -> (b,g,2)"""
            theta1 = theta1.unsqueeze(-1)
            theta2 = theta2.unsqueeze(-1)
            r2 = torch.norm(x, dim=-1, keepdim=True) ** 2
            u0 = r2 * x
            u = theta1[:, 0:1, :] * u0 + theta2[:, 0:1, :]
            return u

        def f_inv(u, theta1, theta2):
            """(b,g,2) (b,1) (b,1) -> (b,g,2)"""
            theta1 = theta1.unsqueeze(-1)
            theta2 = theta2.unsqueeze(-1)
            u0 = (u - theta2[:, 0:1, :]) / theta1[:, 0:1, :]
            r2 = torch.norm(u0, dim=-1, keepdim=True) ** (-2 / 3)
            x = u0 * r2
            return x

        class FInv(diff_newton_inverse.DifferentiableNewtonInverse):
            """implements f_inv using newtons method and implicit function theorem"""

            @staticmethod
            def my_function(x, theta1, theta2):
                return f(x, theta1, theta2)

        y = torch.tensor(
            [
                [[2.5183, 1.8197], [0.2476, 0.9849], [1.1926, -1.2012], [-1.8105, 1.0297]],
                [[0.2667, 0.1598], [-0.0577, 0.7857], [0.0991, -0.8331], [1.1598, -1.8567]],
            ],
            requires_grad=True,
        )
        theta1 = torch.tensor([[0.5], [0.2]], requires_grad=True)
        theta2 = torch.tensor([[0.1], [0.4]], requires_grad=True)

        # compute inverse and grad explicitly
        x = f_inv(y, theta1, theta2)
        torch.sum(x).backward()
        grad_y = y.grad
        grad_theta1 = theta1.grad
        grad_theta2 = theta2.grad

        # compute inverse and grad with newton
        x_init = x.detach().clone() + 0.1
        y = y.detach().clone()
        theta1 = theta1.detach().clone()
        theta2 = theta2.detach().clone()
        y.requires_grad = True
        theta1.requires_grad = True
        theta2.requires_grad = True
        x_newton = FInv.apply(y, x_init, 100, theta1, theta2)
        torch.sum(x_newton).backward()
        grad_y_newton = y.grad
        grad_theta1_newton = theta1.grad
        grad_theta2_newton = theta2.grad

        torch.testing.assert_close(x, x_newton)
        torch.testing.assert_close(grad_y_newton, grad_y)
        torch.testing.assert_close(grad_theta1_newton, grad_theta1)
        torch.testing.assert_close(grad_theta2_newton, grad_theta2)

    # @unittest.skip('Uncomment to skip this test')
    def test_2d_fun(self):
        def f(x, theta1, theta2):
            """(b,g,2) (b,2) (b,2) -> (b,g,2)"""
            a = theta1[:, 0:1]
            b = theta1[:, 1:2]
            c = theta2[:, 0:1]
            d = theta2[:, 1:2]
            y = x[:, :, 1]
            x = x[:, :, 0]
            term1 = (a * x + b) ** 3
            term2 = (c * y + d) ** 3
            u = term1 + term2
            v = term1 - term2
            out = torch.stack((u, v), dim=-1)
            return out

        def f_inv(u, theta1, theta2):
            """(b,g,2) (b,1) (b,1) -> (b,g,2)"""
            a = theta1[:, 0:1]
            b = theta1[:, 1:2]
            c = theta2[:, 0:1]
            d = theta2[:, 1:2]
            v = u[:, :, 1]
            u = u[:, :, 0]
            term1 = (u + v) / 2
            term1 = torch.sign(term1) * (torch.abs(term1) ** (1 / 3))
            term2 = (u - v) / 2
            term2 = torch.sign(term2) * (torch.abs(term2) ** (1 / 3))
            x = (1 / a) * (term1 - b)
            y = (1 / c) * (term2 - d)
            out = torch.stack((x, y), dim=-1)
            return out

        class FInv(diff_newton_inverse.DifferentiableNewtonInverse):
            """implements f_inv using newtons method and implicit function theorem"""

            @staticmethod
            def my_function(x, theta1, theta2):
                return f(x, theta1, theta2)

        y = torch.tensor(
            [
                [[2.5183, 1.8197], [0.2476, 0.9849], [1.1826, -1.2012], [-1.8105, 1.0297]],
                [[0.2667, 0.1598], [-0.0577, 0.7857], [0.0991, -0.8331], [1.1598, -1.8567]],
            ],
            requires_grad=True,
        )
        theta1 = torch.tensor([[1.4, 1.2], [1.0, 0.3]], requires_grad=True)
        theta2 = torch.tensor([[1.3, -1.4], [0.8, -0.3]], requires_grad=True)

        # compute inverse and grad explicitly
        x = f_inv(y, theta1, theta2)
        torch.sum(x).backward()
        grad_y = y.grad
        grad_theta1 = theta1.grad
        grad_theta2 = theta2.grad

        # compute inverse and grad with newton
        x_init = x.detach().clone() + 0.05
        y = y.detach().clone()
        theta1 = theta1.detach().clone()
        theta2 = theta2.detach().clone()
        y.requires_grad = True
        theta1.requires_grad = True
        theta2.requires_grad = True
        x_newton = FInv.apply(y, x_init, 100, theta1, theta2)
        torch.sum(x_newton).backward()
        grad_y_newton = y.grad
        grad_theta1_newton = theta1.grad
        grad_theta2_newton = theta2.grad

        torch.testing.assert_close(x, x_newton)
        torch.testing.assert_close(grad_y_newton, grad_y)
        torch.testing.assert_close(grad_theta1_newton, grad_theta1)
        torch.testing.assert_close(grad_theta2_newton, grad_theta2)


if __name__ == "__main__":
    unittest.main()
    from io import StringIO

    with unittest.mock.patch("sys.stdout", new=StringIO()) as std_out:
        unittest.main()
