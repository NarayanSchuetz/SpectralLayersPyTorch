"""
 Created by Narayan Schuetz at 14/11/2018
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectral2dBase(nn.Module):

    def __init__(self, nrows, ncols, fixed):
        """
        :param nrows: the number of rows of the 2d input (y dimension)
        :type nrows: int
        :param ncols: the number of columns of the 2d input (x dimension)
        :type ncols: int
        :param fixed: whether the layer should be fixed or not
        :type fixed: bool
        """
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.requires_grad = not fixed
        self.register_parameter('bias', None)

    def extra_repr(self):
        return 'nrows_in={}, ncols_in={}, bias=False'.format(
            self.x_dim, self.y_dim
        )


class Fft2d(Spectral2dBase):
    """
    Linear layer with weights initialized as a two dimensional discrete fast fourier transform.
    Dimensionality: input is expected to be matrix-like with an x- and y-axis, the output will be the same along the
    x-axis but double the y-axis of the input -> input: n_x, n_y, output: n_x, 2 x n_y
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)

        real_tensor1, imag_tensor1 = self.create_weight_tensors(signal_length=self.nrows)
        real_tensor2, imag_tensor2 = self.create_weight_tensors(signal_length=self.ncols)

        self.weights_real1 = nn.Parameter(real_tensor1, requires_grad=self.requires_grad)
        self.weights_real2 = nn.Parameter(real_tensor2, requires_grad=self.requires_grad)

        self.weights_imag1 = nn.Parameter(imag_tensor1, requires_grad=self.requires_grad)
        self.weights_imag2 = nn.Parameter(imag_tensor2, requires_grad=self.requires_grad)

    def create_weight_tensors(self, signal_length):
        n = np.arange(0, signal_length, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (signal_length, 1)))
        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
        X_f = np.tile(f.T, (1, signal_length))

        X = np.multiply(X, X_f)
        X = X * ((-2 * np.pi) / signal_length)

        X_r = np.cos(X)
        X_i = np.sin(X)

        return torch.tensor(X_r, dtype=torch.float32), torch.tensor(X_i, dtype=torch.float32)

    def forward(self, input):

        c1 = F.linear(input, self.weights_real1)
        s1 = F.linear(input, self.weights_imag1)

        real_part = F.linear(torch.transpose(c1, -1, -2), self.weights_real2) - \
                    F.linear(torch.transpose(s1, -1, -2), self.weights_imag2)

        imag_part = F.linear(torch.transpose(c1, -1, -2), self.weights_imag2) + \
                    F.linear(torch.transpose(s1, -1, -2), torch.transpose(self.weights_real2, -1, -2))

        return torch.cat((torch.transpose(real_part, -1, -2), torch.transpose(imag_part, -1, -2)), -1)


class DctII2d(Spectral2dBase):
    """
    Linear layer with weights initialized as two dimensional discrete cosine II transform.
    Dimensionality: input is expected to be
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)
        self.x_dim = ncols
        self.y_dim = nrows
        self.requires_grad = not fixed

        self.weights_x = nn.Parameter(self.create_weight_tensor(self.x_dim), requires_grad=self.requires_grad)
        self.weights_y = nn.Parameter(self.create_weight_tensor(self.y_dim), requires_grad=self.requires_grad)

    def create_weight_tensor(self, signal_length):
        """
        Generate matrix with coefficients of discrete cosine transformation
        Here, DCT II is implemented https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
        """
        n = np.arange(0, signal_length, 1, dtype=np.float32) + 0.5
        X = np.asmatrix(np.tile(n, (signal_length, 1)))

        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))

        X_f = np.tile(f.T, (1, signal_length))

        X = np.multiply(X, X_f)
        X = X * ((np.pi) / signal_length)

        X_i = np.cos(X)
        return torch.tensor(X_i, dtype=torch.float32)

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_x)
        return F.linear(torch.transpose(x, -2, -1), self.weights_y)