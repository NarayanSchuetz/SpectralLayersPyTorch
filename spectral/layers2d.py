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

    def forward(self, *input):
        raise NotImplementedError


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
    Dimensionality: input: nrows, ncols, output: 2 x nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)

        self.weights_1 = nn.Parameter(self.create_weight_tensor(self.nrows), requires_grad=self.requires_grad)
        self.weights_2 = nn.Parameter(self.create_weight_tensor(self.ncols), requires_grad=self.requires_grad)

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
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)


class iFft2d(Spectral2dBase):
    """
    NOTE: the forward call might be rather slow, so use this part with caution.
    Linear layer with weights initialized as two dimensional inverse discrete fast fourier transform.
    Dimensionality: input: nrows, ncols, output: 2 x nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False, amplitude=False):
        super().__init__(nrows, ncols, fixed)

        self.amplitude = amplitude

        real_tensor_1, imag_tensor_1 = self.create_weight_tensors(self.nrows)
        self.weights_real_1 = nn.Parameter(real_tensor_1, requires_grad=self.requires_grad)
        self.weights_imag_1 = nn.Parameter(imag_tensor_1, requires_grad=self.requires_grad)

        real_tensor_2, imag_tensor_2 = self.create_weight_tensors(self.ncols)
        self.weights_real_2 = nn.Parameter(real_tensor_2, requires_grad=self.requires_grad)
        self.weights_imag_2 = nn.Parameter(imag_tensor_2, requires_grad=self.requires_grad)

    def create_weight_tensors(self, signal_length):

        n = np.arange(0, signal_length, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (signal_length, 1)))
        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
        X_f = np.tile(f.T, (1, signal_length))
        X = np.multiply(X, X_f)
        X = X * ((2 * np.pi) / signal_length)
        X_r = 1 / (self.in_features_x * self.in_features_y) * np.cos(X)
        X_i = 1 / (self.in_features_x * self.in_features_y) * np.sin(X)

        return torch.tensor(X_r, dtype=torch.float32), torch.tensor(X_i, dtype=torch.float32)

    def forward(self, input):

        c1_real = F.linear(input[:self.nrows], self.weights_real_1)
        c1_imag = F.linear(input[self.nrows:], self.weights_real_1)

        s1_real = F.linear(input[:self.nrows], self.weights_imag_1)
        s1_imag = F.linear(input[self.nrows:], self.weights_imag_1)

        real_part = F.linear(torch.transpose(c1_real, -1, -2), self.weights_real_2) - \
                    F.linear(torch.transpose(s1_real, -1, -2), self.weights_imag_2) - \
                    F.linear(torch.transpose(c1_imag, -1, -2), self.weights_imag_2) - \
                    F.linear(torch.transpose(s1_imag, -1, -2), self.weights_real_2)

        imag_part = F.linear(torch.transpose(c1_real, -1, -2), self.weights_imag_2) + \
                    F.linear(torch.transpose(s1_real, -1, -2), self.weights_real_2) + \
                    F.linear(torch.transpose(c1_imag, -1, -2), self.weights_real_2) - \
                    F.linear(torch.transpose(s1_imag, -1, -2), self.weights_imag2)

        if self.amplitude:
            amplitude = lambda a, b: torch.sqrt(a ** 2 + b ** 2)
            return amplitude(real_part, imag_part)

        return torch.cat((torch.transpose(real_part, -1, -2), torch.transpose(imag_part, -1, -2)), -1)


class iDctII2d(Spectral2dBase):
    """
    Linear Layer with weights initialized as two dimensional inverse discrete cosine II transform.
    Dimensionality: input: nrows, ncols, output: nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)

        self.weights_1 = nn.Parameter(self.create_weight_tensor(self.nrows))
        self.weights_2 = nn.Parameter(self.create_weight_tensor(self.ncols))

    def create_weight_tensor(self, signal_length):

        n = np.arange(0, signal_length, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (signal_length, 1)))
        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32)) + 0.5
        X_f = np.tile(f.T, (1, signal_length))
        X = np.multiply(X, X_f)
        X = X * ((np.pi) / signal_length)
        X = np.cos(X)

        X[:, 0] = 0.5 * X[:, 0]
        X_i = (2 / signal_length) * X

        return torch.tensor(X_i, dtype=torch.float32)

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)



