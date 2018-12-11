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
from .util import build_base_matrix_1d


class Spectral2dBase(nn.Module):

    def __init__(self, nrows, ncols, fixed, base_matrix_builder=None):
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
        self.base_matrix_builder = base_matrix_builder
        self.register_parameter('bias', None)

    def extra_repr(self):
        return 'nrows_in={}, ncols_in={}, bias=False'.format(
            self.x_dim, self.y_dim
        )

    def forward(self, *input):
        raise NotImplementedError


class Dft2d(Spectral2dBase):
    """
    Linear layer with weights initialized as a two dimensional discrete fast fourier transform.
    Dimensionality: input is expected to be matrix-like with an x- and y-axis, the output will be the same along the
    x-axis but double the y-axis of the input -> input: n_x, n_y, output: n_x, 2 x n_y
    """

    def __init__(self, nrows, ncols, fixed=False, base_matrix_builder=build_base_matrix_1d, mode='amp', redundance = False):
        super().__init__(nrows, ncols, fixed, base_matrix_builder)

        self.mode = mode
        self.redundance = redundance

        self._amp = None
        self._phase = None
        self._real = None
        self._imag = None

        real_tensor1, imag_tensor1 = self._create_weight_tensors(signal_length=self.nrows)
        real_tensor2, imag_tensor2 = self._create_weight_tensors(signal_length=self.ncols)

        self.weights_real1 = nn.Parameter(real_tensor1, requires_grad=self.requires_grad)
        self.weights_real2 = nn.Parameter(real_tensor2, requires_grad=self.requires_grad)

        self.weights_imag1 = nn.Parameter(imag_tensor1, requires_grad=self.requires_grad)
        self.weights_imag2 = nn.Parameter(imag_tensor2, requires_grad=self.requires_grad)

    def _create_weight_tensors(self, signal_length):
        X_base = self.base_matrix_builder(signal_length, redundance=self.redundance, forward=True)
        T_real = torch.tensor(np.cos(X_base), dtype=torch.float32)
        T_imag = torch.tensor(np.sin(X_base), dtype=torch.float32)
        return T_real, T_imag

    def _create_amplitude_phase(self):
        self._amp = torch.sqrt(self._real ** 2 + self._imag ** 2)
        self._phase = torch.atan2(self._imag, self._real)

    def forward(self, input):
        c1 = F.linear(input, self.weights_real1)
        s1 = F.linear(input, self.weights_imag1)

        real_part = F.linear(torch.transpose(c1, -1, -2), self.weights_real2) - \
                    F.linear(torch.transpose(s1, -1, -2), self.weights_imag2)

        imag_part = F.linear(torch.transpose(c1, -1, -2), self.weights_imag2) + \
                    F.linear(torch.transpose(s1, -1, -2), self.weights_real2)

        self._real = torch.transpose(real_part, -1, -2)
        self._imag = torch.transpose(imag_part, -1, -2)
        print(self._real.shape)
        print(self._imag.shape)

        if self.mode == 'complex':
            return torch.cat((self._real, self._imag), -1)
        elif self.mode == 'amp':
            self._create_amplitude_phase()
            return torch.cat((self._amp, self._phase), -1)
        else:
            raise AttributeError("'mode' should be 'complex' or 'amp' while %s was found!" % str(self.mode))


class DctII2d(Spectral2dBase):
    """
    Linear layer with weights initialized as two dimensional discrete cosine II transform.
    Dimensionality: input: nrows, ncols, output: 2 x nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)

        self.weights_1 = nn.Parameter(self._create_weight_tensor(self.nrows), requires_grad=self.requires_grad)
        self.weights_2 = nn.Parameter(self._create_weight_tensor(self.ncols), requires_grad=self.requires_grad)

    def _create_weight_tensor(self, signal_length):
        """
        Generate matrix with coefficients of discrete cosine transformation
        Here, DCT II is implemented https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
        """
        n = np.arange(0, signal_length, 1, dtype=np.float32) + 0.5
        X = np.asmatrix(np.tile(n, (signal_length, 1)))

        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))

        X_f = np.tile(f.T, (1, signal_length))

        X = np.multiply(X, X_f)
        X = X * (np.pi / signal_length)

        X_i = np.cos(X)
        return torch.tensor(X_i, dtype=torch.float32)

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)


class iDft2d(Spectral2dBase):
    """
    NOTE: the forward call might be rather slow, so use this part with caution.
    Linear layer with weights initialized as two dimensional inverse discrete fast fourier transform.
    Dimensionality: input: nrows, ncols, output: 2 x nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False, mode='complex'):
        super().__init__(nrows, ncols, fixed)

        self.mode = mode

        self._amp = None
        self._phase = None
        self._real = None
        self._imag = None

        real_tensor_1, imag_tensor_1 = self._create_weight_tensors(self.nrows)
        self.weights_real_1 = nn.Parameter(real_tensor_1, requires_grad=self.requires_grad)
        self.weights_imag_1 = nn.Parameter(imag_tensor_1, requires_grad=self.requires_grad)

        real_tensor_2, imag_tensor_2 = self._create_weight_tensors(self.ncols)
        self.weights_real_2 = nn.Parameter(real_tensor_2, requires_grad=self.requires_grad)
        self.weights_imag_2 = nn.Parameter(imag_tensor_2, requires_grad=self.requires_grad)

    def _create_weight_tensors(self, signal_length):

        n = np.arange(0, signal_length, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (signal_length, 1)))
        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
        X_f = np.tile(f.T, (1, signal_length))
        X = np.multiply(X, X_f)
        X = X * ((2 * np.pi) / signal_length)
        X_r = 1 / (self.nrows * self.ncols) * np.cos(X)
        X_i = 1 / (self.nrows * self.ncols) * np.sin(X)

        return torch.tensor(X_r, dtype=torch.float32), torch.tensor(X_i, dtype=torch.float32)

    def _create_complex(self, input):
        self._amp = input[:,:,:,:self.nrows]
        self._phase = input[:,:,:,self.nrows:]

        self._real = self._amp * torch.cos(self._phase)
        self._imag = self._amp * torch.sin(self._phase)
        return True

    def forward(self, input):
        if self.mode == 'amp':
            self._create_complex(input)
        elif self.mode == 'complex':
            self._real = input[:,:,:,self.nrows]
            self._imag = input[:,:,:,self.nrows:]

        else:
            raise AttributeError("'mode' should be 'complex' or 'amp' while %s was found!" % str(self.mode))


        c1_real = F.linear(self._real, self.weights_real_1)
        c1_imag = F.linear(self._imag, self.weights_real_1)

        s1_real = F.linear(self._real, self.weights_imag_1)
        s1_imag = F.linear(self._imag, self.weights_imag_1)

        real_part = F.linear(torch.transpose(c1_real, -1, -2), self.weights_real_2) - \
                    F.linear(torch.transpose(s1_real, -1, -2), self.weights_imag_2) - \
                    F.linear(torch.transpose(c1_imag, -1, -2), self.weights_imag_2) - \
                    F.linear(torch.transpose(s1_imag, -1, -2), self.weights_real_2)

        imag_part = F.linear(torch.transpose(c1_real, -1, -2), self.weights_imag_2) + \
                    F.linear(torch.transpose(s1_real, -1, -2), self.weights_real_2) + \
                    F.linear(torch.transpose(c1_imag, -1, -2), self.weights_real_2) - \
                    F.linear(torch.transpose(s1_imag, -1, -2), self.weights_imag_2)

        return torch.cat((torch.transpose(real_part, -1, -2), torch.transpose(imag_part, -1, -2)), -1)


class iDctII2d(Spectral2dBase):
    """
    Linear Layer with weights initialized as two dimensional inverse discrete cosine II transform.
    Dimensionality: input: nrows, ncols, output: nrows, ncols (the last two dimensions are supposed to be the part
    where the transform will be applied to - as is usually the case in PyTorch.
    """

    def __init__(self, nrows, ncols, fixed=False):
        super().__init__(nrows, ncols, fixed)

        self.weights_1 = nn.Parameter(self._create_weight_tensor(self.nrows), requires_grad=self.requires_grad)
        self.weights_2 = nn.Parameter(self._create_weight_tensor(self.ncols), requires_grad=self.requires_grad)

    def _create_weight_tensor(self, signal_length):

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
