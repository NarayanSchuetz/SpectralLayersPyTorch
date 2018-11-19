"""
 Created by Narayan Schuetz at 13/11/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import build_base_matrix_1d, build_base_matrix_1d_cos_II


# ----------------------------------------------------------------------------------------------------------------------
# linear transformation layers
# ----------------------------------------------------------------------------------------------------------------------


class Spectral1dBase(nn.Module):

    def __init__(self, in_features, fixed, base_matrix_builder=None, modus='amp'):
        """
        Defines base for linear 1d spectral transformation layers.
        :param in_features: the number of input features (signal length)
        :type in_features: int
        :param fixed: whether the layer should be fixed or not
        :type fixed: bool
        :param base_matrix_builder: a function that can be used to create a base matrix. Helps build the weight tensors.
        :type base_matrix_builder: function
        :param modus: whether amplitude/phase or real/imag signal should be returned
        :type modus: str
                modus='amp':     returns amplitude and phase of signal
                modus='complex': returns complex signal output of fourier transformation
        """
        super().__init__()
        self.register_parameter('bias', None)   # is this necessary?
        self.in_features = in_features
        self.base_matrix_builder = base_matrix_builder
        self.requires_grad = not fixed
        self.modus = modus

    def forward(self, input):
        return F.linear(input, self.weights)

    def extra_repr(self):
        return 'in_features={}, bias=False'.format(self.in_features)


class NaiveDst1d(Spectral1dBase):
    """
    Linear layer with weights initialized as a 'naive' one dimensional discrete sine transform.
    Dimensionality: the length of the input signal is the same as the output -> n_features_in == n_features_out
    """

    def __init__(self, in_features, fixed=False, base_matrix_builder=build_base_matrix_1d):
        super().__init__(in_features, fixed, base_matrix_builder)
        self.weights = nn.Parameter(self.create_weight_tensor(), requires_grad=self.requires_grad)

    def create_weight_tensor(self):
        X_base = self.base_matrix_builder(self.in_features)
        X = np.sin(X_base)
        return torch.tensor(X, dtype=torch.float32)


class NaiveDct1d(Spectral1dBase):
    """
    Linear layer with weights initialized as a 'naive' one dimensional discrete cosine transform.
    Dimensionality: the length the input signal is the same as the output -> n_features_in == n_features_out
    """

    def __init__(self, in_features, fixed=False, base_matrix_builder=build_base_matrix_1d):
        super().__init__(in_features, fixed, base_matrix_builder)
        self.weights = nn.Parameter(self.create_weight_tensor(), requires_grad=self.requires_grad)

    def create_weight_tensor(self):
        X_base = self.base_matrix_builder(self.in_features)
        X = np.cos(X_base)
        return torch.tensor(X, dtype=torch.float32)


class DctII1d(NaiveDct1d):
    """
        Linear layer with weights initialized as a one dimensional discrete cosine transform II
        (https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II).
        Dimensionality: the length the input signal is the same as the output -> n_features_in == n_features_out
    """
    def __init__(self, in_features, fixed, base_matrix_builder=build_base_matrix_1d_cos_II):
        super().__init__(in_features, fixed, base_matrix_builder)


class Fft1d(Spectral1dBase):
    """
        Linear layer with weights initialized as a one dimensional discrete fast fourier transform.
        Dimensionality: the length the input signal is half the output -> n_features_out == 2 x n_features_in
    """

    def __init__(self, in_features, fixed=False, base_matrix_builder=build_base_matrix_1d):
        super().__init__(in_features, fixed, base_matrix_builder)
        T_real, T_imag = self.create_weight_tensors()
        self.weights_real = nn.Parameter(T_real, requires_grad=self.requires_grad)
        self.weights_imag = nn.Parameter(T_imag, requires_grad=self.requires_grad)

    def create_weight_tensors(self):
        X_base = self.base_matrix_builder(self.in_features)
        T_real = torch.tensor(np.cos(X_base), dtype=torch.float32)
        T_imag = torch.tensor(np.sin(X_base), dtype=torch.float32)
        return T_real, T_imag

    def create_amplitude_phase(self):
        self.amp   = torch.sqrt(self.real**2 + self.imag**2)
        self.phase = torch.atan2(self.imag, self.real)

    def forward(self, input):
        self.real = F.linear(input, self.weights_real)
        self.imag = F.linear(input, self.weights_imag)
        if self.modus = 'complex':
            return torch.cat((self.real, self.imag), -1)
        elif self.modus = 'amp':
            create_amplitude_phase()
            return torch.cat((self.amp, self.phase), -1)

# ----------------------------------------------------------------------------------------------------------------------
# inverse linear transformation layers
# ----------------------------------------------------------------------------------------------------------------------


class iDctII1d(Spectral1dBase):
    """
    Linear layer with weights initialized as a 'naive' one dimensional inverse of the discrete cosine II transform.
    Dimensionality: the length of the input signal is the same as the output -> n_features_in == n_features_out.
    """

    def __init__(self, in_features, fixed=False, base_matrix_builder=None):
        super().__init__(in_features, fixed, base_matrix_builder)
        self.weights = nn.Parameter(self.create_weight_tensor(), requires_grad=self.requires_grad)

    def create_weight_tensor(self):
        """
        Generate tensor with coefficients of discrete cosine transformation
        inverse Discrete Cosine Transformation is 2/N * DCT III (x)
        https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
        :return: a tensor containing the weights
        :rtype: torch.Tensor(dtype=torch.float32)
        """
        signal_length = self.in_features

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


class iFft1d(Spectral1dBase):
    """
    Linear layer with weights initialized as inverse one dimensional discrete fast fourier transform.
    Dimensionality: the length of the input is double the size of the output -> n_features_in x 2 == n_features_out.
    """

    def __init__(self, in_features, fixed=False, base_matrix_builder=None):
        super(Spectral1dBase).__init__(in_features, fixed, base_matrix_builder)
        T_real, T_imag = self.create_weight_tensors()
        self.weights_real = nn.Parameter(T_real, requires_grad=self.requires_grad)
        self.weights_imag = nn.Parameter(T_imag, requires_grad=self.requires_grad)

    def create_weight_tensors(self):

        signal_length = self.in_features

        n = np.arange(0, signal_length, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (signal_length, 1)))
        f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
        X_f = np.tile(f.T, (1, signal_length))

        X = np.multiply(X, X_f)
        X = X * ((2 * np.pi) / signal_length)

        X_r = 1 / signal_length * np.cos(X)
        X_i = 1 / signal_length * np.sin(X)

        return torch.tensor(X_r, dtype=torch.float32), torch.tensor(X_i, dtype=torch.float32)

    def create_complex(self):
        # TODO: calculate real/imag from amplitude/phase
        return True

    def forward(self, input):
        return torch.cat((F.linear(input, self.weights_real), F.linear(input, self.weights_imag)), -1)
