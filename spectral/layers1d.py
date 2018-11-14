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


# ----------------------------------------------------------------------------------------------------------------------
# util
# ----------------------------------------------------------------------------------------------------------------------


def build_base_matrix_1d(signal_length):
    """
    Helper function to create base matrix for the one dimensional case of multiple spectral transformations (naive
    sine, cosine and fft).
    :param signal_length: length of the input signal
    :type signal_length: int
    :return: base matrix
    :rtype: np.matrix
    """
    n = np.arange(0, signal_length, 1, dtype=np.float32)
    X = np.asmatrix(np.tile(n, (signal_length, 1)))
    f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
    X_f = np.tile(f.T, (1, signal_length))

    X = np.multiply(X, X_f)
    X = X * ((-2 * np.pi) / signal_length)

    return X


def build_base_matrix_1d_cos_II(signal_length):
    """
    Helper function to create base matrix for the one dimensional case of discrete cosine transform II.
    :param signal_length: length of the input signal
    :type signal_length: int
    :return: base matrix
    :rtype: np.matrix
    """
    n = np.arange(0, signal_length, 1, dtype=np.float32) + 0.5
    X = np.asmatrix(np.tile(n, (signal_length, 1)))
    f = np.asmatrix(np.arange(0, signal_length, dtype=np.float32))
    X_f = np.tile(f.T, (1, signal_length))

    X = np.multiply(X, X_f)
    X = X * (np.pi / signal_length)

    return X


# ----------------------------------------------------------------------------------------------------------------------
# linear transformation layers
# ----------------------------------------------------------------------------------------------------------------------


class Spectral1dBase(nn.Module):

    def __init__(self, in_features, fixed, base_matrix_builder):
        """
        Defines base for linear 1d spectral transformation layers.
        :param in_features: the number of input features (signal length)
        :type in_features: int
        :param fixed: whether the layer should be fixed or not
        :type fixed: bool
        :param base_matrix_builder: a function that can be used to create a base matrix need to build the weight tensors
        :type base_matrix_builder: function
        """
        self.register_parameter('bias', None)   # is this necessary?
        self.in_features = in_features
        self.base_matrix_builder = base_matrix_builder
        self.requires_grad = not fixed

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

    def forward(self, input):
        return torch.cat((F.linear(input, self.weights_real), F.linear(input, self.weights_imag)), -1)


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

    def forward(self, input):
        return torch.cat((F.linear(input, self.weights_real), F.linear(input, self.weights_imag)), -1)
