"""
 Created by Narayan Schuetz at 16/11/2018
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import numpy as np


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