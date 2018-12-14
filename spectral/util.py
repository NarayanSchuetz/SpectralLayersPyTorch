"""
 Created by Narayan Schuetz at 16/11/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import numpy as np


def build_base_matrix_1d(signal_length, redundance=True, forward=True):
    """
    Helper function to create base matrix for the one dimensional case of multiple spectral transformations (naive
    sine, cosine and fft).
    :param signal_length: length of the input signal
    :type signal_length: int
    :param redundance: Returns full DFT (True) or omits redundant part (False)
    :type redundance: bool
    :param forward: choose forward (True) or backward (False) transformation
    :type forward: bool
    :return: base matrix
    :rtype: np.matrix
    """

    if not redundance:
        # If the redundant parts are/were omitted, then weight matrix is not square,
        # the dimension has to be adjusted through parameter 'coef'

        if forward:
            # In DFT, if redundant signal is omitted, length of signal is halfed
            coef = 0.5
        else:
            # In iDFT, if redundant signal was omitted in DFT, when transforming back
            # the resulting signal has to be the full length, again.
            coef = 2

        signal_out = int(signal_length * coef)
    else:
        signal_out = signal_length

    n = np.arange(0, signal_length, 1, dtype=np.float32)
    X = np.asmatrix(np.tile(n, (signal_out, 1)))
    f = np.asmatrix(np.arange(0, signal_out, dtype=np.float32))
    X_f = np.tile(f.T, (1, signal_length))

    X = np.multiply(X, X_f)

    if forward:
        X = X * ((-2 * np.pi) / signal_length)
    else:
        X = X * ((2 * np.pi) / signal_out)

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
