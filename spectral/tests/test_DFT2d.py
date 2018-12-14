"""
 Created by Angela Botros at 03/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""
from unittest import TestCase
import torch
from spectral import Dft2d, iDft2d
import numpy as np

class MockNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dft = Dft2d(12, 12, mode="complex")
        self.idft= iDft2d(12, 12, mode='complex')

    def forward(self, x):
        x = self.dft(x)
        x = self.idft(x)
        return x


class TestIDft2d(TestCase):

    def test_full_pass(self):
        # Generate two random feature maps
        test = np.ones((3,2,12,12))

        # first map
        test[:, 0, ::3, ::3] = 2.5
        test[:, 0, 1::3, 1::3] = -0.47
        for i in range(10):
            test[:, 0, :, i] += 0.1 * np.sin(2 * np.pi * i / 10)

        # first map
        test[:, 1, 2::2, 1::2] = -1.5
        test[:, 1, 1::3, 2::3] = 4
        for i in range(10):
            test[:, 1, :, i] += -0.3 * np.sin(2 * np.pi * i / 10 + np.pi/3)

        print(test.shape)
        for i in range(2):
            print(test[0, i, 1, 1])

        test_torch = torch.tensor(test, dtype=torch.float32)

        net = MockNN()
        out = net(test_torch)

        for i in range(4):
            print(out[0, i, 1, 1])


