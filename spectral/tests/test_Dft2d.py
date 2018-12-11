"""
 Created by Angela Botros at 03/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""
from unittest import TestCase
import torch
from spectral import DctII2d
import numpy as np


class MockNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dft = DctII2d(20, 20, fixed=True)

    def forward(self, x):
        x = self.dft(x)
        return x


class TestIDft2d(TestCase):

    def test_full_pass(self):
        n = np.arange(0, 20, 1, dtype=np.float32)
        X = np.asmatrix(np.tile(n, (20, 1)))
        f = np.asmatrix(np.arange(0, 20, dtype=np.float32))
        X_f = np.tile(f.T, (1, 20))
        X_f = np.sin(2*np.pi*X_f)

        test = torch.ones(2, 3, 20, 20, dtype=torch.float32) + torch.tensor(X_f, dtype=torch.float32)
        net = MockNN()
        out = net(test)
        return out

a = TestIDft2d()
a.test_full_pass()