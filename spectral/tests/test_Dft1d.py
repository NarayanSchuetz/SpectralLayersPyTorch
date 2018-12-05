"""
 Created by Angela Botros at 03/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""
from unittest import TestCase
import torch
from spectral import Dft1d

class MockNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dft = Dft1d(10, mode="amp")

    def forward(self, x):
        x = self.dft(x)
        return x


class TestIDft2d(TestCase):

    def test_full_pass(self):
        test = torch.ones(2, 3, 10)
        net = MockNN()
        out = net(test)
        print(out.shape)
