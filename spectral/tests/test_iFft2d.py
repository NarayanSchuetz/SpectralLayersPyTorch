"""
 Created by Narayan Schuetz at 20/11/2018
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""
from unittest import TestCase
import torch
from spectral import iFft2d


class MockNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ifft = iFft2d(10, 10, mode="amp")

    def forward(self, x):
        x = self.ifft(x)
        return x


class TestIFft2d(TestCase):

    def test_full_pass(self):
        test = torch.ones(2, 3, 10, 20)
        net = MockNN()
        out = net(test)
        print(out.shape)


