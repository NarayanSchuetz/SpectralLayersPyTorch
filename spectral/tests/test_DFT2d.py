"""
 Created by Angela Botros at 03/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""
from unittest import TestCase
import torch
from spectral import Dft2d
import numpy as np
from scipy.fftpack import fft2

class MockNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dft = Dft2d(10, 10, mode="complex")

    def forward(self, x):
        x = self.dft(x)
        return x


class TestIDft2d(TestCase):

    def test_full_pass(self):
        test = np.ones((3,1,10,10))
        test[:, :, ::3, ::3] = 3
        test[:, :, 1::3, 1::3] = 2
        for i in range(10):
            test[:, :, :, i] += 0.1 * np.sin(2 * np.pi * i / 10)
        test_torch = torch.tensor(test, dtype=torch.float32)

        reff = fft2(test[0, 0, :, :])
        refR = np.real(reff)
        refI = np.imag(reff)


        net = MockNN()
        out = net(test_torch)
        print('Network output')
        print(out.shape)
        outR = out[:, 0, :, :].detach().numpy()
        outI = out[:, 1, :, :].detach().numpy()
        print('outR', outR.shape)

        diffR = refR[:, :] - outR[0, :, :]
        diffI = refI[:, :] - outI[0, :, :]

        print(outR[0, :3, :3])
        print(refR[:3, :3])

        print(outI[0, :3, :3])
        print(refI[:3, :3])

        print(sum(sum(diffR)))
        print(sum(sum(diffI)))

