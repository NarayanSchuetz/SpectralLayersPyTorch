"""
 Created by Narayan Schuetz at 10/12/2018
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import torch.nn as nn


class DctII2dPooling(nn.Module):

    def __init__(self, new_width, new_height):
        super().__init__()

        self.width = new_width
        self.height = new_height

    def forward(self, x):
        return x[:,:, :self.new_height+1, :self.new_width+1]