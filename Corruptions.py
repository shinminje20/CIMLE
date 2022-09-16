"""File containing utilities that provide corruptions that can be applied to
images. It can also be run to visualize what these corruptions actually do.
"""
import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Data import *
from utils.Utils import *

import kornia as K


################################################################################
# GENERATOR AUGMENTATIONS. These are for corrupting images before they are fed
# to the generator. The generator is responsible for outputting more than one
# image per input image.
################################################################################

class Corruption(nn.Module):

    def __init__(self, grayscale=1, **kwargs):
        super(Corruption, self).__init__()

        corruptions = []
        if grayscale == 1:
            corruptions.append(K.augmentation.RandomGrayscale(p=1))
        else:
            pass

        self.model = nn.Sequential(*corruptions)

    def forward(self, x): return self.model(x)