from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import hflip

def get_gen_augs(args):
    """Returns a list of base transforms for image generation. Each should be
    able to accept multiple input images and be deterministic between any two
    images input at the same time, and return a list of the transformed images.
    """
    class RandomHorizontalFlips(nn.Module):
        """RandomHorizontalFlip but can be applied to multiple images."""
        def __init__(self, p=0.5):
            super(RandomHorizontalFlips, self).__init__()
            self.p = p

        def forward(self, images):
            """Returns [images] but with all elements flipped in the same
            direction, with the direction chosen randomly.

            Args:
            images  -- list of (PIL Image or Tensor): Images to be flipped
            """
            if torch.rand(1) < self.p:
                return [hflip(img) for img in images]
            return images

        def __repr__(self): return f"{self.__class__.__name__}(p={self.p})"

    class ToTensors(nn.Module):
        def __init__(self):
            super(ToTensors, self).__init__()
            self.to_tensor = transforms.ToTensor()

        def forward(self, images): return [self.to_tensor(x) for x in images]

        def __repr__(self): return self.__class__.__name__

    return transforms.Compose([
            RandomHorizontalFlips(),
            ToTensors()
        ])