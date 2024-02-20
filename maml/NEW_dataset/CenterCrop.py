import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms as T
from patchify import patchify, unpatchify

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        no_cell = True

        while no_cell:
            # image = pad_if_smaller(image, self.size)
            # target = pad_if_smaller(target, self.size, fill=255)
            # crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
            crop_image = F.center_crop(image, self.size)
            crop_target = F.center_crop(target, self.size)

            # print(np.array(target))
            no_cell = (np.sum(np.array(crop_target)) == 0)
            # print(no_cell)
            # print()

        return crop_image, crop_target



