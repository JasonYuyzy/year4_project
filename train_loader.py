# -*- encoding: utf-8 -*-
# -------------------------------------------
# Year 4 personal project code work arguments part
# -------------------------------------------
# Zhengyu Yu

import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms.functional as f


# the training loader: preprocess with the dataset before training
class Dataset(data.Dataset):
    def __init__(self, FG_paths, BG_paths, R_paths, transforms=None):
        self.FG_paths = FG_paths
        self.BG_paths = BG_paths
        self.R_paths = R_paths
        self.transform = transforms

    def __getitem__(self, index):
        fg_path = self.FG_paths[index]
        bg_path = self.BG_paths[index]
        r_path = self.R_paths[index]

        # image transformation to Tensor
        fg_img = self.transform(Image.open(fg_path))
        bg_img = self.transform(Image.open(bg_path))
        r_img = self.transform(Image.open(r_path))

        # resize the fg image into the normal size
        fg_img = f.affine(fg_img, angle=0,
                          translate=(0, 0),
                          shear=(0, 0),
                          scale=0.3,
                          fill=1.0)

        return (fg_img, bg_img, r_img)

    # function return the len of the training sets
    def __len__(self):
        return len(self.FG_paths)