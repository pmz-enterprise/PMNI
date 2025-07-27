import torch
import torch.nn as nn
import pypose as pp
import numpy as np
import cv2 as cv
import ipdb
from pypose import mat2SE3
import torch
import torch.nn as nn

class LearnScale(nn.Module):
    def __init__(self, num_cams=20, learn_scale=False, init_scale=None,fix_scaleN = False):
        """Depth distortion parameters

        Args:
            num_cams (int): Number of cameras.
            learn_scale (bool): Whether to update scale.
            init_scale (torch.Tensor): Initial scale values [num_cams, 1].
        """
        super(LearnScale, self).__init__()
        self.num_cams = num_cams
        self.fix_scaleN = fix_scaleN
        self.init_scale = init_scale
        if learn_scale:
            # If init_scale is provided, use it as the initial value; otherwise, initialize with ones.
            self.global_scales = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=learn_scale)
        else:
            self.global_scales = init_scale

    def forward(self, cam_id):
        cam_id = int(cam_id)
        scale = self.global_scales[cam_id]
        if self.fix_scaleN and cam_id ==(self.num_cams-1):
            scale = torch.tensor([1.0], device=self.global_scales.device)
        return scale