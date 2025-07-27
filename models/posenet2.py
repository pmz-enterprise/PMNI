import torch
import pypose as pp
from torch import nn
import numpy as np
import cv2 as cv
import ipdb
from pytorch3d.renderer.cameras import look_at_rotation
from .init_pose import generate_c2w_matrices
class LearnPose2(nn.Module):
    def __init__(self, num_cams, learn_t = True,init_t=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose2, self).__init__()
        self.num_cams = num_cams
        self.t = nn.Parameter(init_t, requires_grad=True)
        self.init_t = init_t
        self.at = torch.tensor([[0.0, 0.0,  0.0]], dtype=torch.float32) 
        self.up = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)

    def forward(self, cam_id):
        cam_id = int(cam_id)
        t = self.t[cam_id]  # (3, )
        if(cam_id == (self.num_cams-1)):
            t = self.init_t[self.num_cams-1]
        # learn a delta pose between init pose and target pose, if a init pose is provided
        c2w = generate_c2w_matrices(camera_position=t.unsqueeze(0))
        return c2w.squeeze(0)
    def get_t(self):
       return self.t