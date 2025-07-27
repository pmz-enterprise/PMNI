import torch
import pypose as pp
from torch import nn
import numpy as np
import cv2 as cv
import ipdb
from pypose import mat2SE3
from .common import make_c2w
import pytorch3d
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t,init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        #if(cam_id == (self.num_cams-1)):
        #    c2w = self.init_c2w[self.num_cams-1]
        #if(cam_id == 0):
        #    c2w = self.init_c2w[0]
        return c2w
    def get_t(self):
       
        all_t = torch.zeros((self.num_cams, 3), dtype=torch.float32, device=self.t.device)
        
        for cam_id in range(self.num_cams):

            r = self.r[cam_id]  # (3,)
            t = self.t[cam_id]  # (3,)
            c2w = make_c2w(r, t)  # (4, 4)
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            all_t[cam_id] = c2w[:3, 3]  # (3,)

        return all_t  
    def get_all_c2w(self):
        all_c2w = torch.zeros((self.num_cams, 4, 4), dtype=torch.float32, device=self.r.device)
        
        for cam_id in range(self.num_cams):
            r = self.r[cam_id]  # (3,)
            t = self.t[cam_id]  # (3,)
            c2w = make_c2w(r, t)  # (4, 4)
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            all_c2w[cam_id] = c2w

        return all_c2w 