import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.depth_loss_type = 'l1'#cfg['depth_loss_type']

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')

    def get_normal_loss(self, normal_pred, normal_gt, mask, mask_sum,normal_loss_type='l2'):
        normal_error = (normal_pred - normal_gt) * mask
        if normal_loss_type == 'l1':
            normal_loss = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
        elif normal_loss_type == 'l2':
            normal_loss = F.mse_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum

        return normal_loss
    def get_depth_loss(self, depth_pred, depth_gt, mask, mask_sum, depth_loss_type = 'l1'):
        depth_error = (depth_pred - depth_gt) * mask
        depth_error = torch.nan_to_num(depth_error, nan=0.0)
        if depth_loss_type == 'l1':
            depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum
        elif depth_loss_type == 'l2':
            depth_loss = F.mse_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum

        return depth_loss

    def depth_loss_dpt(self, pred_depth, gt_depth, weight=None):
        """
        :param pred_depth:  (H, W)
        :param gt_depth:    (H, W)
        :param weight:      (H, W)
        :return:            scalar
        """
        
        t_pred = torch.median(pred_depth)
        s_pred = torch.mean(torch.abs(pred_depth - t_pred))

        t_gt = torch.median(gt_depth)
        s_gt = torch.mean(torch.abs(gt_depth - t_gt))

        pred_depth_n = (pred_depth - t_pred) / s_pred
        gt_depth_n = (gt_depth - t_gt) / s_gt

        if weight is not None:
            loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-8)
        else:
            loss = F.mse_loss(pred_depth_n, gt_depth_n)
        return loss
    
    def get_depth_loss_2(self, depth_pred, depth_gt):
        if self.depth_loss_type == 'l1':
            loss = self.l1_loss(depth_pred, depth_gt) / float(depth_pred.shape[0])
        elif self.depth_loss_type=='invariant':
            loss = self.depth_loss_dpt(depth_pred, depth_gt)
        return loss

    def get_pc_loss(self, Xt, Yt):

        #ipdb.set_trace()
        # compute  error
        loss1 = self.comp_point_point_error(Xt[0].permute(1, 0), Yt[0].permute(1, 0))
        loss2= self.comp_point_point_error(Yt[0].permute(1, 0), Xt[0].permute(1, 0))
        loss = loss1 + loss2
        return loss
    def comp_closest_pts_idx_with_split(self, pts_src, pts_des):
        """
        :param pts_src:     (3, S)
        :param pts_des:     (3, D)
        :param num_split:
        :return:
        """

        pts_src_list = torch.split(pts_src, 500000, dim=1)
        idx_list = []
        for pts_src_sec in pts_src_list:
            diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)

            dist = torch.linalg.norm(diff, dim=0)  # (S, D)
            closest_idx = torch.argmin(dist, dim=1)  # (S,)
            idx_list.append(closest_idx)
        closest_idx = torch.cat(idx_list)
        return closest_idx
    def comp_point_point_error(self, Xt, Yt):
        if Xt.shape[1] == 0 or Yt.shape[1] == 0:
            raise ValueError("输入张量在第二维度上为空。")
        closest_idx = self.comp_closest_pts_idx_with_split(Xt, Yt)
        pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
        pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
        eng = torch.mean(pt_pt_dist)
        return eng
    
    def get_mask_loss(self, mask, comp_mask):
        mask_loss = F.binary_cross_entropy(comp_mask.clip(1e-5, 1.0 - 1e-5), mask)
        return mask_loss
    
    def get_eikonal_loss(self, gradients):
        gradients_norm = torch.linalg.norm(gradients, ord=2, dim=-1)
        eikonal_loss = F.mse_loss(gradients_norm, torch.ones_like(gradients_norm), reduction='mean')
        return eikonal_loss
    
    def get_sdf_loss(self, sdf_all, sdf_loss_type = 'l1'):
        if sdf_loss_type == 'l1':
            sdf_loss = F.l1_loss(sdf_all, torch.zeros_like(sdf_all), reduction='sum')/float(sdf_all.shape[0])
        elif(sdf_loss_type == 'l2'):
            sdf_loss = F.mse_loss(sdf_all, torch.zeros_like(sdf_all), reduction='sum')/float(sdf_all.shape[0])
        else:
            sdf_loss = torch.tensor(0.0).cuda().float()
        return sdf_loss
    
    def get_depth_loss3(self, depth_pred, depth_gt, mask, mask_sum, depth_loss_type = 'l1'):
        mask = mask.squeeze(-1)  # [batchsize, patchsize, patchsize]
        valid_gt = depth_gt[mask == 1].view(-1)  # [N]
        valid_pred = depth_pred[mask == 1].view(-1)  # [N]
        scale = (valid_gt * valid_pred).sum() / (valid_gt ** 2).sum()
        depth_error = valid_gt*scale - valid_pred
        if depth_loss_type == 'l1':
            depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum
        elif depth_loss_type == 'l2':
            depth_loss = F.mse_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum
        return depth_loss
    
    
    def get_con_loss3(self, 
                  weights_cuda_filtered=None,     # (num_points)
                  normal_world_all=None,          # (num_points, num_cams, 3)
                  visibility_mask=None):          # (num_points, num_cams)

        assert normal_world_all.shape[2] == 3, "normal_world_all should have shape (num_points, num_cams, 3)"

        loss = 0.0
        total_valid_points = 0  

        num_cams = normal_world_all.shape[1]  
        for cam_idx in range(num_cams - 1):  
            normal_cam1 = normal_world_all[:, cam_idx, :]  
            normal_cam2 = normal_world_all[:, cam_idx + 1, :]  
            normal_diff = normal_cam1 - normal_cam2  
            mse_loss = torch.sum(normal_diff ** 2, dim=1)  

            if visibility_mask is not None:
                visible_points = visibility_mask[:, cam_idx] * visibility_mask[:, cam_idx + 1]  
                mse_loss = mse_loss * visible_points  
                total_valid_points += visible_points.sum().item()  

            if weights_cuda_filtered is not None:
                mse_loss = mse_loss * weights_cuda_filtered  

            loss += mse_loss.sum()  

        if total_valid_points > 0:
            loss = loss / total_valid_points  
        else:
            loss = 0  

        return loss

    
    def forward(self, normal_pred= None, normal_gt = None,  depth_pred=None, depth_gt=None, 
                X=None, Y=None, mask = None ,comp_mask = None, gradients = None, sdf = None,
                weights={}, 
                visibility_mask = None, #(num_points, num_cams)
                normal_world_all = None, #(num_points, num_cams, 3)
                gradients_filtered = None, #(num_points, 3)
                weights_cuda_filtered = None, #(num_points, )
                normal_loss_type='l2', depth_loss_type = 'l1',**kwargs):

        mask = (mask > 0.5).float()


        mask_sum = mask.sum() + 1e-5

        
        if weights['normal_weight'] != 0.0:#normal_loss
            normal_loss= self.get_normal_loss(normal_gt=normal_gt, 
                                              normal_pred= normal_pred,
                                              mask = mask,
                                              mask_sum = mask_sum,
                                              normal_loss_type=normal_loss_type)
        else:
            normal_loss = torch.tensor(0.0).cuda().float()


        if weights['depth_weight'] != 0.0:#depth_loss
            depth_loss = self.get_depth_loss3(depth_gt= depth_gt,
                                             depth_pred= depth_pred,
                                             mask = mask,
                                            mask_sum = mask_sum,
                                            depth_loss_type= depth_loss_type)
        else: 
            depth_loss = torch.tensor(0.0).cuda().float()
        
        if weights['pc_weight']!=0.0: 
            pc_loss = self.get_pc_loss(X, Y)
        else:
            pc_loss = torch.tensor(0.0).cuda().float()


        if weights['mask_weight']!=0.0: # mask_loss
            mask_loss = self.get_mask_loss(mask = mask, comp_mask= comp_mask)
        else:
            mask_loss = torch.tensor(0.0).cuda().float()


        if weights['eikonal_weight']!=0.0:
            eikonal_loss = self.get_eikonal_loss(gradients)
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        if(weights['sdf_weight']!=0.0):
            sdf_loss = self.get_sdf_loss(sdf_all= sdf)
        else:
            sdf_loss = torch.tensor(0.0).cuda().float()

        if(weights['con_weight']!=0.0):
            con_loss = self.get_con_loss3(weights_cuda_filtered = weights_cuda_filtered,#(num_points
                                        normal_world_all=normal_world_all,#(num_points, num_cams, 3)
                                        visibility_mask = visibility_mask)#(num_points, num_cams)
        else:
            con_loss = torch.tensor(0.0).cuda().float()

        loss = weights['normal_weight'] * normal_loss + \
                    weights['depth_weight'] * depth_loss + \
                        weights['pc_weight'] * pc_loss+\
                            weights['mask_weight'] * mask_loss+\
                                weights['eikonal_weight'] * eikonal_loss+\
                                    weights['sdf_weight'] * sdf_loss+\
                                        weights['con_weight']*con_loss
        if torch.isnan(loss):
            breakpoint()
        return_dict = {
            'loss': loss,
            'loss_normal': normal_loss,
            'loss_depth': depth_loss,
            'loss_pc': pc_loss,
            'loss_mask':mask_loss,
            'loss_eikonal':eikonal_loss,
            'loss_sdf':sdf_loss,
            'loss_con':con_loss
        }
        return return_dict
