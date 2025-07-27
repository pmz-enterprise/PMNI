import torch,ipdb
import numpy as np
import mcubes
from tqdm import tqdm
from nerfacc import ContractionType, OccupancyGrid, ray_marching, \
    render_weight_from_alpha_patch_based, accumulate_along_rays_patch_based, \
    render_weight_from_alpha, accumulate_along_rays
from .visibility_tracer import VisibilityTracing
def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network,
                 gradient_method="ad", K=None,  H=None, W=None ,intrinsics_all =None, normals = None):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network

        # define the occ grid, see NerfAcc for more details
        self.scene_aabb = torch.as_tensor([-1., -1., -1., 1., 1., 1.], dtype=torch.float32)
        # define the contraction_type for scene contraction
        self.contraction_type = ContractionType.AABB
        # create Occupancy Grid
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=128,  # if res is different along different axis, use [256,128,64]
            contraction_type=self.contraction_type).to("cuda")
        self.sampling_step_size = 0.01  # ray marching step size, will be modified during training
        self.gradient_method = gradient_method   # dfd or fd or ad
        self.visible_ray_tracer = VisibilityTracing()
        self.K = K
        self.H = H
        self.W = W
        self.intrinsics_all = intrinsics_all
        self.normals = normals


    def occ_eval_fn(self, x):
        # function for updating the occ grid given the current sdf
        sdf = self.sdf_network(x)[..., :1]
        alpha = torch.sigmoid(- sdf * 80)  # occ grids with alpha below the occ threshold will be set as 0
        return alpha


    def render(self, rays_o_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     rays_d_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     marching_plane_normal,  # (num_patch, 3)
                     near,  # (num_patch,)
                     far,  # (num_patch,)
                     mask,
                     c2ws,
                     idx,
                     con =False,
                     val_gradient_method='ad',
                     mode='train'):
        # patch size, should be odd
        patch_H = rays_o_patch_all.shape[1]
        patch_W = rays_o_patch_all.shape[2]
        num_patch = rays_o_patch_all.shape[0]

        # extract camera location and ray direction of the patches' center pixels
        rays_o_patch_center = rays_o_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_d_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        def alpha_fn_patch_center(t_starts, t_ends, ray_indices, ret_sdf=False):
            # the function used in ray marching
            ray_indices = ray_indices.long()
            t_origins = rays_o_patch_center[ray_indices]
            t_dirs = rays_d_patch_center[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return
            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_start[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        #ipdb.set_trace()
        patch_indices, t_starts_patch_center, t_ends_patch_center = ray_marching(
            rays_o_patch_center, rays_d_patch_center,
            t_min=near,
            t_max=far,
            grid=self.occupancy_grid,
            render_step_size=self.sampling_step_size,
            stratified=True,
            cone_angle=0.0,
            early_stop_eps=1e-8,
            alpha_fn=alpha_fn_patch_center,
        )
        samples_per_ray = patch_indices.shape[0] / num_patch
        if patch_indices.shape[0] == 0:  # all patch center rays are within the zero region of the occ grid. skip this iteration.
            return {
                "comp_normal": torch.zeros([num_patch, patch_H, patch_W, 3], device=rays_o_patch_center.device)
            }

        num_samples = patch_indices.shape[0]
        patch_indices = patch_indices.long()

        # compute the sampling distance on remaining rays
        t_starts_patch_all = t_starts_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                                 /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]
        t_ends_patch_all = t_ends_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                               /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]
        mid_points_patch_all = (t_starts_patch_all + t_ends_patch_all)/2.0

        t_starts_patch_center_shift_left = t_starts_patch_center[1:]
        t_starts_patch_center_shift_left = torch.cat([t_starts_patch_center_shift_left, t_starts_patch_center[-1:]], 0)
        diff_mask = ((t_ends_patch_center - t_starts_patch_center_shift_left) != 0)[..., 0]
        positions_starts_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_starts_patch_all
        positions_ends_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_ends_patch_all  # (num_samples, patch_H, patch_W, 3)
        positions_ends_diff = positions_ends_patch_all[diff_mask]
        positions_all = torch.cat([positions_starts_patch_all, positions_ends_diff], 0)
        positions_all_flat = positions_all.reshape(-1, 3)
        #ipdb.set_trace()
        sdf_all = self.sdf_network(positions_all_flat)
        sdf_all = sdf_all.reshape(*positions_all.shape[:-1], 1)

        sdf_starts_patch_all = sdf_all[:positions_starts_patch_all.shape[0]]

        sdf_end_diff = sdf_all[positions_starts_patch_all.shape[0]:]
        sdf_ends_patch_all = sdf_starts_patch_all[1:]
        sdf_ends_patch_all = torch.cat([sdf_ends_patch_all, sdf_starts_patch_all[-1:]], 0)
        sdf_ends_patch_all[diff_mask] = sdf_end_diff

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter

        prev_cdf = torch.sigmoid(sdf_starts_patch_all * inv_s)  # (num_samples, patch_H, patch_W, 1)
        next_cdf = torch.sigmoid(sdf_ends_patch_all * inv_s)   # (num_samples, patch_H, patch_W, 1)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # (num_samples, patch_H, patch_W, 1)
        weights_cuda = render_weight_from_alpha_patch_based(alpha.reshape(num_samples, patch_H*patch_W, 1), patch_indices)  # (num_samples, patch_H, patch_W, 1)
        if mode == 'train':
            gradient_method = self.gradient_method
        elif mode == 'eval':
            gradient_method = val_gradient_method

        if gradient_method == "ad":
            gradients = self.sdf_network.gradient(positions_starts_patch_all.reshape(-1, 3)).reshape(num_samples, patch_H, patch_W, 3)
        
        

        weights_sum_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, n_patches=num_patch)  # (num_samples, patch_H, patch_W, 1)
        weights_sum = weights_sum_cuda.reshape(num_patch, patch_H, patch_W, 1)

        comp_normals_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, values=gradients.reshape(num_samples,patch_H * patch_W, 3),n_patches=num_patch)  # (num_samples, patch_H, patch_W, 3)
        comp_normal = comp_normals_cuda.reshape(num_patch, patch_H, patch_W, 3)

        comp_normal_plain = comp_normal.view(-1, 3)

        comp_depth_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, values=mid_points_patch_all.reshape(num_samples,patch_H * patch_W, 1),n_patches=num_patch)
        comp_depth = comp_depth_cuda.reshape(num_patch, patch_H, patch_W, 1)

        #mask ([4096, 1, 1, 1]) (num_patch, patch_H, patch_W, 1)
        
        surface_points = positions_starts_patch_all# (num_samples, patch_H, patch_W, 3)
        #   patch_indices #(num_samples)
        pre_mask = (mask.view(-1) > 0).squeeze()  
        surface_mask = pre_mask[patch_indices]

        cam_t = c2ws[:, :3, 3]  
        w2cs = torch.inverse(c2ws)
        surface_points_plain = surface_points.view(-1, 3).detach()
        idx_expand = torch.full(surface_mask.shape, idx, dtype=torch.long, device=surface_mask.device)
        if(con):
            with torch.no_grad():
                visibility_mask = self.visible_ray_tracer(sdf=lambda x: self.sdf_network.sdf(x),
                                                          unique_camera_centers=cam_t,
                                                          points=surface_points_plain[surface_mask])  # (num_points, num_cams)

                num_vis_points = visibility_mask.shape[0]
                visibility_mask[torch.arange(num_vis_points), idx_expand[surface_mask].long()] = 1
                assert torch.all(visibility_mask.sum(-1) > 0)
                points_homo = torch.cat(
                (surface_points_plain[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device='cuda')), -1).float()
                # project points onto all image planes
                # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                K_3x4 = self.intrinsics_all[:, :3, :]
            projection_matrices = torch.bmm(K_3x4, w2cs)

            pixel_coordinates_homo = torch.einsum("ijk,kp->ijp", projection_matrices.to('cuda'), points_homo.T)

            pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
            pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

            index_axis0 = torch.round(pixel_coordinates_yy)  # (num_points, num_cams)
            index_axis1 = torch.round(pixel_coordinates_xx)  # (num_points, num_cams)
            index_axis0 = torch.clamp(index_axis0, min=0, max=self.H - 1).to(torch.int64)  
            index_axis1 = torch.clamp(index_axis1, min=0, max=self.W - 1).to(torch.int64) 
            num_cams = index_axis0.shape[1]
            normal_world = []
            for cam_idx in range(num_cams):
                idx_normal = self.normals[cam_idx,
                                        index_axis0[:, cam_idx],
                                        index_axis1[:, cam_idx]]  # (num_surface_points)
                rotation_matrix = c2ws[cam_idx][:3, :3]

                idx_normals_world = torch.einsum('ij,nj->ni', rotation_matrix, idx_normal)
                normal_world.append(idx_normals_world)
            normal_world_all = torch.stack(normal_world, dim=1).to('cuda') #(num_points, num_cams, 3)


            weights_cuda_squeezed = weights_cuda.view(-1) 
            gradients_squeezed = gradients.view(-1, 3)     

            weights_cuda_filtered = weights_cuda_squeezed[surface_mask]  
            gradients_filtered = gradients_squeezed[surface_mask]       


        else:
            visibility_mask =None
            normal_world_all =None
            gradients_filtered =None
            weights_cuda_filtered=None



            
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        return {
            's_val': 1/inv_s,
            'weight_sum': weights_sum,
            'gradients': gradients,
            "comp_normal": comp_normal,
            "comp_depth": comp_depth,
            "samples_per_ray": samples_per_ray,
            'visibility_mask':visibility_mask, #(num_points, num_cams)
            'normal_world_all':normal_world_all, #(num_points, num_cams, 3)
            "gradients_filtered":gradients_filtered, #(num_points, 3)
            'weights_cuda_filtered':weights_cuda_filtered #[num_points]
        }

    @torch.no_grad()
    def render_normal_pixel_based(self, rays_o, rays_d, near, far):
        def alpha_fn(t_starts, t_ends, ray_indices, ret_sdf=False):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return

            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            # ic(diff_mask.shape, positions_ends_diff.shape, positions_starts.shape)
            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_start[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            t_min=near.squeeze(),
            t_max=far.squeeze(),
            grid=self.occupancy_grid,
            render_step_size=self.sampling_step_size,
            stratified=True,
            cone_angle=0.0,
            alpha_thre=0.0,
            early_stop_eps=1e-3,
            alpha_fn=alpha_fn,
        )

        alpha = alpha_fn(t_starts, t_ends, ray_indices)

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        gradients = self.sdf_network.gradient(positions).reshape(-1, 3)

        n_rays = rays_o.shape[0]
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)  # [n_samples, 1]
        comp_normal = accumulate_along_rays(weights, ray_indices, values=gradients, n_rays=n_rays)
        comp_depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        return comp_normal, comp_depth

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
    
    
    def mvas_net(self, pts, mask, idx,sdf_network, c2ws =None, device ='cuda',training =True):
        #pts ([4096, 1, 1, 3])
        #mask  ([4096, 1, 1, 1])
        cam_t = c2ws[:3, 3]
        if training:
            points = pts
            points, object_mask, idx = pts, mask, idx
            valid_mask = mask.view(-1) > 0

            num_pixels, _ = pts.shape
            
            surface_mask = object_mask[...,None].repeat(1,128).reshape(-1)
            idx = idx[...,None].repeat(1,128).reshape(-1)
            
            with torch.no_grad():
                visibility_mask = self.visible_ray_tracer(sdf=lambda x: sdf_network(x)[..., 0],
                                                          unique_camera_centers=dataset.unique_camera_centers.to(
                                                              device),
                                                          points=points[surface_mask])  # (num_points, num_cams)

                num_vis_points = visibility_mask.shape[0]
                visibility_mask[torch.arange(num_vis_points), idx[surface_mask].long()] = 1
                assert torch.all(visibility_mask.sum(-1) > 0)
                points_homo = torch.cat(
                    (points[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device=device)), -1).float()
                # project points onto all image planes
                # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                pixel_coordinates_homo = torch.einsum("ijk, kp->ijp", dataset.projection_matrices.to(device),
                                                      points_homo.T).cpu().detach().numpy()
                pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
                pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

                # opencv convention to numpy axis convention
                #  (top left) ----> x    =>  (top left) ---> axis 1
                #    |                           |
                #    |                           |
                #    |                           |
                #    y                         axis 0
                index_axis0 = np.round(pixel_coordinates_yy)  # (num_points, num_cams)
                index_axis1 = np.round(pixel_coordinates_xx)  # (num_points, num_cams)
                index_axis0 = np.clip(index_axis0, int(0), int(self.H - 1)).astype(
                    np.uint)  # (num_points, num_cams)
                index_axis1 = np.clip(index_axis1, int(0), int(self.W - 1)).astype(np.uint)

                num_cams = index_axis0.shape[1]
                tangent_vectors_all_view_list = []
                tangent_vectors_half_pi_all_view_list = []
                for cam_idx in range(num_cams):
                    azimuth_angles = dataset.azimuth_map_all_view[cam_idx,
                                                                  index_axis0[:, cam_idx],
                                                                  index_axis1[:, cam_idx]]  # (num_surface_points)
                    R_list = dataset.W2C_list[cam_idx]
                    r1 = R_list[0, :3]
                    r2 = R_list[1, :3]
                    tangent_vectors_all_view_list.append(
                        r1 * np.sin(azimuth_angles[:, None]) - r2 * np.cos(azimuth_angles[:, None]))
                    tangent_vectors_half_pi_all_view_list.append(r1 * np.sin(azimuth_angles[:, None] + np.pi / 2) -
                                                                 r2 * np.cos(azimuth_angles[:, None] + np.pi / 2))
                    

                tangent_vectors_all_view = torch.stack(tangent_vectors_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
                tangent_vectors_half_pi_all_view = torch.stack(tangent_vectors_half_pi_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
            output = {
                "tangent_vectors_all_view": tangent_vectors_all_view,
                "tangent_vectors_all_view_half_pi": tangent_vectors_half_pi_all_view,
                "visibility_mask": visibility_mask,
                "surface_mask": surface_mask,
                'network_object_mask': 0,
                'surface_normal': 0
            }

        else:
            pass

        return output
