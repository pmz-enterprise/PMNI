import numpy as np
import torch
import pypose as pp
from torch import nn
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
def arange_pixels_2(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                   depth=None, device=torch.device("cpu")):
    ''' Arranges pixels for given resolution in range image_range, only outputs valid points based on depth.

    The function returns the unscaled pixel locations as integers and the
    scaled float values, filtered by valid depth values (non-NaN).

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        depth (tensor): depth map tensor, used to filter valid points
        device (torch.device): device to use
    '''
    h, w = resolution
    # Arrange pixel locations in scaled resolution
    pixel_locations = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    pixel_locations = torch.stack(
        [pixel_locations[1], pixel_locations[0]],  # Switch to (x, y) order
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0]) / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # If depth is provided, apply a mask to keep only valid points
    if depth is not None:
        assert depth.shape == (batch_size, h, w), "Depth shape must match resolution and batch size"
        depth = depth.view(batch_size, -1, 1)  # Flatten depth to match pixels (B x N x 1)
        #valid_mask = ~torch.isnan(depth)  # Create a mask for valid depth values (non-NaN)
        valid_mask = depth != 0
        # Filter pixel locations, scaled values, and depth based on valid_mask
        pixel_locations = pixel_locations[valid_mask.squeeze(-1)]
        pixel_scaled = pixel_scaled[valid_mask.squeeze(-1)]
        depth = depth[valid_mask]  # Filter depth as well

        # Reshape the filtered depth back to (B x N x 1)
        depth = depth.view(batch_size,-1, 1)

        pixel_scaled = pixel_scaled.view(batch_size, -1, 2)

    return pixel_locations, pixel_scaled, depth


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def transform_to_world(pixels, depth, camera_mat, world_mat=None, scale_mat=None,
                       invert=True, device=torch.device("cuda")):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    if world_mat is None:
        world_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]], dtype=torch.float32, device=device)
    if scale_mat is None:
        scale_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]], dtype=torch.float32, device=device)
    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)
    
    
    # Invert camera matrices
    if invert: 
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1) #B x 2 x N
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    pixels_depth = pixels.clone()
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels_depth

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)
def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    #r_lie=pp.so3(r)
    #R_2 = r_lie.matrix()
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R
def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w
def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def visualize_point_clouds_open3d(pc1, pc2):
   
    pc1_np = pc1.detach().cpu().numpy()
    pc2_np = pc2.detach().cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1_np)
    pcd1.paint_uniform_color([1, 0, 0]) 

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2_np)
    pcd2.paint_uniform_color([0, 0, 1]) 

    o3d.visualization.draw_geometries([pcd1, pcd2])


def visualize_point_clouds_matplotlib(pc1, pc2):
    pc1_np = pc1.detach().cpu().numpy()
    pc2_np = pc2.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc1_np[:, 0], pc1_np[:, 1], pc1_np[:, 2], c='r', marker='o', label='Point Cloud 1')
    
    ax.scatter(pc2_np[:, 0], pc2_np[:, 1], pc2_np[:, 2], c='b', marker='^', label='Point Cloud 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

def add_noise_to_pose(pose_all, rotation_noise_std=0.01, translation_noise_std=0.01):
    device = pose_all.device
    rotation_matrices = pose_all[:, :3, :3].to(device)
    translation_vectors = pose_all[:, :3, 3].to(device)
    n_images = rotation_matrices.shape[0]
    
    rotation_noise = pp.randn_so3(n_images, sigma=rotation_noise_std, requires_grad=False, dtype=torch.float32).to(device=pose_all.device)
    rotation_noise_matrices = rotation_noise.matrix().to(device)
    rotation_matrices_noisy = torch.matmul(rotation_noise_matrices, rotation_matrices)
    
    translation_noise = torch.randn_like(translation_vectors) * translation_noise_std
    translation_vectors_noisy = translation_vectors + translation_noise
    
    pose_all_noisy = pose_all.clone()
    pose_all_noisy[:, :3, :3] = rotation_matrices_noisy
    pose_all_noisy[:, :3, 3] = translation_vectors_noisy
    
    return pose_all_noisy #, rotation_noise_matrices, translation_noise


