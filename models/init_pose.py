import torch
from pytorch3d.renderer.cameras import look_at_rotation
import matplotlib.pyplot as plt
import numpy as np


def sample_points_on_sphere_uniform(latitude_deg, radius, num_points, clockwise=True):
    
    latitude_rad = torch.deg2rad(torch.tensor(latitude_deg)) 

    if clockwise:
        longitudes = torch.linspace(0, -2 * torch.pi + 2 * torch.pi / num_points, num_points)
    else:
        longitudes = torch.linspace(0, 2 * torch.pi - 2 * torch.pi / num_points, num_points)

    x = radius * torch.cos(latitude_rad) * torch.cos(longitudes)
    y = radius * torch.cos(latitude_rad) * torch.sin(longitudes)
    z = torch.full((num_points,), radius * torch.sin(latitude_rad)) 
    return torch.stack((x, y, z), dim=1)


def generate_c2w_matrices(camera_position, at=((0, 0, 0),), up=((0, 0, -1),), device='cuda'):
    
    R = look_at_rotation(camera_position, at=at, up=up, device=device)
    R_inv = R

    T = torch.zeros((R.shape[0], 4, 4), device=device)
    T[:, :3, :3] = R_inv 
    T[:, :3, 3] = camera_position  
    T[:, 3, 3] = 1.0  

    return T
if __name__ == '__main__': 
    radius = 15 
    num_points = 20 
    latitude_deg = 25  
    camera_position = sample_points_on_sphere_uniform(latitude_deg, radius, num_points)
    at = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # (1, 3)
    up = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)  # (1, 3)
    c2w_matrices = generate_c2w_matrices(camera_position, at=at, up=up)
   
    c2w_matrices_numpy = c2w_matrices.numpy()

    
    np.save('../c2w_matrices.npy', c2w_matrices_numpy)



def sample_positions_torch(radius_range, num_cameras,seed = 30):
    
    torch.manual_seed(seed)
    min_radius, max_radius = radius_range

    radii = torch.empty(num_cameras).uniform_(min_radius, max_radius)

    thetas = torch.empty(num_cameras).uniform_(0, 2 * torch.pi)  
    phis = torch.empty(num_cameras).uniform_(torch.pi / 6, torch.pi / 2)   

    xs = radii * torch.sin(phis) * torch.cos(thetas)
    ys = radii * torch.sin(phis) * torch.sin(thetas)
    zs = radii * torch.cos(phis)

    positions = torch.stack((xs, ys, zs), dim=1)
    return positions

