import os
import torch
import numpy as np
from PIL import Image
from utils.sh_utils import eval_sh
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)

def setup_renderer(cameras, image_size, radius=0.01, points_per_pixel=10):
    """
    Set up point cloud renderer
    
    Args:
        cameras: PyTorch3D camera object
        image_size: Size of rendered image in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        
    Returns:
        renderer: Configured PyTorch3D point cloud renderer
    """
    # Set up rasterization parameters
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )
    
    # Create renderer
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    
    return renderer

def fov_to_focal_length(fov, image_size):
    """
    Convert field of view to focal length
    
    Args:
        fov: Field of view in radians (a scalar)
        image_size: Size of the image dimension (width for fovx, height for fovy)
        
    Returns:
        focal_length: Focal length in pixels
    """
    # Calculate focal length: f = (image_size/2) / tan(fov/2)
    return (image_size / 2) / torch.tan(fov / 2)

def convert_camera_params(fovx_deg_list, fovy_deg_list, poses, image_size, device):
    """
    Convert camera intrinsic and pose matrices to PyTorch3D camera objects
    
    Args:
        fovx_list: List of horizontal field of view angles in degrees for each camera
        fovy_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] camera-to-world transformation matrices
        image_size: Rendered image size in (H, W) format
        device: Computation device
        
    Returns:
        cameras: PyTorch3D camera object
    """
    # Extract camera parameters
    M = poses.shape[0]  # Number of cameras
    H, W = image_size

    fovxs = torch.deg2rad(torch.tensor(fovx_deg_list, device=device))
    fovys = torch.deg2rad(torch.tensor(fovy_deg_list, device=device))

    focal_x = fov_to_focal_length(fovxs, W)
    focal_y = fov_to_focal_length(fovys, H)
    focal_lengths = torch.stack([focal_x, focal_y], dim=-1)

    principal_points = torch.tensor([[W/2, H/2]], device=device).expand(M, -1)
    
    # Extract rotation and translation from c2w matrices
    R = poses[:, :3, :3]  # [M, 3, 3]

    temp_pose = torch.linalg.inv(poses[0])
    T = (temp_pose[:3, 3]).unsqueeze(0).expand(M, -1)
    
    # PyTorch3D uses different coordinate system convention, need to convert
    # Convert from OpenCV/COLMAP coordinate system (right-down-forward) to PyTorch3D coordinate system (left-up-forward)
    R_convert = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float()
    R_converted = R @ R_convert

    T_converted = T @ R_convert

    # Create PyTorch3D camera
    cameras = PerspectiveCameras(
        focal_length=focal_lengths,
        principal_point=principal_points,
        R=R_converted,
        T=T_converted,
        in_ndc=False,
        image_size=[image_size] * M,  # Use the same image size for each camera
        device=device
    )
    
    return cameras

def pcd_render_multiview(gaussians, fovx_deg_list, fovy_deg_list, poses, image_size=(512, 512), radius=0.01, points_per_pixel=10, device='cuda'):
    """
    Render point cloud from multiple viewpoints, processing one view at a time
    
    Args:
        gaussians: GaussianModel
        fovx_deg_list: List of horizontal field of view angles in degrees for each camera
        fovy_deg_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] Camera pose matrices (camera-to-world)
        image_size: Rendered image size in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        device: Computation device
        
    Returns:
        images: [M, H, W, 3] Rendered RGB images
        view_masks: [M, H, W, 1] Rendered mask images
    """
    pcds = gaussians.get_xyz

    # Ensure inputs are tensors
    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, dtype=torch.float32, device=device)
    else:
        poses = poses.to(device)
    
    # Process one camera at a time to avoid batch dimension issues
    M = poses.shape[0]
    images_list = []
    masks_list = []
    
    for i in tqdm(range(M)):
        # Get single camera parameters
        fovx_deg_i = fovx_deg_list[i:i+1]
        fovy_deg_i = fovy_deg_list[i:i+1]
        pose_i = poses[i:i+1].to(device)

        # camera_center = pose_i[0, :3, 3]
        temp_pose = torch.linalg.inv(pose_i[0])
        camera_center = temp_pose[:3, 3]
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        dir_pp = (gaussians.get_xyz - camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        pcd_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        # Create camera
        cameras = convert_camera_params(fovx_deg_i, fovy_deg_i, pose_i, image_size, device)
        
        # Create renderer
        renderer = setup_renderer(cameras, image_size, radius, points_per_pixel)
        
        # Create point cloud (for a single view)
        point_cloud = Pointclouds(points=[pcds], features=[pcd_colors])
        
        # Render RGB image
        image = renderer(point_cloud)
        images_list.append(image)
        
        # Render mask
        white_colors = torch.ones_like(pcd_colors)
        point_cloud_mask = Pointclouds(points=[pcds], features=[white_colors])
        mask = renderer(point_cloud_mask)
        masks_list.append(mask)
    
    # Concatenate results
    images = torch.cat(images_list, dim=0)
    view_masks = torch.cat(masks_list, dim=0)
    
    return images, view_masks


def init_pcd_render_multiview(pcds, pcd_colors, fovx_deg_list, fovy_deg_list, poses, image_size=(512, 512), radius=0.01, points_per_pixel=10, device='cuda'):
    """
    Render point cloud from multiple viewpoints, processing one view at a time
    
    Args:
        pcds: [N, 3] point cloud
        pcd_colors: [N, 3] point cloud color
        fovx_deg_list: List of horizontal field of view angles in degrees for each camera
        fovy_deg_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] Camera pose matrices (camera-to-world)
        image_size: Rendered image size in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        device: Computation device
        
    Returns:
        images: [M, H, W, 3] Rendered RGB images
        view_masks: [M, H, W, 1] Rendered mask images
    """

    # Ensure inputs are tensors
    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, dtype=torch.float32, device=device)
    else:
        poses = poses.to(device)
    
    # Process one camera at a time to avoid batch dimension issues
    M = poses.shape[0]
    images_list = []
    masks_list = []
    
    for i in tqdm(range(M)):
        # Get single camera parameters
        fovx_deg_i = fovx_deg_list[i:i+1]
        fovy_deg_i = fovy_deg_list[i:i+1]
        pose_i = poses[i:i+1].to(device)
        
        # Create camera
        cameras = convert_camera_params(fovx_deg_i, fovy_deg_i, pose_i, image_size, device)
        
        # Create renderer
        renderer = setup_renderer(cameras, image_size, radius, points_per_pixel)
        
        # Create point cloud (for a single view)
        point_cloud = Pointclouds(points=[pcds], features=[pcd_colors])
        
        # Render RGB image
        image = renderer(point_cloud)
        images_list.append(image)
        
        # Render mask
        white_colors = torch.ones_like(pcd_colors)
        point_cloud_mask = Pointclouds(points=[pcds], features=[white_colors])
        mask = renderer(point_cloud_mask)
        masks_list.append(mask)
    
    # Concatenate results
    images = torch.cat(images_list, dim=0)
    view_masks = torch.cat(masks_list, dim=0)
    
    return images, view_masks



def save_rendered_images(images, output_dir, prefix, image_format="png"):
    """
    Save rendered images
    
    Args:
        images: [M, H, W, 3] Rendered images
        output_dir: Output directory
        prefix: Filename prefix
        image_format: Image format
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and save
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Ensure values are in [0, 1] range
        img = np.clip(img, 0, 1)
        
        # Convert to [0, 255] range as uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Save image
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(os.path.join(output_dir, f"{prefix}{i:06d}.{image_format}"))

