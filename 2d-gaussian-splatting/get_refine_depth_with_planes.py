import torch
from scene import Scene
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import save_img_f32, save_img_u8
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel
import trimesh
import numpy as np
import json
import open3d as o3d

from guidance.cam_utils import generate_see3d_camera, vis_camera_pose
from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
from guidance.See3D_modules.pcd_render_util import pcd_render_multiview, init_pcd_render_multiview, save_rendered_images, ori_init_pcd_render_multiview
import cv2
from PIL import Image

from matcha.pointmap.depthanythingv2 import depth_linear_align

import matplotlib.pyplot as plt

def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

def create_overlay_visualization(rgb_image, mask_obj, transparency=0.6, color=[0, 0, 255]):
    """
    Create overlay visualization of mask on RGB image
    
    Args:
        rgb_image: RGB image (H, W, 3)
        mask_obj: Binary mask (H, W)
        transparency: Mask transparency
        color: Specified color, default is blue
    
    Returns:
        blended: Overlaid image
    """
    # Ensure correct input format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask_obj] = color
    
    # Blend images
    blended = rgb_image.copy()
    mask_indices = mask_obj
    
    for i in range(3):
        blended[:, :, i] = np.where(
            mask_indices,
            rgb_image[:, :, i] * (1 - transparency) + colored_mask[:, :, i] * transparency,
            rgb_image[:, :, i]
        )
    
    return blended.astype(np.uint8)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--vis_mask", action='store_true')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    model_name=os.path.basename(args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()             # train views

    vis_mask = args.vis_mask
    vis_dir = 'mycode/check_error_align_depth/vis_linear_align'
    os.makedirs(vis_dir, exist_ok=True)
    cat_dir = 'mycode/check_error_align_depth/cat_vis_linear_align'
    os.makedirs(cat_dir, exist_ok=True)

    # load depth
    data_path = args.data_path
    for i in range(len(train_viewpoints)):
        depth_path = os.path.join(data_path, f"depth_frame{i:06d}.tiff")
        depth = np.array(Image.open(depth_path))
        depth = torch.from_numpy(depth).to('cuda')

        # load plane mask
        plane_mask_path = os.path.join(data_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)

        conf_path = os.path.join(data_path, f"vis_frequency_frame{i:06d}.npy")
        conf_map = np.load(conf_path)
        conf_map = (conf_map > 0.5)

        plane_id_list = np.unique(plane_mask)
        for plane_id in plane_id_list:
            if plane_id == 0:                       # 0 is default, not plane
                continue

            mask = (plane_mask == plane_id).astype(np.float32)
            mask = (mask > 0.5)

            valid_mask = mask & conf_map
            valid_mask = torch.from_numpy(valid_mask).to('cuda')

            valid_points_num = valid_mask.sum()
            if valid_points_num < 5:        # need at least 5 valid points for fitting
                continue

            mono_depth_path = os.path.join(data_path, f"mono_depth_frame{i:06d}.npy")
            mono_depth = np.load(mono_depth_path)
            mono_disp = 1. / (mono_depth + 1e-6)
            mono_disp = torch.from_numpy(mono_disp).to('cuda')

            # get valid depth for alignment
            aligned_depth, alpha, beta = depth_linear_align(mono_disp, depth, valid_mask, return_alpha_beta=True)

            if vis_mask:
                # draw scatter figure for mono_depth and depth
                true_disp = 1. / depth
                mono_disp_valid = mono_disp[valid_mask].cpu().numpy()
                true_disp_valid = true_disp[valid_mask].cpu().numpy()

                plt.figure(figsize=(10, 6))
                
                # Scatter plot of mono_disp vs true_disp
                plt.scatter(mono_disp_valid, true_disp_valid, alpha=0.6, s=1, c='blue')
                
                # Plot the fitted line: true_disp = beta * mono_disp + alpha
                mono_range = np.linspace(mono_disp_valid.min(), mono_disp_valid.max(), 100)
                fitted_line = beta * mono_range + alpha
                plt.plot(mono_range, fitted_line, 'r-', linewidth=2, 
                               label=f'Fitted line: y = {beta:.4f}x + {alpha:.4f}')
                
                plt.xlabel('Mono Disparity')
                plt.ylabel('True Disparity')
                plt.title(f'Disparity Alignment (Plane {plane_id}, Frame {i}, {valid_points_num} valid points)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save the analysis plot
                analysis_save_path = os.path.join(vis_dir, f'linear_align_analysis_plane{plane_id}_frame{i:06d}.png')
                plt.savefig(analysis_save_path, dpi=150, bbox_inches='tight')
                plt.close()

                rgb_path = os.path.join(data_path, f'rgb_frame{i:06d}.png')
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = create_overlay_visualization(rgb, mask)
                rgb_map = Image.fromarray(rgb)

                # cat scatter figure and rgb map
                ana_map = Image.open(analysis_save_path)

                rgb_height = rgb_map.height
                ana_height = ana_map.height
                assert rgb_height < ana_height
                # pad rgb_map to ana_map height
                temp_rgb_map = Image.new('RGB', (rgb_map.width, ana_height))
                temp_rgb_map.paste(rgb_map, (0, (ana_height - rgb_height) // 2))

                cat_img = Image.new('RGB', (rgb_map.width + ana_map.width, ana_map.height))
                cat_img.paste(temp_rgb_map, (0, 0))
                cat_img.paste(ana_map, (rgb_map.width + 10, 0))
                cat_img.save(os.path.join(cat_dir, f'frame{i:06d}_plane{plane_id:06d}.png'))


            # replace depth use aligned depth in mask region
            mask = torch.from_numpy(mask).to('cuda')
            depth[mask] = aligned_depth[mask]

        # save refine depth
        save_img_f32(depth.cpu().numpy(), os.path.join(data_path, f"refine_depth_frame{i:06d}.tiff"))

        # get points
        view_points = depths_to_points_parallel(depth, [train_viewpoints[i]])
        view_points = view_points.squeeze(0)

        # save points
        save_path = os.path.join(data_path, f"points_{i:06d}.ply")
        save_tensor_as_pcd(view_points, save_path)
        print(f"Saved points to {save_path}")

    print('done')
