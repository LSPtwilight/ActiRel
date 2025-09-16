import torch
from scene import Scene, GaussianModel
from scene.dataset_readers import load_see3d_cameras
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import shutil
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image
import trimesh

from utils.general_utils import safe_state

from guidance.cam_utils import (
    generate_see3d_camera_by_lookat, 
    select_need_inpaint_views, 
    vis_camera_pose, 
    generate_see3d_camera_by_lookat_object_centric, 
    generate_random_perturbed_camera_poses, 
    generate_interpolated_camera_poses, 
    generate_look_around_camera_poses, 
    generate_see3d_camera_by_view_angle
)

from matcha.dm_scene.charts import depths_to_points_parallel


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument('--see3d_root_dir', type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    see3d_stage_path = os.path.join(args.see3d_root_dir, f'stage{args.see3d_stage}')
    see3d_cameras_path = os.path.join(see3d_stage_path, f'stage{args.see3d_stage}_see3d_cameras.npz')
    inpainted_image_root_path = os.path.join(see3d_stage_path, 'select-gs-inpainted')
    see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cameras_path, inpainted_image_root_path)

    alpha_vis_thresh = 0.99
    gs_depths = []
    render_save_root_path = os.path.join(see3d_stage_path, 'select-gs')
    os.makedirs(render_save_root_path, exist_ok=True)
    for idx, see3d_gs_camera in enumerate(see3d_gs_cameras_list):
        render_pkg = render(see3d_gs_camera, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']

        gs_depths.append(depth[0].detach())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(render_save_root_path, f'ori_warp_frame{idx:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(render_save_root_path, f'depth_frame{idx:06d}.tiff'))
        # save .npy
        np.save(os.path.join(render_save_root_path, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh

        none_visible_rate = 1 - alpha_vis_mask.sum() / (alpha_vis_mask.shape[0] * alpha_vis_mask.shape[1])
        save_img_u8(alpha_vis_mask, os.path.join(render_save_root_path, f'alpha_mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(render_save_root_path, f'alpha_warp_frame{idx:06d}.png'))

        print(f'See3D view {idx} save done!')

    print(f'See3D stage {args.see3d_stage} render done!')

    # save need inpaint views points
    need_inpaint_views_depths = torch.stack(gs_depths, dim=0)
    invalid_depth_mask = need_inpaint_views_depths <= 1e-6
    need_inpaint_views_depths[invalid_depth_mask] = 1e-3
    need_inpaint_views_points = depths_to_points_parallel(need_inpaint_views_depths, see3d_gs_cameras_list)

    need_inpaint_views_points = need_inpaint_views_points.reshape(-1, 3)
    invalid_depth_mask_flatten = invalid_depth_mask.reshape(-1)
    need_inpaint_views_points = need_inpaint_views_points[~invalid_depth_mask_flatten]
    
    # save need inpaint views points
    need_inpaint_views_points_path = os.path.join(see3d_stage_path, f'stage{args.see3d_stage}_need_inpaint_views_points.ply')
    trimesh.PointCloud(need_inpaint_views_points.cpu().numpy()).export(need_inpaint_views_points_path)
    print(f'Saved need inpaint views points to {need_inpaint_views_points_path}')

