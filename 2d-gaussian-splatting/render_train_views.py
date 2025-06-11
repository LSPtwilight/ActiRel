import torch
from scene import Scene, GaussianModel
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

from utils.general_utils import safe_state
from matcha.dm_utils.rendering import normal2curv
from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel
from guidance.cam_utils import build_visibility_masks

import cv2
import matplotlib.pyplot as plt

import trimesh
def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

from utils.point_utils import depth_to_normal
def get_surf_cam_normal(view, depth):
    world_normal_map = depth_to_normal(view, depth)
    surf_normal = world_normal_map.permute(2,0,1)
    surf_normal_cam = (surf_normal.permute(1,2,0) @ (view.world_view_transform[:3,:3])).permute(2,0,1)
    return surf_normal_cam

from matcha.dm_utils.rendering import depth2normal_parallel
def get_surf_cam_normal_pseudo_parallel(view, depth):
    views = [view]
    depth = depth.unsqueeze(0)
    world_view_transforms = torch.stack([views[i].world_view_transform for i in range(len(views))])
    full_proj_transforms = torch.stack([views[i].full_proj_transform for i in range(len(views))])
    normals = depth2normal_parallel(depth, world_view_transforms=world_view_transforms, full_proj_transforms=full_proj_transforms)
    normals = normals.permute(0, 3, 1, 2)
    normal = normals[0]        # [3, H, W]
    normal_cam = (normal.permute(1,2,0) @ (view.world_view_transform[:3,:3])).permute(2,0,1)
    return normal_cam

def get_surf_normal_parallel(views, depths):
    world_view_transforms = torch.stack([views[i].world_view_transform for i in range(len(views))])
    full_proj_transforms = torch.stack([views[i].full_proj_transform for i in range(len(views))])
    normals = depth2normal_parallel(depths, world_view_transforms=world_view_transforms, full_proj_transforms=full_proj_transforms)
    normals = normals.permute(0, 3, 1, 2)
    return normals

def create_vis_frequency_heatmap(rgb_image, match_mask, transparency = 0.8, colormap='viridis'):
    """
    Create a heatmap visualization overlaid on an RGB image based on match frequency.
    
    Args:
        rgb_image: The original RGB image (numpy array with shape [H, W, 3])
        match_mask: 2D array with same height and width as rgb_image, containing match frequency counts
        transparency: Transparency of the heatmap overlay (0.0 to 1.0)
        colormap: Matplotlib colormap name to use for the heatmap
    
    Returns:
        Combined visualization with heatmap overlaid on RGB image
    """
    # Ensure inputs have correct shapes
    assert rgb_image.shape[:2] == match_mask.shape, "RGB image and match mask must have same dimensions"

    match_mask = match_mask + 1e-4          # set base color
    
    # Normalize match frequency to 0-1 range for visualization
    if match_mask.max() > 0:
        normalized_mask = match_mask.astype(float) / match_mask.max()
    else:
        normalized_mask = match_mask.astype(float)
    
    # Get colormap from matplotlib
    colormap_func = plt.get_cmap(colormap)
    
    # Apply colormap to the normalized mask (returns RGBA)
    heatmap_colored = colormap_func(normalized_mask)
    
    # Convert to RGB with correct shape for blending
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Ensure RGB image is in the right format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Blend the original image with the heatmap
    blended = rgb_image.copy()
    for i in range(3):
        blended[:, :, i] = rgb_image[:, :, i] * (1 - transparency) + heatmap_rgb[:, :, i] * transparency
    
    return blended.astype(np.uint8)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--train_view_num", required=True, type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    model_name=os.path.basename(args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    dataset.eval = True                 # load all images
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoints = scene.getTrainCameras().copy()             # all views

    train_view_num = args.train_view_num
    view_json_path = os.path.join(args.data_path, f'split-{train_view_num}views.json')
    if os.path.exists(view_json_path):
        with open(view_json_path, 'r') as f:
            view_data = json.load(f)
        train_id_list = view_data['train']
        test_id_list = view_data['test']
    else:
        view_json_path = os.path.join(args.data_path, f'train_test_split_{train_view_num}.json')
        with open(view_json_path, 'r') as f:
            view_data = json.load(f)
        train_id_list = view_data['train_ids']
        test_id_list = view_data['test_ids']
    train_viewpoints = [viewpoints[i] for i in train_id_list]
    test_viewpoints = [viewpoints[i] for i in test_id_list]

    test_novel_views_save_root_path = os.path.join(args.model_path, 'test_see3d_render')
    os.makedirs(test_novel_views_save_root_path, exist_ok=True)

    # render train views
    train_save_root_path = os.path.join(test_novel_views_save_root_path, 'vis-train-views-debug-planes')
    if os.path.exists(train_save_root_path):
        shutil.rmtree(train_save_root_path)
    os.makedirs(train_save_root_path, exist_ok=True)

    train_depths = []
    for idx, train_viewpoint in enumerate(train_viewpoints):
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_depths.append(depth)

        surf_normal = render_pkg['surf_normal']
        rend_normal = render_pkg['rend_normal']
        rend_normal_cam = render_pkg['rend_normal_cam']
        surf_normal_cam = render_pkg['surf_normal_cam']
        rend_curvature = normal2curv(render_pkg['rend_normal'], torch.ones_like(render_pkg['rend_normal'][0:1]))

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(train_save_root_path, f'rgb_frame{idx:06d}.png'))

        depth_vis = depth[0].detach().cpu().numpy()
        save_img_f32(depth_vis, os.path.join(train_save_root_path, f'depth_frame{idx:06d}.tiff'))
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-5) + 1e-6       # not exactly need
        plt.imsave(os.path.join(train_save_root_path, f'depth_frame{idx:06d}.png'), depth_vis, cmap='viridis')

        # save rend_normal and surf_normal as npy
        np.save(os.path.join(train_save_root_path, f'rend_normal_world_frame{idx:06d}.npy'), rend_normal.permute(1,2,0).detach().cpu().numpy())
        np.save(os.path.join(train_save_root_path, f'surf_normal_world_frame{idx:06d}.npy'), surf_normal.permute(1,2,0).detach().cpu().numpy())

        save_img_u8(surf_normal.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, os.path.join(train_save_root_path, f'surf_normal_world_frame{idx:06d}.png'))
        save_img_u8(rend_normal.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, os.path.join(train_save_root_path, f'rend_normal_world_frame{idx:06d}.png'))
        save_img_u8(rend_normal_cam.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, os.path.join(train_save_root_path, f'rend_normal_cam_frame{idx:06d}.png'))
        save_img_u8(surf_normal_cam.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, os.path.join(train_save_root_path, f'surf_normal_cam_frame{idx:06d}.png'))
        
        rend_curvature_vis = rend_curvature[0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'rend_curvature_frame{idx:06d}.png'), rend_curvature_vis, cmap='viridis')

    train_depths = torch.stack(train_depths)            # [N, H, W]
    train_points = depths_to_points_parallel(train_depths, train_viewpoints)

    use_mast3r_matching = False             # not need use mast3r matching
    if use_mast3r_matching:
        # load mast3r matching
        mast3r_matching_path = os.path.join(test_novel_views_save_root_path, 'ref-views-mast3r', 'pixel_correspondences.json')
        if not os.path.exists(mast3r_matching_path):
            print(f"[WARNING] Mast3r matching file {mast3r_matching_path} not found, run mast3r matching now")
            cmd = f'python mast3r/resize_matcher.py -i {args.source_path}/ref-images -o {test_novel_views_save_root_path}/ref-views-mast3r --visualize'
            print(cmd)
            os.system(cmd)
            print(f"Mast3r matching done!")

        with open(mast3r_matching_path, 'r') as f:
            mast3r_matching = json.load(f)
    else:
        print('NOTE: use depth warp to build visibility mask')
        mast3r_matching = None
    
    visibility_times_masks = build_visibility_masks(train_viewpoints, train_depths, train_points, mast3r_matching=mast3r_matching, return_origin_masks=True)
    vis_points = []
    for idx in range(len(visibility_times_masks)):
        rgb_path = os.path.join(train_save_root_path, f'rgb_frame{idx:06d}.png')
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        match_mask = visibility_times_masks[idx][0].detach().cpu().numpy()
        np.save(os.path.join(train_save_root_path, f'vis_frequency_frame{idx:06d}.npy'), match_mask)
        blended = create_vis_frequency_heatmap(rgb_image, match_mask)
        cv2.imwrite(os.path.join(train_save_root_path, f'vis_frequency_frame{idx:06d}.png'), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        ori_points = train_points[idx].detach().cpu().numpy()
        vis_mask = (match_mask > 0.5).reshape(-1)
        vis_points.append(ori_points[vis_mask])
    vis_points = np.concatenate(vis_points, axis=0)
    save_tensor_as_pcd(vis_points, os.path.join(train_save_root_path, 'vis_points.ply'))

    # vis charts data
    sfm_root_path = (args.model_path).replace('free_gaussians', 'mast3r_sfm')
    charts_data_path = os.path.join(sfm_root_path, 'charts_data.npz')
    charts_data = load_charts_data(charts_data_path)
    charts_priors = build_priors_from_charts_data(charts_data, train_viewpoints)
    charts_depths = charts_priors['depths']
    charts_depth_normals = get_surf_normal_parallel(train_viewpoints, charts_depths)            # normal from charts depth
    charts_prior_depths = charts_priors['prior_depths']
    charts_confs = charts_priors['confs']
    charts_mono_normals = charts_priors['normals']                                              # normal from depth-anything-v2 (MAtCha use this as normal prior)
    charts_curvs = charts_priors['curvs']
    charts_prior_pcds = depths_to_points_parallel(charts_prior_depths, train_viewpoints)
    for idx in range(len(charts_prior_pcds)):
        charts_prior_pcd = charts_prior_pcds[idx]
        save_tensor_as_pcd(charts_prior_pcd, os.path.join(train_save_root_path, f'charts_prior_pcd_frame{idx:06d}.ply'))

    for idx in range(len(train_viewpoints)):
        vis_charts_conf = charts_confs[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'charts_conf_frame{idx:06d}.png'), vis_charts_conf, cmap='viridis')

        vis_charts_depth = charts_depths[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'charts_depth_frame{idx:06d}.png'), vis_charts_depth, cmap='viridis')

        # save charts depth as .tiff
        save_img_f32(charts_depths[idx][0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'charts_depth_frame{idx:06d}.tiff'))

        # get normal in world coordinate
        vis_charts_mono_normal_world = charts_mono_normals[idx].permute(1,2,0).detach().cpu().numpy()
        # save world normal as npy
        np.save(os.path.join(train_save_root_path, f'charts_mono_normal_world_frame{idx:06d}.npy'), vis_charts_mono_normal_world)
        # save world normal as png
        save_img_u8(vis_charts_mono_normal_world * 0.5 + 0.5, os.path.join(train_save_root_path, f'charts_mono_normal_world_frame{idx:06d}.png'))

        # get normal in camera coordinate
        vis_charts_mono_normal = (charts_mono_normals[idx].permute(1,2,0) @ train_viewpoints[idx].world_view_transform[:3,:3]).permute(2,0,1)
        vis_charts_mono_normal = vis_charts_mono_normal.permute(1,2,0).detach().cpu().numpy()
        # save normal as npy
        np.save(os.path.join(train_save_root_path, f'charts_mono_normal_frame{idx:06d}.npy'), vis_charts_mono_normal)
        # save normal as png
        save_img_u8(vis_charts_mono_normal * 0.5 + 0.5, os.path.join(train_save_root_path, f'charts_mono_normal_frame{idx:06d}.png'))
        
        # get normal from charts depth in camera coordinate
        vis_charts_depth_normal = (charts_depth_normals[idx].permute(1,2,0) @ train_viewpoints[idx].world_view_transform[:3,:3]).permute(2,0,1)
        vis_charts_depth_normal = vis_charts_depth_normal.permute(1,2,0).detach().cpu().numpy()
        # save normal as npy
        np.save(os.path.join(train_save_root_path, f'charts_depth_normal_frame{idx:06d}.npy'), vis_charts_depth_normal)
        # save normal as png
        save_img_u8(vis_charts_depth_normal * 0.5 + 0.5, os.path.join(train_save_root_path, f'charts_depth_normal_frame{idx:06d}.png'))

        vis_charts_curv = charts_curvs[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'charts_curv_frame{idx:06d}.png'), vis_charts_curv, cmap='viridis')


    print(f'Train views render done!')

