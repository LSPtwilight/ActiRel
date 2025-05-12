import torch
from scene import Scene, GaussianModel
from scene.gaussian_model import get_obj_gaussian_by_mask
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import shutil
from gaussian_renderer import render
from gaussian_renderer.trace import get_weights_by_single_view
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image

from utils.general_utils import safe_state

from guidance.cam_utils import generate_see3d_camera_by_lookat, select_need_inpaint_views, vis_camera_pose


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--output_root_path", required=True, type=str)
    args = get_combined_args(parser)
    model_name=os.path.basename(args.model_path)
    output_root_path = args.output_root_path

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()

    # render train views
    train_view_depths = []
    for idx, train_viewpoint in enumerate(train_viewpoints):
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach().cpu().numpy())

    mapping_id_dict = scene.mapping_id_dict
    obj_id_list = list(mapping_id_dict.values())

    # get suitable mask views for each object
    suitable_mask_views_dict = {}                       # obj_id : view_id
    for obj_id in obj_id_list:
        if obj_id == 0:             # not for background
            continue

        best_view_id = -1
        best_view_score = -1

        for idx, train_viewpoint in enumerate(train_viewpoints):
            instance_map = train_viewpoint.instance_map
            obj_mask = (instance_map == obj_id).cpu().numpy()

            # Skip if the object is not visible in this view
            if not obj_mask.any():
                continue
                
            # Get depth map for this view
            depth_map = train_view_depths[idx]
            
            # Calculate average depth of the object in this view
            obj_depths = depth_map[obj_mask]
            avg_depth = np.mean(obj_depths) ** 3
            
            # Count pixels where object is visible
            pixel_count = np.sum(obj_mask)
            
            # Calculate score: pixel count * average depth
            view_score = pixel_count * avg_depth
            if view_score > best_view_score:
                best_view_score = view_score
                best_view_id = idx
        
        if best_view_id != -1:
            suitable_mask_views_dict[obj_id] = best_view_id
            print(f"Best view for object {obj_id}: view {best_view_id} with score {best_view_score}")
        else:
            print(f"Warning: No suitable view found for object {obj_id}")

    # save mask views
    suitable_mask_root_path = os.path.join(output_root_path, "suitable_mask")
    os.makedirs(suitable_mask_root_path, exist_ok=True)
    for obj_id, view_id in suitable_mask_views_dict.items():
        obj_mask_map = train_viewpoints[view_id].instance_map.cpu().numpy()
        obj_mask_map = (obj_mask_map == obj_id).astype(np.uint8)
        rgb_map = train_viewpoints[view_id].original_image.permute(1, 2, 0).cpu().numpy()
        rgb_map = rgb_map * obj_mask_map[:, :, np.newaxis]
        rgb_map = (rgb_map * 255).astype(np.uint8)
        rgb_map_path = os.path.join(suitable_mask_root_path, f"obj_{obj_id}_view_{view_id}.png")
        Image.fromarray(rgb_map).save(rgb_map_path)
    print(f"Saved {len(suitable_mask_views_dict)} suitable mask views to {suitable_mask_root_path}")

    test_obj_id = 4             # the table
    suitable_view_id = suitable_mask_views_dict[test_obj_id]
    print(f"Suitable view for object {test_obj_id}: view {suitable_view_id}")

    unseen_value = -1
    obj_mask_thresh = 0.6

    with torch.no_grad():
        view_weights = get_weights_by_single_view(gaussians, train_viewpoints[suitable_view_id], pipe, background)
        obj_gs_mask = (view_weights == test_obj_id).sum(-1)/((view_weights != unseen_value).sum(-1) + 1e-6) > obj_mask_thresh
    if obj_gs_mask.sum() < 100:
        raise ValueError(f"No enough Gaussians for object {test_obj_id} in view {suitable_view_id}")

    # save obj gs
    obj_gs = get_obj_gaussian_by_mask(gaussians, obj_gs_mask)
    obj_gs_path = os.path.join(output_root_path, f"obj_{test_obj_id}_view_{suitable_view_id}_gs.ply")
    obj_gs.save_ply(obj_gs_path)
    print(f"Saved obj gs to {obj_gs_path}")

