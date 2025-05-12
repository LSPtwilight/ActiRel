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

from guidance.cam_utils import generate_see3d_camera_by_lookat, select_need_inpaint_views, vis_camera_pose
from guidance.See3D_modules.pcd_render_util import init_pcd_render_multiview, save_rendered_images, filter_pcd_by_edge, downsample_pcd, vis_depth

from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel

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

    # copy reference images
    ref_views_save_root_path = os.path.join(args.model_path, 'test_see3d_render', 'ref-views')
    os.makedirs(ref_views_save_root_path, exist_ok=True)

    img_data_path = os.path.join(args.data_path, 'images')
    img_data_list = os.listdir(img_data_path)
    img_data_list.sort()
    for ref_view_id in train_id_list:
        # shutil.copy(os.path.join(args.data_path, 'images', f'{ref_view_id:06d}_rgb.png'), os.path.join(ref_views_save_root_path, f'{ref_view_id:06d}_rgb.png'))
        shutil.copy(os.path.join(img_data_path, img_data_list[ref_view_id]), os.path.join(ref_views_save_root_path, img_data_list[ref_view_id]))

    # render test views
    test_save_root_path = os.path.join(test_novel_views_save_root_path, 'render-test-views')
    os.makedirs(test_save_root_path, exist_ok=True)
    for idx, test_viewpoint in enumerate(test_viewpoints):
        render_pkg = render(test_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']

        alpha_vis_thresh = 0.99
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
        save_img_u8(alpha_vis_mask, os.path.join(test_save_root_path, f'mask_frame{idx:06d}.png'))

        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(test_save_root_path, f'warp_frame{idx:06d}.png'))

    print(f'Test views render done!')

    # use see3d inpaint test views
    inpaint_save_root_path = os.path.join(test_novel_views_save_root_path, 'inpaint-test-views')
    os.makedirs(inpaint_save_root_path, exist_ok=True)
    cmd = f'python 2d-gaussian-splatting/guidance/see3d_util.py --source_imgs_dir {ref_views_save_root_path} --warp_root_dir {test_save_root_path} --output_root_dir {inpaint_save_root_path}'
    print(cmd)
    os.system(cmd)
    print(f'Test views inpaint done!')
