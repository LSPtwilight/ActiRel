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

from guidance.cam_utils import generate_see3d_camera_by_lookat, select_need_inpaint_views, vis_camera_pose, generate_see3d_camera_by_lookat_object_centric
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
    parser.add_argument("--output_root_path", required=True, type=str)
    parser.add_argument("--select_inpaint_num", required=True, type=str)
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
    else:
        view_json_path = os.path.join(args.data_path, f'train_test_split_{train_view_num}.json')
        with open(view_json_path, 'r') as f:
            view_data = json.load(f)
        train_id_list = view_data['train_ids']
    train_viewpoints = [viewpoints[i] for i in train_id_list]

    novel_views_save_root_path = os.path.join(args.model_path, 'see3d_render')
    os.makedirs(novel_views_save_root_path, exist_ok=True)

    # copy reference images
    ref_views_save_root_path = os.path.join(args.model_path, 'see3d_render', 'ref-views')
    os.makedirs(ref_views_save_root_path, exist_ok=True)

    img_data_path = os.path.join(args.data_path, 'images')
    img_data_list = os.listdir(img_data_path)
    img_data_list.sort()
    for ref_view_id in train_id_list:
        # shutil.copy(os.path.join(args.data_path, 'images', f'{ref_view_id:06d}_rgb.png'), os.path.join(ref_views_save_root_path, f'{ref_view_id:06d}_rgb.png'))
        shutil.copy(os.path.join(img_data_path, img_data_list[ref_view_id]), os.path.join(ref_views_save_root_path, img_data_list[ref_view_id]))

    # render train views
    train_save_root_path = os.path.join(novel_views_save_root_path, 'render-train-views')
    os.makedirs(train_save_root_path, exist_ok=True)
    train_view_depths = []
    for idx, train_viewpoint in enumerate(train_viewpoints):
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach().cpu().numpy())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(train_save_root_path, f'{idx:05d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'depth_{idx:05d}.tiff'))
        # save .npy
        np.save(os.path.join(train_save_root_path, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_thresh = 0.99
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
        save_img_u8(alpha_vis_mask, os.path.join(train_save_root_path, f'alpha_mask_{idx:06d}.png'))
    print(f'Train views render done!')

    gs_train_view_depths = np.array(train_view_depths)
    gs_train_view_depths = torch.from_numpy(gs_train_view_depths).cuda()
    gs_train_view_depths = gs_train_view_depths.unsqueeze(1)
    gs_train_view_points = depths_to_points_parallel(gs_train_view_depths, train_viewpoints)

    # generate novel cameras
    # novel_poses, novel_cams = generate_see3d_camera_by_lookat(train_viewpoints, gs_train_view_depths.squeeze(1), gs_train_view_points)
    novel_poses, novel_cams = generate_see3d_camera_by_lookat_object_centric(train_viewpoints)

    # # vis train camera
    # train_c2ws = []
    # for train_cam in train_viewpoints:
    #     w2c = (train_cam.world_view_transform).transpose(0, 1).cpu().numpy()
    #     c2w = np.linalg.inv(w2c)
    #     train_c2ws.append(c2w)
    # train_c2ws = np.array(train_c2ws)

    # # temp_mesh_path = '/home/nijunfeng/mycode/project/gs-recon/priorgs/data/replica/scan6/gt_mesh/scene_mesh.ply'
    # temp_mesh_path = '/home/nijunfeng/mycode/project/gs-recon/priorgs/output/mipnerf360-6-views/bonsai-t1-scratch/tetra_meshes/tetra_mesh_binary_search_7_iter_14000.ply'
    # vis_camera_pose(novel_poses, mesh_path=temp_mesh_path)
    # # vis_camera_pose(train_c2ws, mesh_path=temp_mesh_path)
    # exit()

    # First render gs
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs')
    os.makedirs(gs_output_dir, exist_ok=True)

    gs_depths = []
    gs_none_visible_rate = []
    for idx, novel_cam in enumerate(novel_cams):

        render_pkg = render(novel_cam, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']

        gs_depths.append(depth[0].detach())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(gs_output_dir, f'depth_frame{idx:06d}.tiff'))
        # save .npy
        np.save(os.path.join(gs_output_dir, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_thresh = 0.99
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh

        none_visible_rate = 1 - alpha_vis_mask.sum() / (alpha_vis_mask.shape[0] * alpha_vis_mask.shape[1])
        gs_none_visible_rate.append(none_visible_rate)
        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

        print(f'Novel view {idx} save done!')

    need_inpaint_views = select_need_inpaint_views(novel_cams, gs_none_visible_rate, gaussians, int(args.select_inpaint_num))
    print(f'Need inpaint views: {need_inpaint_views}')

    select_gs_output_dir = args.output_root_path
    os.makedirs(select_gs_output_dir, exist_ok=True)
    need_inpaint_views_cams = [novel_cams[i] for i in need_inpaint_views]
    # save need inpaint views cameras
    save_cameras = {}
    for idx, need_inpaint_view_cam in enumerate(need_inpaint_views_cams):
        save_cameras[f'R_{idx:06d}'] = need_inpaint_view_cam.R
        save_cameras[f'T_{idx:06d}'] = need_inpaint_view_cam.T
        save_cameras[f'FoVx_{idx:06d}'] = need_inpaint_view_cam.FoVx
        save_cameras[f'FoVy_{idx:06d}'] = need_inpaint_view_cam.FoVy
        save_cameras[f'image_width_{idx:06d}'] = need_inpaint_view_cam.image_width
        save_cameras[f'image_height_{idx:06d}'] = need_inpaint_view_cam.image_height

        ori_id = need_inpaint_views[idx]
        # copy ori_id warp_frame into select_gs_output_dir
        shutil.copy(os.path.join(gs_output_dir, f'ori_warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'ori_warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'depth_frame{ori_id:06d}.tiff'), os.path.join(select_gs_output_dir, f'depth_frame{idx:06d}.tiff'))
        shutil.copy(os.path.join(gs_output_dir, f'alpha_{ori_id:06d}.npy'), os.path.join(select_gs_output_dir, f'alpha_{idx:06d}.npy'))
        shutil.copy(os.path.join(gs_output_dir, f'mask_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'mask_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'warp_frame{idx:06d}.png'))

    # save need inpaint views cameras
    save_cameras['n_views'] = len(need_inpaint_views_cams)
    np.savez(os.path.join(select_gs_output_dir, 'see3d_cameras.npz'), **save_cameras)
    print(f'See3D cameras save to {os.path.join(select_gs_output_dir, "see3d_cameras.npz")}')
