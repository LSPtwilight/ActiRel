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

from guidance.vis_grid import VisibilityGrid

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)                 # 1: perturb input views, 2: interpolate input views, 3: random search
    parser.add_argument("--select_inpaint_num", required=True, type=str)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()
    input_view_num = len(train_viewpoints)

    see3d_render_path = os.path.join(args.source_path, 'see3d_render')
    os.makedirs(see3d_render_path, exist_ok=True)

    # copy reference images
    ref_views_save_root_path = os.path.join(see3d_render_path, 'ref-views')
    if not os.path.exists(ref_views_save_root_path):
        os.makedirs(ref_views_save_root_path, exist_ok=True)

        # copy ref-views from source_path
        src_image_root_path = os.path.join(args.source_path, 'images')
        temp_image_name = os.listdir(src_image_root_path)[0]
        postfix = temp_image_name.split('.')[-1]
        for viewpoint in train_viewpoints:
            image_name = f'{viewpoint.image_name}.{postfix}'
            shutil.copy(os.path.join(src_image_root_path, image_name), os.path.join(ref_views_save_root_path, image_name))

    # load see3d cameras
    see3d_cam_path = os.path.join(see3d_render_path, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        see3d_viewpoints, _ = load_see3d_cameras(see3d_cam_path, os.path.join(see3d_render_path, 'inpainted_images'))
        train_viewpoints.extend(see3d_viewpoints)
        print(f'Stage {args.see3d_stage} See3D cameras loaded from {see3d_cam_path}')
    else:
        if args.see3d_stage > 1:
            assert False, 'See3D cameras not found, but see3d_stage > 1'

    # save this stage see3d_render
    novel_views_save_root_path = os.path.join(see3d_render_path, f'stage{args.see3d_stage}')
    os.makedirs(novel_views_save_root_path, exist_ok=True)

    # render train views
    alpha_vis_thresh = 0.99
    train_save_root_path = os.path.join(novel_views_save_root_path, 'render-train-views')
    os.makedirs(train_save_root_path, exist_ok=True)
    train_view_depths = []
    for train_viewpoint in train_viewpoints:
        idx = train_viewpoint.colmap_id                                         # colmap_id is the unique index of the train viewpoint and see3d viewpoint
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach().cpu().numpy())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(train_save_root_path, f'{idx:05d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'depth_{idx:05d}.tiff'))

    print(f'Train views render done!')

    input_view_depths = train_view_depths[:input_view_num]
    input_viewpoints = train_viewpoints[:input_view_num]
    gs_input_view_depths = np.array(input_view_depths)
    gs_input_view_depths = torch.from_numpy(gs_input_view_depths).cuda()
    gs_input_view_depths = gs_input_view_depths.unsqueeze(1)
    gs_input_view_points = depths_to_points_parallel(gs_input_view_depths, input_viewpoints)

    # init visibility grid
    bbox_min = torch.min(gaussians.get_xyz, dim=0).values
    bbox_max = torch.max(gaussians.get_xyz, dim=0).values
    grid_resolution = 256
    visibility_grid = VisibilityGrid(bbox_min, bbox_max, grid_resolution, train_viewpoints, train_view_depths)
    visibility_grid.vis_invisible_pnts(os.path.join(novel_views_save_root_path, 'invisible_points.ply'))

    # generate novel cameras
    novel_poses, novel_cams = [], []
    large_fov_deg = 80
    if args.see3d_stage == 1:
        # interpolate between train_viewpoints
        novel_poses_1, novel_cams_1 = generate_interpolated_camera_poses(train_viewpoints, visibility_grid, interpolate_num=20)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

        # generate random perturbed camera poses
        novel_poses_2, novel_cams_2 = generate_random_perturbed_camera_poses(train_viewpoints, visibility_grid, position_std=0.15, rotation_std=0.1)
        novel_poses.extend(novel_poses_2)
        novel_cams.extend(novel_cams_2)
    elif args.see3d_stage == 2:
        # look at scene center
        novel_poses_1, novel_cams_1 = generate_see3d_camera_by_lookat_object_centric(train_viewpoints, visibility_grid, n_frames=40, fovy_deg=large_fov_deg)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

        # look at scene around
        novel_poses_2, novel_cams_2 = generate_see3d_camera_by_lookat(input_viewpoints, visibility_grid, gs_input_view_depths.squeeze(1), gs_input_view_points, n_frames=40, fovy_deg=large_fov_deg)
        novel_poses.extend(novel_poses_2)
        novel_cams.extend(novel_cams_2)
    else:
        # look around in input views position
        novel_poses_1, novel_cams_1 = generate_see3d_camera_by_view_angle(input_viewpoints, visibility_grid, fovy_deg=large_fov_deg, n_frames=60)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

        # # look around in scene center
        # novel_poses_2, novel_cams_2 = generate_look_around_camera_poses(input_viewpoints, visibility_grid, fovy_deg=80)
        # novel_poses.extend(novel_poses_2)
        # novel_cams.extend(novel_cams_2)

    # # vis train camera
    # train_c2ws = []
    # for train_cam in train_viewpoints:
    #     w2c = (train_cam.world_view_transform).transpose(0, 1).cpu().numpy()
    #     c2w = np.linalg.inv(w2c)
    #     train_c2ws.append(c2w)
    # train_c2ws = np.array(train_c2ws)
    # temp_mesh_path = './data/replica/scan5/gt_mesh/scene_mesh.ply'
    # vis_camera_pose(novel_poses, mesh_path=temp_mesh_path)
    # # vis_camera_pose(train_c2ws, mesh_path=temp_mesh_path)
    # exit()


    # render gs
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs')
    os.makedirs(gs_output_dir, exist_ok=True)

    gs_depths = []
    gs_none_visible_rate = []
    alpha_list = []
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
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
        alpha_list.append(alpha_vis_mask)

        none_visible_rate = 1 - alpha_vis_mask.sum() / (alpha_vis_mask.shape[0] * alpha_vis_mask.shape[1])
        gs_none_visible_rate.append(none_visible_rate)
        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))

        print(f'Novel view {idx} save done!')

    # render visibility map
    visibility_maps = visibility_grid.render_visibility_map(novel_cams, gs_depths)
    grid_none_visible_rate = []
    for idx in range(len(visibility_maps)):
        rgb_path = os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png')
        rgb_image = Image.open(rgb_path)
        rgb_image = np.array(rgb_image)

        vis_map = visibility_maps[idx].detach().cpu().numpy() > 0.5
        alpha_map = alpha_list[idx]
        vis_map = vis_map & alpha_map                            # filter vis_map by alpha_map
        vis_map = vis_map.astype(np.float32)

        none_visible_rate = 1 - vis_map.sum() / (vis_map.shape[0] * vis_map.shape[1])
        grid_none_visible_rate.append(none_visible_rate)

        rgb_image = rgb_image * vis_map[:,:,None]
        rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
        rgb_image.save(os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

        # save vis_map
        vis_map = vis_map.astype(np.uint8) * 255
        vis_map = Image.fromarray(vis_map)
        vis_map.save(os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

    print(f'Render visibility map done!')

    need_inpaint_views = select_need_inpaint_views(novel_cams, grid_none_visible_rate, gaussians, int(args.select_inpaint_num), none_visible_rate_high_bound=0.5, covisible_rate_high_bound=0.9)
    print(f'Need inpaint views: {need_inpaint_views}')

    select_gs_output_dir = os.path.join(novel_views_save_root_path, 'select-gs')
    os.makedirs(select_gs_output_dir, exist_ok=True)
    need_inpaint_views_cams = [novel_cams[i] for i in need_inpaint_views]
    need_inpaint_views_depths = [gs_depths[i] for i in need_inpaint_views]
    need_inpaint_views_depths = torch.stack(need_inpaint_views_depths, dim=0)

    # get need inpaint views points
    invalid_depth_mask = need_inpaint_views_depths <= 1e-6
    need_inpaint_views_depths[invalid_depth_mask] = 1e-3
    need_inpaint_views_points = depths_to_points_parallel(need_inpaint_views_depths, need_inpaint_views_cams)

    # # vis each view need inpaint views points
    # for idx in range(len(need_inpaint_views_cams)):
    #     vis_points = (need_inpaint_views_points[idx].cpu().numpy()).reshape(-1, 3)
    #     invalid_points_mask = (invalid_depth_mask[idx].cpu().numpy()).reshape(-1)
    #     vis_points = vis_points[~invalid_points_mask]
    #     trimesh.PointCloud(vis_points).export(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_need_inpaint_views_points_frame{idx:06d}.ply'))

    need_inpaint_views_points = need_inpaint_views_points.reshape(-1, 3)
    invalid_depth_mask_flatten = invalid_depth_mask.reshape(-1)
    need_inpaint_views_points = need_inpaint_views_points[~invalid_depth_mask_flatten]
    
    # save need inpaint views points
    need_inpaint_views_points_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_need_inpaint_views_points.ply')
    trimesh.PointCloud(need_inpaint_views_points.cpu().numpy()).export(need_inpaint_views_points_path)
    print(f'Saved need inpaint views points to {need_inpaint_views_points_path}')

    # save need inpaint views cameras
    save_cameras = {}
    save_cameras['train_views'] = len(train_viewpoints)
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
        shutil.copy(os.path.join(gs_output_dir, f'alpha_mask_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'alpha_warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'warp_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'warp_frame{idx:06d}.png'))
        shutil.copy(os.path.join(gs_output_dir, f'mask_frame{ori_id:06d}.png'), os.path.join(select_gs_output_dir, f'mask_frame{idx:06d}.png'))

    # save need inpaint views cameras
    save_cameras['n_views'] = len(need_inpaint_views_cams)
    np.savez(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_see3d_cameras.npz'), **save_cameras)

    print(f'See3D stage {args.see3d_stage} save done!')

