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
from PIL import Image, ImageDraw, ImageFont
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
    generate_see3d_camera_by_view_angle,
    generate_see3d_camera_by_lookat_none_vis_plane,
    generate_see3d_camera_by_lookat_all_plane,
    generate_cameras_from_positions_looking_at,
    generate_see3d_camera_by_fps_planes,
    select_views_by_plane_coverage
)

from matcha.dm_scene.charts import depths_to_points_parallel

from guidance.vis_grid import VisibilityGrid
from planes.get_global_3Dpnts import get_none_vis_global_3Dpnts, get_visible_mask_for_input_views, get_all_global_3Dpnts

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

    REAL_H=512
    REAL_W=512
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
    alpha_vis_thresh = 0.95
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
    #visibility_grid.vis_visible_vs_invisible(os.path.join(novel_views_save_root_path, 'invisible_points.ply'), downsample=15)

    # generate novel cameras
    novel_poses, novel_cams = [], []
    plane_root_path = os.path.join(args.source_path, 'plane-refine-depths')
    vis_plane_pnts_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_vis_global_3Dplane_points')
    used_top_k = 8
    plane_all_points_dict = get_all_global_3Dpnts(args.source_path, plane_root_path, see3d_render_path, vis_plane_pnts_path, top_k=used_top_k)

    ### filter global target points within room AABB
    if len(plane_all_points_dict) > 0:
        all_plane_pts_np = np.concatenate(list(plane_all_points_dict.values()), axis=0)
        all_plane_pts_torch = torch.from_numpy(all_plane_pts_np).float().cuda()
        room_min = all_plane_pts_torch.min(dim=0).values - 0.2
        room_max = all_plane_pts_torch.max(dim=0).values + 0.2
        
        vg = visibility_grid
        nx, ny, nz = vg.resolution, vg.resolution, vg.resolution
        
        x_indices = torch.arange(nx, device=vg.device)
        y_indices = torch.arange(ny, device=vg.device)
        z_indices = torch.arange(nz, device=vg.device)
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        grid_centers = torch.stack([
            vg.bbox_min[0] + (X + 0.5) * vg.grid_size[0],
            vg.bbox_min[1] + (Y + 0.5) * vg.grid_size[1], 
            vg.bbox_min[2] + (Z + 0.5) * vg.grid_size[2]
        ], dim=-1)

        global_target_mask = (vg.visibility_grid < 0.5) & \
                             (grid_centers[..., 0] >= room_min[0]) & (grid_centers[..., 0] <= room_max[0]) & \
                             (grid_centers[..., 1] >= room_min[1]) & (grid_centers[..., 1] <= room_max[1]) & \
                             (grid_centers[..., 2] >= room_min[2]) & (grid_centers[..., 2] <= room_max[2])
        
        global_target_pnts = grid_centers[global_target_mask]
        print(f">>> [Setup] Extracted {global_target_pnts.shape[0]} valid target voxels within room bounds.")
    else:
        global_target_pnts = None
        print(">>> [Setup] Warning: No planes found, global_target_pnts is empty.")


    used_fov_deg = 80
    only_warp_input_views = False
    select_view_method = 'depth_quality'
    if args.see3d_stage == 1:
        novel_poses_1, novel_cams_1 = generate_see3d_camera_by_fps_planes(train_viewpoints, visibility_grid, plane_all_points_dict, n_samples=80, traj_center=None, width=512, height=512, fovy_deg=used_fov_deg, fovx_deg=None)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

    elif args.see3d_stage == 2:
        novel_poses_1, novel_cams_1 = generate_see3d_camera_by_fps_planes(train_viewpoints, visibility_grid, plane_all_points_dict, n_samples=80, traj_center=None, width=512, height=512, fovy_deg=used_fov_deg, fovx_deg=None)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

    elif args.see3d_stage == 3:
        novel_poses_1, novel_cams_1 = generate_see3d_camera_by_fps_planes(train_viewpoints, visibility_grid, plane_all_points_dict, n_samples=80, traj_center=None, width=512, height=512, fovy_deg=used_fov_deg, fovx_deg=None)
        novel_poses.extend(novel_poses_1)
        novel_cams.extend(novel_cams_1)

    else:
        raise ValueError(f'Invalid see3d_stage: {args.see3d_stage}')
      
    # render gs
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs')
    os.makedirs(gs_output_dir, exist_ok=True)

    gs_depths = []
    none_visible_rate_list = []
    alpha_list = []
    input_view_depths_cuda = [torch.from_numpy(depth).float().cuda() for depth in input_view_depths]
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

        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))

        if only_warp_input_views:
            # get visible mask in input views
            gs_points = depths_to_points_parallel(depth.unsqueeze(0), [novel_cam]).squeeze(0)
            visible_mask = get_visible_mask_for_input_views(input_viewpoints, input_view_depths_cuda, gs_points)
            visible_mask = visible_mask.reshape(rgb.shape[1], rgb.shape[2]).detach().cpu().numpy()
            save_img_u8(visible_mask, os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

            # filter rgb use visible mask
            rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * visible_mask[:,:,None]
            save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

            none_visible_rate = 1 - visible_mask.sum() / (visible_mask.shape[0] * visible_mask.shape[1])
            none_visible_rate_list.append(none_visible_rate)

        print(f'Novel view {idx} save done!')

    if not only_warp_input_views:
        # render visibility map
        visibility_maps = visibility_grid.render_visibility_map(novel_cams, gs_depths)
        for idx in range(len(visibility_maps)):
            rgb_path = os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png')
            rgb_image = Image.open(rgb_path)
            rgb_image = np.array(rgb_image)

            vis_map = visibility_maps[idx].detach().cpu().numpy() > 0.5
            alpha_map = alpha_list[idx]
            vis_map = vis_map & alpha_map                            # filter vis_map by alpha_map
            vis_map = vis_map.astype(np.float32)

            none_visible_rate = 1 - vis_map.sum() / (vis_map.shape[0] * vis_map.shape[1])
            none_visible_rate_list.append(none_visible_rate)

            rgb_image = rgb_image * vis_map[:,:,None]
            rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
            rgb_image.save(os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

            # save vis_map
            vis_map = vis_map.astype(np.uint8) * 255
            vis_map = Image.fromarray(vis_map)
            vis_map.save(os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

        print(f'Render visibility map done!')

    max_none_visible_thresh = 0.6
    all_data_dir = os.path.join(novel_views_save_root_path, 'all-data')
    os.makedirs(all_data_dir, exist_ok=True)

    if select_view_method == 'depth_quality':
        need_inpaint_views = select_views_by_plane_coverage(
            novel_cams, 
            plane_all_points_dict, 
            gaussians, 
            select_num=32, 
            gs_depths=gs_depths,
            covisible_rate_high_bound=0.9,
            none_visible_rate_list=none_visible_rate_list
        )
                
    else:
        raise ValueError(f'Invalid select_view_method: {select_view_method}')

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


    # =========================================================================
    # 针对 Replica/ScanNet++ 优化的可见性矩阵预计算
    # =========================================================================
    
    print(">>> Pre-calculating Masked Visibility Matrix for Active Vision...")

    if len(plane_all_points_dict) > 0:
        num_pnts = global_target_pnts.shape[0]
        
        print(f"Total target voxels to fill: {num_pnts}")

        # 3. 计算可见性矩阵 (基于你筛选出的 need_inpaint_views)
        num_candidates = len(need_inpaint_views)
        if num_pnts > 0 and num_candidates > 0:
            # 使用 uint8 节省存储空间
            vis_matrix = torch.zeros((num_candidates, num_pnts), dtype=torch.uint8, device='cuda')
            target_pnts_homo = torch.cat([global_target_pnts, torch.ones((num_pnts, 1), device='cuda')], dim=-1)

            for i, ori_idx in enumerate(need_inpaint_views):
                cam = novel_cams[ori_idx]
                depth_map = gs_depths[ori_idx] # [H, W] Tensor

                # 投影到相机坐标系
                p_cam = (target_pnts_homo @ cam.world_view_transform) 
                z_vals = p_cam[:, 2]

                # 投影到像素坐标 (NDC)
                p_proj = (target_pnts_homo @ cam.full_proj_transform)
                p_ndc = p_proj[:, :3] / p_proj[:, 3:4]

                # 筛选在 FOV 内且深度为正的点
                in_fov = (p_ndc[:, 0].abs() < 1.0) & (p_ndc[:, 1].abs() < 1.0) & (z_vals > 0.05)
                valid_indices = torch.where(in_fov)[0]

                if len(valid_indices) > 0:
                    x_pix = ((p_ndc[valid_indices, 0] + 1.0) * cam.image_width / 2.0).long()
                    y_pix = ((p_ndc[valid_indices, 1] + 1.0) * cam.image_height / 2.0).long()
                    
                    x_pix = torch.clamp(x_pix, 0, cam.image_width - 1)
                    y_pix = torch.clamp(y_pix, 0, cam.image_height - 1)

                    # 遮挡剔除：对比点深度与 GS 渲染深度
                    # Replica/ScanNet++ 场景比较紧凑，0.03m 的 epsilon 较合适
                    rendered_depth = depth_map[y_pix, x_pix]
                    visible_in_view = (z_vals[valid_indices] <= rendered_depth + 0.03)
                    
                    # 填充矩阵
                    vis_matrix[i, valid_indices[visible_in_view]] = 1

            # 4. 导出结果，供第二个脚本使用
            matrix_save_path = os.path.join(novel_views_save_root_path, 'candidate_vis_matrix.npy')
            np.save(matrix_save_path, vis_matrix.cpu().numpy())
            
            
            print(f"✅ Matrix {vis_matrix.shape} and PLY saved to {novel_views_save_root_path}")
        else:
            print("⚠️ No target points within room bounds. Matrix not saved.")
    else:
        print("⚠️ Plane dictionary is empty. Skipping spatial filtering.")

    # save need inpaint views cameras
    save_cameras['n_views'] = len(need_inpaint_views_cams)
    np.savez(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_see3d_cameras.npz'), **save_cameras)

    # save all cameras 
    all_cameras = {}
    all_cameras['train_views'] = len(train_viewpoints)
    all_cameras['n_views'] = len(novel_cams)

    for idx, (cam, none_vis_rate) in enumerate(zip(novel_cams, none_visible_rate_list)):
        all_cameras[f'R_{idx:06d}'] = cam.R
        all_cameras[f'T_{idx:06d}'] = cam.T
        all_cameras[f'FoVx_{idx:06d}'] = cam.FoVx
        all_cameras[f'FoVy_{idx:06d}'] = cam.FoVy
        all_cameras[f'image_width_{idx:06d}'] = cam.image_width
        all_cameras[f'image_height_{idx:06d}'] = cam.image_height
        all_cameras[f'none_visible_rate_{idx:06d}'] = none_vis_rate

    all_cam_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_all_cameras.npz')
    np.savez(all_cam_path, **all_cameras)

    for idx in range(len(novel_cams)):
        src_rgb_path = os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png')
        dst_rgb_path = os.path.join(all_data_dir, f'all_rgb_{idx:06d}.png')
        if os.path.exists(src_rgb_path):
            shutil.copy(src_rgb_path, dst_rgb_path)
        
        src_mask_path = os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png')
        dst_mask_path = os.path.join(all_data_dir, f'all_mask_{idx:06d}.png')
        if os.path.exists(src_mask_path):
            shutil.copy(src_mask_path, dst_mask_path)

    print(f'Saved all generated cameras (in original order) to {all_cam_path}')
    print(f'Saved all RGB and mask files to {all_data_dir}')


    print("#######################################################################")
    print(f'See3D stage {args.see3d_stage} save done!')
    print("#######################################################################")