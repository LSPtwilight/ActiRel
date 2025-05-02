import torch
from scene import Scene, GaussianModel
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np

from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image

from utils.general_utils import safe_state

from guidance.cam_utils import generate_see3d_camera, vis_camera_pose, check_valid_camera_center, generate_see3d_camera_by_lookat
from guidance.See3D_modules.pcd_render_util import init_pcd_render_multiview, save_rendered_images, filter_pcd_by_edge, downsample_pcd, vis_depth

from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--train_view_num", default=5, type=int)
    parser.add_argument("--use_downsample", action='store_true')
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
    with open(view_json_path, 'r') as f:
        view_data = json.load(f)
    train_id_list = view_data['train']
    train_viewpoints = [viewpoints[i] for i in train_id_list]

    novel_views_save_root_path = os.path.join(args.model_path, 'see3d_render')
    os.makedirs(novel_views_save_root_path, exist_ok=True)

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

    # load charts data
    charts_data_path = os.path.join(args.source_path, 'charts_data.npz')
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    charts_priors = build_priors_from_charts_data(charts_data, train_viewpoints)
    charts_depths = charts_priors['depths']
    charts_points = depths_to_points_parallel(charts_depths, train_viewpoints)

    gs_train_view_depths = np.array(train_view_depths)
    gs_train_view_depths = torch.from_numpy(gs_train_view_depths).cuda()
    gs_train_view_depths = gs_train_view_depths.unsqueeze(1)
    gs_train_view_points = depths_to_points_parallel(gs_train_view_depths, train_viewpoints)

    # # vis confs map
    # charts_confs = charts_priors['confs']
    # charts_confs = charts_confs.permute(0, 2, 3, 1).detach().cpu().numpy()
    # for i in range(charts_confs.shape[0]):
    #     vis_depth(charts_confs[i], cmap='viridis', save_path=os.path.join(train_save_root_path, f'charts_confs_{i:06d}.png'))

    #     # save confs mask
    #     confs_thresh = np.percentile(charts_confs[i], 5)           # 5% of the confs
    #     confs_mask = (charts_confs[i] < confs_thresh).astype(np.uint8).squeeze(-1)
    #     save_img_u8(confs_mask, os.path.join(train_save_root_path, f'charts_low_confs_mask_{i:06d}.png'))

    # get input image
    input_img = []
    for train_cam in train_viewpoints:
        input_img.append(train_cam.original_image)
    input_img = torch.stack(input_img, dim=0)                       # [n_views, 3, H, W]
    charts_pcd_colors = input_img.permute(0, 2, 3, 1)                      # [n_views, H, W, 3]
    charts_pcd_colors = charts_pcd_colors.reshape(charts_points.shape[0], -1, 3)  # [n_views, H * W, 3], range: [0, 1]

    filtered_charts_points, filtered_pcd_colors = filter_pcd_by_edge(charts_points, charts_pcd_colors, train_view_depths)
    scene_pcds = filtered_charts_points.reshape(-1, 3)
    pcd_colors = filtered_pcd_colors.reshape(-1, 3)

    # # save scene_pcds
    # save_root_path = f'{args.model_path}/charts_pcd_render/charts-pcds'
    # os.makedirs(save_root_path, exist_ok=True)
    # save_tensor_as_pcd(scene_pcds, os.path.join(save_root_path, 'charts-pcds.ply'), (pcd_colors * 255.0).to(torch.uint8))

    if args.use_downsample:
        origin_pcd_num = scene_pcds.shape[0]
        scene_pcds, pcd_colors = downsample_pcd(scene_pcds, pcd_colors, method='voxel')
        now_pcd_num = scene_pcds.shape[0]
        print(f'Downsample from {origin_pcd_num} to {now_pcd_num}')
    else:
        origin_pcd_num = scene_pcds.shape[0]
        print(f'No downsample, pcd_num: {origin_pcd_num}')

    # # NOTE: hard code set ref view id
    # ref_view_id_list = [63, 83]
    # ref_c2ws = []
    # for ref_view_id in ref_view_id_list:
    #     viewpoint = viewpoints[ref_view_id]
    #     w2c = (viewpoint.world_view_transform).transpose(0, 1).cpu().numpy()
    #     c2w = np.linalg.inv(w2c)
    #     ref_c2ws.append(c2w)

    # _, temp_novel_cams = generate_see3d_camera(ref_c2ws, interpolate_num=10, camera_type='ellipse', scale=5, width=512, height=512, fovy_deg=60)

    # # delete_list = [33, 34, 35, 36, 37, 38, 39]
    # # # delete_list = []

    # # novel_cams = []
    # # for idx, temp_novel_cam in enumerate(temp_novel_cams):
    # #     if idx in delete_list:
    # #         continue
    # #     novel_cams.append(temp_novel_cam)

    # # delete novel_cams that are not visible from any of the training cameras
    # novel_cam_centers = torch.stack([cam.camera_center for cam in temp_novel_cams], dim=0)
    # cam_valid_mask = check_valid_camera_center(train_viewpoints, train_view_depths, novel_cam_centers)
    # novel_cams = [temp_novel_cams[i] for i in range(len(temp_novel_cams)) if cam_valid_mask[i]]


    # generate novel cameras
    novel_poses, novel_cams = generate_see3d_camera_by_lookat(train_viewpoints, gs_train_view_depths.squeeze(1), gs_train_view_points)

    # # vis train camera
    # train_c2ws = []
    # for train_cam in train_viewpoints:
    #     w2c = (train_cam.world_view_transform).transpose(0, 1).cpu().numpy()
    #     c2w = np.linalg.inv(w2c)
    #     train_c2ws.append(c2w)
    # train_c2ws = np.array(train_c2ws)

    # temp_mesh_path = '/home/nijunfeng/mycode/project/gs-recon/priorgs/data/replica/scan6/gt_mesh/scene_mesh.ply'
    # vis_camera_pose(novel_poses, mesh_path=temp_mesh_path)
    # # vis_camera_pose(train_c2ws, mesh_path=temp_mesh_path)
    # exit()


    # First render gs
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs-t1')
    os.makedirs(gs_output_dir, exist_ok=True)

    save_cameras = {}
    use_view_num = 0

    gs_depths = []
    for idx, novel_cam in enumerate(novel_cams):

        render_pkg = render(novel_cam, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']

        gs_depths.append(depth[0].detach())

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(gs_output_dir, f'ori_warp_frame{use_view_num:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(gs_output_dir, f'depth_frame{use_view_num:06d}.tiff'))
        # save .npy
        np.save(os.path.join(gs_output_dir, f'alpha_{use_view_num:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_thresh = 0.99
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'mask_frame{use_view_num:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(gs_output_dir, f'warp_frame{use_view_num:06d}.png'))

        # save camera
        save_cameras[f'R_{use_view_num:06d}'] = novel_cam.R
        save_cameras[f'T_{use_view_num:06d}'] = novel_cam.T
        save_cameras[f'FoVx_{use_view_num:06d}'] = novel_cam.FoVx
        save_cameras[f'FoVy_{use_view_num:06d}'] = novel_cam.FoVy
        save_cameras[f'image_width_{use_view_num:06d}'] = novel_cam.image_width
        save_cameras[f'image_height_{use_view_num:06d}'] = novel_cam.image_height
        use_view_num += 1

        print(f'Novel view {use_view_num} save done!')

    # save cameras as .npz file
    save_cameras['n_views'] = use_view_num
    np.savez(os.path.join(gs_output_dir, 'see3d_cameras.npz'), **save_cameras)
    print(f'See3D cameras save to {os.path.join(gs_output_dir, "see3d_cameras.npz")}')

    gs_depths = torch.stack(gs_depths, dim=0)
    gs_depths = gs_depths.to(scene_pcds.device)
    depth_threshold = 0.1

    # Second render pcd
    if args.use_downsample:
        pcd_output_dir = os.path.join(novel_views_save_root_path, 'raw-downsampled-pcd')
    else:
        pcd_output_dir = os.path.join(novel_views_save_root_path, 'raw-pcd')
    os.makedirs(pcd_output_dir, exist_ok=True)

    images, masks = init_pcd_render_multiview(scene_pcds, pcd_colors, novel_cams, gs_depths=gs_depths, depth_threshold=depth_threshold, image_size=(512, 512), radius=0.01, points_per_pixel=10, device="cuda")
    save_rendered_images(images, pcd_output_dir, prefix="warp_frame")
    save_rendered_images(masks, pcd_output_dir, prefix="mask_frame")

    print(f'All novel views save done!')

