#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras

import gc
import copy
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_with_conf
from utils.sh_utils import SH2RGB
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import get_gaussian_normal, get_obj_gaussian_by_mask
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import depth_to_normal
from matcha.dm_scene.charts import (
    load_charts_data, 
    build_priors_from_charts_data,
    schedule_regularization_factor_1,
    schedule_regularization_factor_2,
    get_gaussian_parameters_from_charts_data,
    get_gaussian_parameters_from_pda_data,
    depths_to_points_parallel,
    depth2normal_parallel,
    normal2curv_parallel,
    voxel_downsample_gaussians,
)
from matcha.dm_regularization.depth import compute_depth_order_loss
from matcha.dm_utils.rendering import normal2curv
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import pickle
from gaussian_renderer.trace import get_weights_by_single_view_with_mask
import trimesh
import cv2
import open3d as o3d
from utils.graphics_utils import BasicPointCloud

# Old confidence to increasing weight function
# def confidence_to_weight(confidence:torch.Tensor):
#     conf_weights = confidence - 1.
#     return 1. - torch.exp(-conf_weights**2 / 2)

# New confidence to increasing weight function
def confidence_to_weight(confidence:torch.Tensor):
    conf_weights = confidence - 1.
    return torch.sigmoid((conf_weights - 2.) * 2.)


def training(
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
    use_refined_charts, use_mip_filter, dense_data_path, use_chart_view_every_n_iter,
    normal_consistency_from, distortion_from,
    depthanythingv2_checkpoint_dir, depthanything_encoder, 
    dense_regul, refine_depth_path, use_downsample_gaussians
):
    
    save_log_images = False
    save_log_images_every_n_iter = 200

    gaussian_points_count = []
    gaussian_points_iterations = []
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    if dense_data_path is not None:
        use_dense_supervision = True
    
    # Sparse data
    scene = Scene(dataset, gaussians, shuffle=False)

    if dense_data_path is not None:
        use_dense_supervision = True
        
        print(f"[INFO] Loading dense supervision data from: {dense_data_path}")
        dense_dataset = copy.deepcopy(dataset)
        dense_dataset.source_path = dense_data_path
        dense_dataset.model_path = os.path.join(dataset.model_path, 'dense_data')
        os.makedirs(dense_dataset.model_path, exist_ok=True)
        dense_gaussians = GaussianModel(dataset.sh_degree)
        dense_scene = Scene(dense_dataset, dense_gaussians, shuffle=False)

    else:
        use_dense_supervision = False
        use_chart_view_every_n_iter = 1

    # NOTE: hard code for See3D root path
    see3d_root_path = os.path.join(dataset.source_path, 'see3d_render')
    see3d_cam_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    if os.path.exists(see3d_cam_path):
        see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cam_path, inpaint_root_dir)
        target_dir = os.path.join(see3d_root_path, 'stage0', 'select-gs-inpainted')
        if not os.path.exists(target_dir):
            print(f"dense view path do not exit: {target_dir}")
            dense_view_num = 0
        else:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            images = [
                f for f in os.listdir(target_dir)
                if os.path.isfile(os.path.join(target_dir, f)) and
                os.path.splitext(f)[1].lower() in image_extensions
            ]
            dense_view_num = len(images)
    else:
        see3d_gs_cameras_list = []
        dense_view_num = 0
    
    print(f"[INFO] Charts will be used for regularization every {use_chart_view_every_n_iter} iteration(s).")
    
    # ===================================================================================
    # Create gaussians from charts data
    print("[INFO] Loading charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    if use_refined_charts:
        charts_data_path = f'{dataset.source_path}/refined_charts_data.npz'
    print("Using charts data from: ", charts_data_path)
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())

    # ===================================================================================
    # Load refine pda depth and points  # pda: prior depth anything
    create_gaussians_from_pda_depth = True          
    print(f'WARNING: PDA depth root path: {refine_depth_path}')
    pda_depths = []
    pda_points_list = []
    pda_confident_maps_list = []

    input_view_num = len(scene.getTrainCameras())
    see3d_view_num = len(see3d_gs_cameras_list)
    training_view_num = input_view_num + see3d_view_num
    for idx in range(training_view_num):
        pda_depth_path = os.path.join(refine_depth_path, f'refine_depth_frame{idx:06d}.tiff')
        pda_depth = Image.open(pda_depth_path)
        pda_depth = np.array(pda_depth)
        pda_depth = torch.from_numpy(pda_depth).cuda()
        pda_depths.append(pda_depth)

        pda_point_path = os.path.join(refine_depth_path, f'refine_points_frame{idx:06d}.ply')
        pda_point = trimesh.load(pda_point_path)
        pda_point = np.array(pda_point.vertices)
        pda_points_list.append(pda_point)

        pda_confident_map_path = os.path.join(refine_depth_path, f'confident_map_frame{idx:06d}.png')
        pda_confident_map = Image.open(pda_confident_map_path)
        pda_confident_map = np.array(pda_confident_map) / 255                   # 0 or 1
        pda_confident_map = torch.from_numpy(pda_confident_map).cuda()
        pda_confident_maps_list.append(pda_confident_map)

    # ===================================================================================
    # Initialize gaussians
    max_gaussians_num = 9_000_000
    print(f"Max gaussians num: {max_gaussians_num}, use downsample gaussians: {use_downsample_gaussians}")

    # --- Process Input Views ---
    input_view_depths = pda_depths[:input_view_num]
    input_view_depths_stack = torch.stack(input_view_depths, dim=0).cuda()
    _images_input = [cam.original_image.cuda().permute(1, 2, 0) for cam in scene.getTrainCameras()]
    pda_points_input = depths_to_points_parallel(input_view_depths_stack, scene.getTrainCameras())
    N, H, W = input_view_depths_stack.shape
    pda_points_input = pda_points_input.reshape(N, H, W, 3)
    input_view_gaussian_params = get_gaussian_parameters_from_pda_data(
        pda_points=pda_points_input,
        images=_images_input,
        conf_th=-1.,
        ratio_th=5.,
        normal_scale=1e-10,
        normalized_scales=0.5,
    )

    # --- Process See3D Views (Separate Dense and Others) ---
    if see3d_view_num > 0:
        see3d_view_depths = pda_depths[input_view_num:]
        _images_see3d = [cam.original_image.cuda().permute(1, 2, 0) for cam in see3d_gs_cameras_list]

        gaussian_params_list = [input_view_gaussian_params]

        # --- Separate Dense Views ---
        if dense_view_num > 0:
            dense_view_depths = see3d_view_depths[:dense_view_num]
            dense_gs_cameras_list = see3d_gs_cameras_list[:dense_view_num]
            dense_images = _images_see3d[:dense_view_num]
            dense_view_depths_stack = torch.stack(dense_view_depths, dim=0).cuda()
            dense_points = depths_to_points_parallel(dense_view_depths_stack, dense_gs_cameras_list)
            N_d, H_d, W_d = dense_view_depths_stack.shape
            dense_points = dense_points.reshape(N_d, H_d, W_d, 3)
            dense_gaussian_params = get_gaussian_parameters_from_pda_data(
                pda_points=dense_points,
                images=dense_images,
                conf_th=-1.,
                ratio_th=5.,
                normal_scale=1e-10,
                normalized_scales=0.5,
            )
            gaussian_params_list.append(dense_gaussian_params)

        # --- Separate Other See3D Views ---
        other_see3d_view_num = see3d_view_num - dense_view_num
        if other_see3d_view_num > 0:
            other_see3d_view_depths = see3d_view_depths[dense_view_num:]
            other_see3d_gs_cameras_list = see3d_gs_cameras_list[dense_view_num:]
            other_see3d_images = _images_see3d[dense_view_num:]

            if other_see3d_view_num > 30:
                print(f'WARNING: Other See3D view num is large: {other_see3d_view_num}, using first 30 for initialization...')
                used_see3d_init_gs_view_list = list(range(10)) + list(range(15, 25)) + list(range(30, see3d_view_num))
                init_gs_see3d_view_depths = [other_see3d_view_depths[i] for i in used_see3d_init_gs_view_list]
                init_gs_see3d_view_depths_stack = torch.stack(init_gs_see3d_view_depths, dim=0).cuda()
                init_gs_see3d_gs_cameras_list = [other_see3d_gs_cameras_list[i] for i in used_see3d_init_gs_view_list]
                init_gs_see3d_images = [other_see3d_images[i] for i in used_see3d_init_gs_view_list]
                init_gs_see3d_points = depths_to_points_parallel(init_gs_see3d_view_depths_stack, init_gs_see3d_gs_cameras_list)
                N_o, H_o, W_o = init_gs_see3d_view_depths_stack.shape
                init_gs_see3d_points = init_gs_see3d_points.reshape(N_o, H_o, W_o, 3)
                other_see3d_gaussian_params = get_gaussian_parameters_from_pda_data(
                    pda_points=init_gs_see3d_points,
                    images=init_gs_see3d_images,
                    conf_th=-1.,
                    ratio_th=5.,
                    normal_scale=1e-10,
                    normalized_scales=0.5,
                )
            else:
                other_see3d_view_depths_stack = torch.stack(other_see3d_view_depths, dim=0).cuda()
                other_see3d_points = depths_to_points_parallel(other_see3d_view_depths_stack, other_see3d_gs_cameras_list)
                N_o, H_o, W_o = other_see3d_view_depths_stack.shape
                other_see3d_points = other_see3d_points.reshape(N_o, H_o, W_o, 3)
                other_see3d_gaussian_params = get_gaussian_parameters_from_pda_data(
                    pda_points=other_see3d_points,
                    images=other_see3d_images,
                    conf_th=-1.,
                    ratio_th=5.,
                    normal_scale=1e-10,
                    normalized_scales=0.5,
                )
            gaussian_params_list.append(other_see3d_gaussian_params)

        # Concatenate all parameter tensors
        gaussian_params = {}
        for key in gaussian_params_list[0].keys():
            gaussian_params[key] = torch.cat([gp[key] for gp in gaussian_params_list if gp is not None], dim=0)

    else:
        gaussian_params = input_view_gaussian_params

    # ===================================================================================
    # downsample gaussians 
    if len(gaussian_params['means']) > max_gaussians_num and use_downsample_gaussians:
        sample_idx, downsample_factor = voxel_downsample_gaussians(gaussian_params, voxel_size=0.005)
        print(f"Downsampled {len(gaussian_params['means'])} gaussians to {len(sample_idx)} gaussians...")
    else:
        sample_idx = torch.arange(len(gaussian_params['means']))
        downsample_factor = 1.0
        print(f"Not downsampling gaussians, using all {len(gaussian_params['means'])} gaussians...")
    print(f"Final number of gaussians: {len(sample_idx)}")

    _means = gaussian_params['means'][sample_idx]
    _scales = gaussian_params['scales'][..., :2][sample_idx] * downsample_factor
    _quaternions = gaussian_params['quaternions'][sample_idx]
    _colors = gaussian_params['colors'][sample_idx]

    if use_dense_supervision:
        with torch.no_grad():
            _means = torch.cat([_means, dense_gaussians.get_xyz.detach()], dim=0)
            _scales = torch.cat([_scales, dense_gaussians.get_scaling.detach()], dim=0)
            _quaternions = torch.cat([_quaternions, dense_gaussians.get_rotation.detach()], dim=0)
            _colors = torch.cat([_colors, SH2RGB(dense_gaussians._features_dc.detach()[:, 0])], dim=0)

    gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale)
    print("[INFO] Gaussians created from pnts data.")

    if use_dense_supervision:
        del dense_gaussians, dense_scene
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================================
    
    # ===================================================================================
    # Training setup
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0
    
    input_cams = scene.getTrainCameras()
    
    # ===================================================================================
    # Build priors from charts data
    print("[INFO] Building priors from charts data...")
    charts_priors = build_priors_from_charts_data(charts_data, input_cams)
    charts_scale_factor = charts_priors['scale_factor']
    charts_prior_depths = charts_priors['prior_depths']
    charts_depths = charts_priors['depths']
    charts_confs = charts_priors['confs']
    charts_normals = charts_priors['normals']
    charts_curvs = charts_priors['curvs']
    print("[INFO] Charts priors built.")

    refine_charts_depth = pda_depths[:input_view_num]
    refine_charts_depth_stack = torch.stack(refine_charts_depth, dim=0).cuda()

    # ===================================================================================
    # Prepare See3D priors (normals, curvs) if see3d_view_num > 0
    see3d_pseudo_confs_list = []
    see3d_prior_normals_list = []
    see3d_prior_curvs_list = []
    see3d_refine_depths_list = pda_depths[input_view_num:]

    if see3d_view_num > 0:
        see3d_refine_depths = pda_depths[input_view_num:]

        if dense_view_num > 0:
            dense_refine_depths = see3d_refine_depths[:dense_view_num]
            dense_gs_cameras_list = see3d_gs_cameras_list[:dense_view_num]
            dense_refine_depths_stack = torch.stack(dense_refine_depths, dim=0).cuda()
            dense_world_view_transforms = torch.stack([cam.world_view_transform for cam in dense_gs_cameras_list])
            dense_full_proj_transforms = torch.stack([cam.full_proj_transform for cam in dense_gs_cameras_list])
            dense_prior_normals = depth2normal_parallel(
                dense_refine_depths_stack,
                world_view_transforms=dense_world_view_transforms,
                full_proj_transforms=dense_full_proj_transforms
            ).permute(0, 3, 1, 2)
            dense_pseudo_confs = pda_confident_maps_list[input_view_num : input_view_num + dense_view_num]
            dense_pseudo_confs_stack = torch.stack(dense_pseudo_confs, dim=0).cuda()
            dense_prior_curvs = normal2curv_parallel(dense_prior_normals, torch.ones_like(dense_prior_normals[:, 0:1]))
            see3d_pseudo_confs_list.extend([dense_pseudo_confs_stack[i] for i in range(dense_view_num)])
            see3d_prior_normals_list.extend([dense_prior_normals[i] for i in range(dense_view_num)])
            see3d_prior_curvs_list.extend([dense_prior_curvs[i] for i in range(dense_view_num)])

        other_see3d_view_num = see3d_view_num - dense_view_num
        if other_see3d_view_num > 0:
            other_see3d_refine_depths = see3d_refine_depths[dense_view_num:]
            other_see3d_gs_cameras_list = see3d_gs_cameras_list[dense_view_num:]
            other_see3d_refine_depths_stack = torch.stack(other_see3d_refine_depths, dim=0).cuda()
            other_see3d_world_view_transforms = torch.stack([cam.world_view_transform for cam in other_see3d_gs_cameras_list])
            other_see3d_full_proj_transforms = torch.stack([cam.full_proj_transform for cam in other_see3d_gs_cameras_list])
            other_see3d_prior_normals = depth2normal_parallel(
                other_see3d_refine_depths_stack,
                world_view_transforms=other_see3d_world_view_transforms,
                full_proj_transforms=other_see3d_full_proj_transforms
            ).permute(0, 3, 1, 2)
            other_see3d_pseudo_confs = pda_confident_maps_list[input_view_num + dense_view_num:]
            other_see3d_pseudo_confs_stack = torch.stack(other_see3d_pseudo_confs, dim=0).cuda()
            other_see3d_prior_curvs = normal2curv_parallel(other_see3d_prior_normals, torch.ones_like(other_see3d_prior_normals[:, 0:1]))
            see3d_pseudo_confs_list.extend([other_see3d_pseudo_confs_stack[i] for i in range(other_see3d_view_num)])
            see3d_prior_normals_list.extend([other_see3d_prior_normals[i] for i in range(other_see3d_view_num)])
            see3d_prior_curvs_list.extend([other_see3d_prior_curvs[i] for i in range(other_see3d_view_num)])

        print('See3D priors (normals, curvs) computed for all views!')

    if create_gaussians_from_pda_depth:
        charts_depths = refine_charts_depth_stack.unsqueeze(1)
        print(f'WARNING: Charts depths are now PDA depths (stacked input views).')

    if see3d_view_num > 0:
        total_views_list = input_cams + see3d_gs_cameras_list
        charts_confs_list = [charts_confs[idx] for idx in range(len(charts_confs))]
        charts_depths_list = [charts_depths[i] for i in range(charts_depths.shape[0])]
        see3d_refine_depths_unsqueezed_list = [d.unsqueeze(0) for d in see3d_refine_depths_list]
        total_confs_list = charts_confs_list + see3d_pseudo_confs_list
        total_depths_list = charts_depths_list + see3d_refine_depths_unsqueezed_list
        charts_normals_list = [charts_normals[idx] for idx in range(len(charts_normals))]
        total_normals_list = charts_normals_list + see3d_prior_normals_list
        charts_curvs_list = [charts_curvs[idx] for idx in range(len(charts_curvs))]
        total_curvs_list = charts_curvs_list + see3d_prior_curvs_list
    else:
        total_views_list = input_cams
        total_confs_list = [c for c in charts_confs]
        charts_depths_list = [charts_depths[i] for i in range(charts_depths.shape[0])]
        total_depths_list = charts_depths_list
        total_normals_list = [n for n in charts_normals]
        total_curvs_list = [c for c in charts_curvs]


    # ===================================================================================
    
    # ===================================================================================
    # Set mip filter
    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.set_mip_filter(use_mip_filter)

        # Compute the mip filter for input views
        gaussians.compute_mip_filter(cameras=total_views_list)
        #gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list)


    # ===================================================================================
    # do not use , we use refined pda depth
    # if use_dense_supervision:
    #     print("[INFO] Building depth priors from dense data...")
    #     # TODO: Build priors from dense data
    #     from matcha.pointmap.depthanythingv2 import load_model as load_depthanythingv2
    #     from matcha.pointmap.depthanythingv2 import apply_depthanything

    #     dav2 = load_depthanythingv2(checkpoint_dir=depthanythingv2_checkpoint_dir, encoder=depthanything_encoder, device='cuda')
    #     dav2.eval()
    #     dense_depth_priors = []
    #     with torch.no_grad():
    #         for i_image in range(len(dense_viewpoint_cams)):
    #             gt_image = dense_viewpoint_cams[i_image].original_image.permute(1, 2, 0)
    #             supervision_disparity = apply_depthanything(dav2, image=gt_image)
    #             supervision_disparity = (supervision_disparity - supervision_disparity.min()) / (supervision_disparity.max() - supervision_disparity.min())
    #             supervision_depth = 1. / (0.1 + 0.9 * supervision_disparity)
    #             dense_depth_priors.append(supervision_depth.squeeze().unsqueeze(0).to('cpu'))
    #     del dav2
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print("[INFO] Depth priors for dense data built.")
    # # ===================================================================================
    
    print(f"\n[INFO] Normal consistency from iteration {normal_consistency_from} with lambda_normal {opt.lambda_normal}")
    print(f"[INFO] Distortion from iteration {distortion_from} with lambda_dist {opt.lambda_dist}")
    if use_dense_supervision:
        print(f"[INFO] Dense regularization strength: {dense_regul}")
    # TODO: Should the sparse depth order regularization be used during dense supervision? It's not clear.
    # use_depth_order_regularization = not use_dense_supervision
    use_depth_order_regularization = True
    if use_depth_order_regularization:
        print(f"[INFO] Using depth order regularization for charts.")

    print("==================================== [view num info] ====================================")
    print(f"[INFO] Total number of views: {len(total_views_list)}, "
          f"input views: {len(input_cams)}, "
          f"see3d views: {len(see3d_gs_cameras_list)-dense_view_num}, "
          f"dense views: {dense_view_num}")
    print("=========================================================================================")
    print("[INFO] Starting training...")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random view for this iteration
        if not viewpoint_idx_stack or len(viewpoint_idx_stack) == 0:
            viewpoint_idx_stack = list(range(len(total_views_list)))
        viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))
        viewpoint_cam = total_views_list[viewpoint_idx]
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # Determine the type of view and calculate loss accordingly
        is_see3d_view = (input_view_num + dense_view_num <= viewpoint_idx) 
        is_dense_view = (input_view_num <= viewpoint_idx) and ( viewpoint_idx < input_view_num + dense_view_num)

        if is_dense_view:
            rgb_current_conf = total_confs_list[viewpoint_idx]
            Ll1 = l1_loss_with_conf(image, gt_image, rgb_current_conf) * 0.01 # Example scaling
            loss = Ll1

        elif is_see3d_view:
            rgb_current_conf = total_confs_list[viewpoint_idx]
            Ll1 = l1_loss_with_conf(image, gt_image, rgb_current_conf) * 0.01 # Example scaling
            loss = Ll1

        else: # is_input_view
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > normal_consistency_from else 0.0
        lambda_dist = opt.lambda_dist if iteration > distortion_from else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        # loss
        total_loss = loss + dist_loss + normal_loss
        
        # ===================================================================================

        # Get the correct confidence, depth, normal, and curvature for the current view
        current_conf = total_confs_list[viewpoint_idx]

        #### adjust dense view confidence 
        current_depth = total_depths_list[viewpoint_idx]
        current_normal = total_normals_list[viewpoint_idx]
        current_curv = total_curvs_list[viewpoint_idx]

        surf_depth = render_pkg['surf_depth']
        total_regularization_loss = 0.
        lambda_anisotropy = 0.1  # 0.01 works well for depth_ratio = 0.
        anisotropy_max_ratio = 5.
        
        # if iteration % use_chart_view_every_n_iter == 0:
        if True:
            rend_curvature = normal2curv(render_pkg['rend_normal'], torch.ones_like(render_pkg['rend_normal'][0:1]))
            
            # ---Charts regularization---
            initial_regularization_factor = 0.5
            regularization_factor = schedule_regularization_factor_2(iteration, initial_regularization_factor)
            lambda_prior_depth = regularization_factor * 0.75
            lambda_prior_depth_derivative = regularization_factor * 0.5  # TODO: Changed. Was 0. before.
            lambda_prior_normal = regularization_factor * 0.5
            lambda_prior_curvature = regularization_factor * 0.25  # 0.5?
            
            confidence_weighting = 0.5
            
            # TODO: Weight charts_normals with rendered alpha, as rendered normals (and surface normals) are not normalized
            
            if False:
                depths_to_regularize = [surf_depth]
            else:
                # depth_to_regularize = render_pkg['rend_depth']
                depths_to_regularize = [surf_depth, render_pkg['rend_depth']]
            
            # Depth regularization
            if False:  # New (should be used with depth_ratio = 1.)
                depth_prior_loss = torch.zeros_like(total_loss)
                for depth_to_regularize in depths_to_regularize:
                    depth_prior_loss += lambda_prior_depth * (
                        # confidence_weighting * charts_confs[viewpoint_idx] *  # (1, h, w)
                        1.5 * (1. - torch.exp(-(charts_confs[viewpoint_idx] - 1.) ** 2 / 2.)) *  # (1, h, w)
                        # 1.5 * confidence_to_weight(charts_confs[viewpoint_idx]) *  # (1, h, w)
                        torch.log(1. + charts_scale_factor * (charts_depths[viewpoint_idx] - depth_to_regularize).abs())
                        # (charts_scale_factor * (charts_depths[viewpoint_idx] - surf_depth).abs())
                    ).mean()
                    if lambda_prior_depth_derivative > 0:
                        surf_normal_to_regularize = depth_to_normal(viewpoint_cam, depth_to_regularize)
                        surf_normal_to_regularize = surf_normal_to_regularize.permute(2,0,1)
                        depth_prior_loss += (
                            lambda_prior_depth_derivative 
                            * torch.exp(-(charts_confs[viewpoint_idx] - 1.) ** 2 / 2.)
                            # * (1. - confidence_to_weight(charts_confs[viewpoint_idx]))
                            * (1. - (surf_normal_to_regularize * charts_normals[viewpoint_idx]).sum(dim=0))
                        ).mean()
            else:  # Old (should be used with depth_ratio = 0. or 0.5)
                # Depth regularization
                depth_prior_loss = lambda_prior_depth * (
                    confidence_weighting * current_conf *  # (1, h, w)
                    torch.log(1. + charts_scale_factor * (current_depth - surf_depth).abs())
                    # (charts_scale_factor * (charts_depths[viewpoint_idx] - surf_depth).abs())
                ).mean()
                if lambda_prior_depth_derivative > 0:
                    depth_prior_loss += (
                        lambda_prior_depth_derivative 
                        * torch.exp(-(current_conf - 1.) ** 2 / 2.)
                        * (1. - (surf_normal * current_normal).sum(dim=0))
                    ).mean()

            normal_prior_loss = lambda_prior_normal * (1. - (rend_normal * current_normal).sum(dim=0)).mean()
            curv_prior_loss = lambda_prior_curvature * (current_curv - rend_curvature).abs().mean()
            # TODO: Should the curvature be applied to the surf normal?
            
            # Depth order regularization
            if use_depth_order_regularization:
                # TODO: Hyperparameters and scheduling should not be hardcoded
                depth_order_loss_max_pixel_shift_ratio = 0.05
                depth_order_loss_log_space = True
                depth_order_loss_log_scale = 20.
                
                # Scheduling
                lambda_depth_order = 0.
                if iteration > 1_500:
                    lambda_depth_order = 1.
                if iteration > 3_000:
                    lambda_depth_order = 0.1
                if iteration > 4_500:
                    lambda_depth_order = 0.01
                if iteration > 6_000:
                    lambda_depth_order = 0.001
                    
                # Compute depth prior loss
                # order_supervision_depth = charts_prior_depths[viewpoint_idx].to(surf_depth.device)
                order_supervision_depth = total_depths_list[viewpoint_idx].to(surf_depth.device)
                if lambda_depth_order > 0:
                    depth_order_prior_loss = lambda_depth_order * compute_depth_order_loss(
                        depth=surf_depth, 
                        prior_depth=order_supervision_depth, 
                        scene_extent=gaussians.spatial_lr_scale,  # TODO: Check if this is correct
                        max_pixel_shift_ratio=depth_order_loss_max_pixel_shift_ratio,
                        normalize_loss=True,
                        log_space=depth_order_loss_log_space,
                        log_scale=depth_order_loss_log_scale,
                        reduction="mean",
                        debug=False,
                    )
                else:
                    depth_order_prior_loss = torch.zeros_like(loss.detach())
                depth_prior_loss = depth_prior_loss + depth_order_prior_loss
            
            # Total loss
            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
        
        else:
            # TODO: Hyperparameters and scheduling should not be hardcoded
            dense_depth_loss_max_pixel_shift_ratio = 0.05
            dense_depth_loss_log_space = True
            dense_depth_loss_log_scale = 20.
            lambda_anisotropy = 0.                
            dense_depth_regularization_schedule = dense_regul
            
            # ---Scheduling: default---
            # Gives more weight to the dense depth regularization than charts regularization,
            # and progressively decreases the weight of the dense depth regularization.
            if dense_depth_regularization_schedule == "default":
                lambda_dense_depth = 0.  # TODO: Should it be 0. or 1.?
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 1.
                if iteration > 7_000:
                    lambda_dense_depth = 0.1
                if iteration > 15_000:
                    lambda_dense_depth = 0.01
                if iteration > 20_000:
                    lambda_dense_depth = 0.001
                if iteration > 25_000:
                    lambda_dense_depth = 0.0001
                
            # ---Scheduling: strong---
            # Removes the charts regularization and only uses a strong dense depth regularization.
            elif dense_depth_regularization_schedule == "strong":
                # lambda_dense_depth = 0.
                lambda_dense_depth = 1.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 1.
                    
            # ---Scheduling: weak---
            # Applies a weak dense depth regularization, while still using the charts regularization.
            elif dense_depth_regularization_schedule == "weak":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 15
                    lambda_dense_depth = 0.1
            
            # ---Scheduling: none---
            # Disables the dense depth regularization, while still using the charts regularization.
            elif dense_depth_regularization_schedule == "none":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                    
            else:
                raise ValueError(f"Invalid dense depth regularization schedule: {dense_depth_regularization_schedule}")
            
            # Compute depth prior loss
            dense_supervision_depth = dense_depth_priors[viewpoint_idx].to(surf_depth.device)
            if lambda_dense_depth > 0:
                depth_prior_loss = lambda_dense_depth * compute_depth_order_loss(
                    depth=surf_depth, 
                    prior_depth=dense_supervision_depth, 
                    scene_extent=gaussians.spatial_lr_scale,  # TODO: Check if this is correct
                    max_pixel_shift_ratio=dense_depth_loss_max_pixel_shift_ratio,
                    normalize_loss=True,
                    log_space=dense_depth_loss_log_space,
                    log_scale=dense_depth_loss_log_scale,
                    reduction="mean",
                    debug=False,
                )
            else:
                depth_prior_loss = torch.zeros_like(loss.detach())
            
            # TODO: Add dense regularization for normal and curvature
            normal_prior_loss = torch.zeros_like(loss.detach())
            curv_prior_loss = torch.zeros_like(loss.detach())
            
            # Total loss
            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
            
        # Anisotropy regularization
        if lambda_anisotropy > 0.:
            gaussians_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gaussians_scaling.max(dim=1).values / gaussians_scaling.min(dim=1).values, anisotropy_max_ratio) 
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss

        total_loss = total_loss + total_regularization_loss
        
        # ===================================================================================

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log
            if lambda_anisotropy > 0.:
                ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log

            if iteration % 10 == 0:

                current_points = len(gaussians.get_xyz.detach())
                gaussian_points_count.append(current_points)
                gaussian_points_iterations.append(iteration)

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz.detach())}",
                    "p_depth": f"{ema_prior_depth_for_log:.{5}f}",
                    "p_normal": f"{ema_prior_normal_for_log:.{5}f}",
                    "pc": f"{ema_prior_curvature_for_log:.{5}f}",
                }
                if lambda_anisotropy > 0:
                    loss_dict["aniso"] = f"{ema_prior_anisotropy_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
                
            if (iteration % save_log_images_every_n_iter == 0) or (iteration == 1):
                # Save log image with rgb, depth, normal, curvature
                # if iteration % use_chart_view_every_n_iter == 0:
                if True:
                    supervision_depth = total_depths_list[viewpoint_idx]
                    supervision_normal = total_normals_list[viewpoint_idx]
                else:
                    supervision_depth = dense_supervision_depth.squeeze().unsqueeze(0)
                    supervision_normal = torch.zeros_like(charts_normals[0])
                
                if save_log_images:
                    figsize = 30
                    height, width = gt_image.shape[-2:]
                    nrows = 2
                    ncols = 3
                    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
                    plt.subplot(nrows, ncols, 1)
                    plt.title("GT Image")
                    plt.imshow(gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 2)
                    plt.title("Charts Depth")
                    plt.imshow(supervision_depth[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 3)
                    plt.title("Charts Normal")
                    plt.imshow((-supervision_normal + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    plt.subplot(nrows, ncols, 4)
                    plt.title("Rendered Image")
                    plt.imshow(image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 5)
                    plt.title("Rendered Depth")
                    plt.imshow(surf_depth.detach()[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 6)
                    plt.title("Rendered Normal")
                    plt.imshow((-rend_normal.detach() + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    # save image
                    plt.savefig(f"{dataset.model_path}/{iteration}.png")
                    plt.close()
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=total_views_list)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:  # don't update in the end of training
                    torch.cuda.empty_cache()
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=total_views_list)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    if len(gaussian_points_count) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(gaussian_points_iterations, gaussian_points_count)
        plt.xlabel('Iterations')
        plt.ylabel('Number of Gaussian Points')
        plt.title('Gaussian Points Count During Training')
        plt.grid(True)
        plt.savefig(f"{dataset.model_path}/gaussian_points_count.png")
        plt.close()
    print("Training complete.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--refine_depth_path", type=str, required=True)
    parser.add_argument("--use_downsample_gaussians", action="store_true", help="Use downsample gaussians")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=None)  # 6009
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_refined_charts", action="store_true", default=False)
    parser.add_argument("--use_mip_filter", action="store_true", default=False)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument("--use_chart_view_every_n_iter", type=int, default=999_999)
    parser.add_argument("--normal_consistency_from", type=int, default=3500)
    parser.add_argument("--distortion_from", type=int, default=1500)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='../Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.port is None:
        import time
        current_time = time.strftime("%H%M%S", time.localtime())[2:]
        args.port = int(current_time)
        print(f"Randomly selected port: {args.port}")
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.use_refined_charts, args.use_mip_filter, 
        args.dense_data_path, args.use_chart_view_every_n_iter,
        args.normal_consistency_from, args.distortion_from,
        args.depthanythingv2_checkpoint_dir, args.depthanything_encoder,
        args.dense_regul, args.refine_depth_path, args.use_downsample_gaussians
    )

    # All done
    print("\nTraining complete.")