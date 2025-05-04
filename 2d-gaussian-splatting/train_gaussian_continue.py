import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import gc
import torch
from random import randint
from tqdm import tqdm
import uuid

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from scene.gaussian_model import combine_gslist, combine_gslist_simple
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, BasicPointCloud
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_gslist
from utils.render_utils import save_img_f32, save_img_u8
from utils.image_utils import psnr
from utils.point_utils import depth_to_normal

from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
from matcha.pointmap.depthanythingv2 import get_pointmap_from_see3d_inpainting_with_depthanything, get_pointmap_from_see3d_with_depthanything
import trimesh
import cv2
from matcha.dm_scene.charts import (
    load_charts_data, 
    build_priors_from_charts_data,
    schedule_regularization_factor_1,
    schedule_regularization_factor_2,
    depth2normal_parallel,
    normal2curv_parallel,
    depths_to_points_parallel
)
from matcha.dm_regularization.depth import compute_depth_order_loss
from matcha.dm_utils.rendering import normal2curv
from guidance.See3D_modules.pcd_render_util import save_tensor_as_pcd, downsample_pcd
from scripts.see3d_align_charts import see3d_align_charts

import math
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# New confidence to increasing weight function
def confidence_to_weight(confidence:torch.Tensor):
    conf_weights = confidence - 1.
    return torch.sigmoid((conf_weights - 2.) * 2.)

def prepare_output_and_logger(output_path):
    # Set up output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(output_path, tb_writer, iteration, Ll1, loss, dist_loss, normal_loss, elapsed, testing_iterations, 
                    gaussians, test_cameras, render_func, render_args):
    """Generate and log training reports and visualizations"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)

    # Report test
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        l1_test = 0.0
        psnr_test = 0.0
        
        sample_cameras = test_cameras
        
        for idx, viewpoint in enumerate(sample_cameras):
            render_pkg = render_func(viewpoint, gaussians, *render_args)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = viewpoint.original_image
            l1 = l1_loss(image, gt_image).mean().double()
            psnr_value = psnr(image, gt_image).mean().double()
            l1_test += l1
            psnr_test += psnr_value
            
            if tb_writer and (idx < 5):
                from utils.general_utils import colormap
                depth = render_pkg["surf_depth"]
                norm = depth.max()
                depth = depth / norm
                depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/depth", depth[None], global_step=iteration)
                tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/render", image[None], global_step=iteration)

                try:
                    rend_alpha = render_pkg['rend_alpha']
                    rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                    surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                    tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/rend_normal", rend_normal[None], global_step=iteration)
                    tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/surf_normal", surf_normal[None], global_step=iteration)
                    tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/rend_alpha", rend_alpha[None], global_step=iteration)

                    rend_dist = render_pkg["rend_dist"]
                    rend_dist = colormap(rend_dist.cpu().numpy()[0])
                    tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/rend_dist", rend_dist[None], global_step=iteration)
                except:
                    pass
                
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"test_view_{viewpoint.colmap_id}/ground_truth", gt_image[None], global_step=iteration)

            # # Save render
            # save_render_path = os.path.join(output_path, f"test_view_{viewpoint.colmap_id}")
            # os.makedirs(save_render_path, exist_ok=True)
            # save_img_u8(image.permute(1, 2, 0).detach().cpu().numpy(), os.path.join(save_render_path, f"rgb_iter_{iteration}.png"))
            # save_img_u8(surf_normal.permute(1, 2, 0).detach().cpu().numpy(), os.path.join(save_render_path, f"surf_normal_iter_{iteration}.png"))
            # save_img_u8(rend_normal.permute(1, 2, 0).detach().cpu().numpy(), os.path.join(save_render_path, f"rend_normal_iter_{iteration}.png"))

        if len(sample_cameras) > 0:
            psnr_test /= len(sample_cameras)
            l1_test /= len(sample_cameras)
            print(f"\n[ITER {iteration}] Evaluating test: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar('test/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar('test/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def save_model(gaussians, iteration, output_path):
    """Save the Gaussian model at specific iterations"""
    # Export gaussians
    ply_root_path = os.path.join(output_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(ply_root_path, exist_ok=True)
    gaussians_ply_path = os.path.join(ply_root_path, "point_cloud.ply")
    gaussians.save_ply(gaussians_ply_path)

    # # Save model parameters
    # model_path = os.path.join(output_path, f"model_{iteration}.pth")
    # model_params = gaussians.capture()
    # torch.save((model_params, iteration), model_path)

    print(f"Model at iteration {iteration} saved")

if __name__ == "__main__":

    # Load training views
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--warp_root_path", type=str, required=True)
    parser.add_argument("--inpaint_root_path", type=str, required=True)
    parser.add_argument("--output_root_path", type=str, required=True)
    parser.add_argument("--load_iteration", type=int, default=7_000)
    parser.add_argument("--train_iterations", type=int, default=7_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--use_downsample", action="store_true")
    args = get_combined_args(parser)

    raw_gs_root_path = args.warp_root_path
    inpaint_root_path = args.inpaint_root_path
    output_path = args.output_root_path
    os.makedirs(output_path, exist_ok=True)
    
    # Extract parameters
    dataset, pipe = model.extract(args), pipeline.extract(args)
    opt_params = opt.extract(args)
    
    # Initialize system state
    safe_state(False)

    # Load base Gaussian model
    old_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, old_gaussians, load_iteration=args.load_iteration, shuffle=False)        # continue training from the last iteration
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    INIT_iter = args.load_iteration

    # Load input view cameras
    input_viewpoint_cams = scene.getTrainCameras()
    print(f"Loaded {len(input_viewpoint_cams)} input view cameras")

    # Load input view charts data
    print("[INFO] Loading input view charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())

    print("[INFO] Building priors from charts data...")
    charts_priors = build_priors_from_charts_data(charts_data, input_viewpoint_cams)
    charts_scale_factor = charts_priors['scale_factor']
    charts_prior_depths = charts_priors['prior_depths']
    charts_depths = charts_priors['depths']
    charts_confs = charts_priors['confs']
    charts_normals = charts_priors['normals']
    charts_curvs = charts_priors['curvs']
    print("[INFO] Charts priors built.")

    # save charts pts
    charts_pts = depths_to_points_parallel(charts_depths, input_viewpoint_cams)
    charts_pts = charts_pts.reshape(-1, 3)
    charts_pts_colors = [input_viewpoint_cams[i].original_image for i in range(len(input_viewpoint_cams))]
    charts_pts_colors = torch.stack(charts_pts_colors, dim=0)
    charts_pts_colors = charts_pts_colors.permute(0, 2, 3, 1).reshape(-1, 3)
    save_tensor_as_pcd(charts_pts, os.path.join(output_path, 'charts_pts.ply'), charts_pts_colors)
    
    # Load See3D cameras
    see3d_cameras = np.load(os.path.join(raw_gs_root_path, 'see3d_cameras.npz'))
    see3d_gs_cameras_list = []
    n_views = see3d_cameras['n_views']
    train_view_num = len(input_viewpoint_cams)
    for i in range(n_views):
        R = see3d_cameras[f'R_{i:06d}']
        T = see3d_cameras[f'T_{i:06d}']
        FoVx = see3d_cameras[f'FoVx_{i:06d}']
        FoVy = see3d_cameras[f'FoVy_{i:06d}']
        image_width = int(see3d_cameras[f'image_width_{i:06d}'])
        image_height = int(see3d_cameras[f'image_height_{i:06d}'])

        inpainted_image_path = os.path.join(inpaint_root_path, f"predict_warp_frame{i:06d}.png")
        inpainted_image = cv2.imread(inpainted_image_path)
        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB) / 255.0
        inpainted_image = torch.from_numpy(inpainted_image).float().to("cuda").permute(2, 0, 1)

        see3d_gs_cameras_list.append(GSCamera(
            colmap_id=i+train_view_num,                     # avoid colmap id conflict with input viewpoint cameras
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=inpainted_image,
            image_width=None,
            image_height=None,
            gt_alpha_mask=None,
            image_name=None,
            uid=None,
            data_device='cuda',
        ))
    
    # Create See3D pointmap cameras
    see3d_pointmap_cameras = CamerasWrapper(see3d_gs_cameras_list)
    print('See3D pointmap cameras loaded!')

    # Get pointmap
    see3d_pm, reference_depths, none_visible_pcds, none_visible_pcd_colors = get_pointmap_from_see3d_inpainting_with_depthanything(
        see3d_pointmap_cameras=see3d_pointmap_cameras,
        inpaint_images_dir=inpaint_root_path,
        visible_mask_dir=raw_gs_root_path,
        return_none_visible_pcds=True,
        visible_threshold=0.9,
    )
    see3d_prior_depths = torch.stack(reference_depths, dim=0)         # [n_views, h, w]
    see3d_prior_confs = torch.ones_like(see3d_prior_depths) * 1.5     # NOTE: hard code 1.5 as in the original implementation

    world_view_transforms = torch.stack([see3d_pointmap_cameras.gs_cameras[i].world_view_transform for i in range(len(see3d_pointmap_cameras))])
    full_proj_transforms = torch.stack([see3d_pointmap_cameras.gs_cameras[i].full_proj_transform for i in range(len(see3d_pointmap_cameras))])
    
    see3d_prior_normals = depth2normal_parallel(
        see3d_prior_depths, 
        world_view_transforms=world_view_transforms, 
        full_proj_transforms=full_proj_transforms
    ).permute(0, 3, 1, 2)  # Shape (n_charts, 3, h ,w)

    see3d_prior_curvs = normal2curv_parallel(see3d_prior_normals, torch.ones_like(see3d_prior_normals[:, 0:1]))
    print('See3D pointmap loaded!')

    # Use none visible pcds for initialization
    none_visible_pcds = torch.cat(none_visible_pcds, dim=0)
    none_visible_pcd_colors = torch.cat(none_visible_pcd_colors, dim=0)
    save_tensor_as_pcd(none_visible_pcds, os.path.join(output_path, 'none_visible_pcds.ply'), none_visible_pcd_colors)

    if args.use_downsample:
        none_visible_pcds, none_visible_pcd_colors = downsample_pcd(none_visible_pcds, none_visible_pcd_colors, method='voxel')
        save_tensor_as_pcd(none_visible_pcds, os.path.join(output_path, 'none_visible_pcds_downsampled.ply'), none_visible_pcd_colors)

    # Initialize new Gaussians
    init_pcd = BasicPointCloud(points=none_visible_pcds.cpu().numpy(), colors=none_visible_pcd_colors.cpu().numpy(), normals=None)
    new_gaussians = GaussianModel(dataset.sh_degree)
    new_gaussians.create_from_pcd(init_pcd, old_gaussians.spatial_lr_scale)
    print(f"Created {new_gaussians.get_xyz.shape[0]} new Gaussians")

    # Combine old and new Gaussians
    gaussians = combine_gslist_simple([old_gaussians, new_gaussians])

    # Set up MIP filter if used
    gaussians.set_mip_filter(True)
    gaussians.compute_mip_filter(cameras=see3d_gs_cameras_list)

    # Set up training parameters
    gaussians.training_setup(opt_params)

    # delete old gaussians
    del old_gaussians, new_gaussians
    gc.collect()
    torch.cuda.empty_cache()

    # use training views and see3d views
    total_views_list = input_viewpoint_cams + see3d_gs_cameras_list
    total_confs_list = [charts_confs[idx] for idx in range(len(charts_confs))] + [see3d_prior_confs[idx].unsqueeze(0) for idx in range(len(see3d_prior_confs))]
    total_depths_list = [charts_depths[idx] for idx in range(len(charts_depths))] + [see3d_prior_depths[idx].unsqueeze(0) for idx in range(len(see3d_prior_depths))]
    total_normals_list = [charts_normals[idx] for idx in range(len(charts_normals))] + [see3d_prior_normals[idx] for idx in range(len(see3d_prior_normals))]
    total_curvs_list = [charts_curvs[idx] for idx in range(len(charts_curvs))] + [see3d_prior_curvs[idx] for idx in range(len(see3d_prior_curvs))]

    # Initialize parameters for training
    tb_writer = prepare_output_and_logger(output_path)
    
    # Set up optimization parameters
    first_iter = 0
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0
    
    # Define training iterations and testing/saving points
    train_iterations = args.train_iterations
    testing_iterations = args.test_iterations
    save_iterations = args.save_iterations
    
    # Set hyperparameters
    lambda_anisotropy = 0.1  # Controls anisotropy regularization strength
    anisotropy_max_ratio = 5.  # Maximum allowed ratio for gaussian scaling
    use_depth_order_regularization = False
    lambda_normal = opt_params.lambda_normal
    lambda_dist = opt_params.lambda_dist
    normal_consistency_from = 3500  # Start normal consistency loss from this iteration
    distortion_from = 1500  # Start distortion loss from this iteration
    
    # Create a stack for view selection
    viewpoint_idx_stack = None
    
    # Set up progress bar
    progress_bar = tqdm(range(first_iter, train_iterations), desc="Training progress")
    first_iter += 1
    
    # Main training loop
    for iteration in range(first_iter, train_iterations + 1):
        # Start timing the iteration
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()
        
        # Update learning rate
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random view for this iteration
        if not viewpoint_idx_stack or len(viewpoint_idx_stack) == 0:
            viewpoint_idx_stack = list(range(len(total_views_list)))
        viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))
        viewpoint_cam = total_views_list[viewpoint_idx]
        
        # Render the view
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # Get the ground truth image
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Calculate L1 and SSIM losses
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Apply regularization based on iteration schedule
        lambda_normal_current = lambda_normal if iteration > normal_consistency_from else 0.0
        lambda_dist_current = lambda_dist if iteration > distortion_from else 0.0
        
        # Get rendered distortion and normals for regularization
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        
        # Calculate normal consistency loss
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal_current * normal_error.mean()
        
        # Calculate distortion loss
        dist_loss = lambda_dist_current * rend_dist.mean()
        
        # Combine main losses
        total_loss = loss + dist_loss + normal_loss
        
        # Check if we should use chart regularization for this iteration
        surf_depth = render_pkg['surf_depth']
        total_regularization_loss = 0.0
        
        # Get the correct confidence, depth, normal, and curvature for the current view
        current_conf = total_confs_list[viewpoint_idx]
        current_depth = total_depths_list[viewpoint_idx]
        current_normal = total_normals_list[viewpoint_idx]
        current_curv = total_curvs_list[viewpoint_idx]
        
        # Calculate depth prior loss
        regularization_factor = schedule_regularization_factor_2(iteration, 0.5)
        lambda_prior_depth = regularization_factor * 0.75
        lambda_prior_depth_derivative = regularization_factor * 0.5
        lambda_prior_normal = regularization_factor * 0.5
        lambda_prior_curvature = regularization_factor * 0.25
        
        # Define depth scaling for regularization
        charts_scale_factor = 1.0  # This should be set appropriately based on your data
        
        # Depth regularization
        depth_prior_loss = lambda_prior_depth * (
            confidence_to_weight(current_conf) *
            torch.log(1. + charts_scale_factor * (current_depth - surf_depth).abs())
        ).mean()
        
        if lambda_prior_depth_derivative > 0:
            depth_prior_loss += (
                lambda_prior_depth_derivative *
                torch.exp(-(current_conf - 1.) ** 2 / 2.) *
                (1. - (surf_normal * current_normal).sum(dim=0))
            ).mean()
        
        # Normal regularization
        normal_prior_loss = lambda_prior_normal * (1. - (rend_normal * current_normal).sum(dim=0)).mean()
        
        # Curvature regularization using rendered normal
        rend_curvature = normal2curv(rend_normal, torch.ones_like(rend_normal[0:1]))
        curv_prior_loss = lambda_prior_curvature * (current_curv - rend_curvature).abs().mean()
        
        # Depth order regularization if enabled
        if use_depth_order_regularization:
            # Schedule depth order regularization strength
            lambda_depth_order = 0.0
            if iteration > 1500:
                lambda_depth_order = 1.0
            if iteration > 3000:
                lambda_depth_order = 0.1
            if iteration > 4500:
                lambda_depth_order = 0.01
            if iteration > 6000:
                lambda_depth_order = 0.001
            
            # Parameters for depth order loss
            depth_order_loss_max_pixel_shift_ratio = 0.05
            depth_order_loss_log_space = True
            depth_order_loss_log_scale = 20.0
            
            # Get depth prior for the current view
            if viewpoint_idx < len(charts_prior_depths):
                order_supervision_depth = charts_prior_depths[viewpoint_idx].to(surf_depth.device)
            else:
                # For see3d views, use see3d_prior_depths
                see3d_idx = viewpoint_idx - len(charts_prior_depths)
                order_supervision_depth = see3d_prior_depths[see3d_idx].to(surf_depth.device)
            
            # Compute depth order loss if lambda > 0
            if lambda_depth_order > 0:
                depth_order_prior_loss = lambda_depth_order * compute_depth_order_loss(
                    depth=surf_depth,
                    prior_depth=order_supervision_depth,
                    scene_extent=gaussians.spatial_lr_scale,
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
        
        # Anisotropy regularization
        if lambda_anisotropy > 0:
            gaussians_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gaussians_scaling.max(dim=1).values / gaussians_scaling.min(dim=1).values, anisotropy_max_ratio)
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss
        
        # Total regularization loss
        total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
        
        # Final loss
        total_loss = total_loss + total_regularization_loss
        
        # Backward pass
        total_loss.backward()
        
        # Record iteration end time
        iter_end.record()
        
        # Update EMA values for logging
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log
            if lambda_anisotropy > 0:
                ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log
            
            # Update progress bar every 10 iterations
            if iteration % 10 == 0:
                current_points = len(gaussians.get_xyz.detach())
                
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{current_points}",
                    "p_depth": f"{ema_prior_depth_for_log:.{5}f}",
                    "p_normal": f"{ema_prior_normal_for_log:.{5}f}",
                    "p_curvature": f"{ema_prior_curvature_for_log:.{5}f}"
                }
                if lambda_anisotropy > 0:
                    loss_dict["aniso"] = f"{ema_prior_anisotropy_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            # Generate and log training report
            if iteration % 100 == 0 or iteration == testing_iterations[0]:
                training_report(
                    output_path, tb_writer, iteration, Ll1, loss, dist_loss, normal_loss,
                    iter_start.elapsed_time(iter_end), testing_iterations,
                    gaussians, total_views_list, render, (pipe, background)
                )
            
            # Save model at specified iterations
            if iteration in save_iterations:
                save_model(gaussians, iteration+INIT_iter, args.model_path)
            
            # Densification logic
            if iteration < opt_params.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt_params.densify_grad_threshold, 
                        opt_params.opacity_cull, 
                        scene.cameras_extent, 
                        size_threshold
                    )
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=total_views_list)
                
                if iteration % opt_params.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt_params.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Periodically update MIP filter after densification phase
            if iteration % 100 == 0 and iteration > opt_params.densify_until_iter:
                if iteration < train_iterations - 100:  # Don't update at end of training
                    torch.cuda.empty_cache()
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=total_views_list)
            
            # Optimizer step
            if iteration < train_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
    # Print training completion message
    print("Gaussian continue training complete.")
