import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from random import randint
from tqdm import tqdm
import uuid

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from scene.gaussian_model import combine_gslist
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, BasicPointCloud
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_gslist
from utils.render_utils import save_img_f32, save_img_u8
from utils.image_utils import psnr

from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
from matcha.dm_utils.rendering import depths_to_points_parallel
from matcha.pointmap.depthanythingv2 import get_pointmap_from_see3d_inpainting_with_depthanything, get_pointmap_from_see3d_with_depthanything
import trimesh
import cv2
from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data
import math
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_tensor_as_pcd(pcd, path, pcd_colors=None):
    pcd = pcd.cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        pcd.colors = pcd_colors.cpu().numpy()
    pcd.export(path)

def convert_camera_to_gscamera(camera):
    return GSCamera(
        colmap_id=camera.colmap_id,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image,
        image_height=camera.image_height,
        image_width=camera.image_width,
        gt_alpha_mask=camera.gt_alpha_mask,
        image_name=camera.image_name,
        uid=camera.uid,
        trans=camera.trans,
        scale=camera.scale,
        data_device=camera.data_device,
    )

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
def training_report(tb_writer, iteration, Ll1, loss, dist_loss, normal_loss, elapsed, testing_iterations, 
                    scene_gs_list, test_cameras, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene_gs_list[1].get_xyz.shape[0], iteration)  # Only count new gaussians

    # Report test
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        l1_test = 0.0
        psnr_test = 0.0
        
        sample_cameras = test_cameras[:5]  # Sample first 5 cameras for visualization
        
        for idx, viewpoint in enumerate(sample_cameras):
            render_pkg = renderFunc(viewpoint, scene_gs_list, *renderArgs)
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

        if len(sample_cameras) > 0:
            psnr_test /= len(sample_cameras)
            l1_test /= len(sample_cameras)
            print(f"\n[ITER {iteration}] Evaluating test: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar('test/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar('test/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def save_model(scene_gs_list, iteration, output_path):
    # Export new gaussians
    new_gaussians = scene_gs_list[1]
    new_gaussians_ply_path = os.path.join(output_path, f"point_cloud_new_{iteration}.ply")
    new_gaussians.save_ply(new_gaussians_ply_path)

    # Export old gaussians
    gaussians = scene_gs_list[0]
    old_gaussians_ply_path = os.path.join(output_path, f"point_cloud_old_{iteration}.ply")
    gaussians.save_ply(old_gaussians_ply_path)

    # # Export total gaussians
    total_gaussians_ply_path = os.path.join(output_path, f"point_cloud_total_{iteration}.ply")
    total_gaussians = combine_gslist(scene_gs_list)
    total_gaussians.save_ply(total_gaussians_ply_path)

    # save params
    model_path = os.path.join(output_path, f"model_{iteration}.pth")
    model_params = total_gaussians.capture()
    torch.save((model_params, iteration), model_path)

    print(f"{iteration} saved")

def train_hierarchical_gaussians():

    # Load training views
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--warp_root_path", type=str, required=True)
    parser.add_argument("--inpaint_root_path", type=str, required=True)
    parser.add_argument("--output_root_path", type=str, required=True)
    parser.add_argument("--train_iterations", type=int, default=7_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 3_000, 7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3_000, 7_000])
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
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=7000, shuffle=False)           # NOTE: hard code for 7000 iteration
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Load See3D cameras
    see3d_cameras = np.load(os.path.join(raw_gs_root_path, 'see3d_cameras.npz'))
    see3d_gs_cameras_list = []
    n_views = see3d_cameras['n_views']
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
            colmap_id=i,
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
    see3d_pm, none_visible_pcds, none_visible_pcd_colors = get_pointmap_from_see3d_inpainting_with_depthanything(
        see3d_pointmap_cameras=see3d_pointmap_cameras,
        inpaint_images_dir=inpaint_root_path,
        visible_mask_dir=raw_gs_root_path,
        return_none_visible_pcds=True,
    )
    print('See3D pointmap loaded!')

    # Choose specific views for initialization
    choose_pcds = torch.cat(none_visible_pcds, dim=0).cpu().numpy()
    choose_pcd_colors = torch.cat(none_visible_pcd_colors, dim=0).cpu().numpy() / 255.0

    # Initialize new Gaussians
    choose_init_pcd = BasicPointCloud(points=choose_pcds, colors=choose_pcd_colors, normals=None)
    new_gaussians = GaussianModel(dataset.sh_degree)
    new_gaussians.create_from_pcd(choose_init_pcd, gaussians.spatial_lr_scale)
    new_gaussians.set_mip_filter(True)
    new_gaussians.compute_mip_filter(cameras=see3d_pointmap_cameras.gs_cameras)

    # Freeze original gaussians
    gaussians.freeze_params()
    
    # Create scene_gs_list
    scene_gs_list = [gaussians, new_gaussians]
    
    # Setup training
    new_gaussians.training_setup(opt_params)
    tb_writer = prepare_output_and_logger(output_path)
    
    # Set up training parameters
    iterations = args.train_iterations
    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    
    # Initialize statistics
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    
    # Set up timers
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    # Create progress bar
    progress_bar = tqdm(range(1, iterations + 1), desc="Training hierarchical model")
    
    # Training loop
    for iteration in range(1, iterations + 1):
        iter_start.record()
        
        # Update learning rate
        new_gaussians.update_learning_rate(iteration)
        
        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            new_gaussians.oneupSHdegree()
        
        # Pick a random camera for training
        view_idx = randint(0, len(see3d_gs_cameras_list) - 1)
        viewpoint_cam = see3d_gs_cameras_list[view_idx]
        gt_image = viewpoint_cam.original_image
        
        # Render
        render_pkg = render_gslist(viewpoint_cam, scene_gs_list, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Calculate loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Regularization
        lambda_normal = opt_params.lambda_normal if iteration > 3500 else 0.0
        lambda_dist = opt_params.lambda_dist if iteration > 3500 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Total loss
        total_loss = loss + dist_loss + normal_loss
        
        # Backward pass
        total_loss.backward()
        
        iter_end.record()
        
        with torch.no_grad():
            # Update progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            
            # Update progress bar every 10 iterations
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(new_gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            # Close progress bar at the end
            if iteration == iterations:
                progress_bar.close()
            
            # Log to tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
            
            # Generate reports
            training_report(tb_writer, iteration, Ll1, loss, dist_loss, normal_loss, 
                           iter_start.elapsed_time(iter_end), testing_iterations, 
                           scene_gs_list, see3d_gs_cameras_list, render_gslist, (pipe, background))
            
            # Save model
            if iteration in saving_iterations:
                save_model(scene_gs_list, iteration, output_path)
            
            # Densification
            if iteration < opt_params.densify_until_iter:
                # Get indices for the new_gaussians in the visibility_filter
                model_indices = render_pkg['model_start_indices']
                new_gaussians_start = model_indices[1]
                new_gaussians_visibility = visibility_filter[new_gaussians_start:]
                new_gaussians_viewspace = viewspace_point_tensor[new_gaussians_start:]
                new_gaussians_viewspace.grad = viewspace_point_tensor.grad[new_gaussians_start:]
                new_gaussians_radii = radii[new_gaussians_start:]
                
                # Update maximum radii
                new_gaussians.max_radii2D[new_gaussians_visibility] = torch.max(
                    new_gaussians.max_radii2D[new_gaussians_visibility], 
                    new_gaussians_radii[new_gaussians_visibility]
                )
                
                # Add densification stats
                new_gaussians.add_densification_stats(new_gaussians_viewspace, new_gaussians_visibility)
                
                # Perform densification
                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    new_gaussians.densify_and_prune(opt_params.densify_grad_threshold, opt_params.opacity_cull, 
                                                  scene.cameras_extent, size_threshold)
                    if new_gaussians.use_mip_filter:
                        new_gaussians.compute_mip_filter(cameras=see3d_pointmap_cameras.gs_cameras)
                
                # Reset opacity
                if iteration % opt_params.opacity_reset_interval == 0:
                    new_gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < iterations:
                new_gaussians.optimizer.step()
                new_gaussians.optimizer.zero_grad(set_to_none=True)
    
    # Final save
    save_model(scene_gs_list, iterations, output_path)
    print("\nTraining complete.")

if __name__ == "__main__":
    train_hierarchical_gaussians()