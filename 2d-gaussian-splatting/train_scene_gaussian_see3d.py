import torch
from scene import Scene, GaussianModel
import os
import json
import numpy as np

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image

from utils.general_utils import safe_state

from guidance.cam_utils import generate_see3d_camera, vis_camera_pose
from guidance.See3D_modules.pcd_render_util import pcd_render_multiview, save_rendered_images

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_type", default="gs", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    model_name=os.path.basename(args.model_path)
    render_type = args.render_type

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    dataset.eval = True                 # load all images
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoints = scene.getTrainCameras().copy()             # all views

    # NOTE: hard code set ref view id
    ref_view_id_list = [63, 83]
    ref_c2ws = []
    for ref_view_id in ref_view_id_list:
        viewpoint = viewpoints[ref_view_id]
        w2c = (viewpoint.world_view_transform).transpose(0, 1).cpu().numpy()
        c2w = np.linalg.inv(w2c)
        ref_c2ws.append(c2w)

    # vis ref camera
    temp_total_mesh_path = '/home/nijunfeng/mycode/project/gs-recon/GSFusion/output/replica/scan6/scan6_10views_use_sparsepc_init_aligned_t1_2025-01-19-04-26-33/mesh/origin/whole_scene_30000.ply'
    # vis_camera_pose(ref_c2ws, mesh_path=temp_total_mesh_path)

    # generate novel camera
    # novel_poses, novel_cams = generate_see3d_camera(ref_c2ws, interpolate_num=50, only_interpolate=True, scale=5, width=512, height=512, fovy_deg=60)
    novel_poses, novel_cams = generate_see3d_camera(ref_c2ws, interpolate_num=10, camera_type='ellipse', scale=5, width=512, height=512, fovy_deg=60)

    # # visualize novel camera
    # all_poses = novel_poses

    # # cat ref and novel camera
    # all_poses = np.concatenate([ref_c2ws, novel_poses], axis=0)
    # vis_camera_pose(all_poses, mesh_path=temp_total_mesh_path)

    # exit()

    if render_type == "gs":                 # render novel views (use gaussian rendering)

        output_dir = os.path.join(args.model_path, 'see3d_render', 'raw-gs')
        if os.path.exists(output_dir):
            os.system(f'rm -rf {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        save_cameras = {}
        use_view_num = 0

        for idx, novel_cam in enumerate(novel_cams):
            render_pkg = render(novel_cam, gaussians, pipe, background)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            depth = render_pkg['surf_depth']

            save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(output_dir, f'warp_frame{idx:06d}.png'))
            save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(output_dir, f'depth_frame{idx:06d}.tiff'))
            # save .npy
            np.save(os.path.join(output_dir, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
            alpha_vis_thresh = 0.99
            alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh
            save_img_u8(alpha_vis_mask, os.path.join(output_dir, f'mask_frame{idx:06d}.png'))

            # save camera
            save_cameras[f'R_{use_view_num:06d}'] = novel_cam.R
            save_cameras[f'T_{use_view_num:06d}'] = novel_cam.T
            save_cameras[f'FoVx_{use_view_num:06d}'] = novel_cam.FoVx
            save_cameras[f'FoVy_{use_view_num:06d}'] = novel_cam.FoVy
            save_cameras[f'image_width_{use_view_num:06d}'] = novel_cam.image_width
            save_cameras[f'image_height_{use_view_num:06d}'] = novel_cam.image_height
            use_view_num += 1

            print(f'Novel view {idx} save done!')

        # save cameras as .npz file
        save_cameras['n_views'] = use_view_num
        np.savez(os.path.join(output_dir, 'see3d_cameras.npz'), **save_cameras)
        print(f'See3D cameras save to {os.path.join(output_dir, "see3d_cameras.npz")}')


    elif render_type == "pcd":             # render novel views (use pytorch3d)

        output_dir = os.path.join(args.model_path, 'see3d_render', 'raw-pcd')
        if os.path.exists(output_dir):
            os.system(f'rm -rf {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        M = len(novel_cams)
        fovx_deg_list = [60.0] * M
        fovy_deg_list = [60.0] * M
        images, masks = pcd_render_multiview(gaussians, fovx_deg_list, fovy_deg_list, novel_poses, image_size=(512, 512), radius=0.01, points_per_pixel=10, device="cuda")
        save_rendered_images(images, output_dir, prefix="warp_frame")
        save_rendered_images(masks, output_dir, prefix="mask_frame")

    else:
        raise ValueError(f'Invalid render type: {render_type}')

    print(f'All novel views save done!')

