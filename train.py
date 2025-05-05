import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to the source directory')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to the output directory')
    
    # Image selection parameters
    parser.add_argument('--n_images', type=int, default=None, 
        help='Number of images to use for optimization, sampled with constant spacing. If not provided, all images will be used.')
    parser.add_argument('--use_view_config', action='store_true', 
        help='Use view config file to select images for optimization. If provided, this will override the --n_images and --image_idx arguments.')
    parser.add_argument('--config_view_num', type=int, default=10, 
        help='View number of the config file. If provided, this will override the --n_images.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images.')
    parser.add_argument('--randomize_images', action='store_true', 
        help='Shuffle training images before sampling with constant spacing. If image_idx is provided, this will be ignored.')
    
    # Dense supervision (Optional)
    parser.add_argument('--dense_supervision', action='store_true', 
        help='Use dense RGB supervision with a COLMAP dataset. Should only be used with --sfm_config posed.')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    
    # Output mesh parameters
    parser.add_argument('--use_multires_tsdf', action='store_true', help='Use multi-resolution TSDF fusion instead of adaptive tetrahedralization for mesh extraction (not recommended).')
    parser.add_argument('--no_interpolated_views', action='store_true', help='Disable interpolated views for mesh extraction.')
    
    # SfM config
    parser.add_argument('--sfm_config', type=str, default='unposed', help='Config for SfM. Should be "unposed" or "posed".')
    
    # Chart alignment config
    parser.add_argument('--alignment_config', type=str, default='default', help='Config for charts alignment')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    
    # Free Gaussians config
    parser.add_argument('--free_gaussians_config', type=str, default=None, 
        help='Config for Free Gaussians refinement. '\
        'By default, the config used is "default" for sparse supervision, and "long" for dense supervision.'
    )
    
    # Multi-resolution TSDF config
    parser.add_argument('--tsdf_config', type=str, default='default', help='Config for multi-resolution TSDF fusion')
    
    # Tetrahedralization config
    parser.add_argument('--tetra_config', type=str, default='default', help='Config for adaptive tetrahedralization')
    parser.add_argument('--tetra_downsample_ratio', type=float, default=0.5, 
        help='Downsample ratio for tetrahedralization. We recommend starting with 0.5 and then decreasing to 0.25 '\
        'if the mesh is too dense, or increasing to 1.0 if the mesh is too sparse.'
    )
    
    # Run specific step
    parser.add_argument('--sfm_only', action='store_true', help='Only run the SfM step')
    parser.add_argument('--alignment_only', action='store_true', help='Only run the chart alignment step')
    parser.add_argument('--refinement_only', action='store_true', help='Only run the chart refinement step')
    parser.add_argument('--mesh_only', action='store_true', help='Only run the mesh extraction step')
    parser.add_argument('--render_only', action='store_true', help='Only run the render all img step')

    parser.add_argument('--select_inpaint_num', type=int, default=20, help='Number of views to select for inpainting.')
    parser.add_argument('--scratch_train', action='store_true', help='Run the scratch training step')
    args = parser.parse_args()
    
    # Set output paths
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
    mast3r_scene_path = os.path.join(args.output_path, 'mast3r_sfm')
    aligned_charts_path = os.path.join(args.output_path, 'mast3r_sfm')
    free_gaussians_path = os.path.join(args.output_path, 'free_gaussians')
    tsdf_meshes_path = os.path.join(args.output_path, 'tsdf_meshes')
    tetra_meshes_path = os.path.join(args.output_path, 'tetra_meshes')
    warp_root_dir = os.path.join(free_gaussians_path, 'see3d_render', 'select-gs')
    ref_views_save_root_path = os.path.join(free_gaussians_path, 'see3d_render', 'ref-views')
    inpaint_root_dir = os.path.join(free_gaussians_path, 'see3d_render', 'select-gs-inpainted')
    continue_train_root_dir = os.path.join(free_gaussians_path, 'gs-continue-training')
    scratch_train_root_dir = os.path.join(free_gaussians_path, 'gs-scratch-training')
    
    # Dense supervision (Optional)
    if args.dense_supervision:
        dense_arg = " ".join([
            "--dense_data_path", args.source_path,
        ])
        if args.sfm_config != 'posed':
            print("[WARNING] Dense supervision is only supported for posed SfM. Switching to posed SfM.")
            args.sfm_config = 'posed'
    else:
        dense_arg = ""
        
    # Free Gaussians refinement default config
    if args.free_gaussians_config is None:
        args.free_gaussians_config = 'long' if args.dense_supervision else 'default'

    if args.use_view_config:
        view_config_path = os.path.join(args.source_path, f'split-{args.config_view_num}views.json')
        with open(view_config_path, 'r') as f:
            view_config = json.load(f)
        n_images = None
        image_idx_list = view_config['train']
    else:
        n_images = args.n_images
        image_idx_list = args.image_idx
    
    # Defining commands
    sfm_command = " ".join([
        "python", "scripts/run_sfm.py",
        "--source_path", args.source_path,
        "--output_path", mast3r_scene_path,
        "--config", args.sfm_config,
        # "--env", args.sfm_env,
        "--n_images" if n_images is not None else "", str(n_images) if n_images is not None else "",
        "--image_idx" if image_idx_list is not None else "", " ".join([str(i) for i in image_idx_list]) if image_idx_list is not None else "",
        "--randomize_images" if args.randomize_images else "",
    ])
    
    align_charts_command = " ".join([
        "python", "scripts/align_charts.py",
        "--source_path", mast3r_scene_path,
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", aligned_charts_path,
        "--config", args.alignment_config,
        "--depth_model", args.depth_model,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
    ])
    
    refine_free_gaussians_command = " ".join([
        "python", "scripts/refine_free_gaussians.py",
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", free_gaussians_path,
        "--config", args.free_gaussians_config,
        dense_arg,
        "--dense_regul", args.dense_regul,
    ])

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
        "--use_default_output_dir",
    ])
    
    tsdf_command = " ".join([
        "python", "scripts/extract_tsdf_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tsdf_meshes_path,
        "--config", args.tsdf_config,
    ])
    
    tetra_command = " ".join([
        "python", "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views" if not args.no_interpolated_views else "",
        dense_arg,
    ])

    see3d_render_command = " ".join([
        "python", "2d-gaussian-splatting/render_novel_views.py",
        "--model_path", free_gaussians_path,
        "--iteration", '7000',
        "--train_view_num", str(args.config_view_num),
        "--data_path", args.source_path,
        "--output_root_path", warp_root_dir,
        "--select_inpaint_num", str(args.select_inpaint_num),
    ])

    see3d_inpaint_command = " ".join([
        "python", "2d-gaussian-splatting/guidance/see3d_util.py",
        "--source_imgs_dir", ref_views_save_root_path,
        "--warp_root_dir", warp_root_dir,
        "--output_root_dir", inpaint_root_dir,
    ])

    continue_train_command = " ".join([
        "python", "2d-gaussian-splatting/train_gaussian_continue.py",
        "-s", mast3r_scene_path,
        "-m", free_gaussians_path,
        "--warp_root_path", warp_root_dir,
        "--inpaint_root_path", inpaint_root_dir,
        "--output_root_path", continue_train_root_dir,
        "--load_iteration", '7000',
        "--train_iterations", '7000',
    ])

    scratch_train_command = " ".join([
        "python", "2d-gaussian-splatting/train_gaussian_from_scratch.py",
        "-s", mast3r_scene_path,
        "-m", free_gaussians_path,
        "--warp_root_path", warp_root_dir,
        "--inpaint_root_path", inpaint_root_dir,
        "--output_root_path", scratch_train_root_dir,
        "--load_iteration", '7000',
        "--train_iterations", '7000',
    ])

    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", args.source_path,
        "--model_path", args.output_path,
        "--sparse_view_num", str(args.config_view_num),
    ])

    t1 = time.time()
    
    # run MAtCha training
    os.system(sfm_command)
    os.system(align_charts_command)
    os.system(refine_free_gaussians_command)

    # render all images, export mesh, and evaluate
    os.system(render_all_img_command)
    os.system(tetra_command)
    os.system(eval_command)

    # see3d inpainting
    os.system(see3d_render_command)
    os.system(see3d_inpaint_command)

    if args.scratch_train:
        # scratch training
        os.system(scratch_train_command)
    else:
        # continue training
        os.system(continue_train_command)

    # render all images, export mesh, and evaluate
    os.system(render_all_img_command)
    os.system(tetra_command)
    os.system(eval_command)


    t2 = time.time()
    print(f"Total running time: {t2 - t1} seconds")
