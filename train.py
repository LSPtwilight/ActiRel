import os
import sys
import argparse
import json
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import shutil

def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")

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
    parser.add_argument('--use_refine_depth', action='store_true', help='Use refine depth for training')
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians for training')
    parser.add_argument('--is_forward_facing_scene', action='store_true', help='This is a forward facing scene, only use stage 1 prior')
    parser.add_argument('--use_difix3d_refine', action='store_true', help='Use difix3d for refine rgb rendering')
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
        n_images = None
        view_config_path = os.path.join(args.source_path, f'split-{args.config_view_num}views.json')
        if os.path.exists(view_config_path):
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train']
        else:
            view_config_path = os.path.join(args.source_path, f'train_test_split_{args.config_view_num}.json')
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train_ids']
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
    
    if args.use_refine_depth:
        plane_root_path = os.path.join(mast3r_scene_path, 'plane-refine-depths')
        refine_depth_path = plane_root_path

    refine_free_gaussians_command = " ".join([
        "python", "scripts/refine_free_gaussians.py",
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", free_gaussians_path,
        "--config", args.free_gaussians_config,
        dense_arg,
        "--dense_regul", args.dense_regul,
        "--refine_depth_path", refine_depth_path,
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
    ])

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        # "--skip_test",
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

    def get_see3d_inpaint_command(stage, select_inpaint_num):
        print('NOTE: not use difix3d')
        return " ".join([
        "python", "scripts/see3d_inpaint.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--plane_root_dir", plane_root_path,
        "--data_path", args.source_path,
        "--iteration", '7000',
        "--see3d_stage", str(stage),
        "--select_inpaint_num", str(select_inpaint_num),
        "--none_difix",
    ])

    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", args.source_path,
        "--model_path", args.output_path,
        "--sparse_view_num", str(args.config_view_num),
    ])

    render_charts_command = " ".join([
        "python", "2d-gaussian-splatting/render_chart_views.py",
        "--source_path", mast3r_scene_path,
        "--save_root_path", plane_root_path,
    ])

    generate_2Dplane_command = " ".join([
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", plane_root_path,
    ])

    pnts_path = os.path.join(mast3r_scene_path, 'chart_pcd.ply')
    vis_plane_path = os.path.join(mast3r_scene_path, 'vis_plane')

    def get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None):
        if see3d_root_path is not None:
            if anchor_view_id_json_path is not None:
                return " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--anchor_view_id_json_path", anchor_view_id_json_path,
                    "--see3d_root_path", see3d_root_path,
                ])
            else:
                return " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--see3d_root_path", see3d_root_path,
                ])
        else:
            return " ".join([
                "python", "scripts/plane_refine_depth.py",
                "--source_path", mast3r_scene_path,
                "--plane_root_path", plane_root_path,
                "--pnts_path", pnts_path,
            ])
        
    see3d_root_path = os.path.join(mast3r_scene_path, 'see3d_render')

    render_eval_path = os.path.join(free_gaussians_path, 'train', 'ours_7000', 'renders')
    difix_output_dir = os.path.join(free_gaussians_path, 'train', 'ours_7000', 'renders_difix3d')
    difix_command = " ".join([
        "python", "2d-gaussian-splatting/guidance/difix3d_util.py",
        "--input_dir", render_eval_path,
        "--output_dir", difix_output_dir,
        "--need_cat_result"
    ])

    pseudo_see3d_command = " ".join([
        "python", "scripts/see3d_inpaint.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--data_path", args.source_path,
        "--plane_root_dir", plane_root_path,
        "--iteration", '7000',
        "--select_inpaint_num",'20',
        "--see3d_stage", '0',
        "--none_difix",
    ])

    plane_refine_depth_command_2 = " ".join([
        "python", "scripts/plane_refine_depth.py",
        "--source_path", mast3r_scene_path,
        "--plane_root_path", plane_root_path,
        "--pnts_path", pnts_path,
        "--see3d_root_path", see3d_root_path,
        # "--vis_plane_path", vis_plane_path,
    ])

    t1 = time.time()
    
    # run MAtCha training
    run_command_safe(sfm_command)
    run_command_safe(align_charts_command)

    # generate 2D planes + refine depth for input views + init gaussian training
    run_command_safe(render_charts_command)
    run_command_safe(generate_2Dplane_command)
    run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None))
    run_command_safe(refine_free_gaussians_command)

    # copy split.json to free_gaussians_path for eval
    cp_cmd_1 = f'cp {args.source_path}/split-{args.config_view_num}views.json {free_gaussians_path}'
    run_command_safe(cp_cmd_1)

    # use dense views as pseudo see3d data
    if args.dense_supervision:
        print('********* NOTE: use dense views as pseudo see3d data *********')
        
        run_command_safe(pseudo_see3d_command)
        run_command_safe(plane_refine_depth_command_2)
        print("=======================================================================================================")
        print(f'Get  dense views done! ')
        print("=======================================================================================================")
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-dense'
        run_command_safe(mv_cmd)
        # if args.dense_supervision:
        #     source_plane_root_dir = os.path.join(mast3r_scene_path, 'see3d_render', 'stage0', 'select-gs-inpainted')
        #     direct_path = os.path.join(mast3r_scene_path, 'see3d_render', 'inpainted_images')
            
        #     os.makedirs(direct_path, exist_ok=True)  

        #     for filename in os.listdir(source_plane_root_dir):
        #         source_file = os.path.join(source_plane_root_dir, filename)
        #         target_file = os.path.join(direct_path, filename)

        #         if os.path.isfile(source_file):
        #             shutil.copy2(source_file, target_file)
        #             print(f"Copied and overwritten: {target_file}")
        run_command_safe(refine_free_gaussians_command)


    # # see3d inpainting stage 1 + refine depth with 2D planes + continue gaussian training
    # run_command_safe(get_see3d_inpaint_command(1, args.select_inpaint_num))
    # run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    # mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-ori'
    # run_command_safe(mv_cmd)
    # if args.dense_supervision:
    #     source_plane_root_dir = os.path.join(mast3r_scene_path, 'see3d_render', 'stage0', 'select-gs-inpainted')
    #     direct_path = os.path.join(mast3r_scene_path, 'see3d_render', 'inpainted_images')
        
    #     os.makedirs(direct_path, exist_ok=True)  

    #     for filename in os.listdir(source_plane_root_dir):
    #         source_file = os.path.join(source_plane_root_dir, filename)
    #         target_file = os.path.join(direct_path, filename)

    #         if os.path.isfile(source_file):
    #             shutil.copy2(source_file, target_file)
    #             print(f"Copied and overwritten: {target_file}")
    # run_command_safe(refine_free_gaussians_command)

    # # see3d inpainting stage 2 + refine depth with 2D planes + continue gaussian training
    # run_command_safe(get_see3d_inpaint_command(2, args.select_inpaint_num))
    # run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    # mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s1'
    # run_command_safe(mv_cmd)
    # if args.dense_supervision:
    #     source_plane_root_dir = os.path.join(mast3r_scene_path, 'see3d_render', 'stage0', 'select-gs-inpainted')
    #     direct_path = os.path.join(mast3r_scene_path, 'see3d_render', 'inpainted_images')
        
    #     os.makedirs(direct_path, exist_ok=True)  

    #     for filename in os.listdir(source_plane_root_dir):
    #         source_file = os.path.join(source_plane_root_dir, filename)
    #         target_file = os.path.join(direct_path, filename)

    #         if os.path.isfile(source_file):
    #             shutil.copy2(source_file, target_file)
    #             print(f"Copied and overwritten: {target_file}")
    # run_command_safe(refine_free_gaussians_command)

    # # see3d inpainting stage 3 + refine depth with 2D planes + continue gaussian training
    # run_command_safe(get_see3d_inpaint_command(3, args.select_inpaint_num))
    # anchor_view_id_json_path = os.path.join(see3d_root_path, 'stage3', 'anchor_view_id.json')
    # run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=anchor_view_id_json_path, see3d_root_path=see3d_root_path))
    # mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s2'
    # run_command_safe(mv_cmd)
    # if args.dense_supervision:
    #     source_plane_root_dir = os.path.join(mast3r_scene_path, 'see3d_render', 'stage0', 'select-gs-inpainted')
    #     direct_path = os.path.join(mast3r_scene_path, 'see3d_render', 'inpainted_images')
        
    #     os.makedirs(direct_path, exist_ok=True)  

    #     for filename in os.listdir(source_plane_root_dir):
    #         source_file = os.path.join(source_plane_root_dir, filename)
    #         target_file = os.path.join(direct_path, filename)

    #         if os.path.isfile(source_file):
    #             shutil.copy2(source_file, target_file)
    #             print(f"Copied and overwritten: {target_file}")
    # run_command_safe(refine_free_gaussians_command)

    # # move split.json to free_gaussians_path for eval
    # cp_cmd_2 = f'cp {args.source_path}/split-{args.config_view_num}views.json {free_gaussians_path}'
    # run_command_safe(cp_cmd_2)

    # if not args.is_forward_facing_scene:                # not forward facing scene, use stage 2 and 3
    #     # see3d inpainting stage 2 + refine depth with 2D planes + continue gaussian training
    #     run_command_safe(get_see3d_inpaint_command(2, args.select_inpaint_num))
    #     run_command_safe(plane_refine_depth_command_2)
    #     mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s1'
    #     run_command_safe(mv_cmd)
    #     run_command_safe(refine_free_gaussians_command)

    #     # see3d inpainting stage 3 + refine depth with 2D planes + continue gaussian training
    #     run_command_safe(get_see3d_inpaint_command(3, args.select_inpaint_num))
    #     run_command_safe(plane_refine_depth_command_2)
    #     mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s2'
    #     run_command_safe(mv_cmd)
    #     run_command_safe(refine_free_gaussians_command)
    # else:
    #     print('NOTE: this is a forward facing scene, only use stage 1 prior')

    # see3d inpainting stage 4 + refine depth with 2D planes + continue gaussian training
    # run_command_safe(get_see3d_inpaint_command(4, args.select_inpaint_num))
    # run_command_safe(plane_refine_depth_command_2)
    # mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s3'
    # run_command_safe(mv_cmd)
    # run_command_safe(refine_free_gaussians_command)

    # render all images, export mesh, and evaluate
    run_command_safe(render_all_img_command)
    run_command_safe(tetra_command)

    # if args.is_forward_facing_scene:
    #     # use mesh filter for forward facing scene
    #     mesh_path = os.path.join(tetra_meshes_path, 'tetra_mesh_binary_search_7_iter_7000.ply')
    #     length_threshold = 0.5
    #     filtered_mesh_path = os.path.join(tetra_meshes_path, f'tetra_mesh_binary_search_7_iter_7000_filtered_t{length_threshold}.ply')
    #     filter_mesh_command = " ".join([
    #         "python", "2d-gaussian-splatting/utils/mesh_filter.py",
    #         "--mesh_path", mesh_path,
    #         "--output_path", filtered_mesh_path,
    #     ])
    #     run_command_safe(filter_mesh_command)
    #     mv_cmd = f'mv {mesh_path} {tetra_meshes_path}/tetra_mesh_binary_search_7_iter_7000_ori.ply'
    #     run_command_safe(mv_cmd)
    #     mv_cmd = f'mv {filtered_mesh_path} {mesh_path}'
    #     run_command_safe(mv_cmd)

    if args.use_difix3d_refine:
        print('********* NOTE: use difix3d for refine rgb rendering *********')
        run_command_safe(difix_command)
        ori_render_path = os.path.join(free_gaussians_path, 'train', 'ours_7000', 'renders-ori')
        mv_cmd = f'mv {render_eval_path} {ori_render_path}'
        run_command_safe(mv_cmd)
        mv_cmd = f'mv {difix_output_dir} {render_eval_path}'
        run_command_safe(mv_cmd)

    run_command_safe(eval_command)

    # # vis global 3D plane by mesh
    # mesh_list = os.listdir(tetra_meshes_path)
    # mesh_list = [mesh_name for mesh_name in mesh_list if mesh_name.endswith('.ply')]
    # mesh_list.sort()
    # mesh_name = mesh_list[-1]
    # mesh_path = os.path.join(tetra_meshes_path, mesh_name)
    # print(f"Mesh path: {mesh_path}")
    # vis_global_3Dplane_by_mesh_command = " ".join([
    #     "python", "2d-gaussian-splatting/planes/vis_global_3Dplane_by_mesh.py",
    #     "--source_path", mast3r_scene_path,
    #     "--mesh_path", mesh_path,
    #     "--plane_root_path", plane_root_path,
    #     "--see3d_root_path", see3d_root_path,
    #     "--output_path", os.path.join(args.output_path, 'vis_global_plane_color_mesh.ply'),
    # ])
    # run_command_safe(vis_global_3Dplane_by_mesh_command)

    t2 = time.time()
    print(f"Total running time: {t2 - t1} seconds")
