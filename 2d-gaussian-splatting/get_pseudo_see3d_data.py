import os
import torch
from scene import Scene
import shutil
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json



if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--see3d_root_dir', type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--iteration", required=True, type=int)
    args = get_combined_args(parser)

    data_path = args.data_path
    see3d_root_dir = args.see3d_root_dir
    see3d_stage = args.see3d_stage

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.eval = True                                 # load all cameras
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    all_viewpoints = scene.getTrainCameras()
    
    file_list = os.listdir(data_path)
    split_json_name_list = [file for file in file_list if 'split' in file]
    assert len(split_json_name_list) == 1, f'There should be only one split-*.json file in {data_path}'
    split_json_name = split_json_name_list[0]
    split_json_path = os.path.join(data_path, split_json_name)
    split_json = json.load(open(split_json_path))
    
    images_root_path = os.path.join(data_path, 'images')
    total_images_num = len(os.listdir(images_root_path))

    ### get pseudo see3d views 
    split_train_ids = [int(view_id) for view_id in split_json['train']]
    split_test_ids = [int(view_id) for view_id in split_json['test']]

    image_files = os.listdir(images_root_path)
    all_image_ids = []
    for file in image_files:
        if file.endswith('.png'):
            try:
                img_id = int(file.replace('.png', ''))
                all_image_ids.append(img_id)
            except:
                continue

    pseudo_view_ids = [vid for vid in all_image_ids if vid not in split_train_ids and vid not in split_test_ids]

    chart_view_id = [view_id + 1 for view_id in split_train_ids]
    see3d_view_id = [view_id + 1 for view_id in pseudo_view_ids]

    see3d_stage_path = os.path.join(see3d_root_dir, f'stage{see3d_stage}')
    os.makedirs(see3d_stage_path, exist_ok=True)

    # save pseudo see3d images and poses
    inpaint_path = os.path.join(see3d_stage_path, 'select-gs-inpainted')
    os.makedirs(inpaint_path, exist_ok=True)

    save_cameras = {}
    save_cameras['train_views'] = len(chart_view_id)
    cam_dict = {vp.image_name.replace('.png', ''): vp for vp in all_viewpoints}

    valid_count = 0
    for i, view_id in enumerate(see3d_view_id):
        # 检查图像是否存在
        src_img_path = os.path.join(images_root_path, f'{view_id:06d}.png')
        if not os.path.exists(src_img_path):
            print(f"[Warning] Image not found: {src_img_path}, skipping...")
            continue
            
        # 检查相机参数是否存在
        key = f'{view_id:06d}'
        if key not in cam_dict:
            print(f"[Warning] Camera not found for view_id: {view_id}, skipping...")
            continue
            
        # 只有当两者都存在时才保存
        # 保存图像
        dst_img_path = os.path.join(inpaint_path, f'predict_warp_frame{valid_count:06d}.png')
        shutil.copy(src_img_path, dst_img_path)
        print(f'predict_warp_frame{valid_count:06d}.png copied')
        
        # 保存相机参数
        see3d_viewpoint = cam_dict[key]          
        assert see3d_viewpoint.image_name.replace('.png', '') == key
        
        save_cameras[f'R_{valid_count:06d}'] = see3d_viewpoint.R
        save_cameras[f'T_{valid_count:06d}'] = see3d_viewpoint.T
        save_cameras[f'FoVx_{valid_count:06d}'] = see3d_viewpoint.FoVx
        save_cameras[f'FoVy_{valid_count:06d}'] = see3d_viewpoint.FoVy
        save_cameras[f'image_width_{valid_count:06d}'] = see3d_viewpoint.image_width
        save_cameras[f'image_height_{valid_count:06d}'] = see3d_viewpoint.image_height
        
        valid_count += 1

    # save need inpaint views cameras
    save_cameras['n_views'] = valid_count
    np.savez(os.path.join(see3d_stage_path, f'stage{args.see3d_stage}_see3d_cameras.npz'), **save_cameras)

    print(f'{valid_count} images and cameras saved')
    print(f'See3D stage {args.see3d_stage} save done!')