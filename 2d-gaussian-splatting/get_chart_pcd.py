import torch
from scene import Scene
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import save_img_f32, save_img_u8
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel
import trimesh
import numpy as np
import json
import open3d as o3d
from PIL import Image

from matcha.dm_scene.charts import (
    load_charts_data, 
    build_priors_from_charts_data,
    depths_to_points_parallel,
)

def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    print("Rendering " + args.model_path)
    model_name=os.path.basename(args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    args.resolution = -1
    args.sh_degree = 3

    os.makedirs(args.model_path, exist_ok=True)
    mast3r_pcd = os.path.join(args.source_path, 'points.ply')
    input_ply = os.path.join(args.model_path, 'input.ply')
    if not os.path.exists(input_ply):
        os.system(f'cp {mast3r_pcd} {input_ply}')

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()             # train views

    ref_img_root_path = os.path.join(dataset.source_path, 'ref-images')
    os.makedirs(ref_img_root_path, exist_ok=True)
    for train_view in train_viewpoints:
        ref_img_path = os.path.join(ref_img_root_path, f'{train_view.image_name}.png')
        ref_img_map = train_view.original_image
        ref_img_map = ref_img_map.permute(1, 2, 0).cpu().numpy()
        ref_img_map = (ref_img_map * 255).astype(np.uint8)
        ref_img_map = Image.fromarray(ref_img_map)
        ref_img_map.save(ref_img_path)

    print("[INFO] Loading charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    print("Using charts data from: ", charts_data_path)
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())

    # Build priors from charts data
    print("[INFO] Building priors from charts data...")
    charts_priors = build_priors_from_charts_data(charts_data, train_viewpoints)
    charts_scale_factor = charts_priors['scale_factor']
    charts_prior_depths = charts_priors['prior_depths']
    charts_depths = charts_priors['depths']
    charts_confs = charts_priors['confs']
    charts_normals = charts_priors['normals']
    charts_curvs = charts_priors['curvs']
    print("[INFO] Charts priors built.")

    chart_pcd = depths_to_points_parallel(charts_depths, train_viewpoints)
    chart_pcd = chart_pcd.reshape(-1, 3)

    save_path = f'{dataset.source_path}/chart_pcd.ply'
    save_tensor_as_pcd(chart_pcd, save_path)
    print(f"[INFO] Saved chart pcd to {save_path}")
