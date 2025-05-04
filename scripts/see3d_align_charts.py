import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml

from matcha.pointmap.depthanythingv2 import get_pointmap_from_mast3r_scene_with_depthanything
from matcha.dm_scene.cameras import CamerasWrapper, rescale_cameras, create_gs_cameras_from_pointmap
from matcha.dm_trainers.charts_alignment import align_charts_in_parallel

from rich.console import Console


def see3d_align_charts(see3d_pm, reference_data, output_path, config_name='default'):

    # Set console
    CONSOLE = Console(width=120)

    os.makedirs(output_path, exist_ok=True)

    # Load config
    config_path = os.path.join('configs/charts_alignment', config_name + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config = config['alignment']
    masking_config = config['masking']

    # Reprojection loss
    if align_config['use_reprojection_loss']:
        raise NotImplementedError("Reprojection loss is not implemented yet.")

    # Not using masks for alignment
    mast3r_masks = None
    CONSOLE.print("[INFO] All MASt3R-SfM points will be used for charts alignment.")
    
    # Align the charts
    output = align_charts_in_parallel(
        # Scene
        see3d_pm,
        # Data parameters
        reference_data,
        masks=mast3r_masks,
        rendering_size=pm_config['max_img_size'],
        target_scale=scene_config['target_scale'],
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        charts_data_path=output_path,
        **align_config,
    )

    if align_config['use_learnable_confidence']:
        output_verts, output_depths, output_confs, training_losses = output
        output_confs = output_confs - 1.
    else:
        output_verts, output_depths, training_losses = output

    CONSOLE.print("\nInitialization complete!")
    CONSOLE.print("Output vertices shape:", output_verts.shape)
    CONSOLE.print("Output depths shape:", output_depths.shape)
    if align_config['use_learnable_confidence']:
        CONSOLE.print("Output confidence shape:", output_confs.shape)
