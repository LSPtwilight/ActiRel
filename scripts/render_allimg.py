import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True, help='Path to the MASt3R-SfM scene.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the 2D Gaussian Splatting model.')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to save the output mesh.')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, 'all_rendering')
    os.makedirs(args.output_path, exist_ok=True)
    
    # Define command
    render_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", args.mast3r_scene,
        "--model_path", args.model_path,
        "--output_dir", args.output_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
    ])
    
    # Run command
    print(render_command)
    os.system(render_command)