import os
import numpy as np
from pathlib import Path
import open3d as o3d
import subprocess
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary,\
                          read_points3D_binary

def convert_points3D_to_ply(points3D_path, ply_path):
    print(f"[INFO] Reading points3D from {points3D_path}")
    points3D = read_points3D_binary(points3D_path)

    xyz = np.stack([point.xyz for _, point in points3D.items()])
    rgb = np.stack([point.rgb for _, point in points3D.items()])
    storePly(ply_path, xyz, rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)

    return xyz, pcd

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_transform_matrix(scale, rot_matrix, save_path):
    """
    Args:
        scale: float, scale factor
        rot_matrix: (3,3) numpy.ndarray, rotation matrix
        save_path: str, save path
    """
    assert rot_matrix.shape == (3, 3), "Rotation matrix must be 3x3"

    # Convert to quaternion [x, y, z, w]
    quat_xyzw = R.from_matrix(rot_matrix).as_quat()

    # Rearrange to [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    # Concatenate to [scale, w, x, y, z, 0, 0, 0]
    arr = np.concatenate([[scale], quat_wxyz, [0.0, 0.0, 0.0]]).astype(np.float32)

    # Save as a single row
    np.savetxt(save_path, arr[None], fmt='%.8f')
    print(f"[INFO] Saved transform matrix to {save_path}")


def run_colmap_model_transformer(input_model, output_model, transform_txt):
    cmd = [
        "colmap", "model_transformer",
        "--input_path", str(input_model),
        "--output_path", str(output_model),
        "--transform_path", str(transform_txt)
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to COLMAP sparse model directory (e.g., sparse/0)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    scale = 1.0                         # not use scale

    # Step 1: Save transform.txt
    transform_path = model_dir / "transform.txt"
    rot_matrix = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
    ])  
    save_transform_matrix(scale, rot_matrix, transform_path)

    # Step 2: Apply model_transformer
    output_model = model_dir.parent / "scaled"
    output_model.mkdir(parents=True, exist_ok=True)
    run_colmap_model_transformer(model_dir, output_model, transform_path)

    transformed_points3D = output_model / "points3D.bin"
    transformed_ply = output_model / "points3D.ply"    
    convert_points3D_to_ply(transformed_points3D, transformed_ply)
    print(f"[INFO] Converted transformed model to {transformed_ply}")