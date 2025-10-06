import os
import numpy as np
import open3d as o3d
import re

def load_mesh(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.85, 0.85, 0.85])  # 浅灰色
    return mesh

def extract_camera_positions_from_R_T(cam_data):
    """从 R_XXXXXX 和 T_XXXXXX 键中提取相机中心"""
    # 找出所有 R_ 开头的键
    r_keys = [k for k in cam_data.keys() if k.startswith('R_')]
    if not r_keys:
        raise ValueError("No 'R_XXXXXX' keys found in camera file.")
    
    # 按编号排序，确保顺序正确
    def get_index(key):
        return int(re.search(r'R_(\d+)', key).group(1))
    r_keys.sort(key=get_index)
    
    centers = []
    for r_key in r_keys:
        idx_str = r_key.split('_')[1]
        t_key = f'T_{idx_str}'
        
        if t_key not in cam_data:
            raise KeyError(f"Missing {t_key} for {r_key}")
        
        R = cam_data[r_key]  # (3, 3)
        T = cam_data[t_key]  # (3,)
        
        # 相机中心（世界坐标） = -R^T @ T
        center = -R.T @ T
        centers.append(center)
    
    return np.array(centers)

def create_colored_camera_points(positions):
    """根据 Z 坐标生成蓝（低）->红（高）颜色映射"""
    points = o3d.utility.Vector3dVector(positions)
    z_vals = positions[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    
    if z_max == z_min:
        colors = np.tile([0, 0, 1], (len(positions), 1))  # 全蓝
    else:
        norm_z = (z_vals - z_min) / (z_max - z_min)
        colors = np.stack([
            norm_z,          # R
            np.zeros_like(norm_z),  # G
            1 - norm_z       # B
        ], axis=1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    # --- 1. 加载 mesh ---
    tetra_dir = "tetra_meshes"
    ply_files = [f for f in os.listdir(tetra_dir) if f.endswith('.ply')]
    if not ply_files:
        raise FileNotFoundError("No .ply file in tetra_meshes/")
    ply_path = os.path.join(tetra_dir, ply_files[0])
    mesh = load_mesh(ply_path)
    print(f"Loaded mesh: {ply_path}")

    # --- 2. 加载相机位置 ---
    cam_path = "mast3r_sfm/see3d_render/see3d_cameras.npz"
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"Camera file not found: {cam_path}")
    
    cam_data = np.load(cam_path)
    positions = extract_camera_positions_from_R_T(cam_data)
    print(f"Loaded {len(positions)} camera positions.")
    print(f"Z range: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")

    # --- 3. 创建彩色点云 ---
    cam_points = create_colored_camera_points(positions)

    # --- 4. 使用 Visualizer 并设置点大小 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Positions (Red=High / Blue=Low)", width=1200, height=800)
    
    vis.add_geometry(mesh)
    vis.add_geometry(cam_points)
    
    # 获取渲染选项并设置点大小
    render_option = vis.get_render_option()
    render_option.point_size = 6.0  # 👈 关键：调大这个值（默认通常是 1~2）
    render_option.background_color = np.asarray([1, 1, 1])  # 可选：白色背景更清晰
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
