import os
import shutil
import struct
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def matrix_to_quaternion_and_translation(T):
    rotation_matrix = T[:3, :3]
    translation_vector = T[:3, 3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()  # (x, y, z, w)
    # (q.x, q.y, q.z, q.w)
    quaternion = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
    return translation_vector, quaternion

def transform_pose(input_file, output_file):
    pose_data = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for index, line in enumerate(lines):

        line = line.split() 
        T = np.array(list(map(float, line))).reshape(4, 4)
        t, q = matrix_to_quaternion_and_translation(T)

        pose_data.append((index, t, q))

    pose_data.sort(key=lambda x: x[0])

    with open(output_file, 'w') as out_file:
        for index, t, q in pose_data:
            out_file.write(f"{index} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

# copy from https://github.com/MichaelGrupp/evo/blob/bafb629c8a5738d6c3dd6e83700597f8f0672a6a/evo/core/geometry.py#L35
def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor

    NOTE: x mapping to y
    """
    if x.shape != y.shape:
        raise print("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise print("Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def read_pose(pose_txt_path):
    with open(pose_txt_path, 'r') as f:
        lines = f.readlines()

    xyz_list = []
    for line in lines:
        line = line.split()
        T = np.array(list(map(float, line))).reshape(4, 4)
        t, q = matrix_to_quaternion_and_translation(T)
        xyz_list.append([t[0], t[1], t[2]])

    return np.array(xyz_list)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)            # NOTE: dummy normal, gaussian init not use normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    os.makedirs(os.path.dirname(path),exist_ok=True)
    ply_data.write(path)


root_path = '/home/nijunfeng/mycode/project/gs-recon/merge/priorgs-merge-total/data/replica'
scan_id = 3
ori_gt_mesh_path = os.path.join(root_path, f'scan{scan_id}-ori', 'gt_mesh', 'scene_mesh.ply')
save_gt_mesh_path = os.path.join(root_path, f'scan{scan_id}', 'gt_mesh', 'scene_mesh.ply')

est_pose_path = os.path.join(root_path, f'scan{scan_id}-ori', 'traj.txt')
ref_pose_path = os.path.join(root_path, f'scan{scan_id}', 'traj.txt')

est_xyz = read_pose(est_pose_path)
ref_xyz = read_pose(ref_pose_path)

# NOTE: not need these code, because we fix the training code camera coordinate
# ref_xyz[:, 1] = -ref_xyz[:, 1]
# ref_xyz[:, 2] = -ref_xyz[:, 2]

r, t, c = umeyama_alignment(est_xyz.T, ref_xyz.T, with_scale=True)

ori_mesh = trimesh.load(ori_gt_mesh_path)
ori_xyz = ori_mesh.vertices

align_xyz = ori_xyz.dot(r.T) * c + t
ori_mesh.vertices = align_xyz
ori_mesh.export(save_gt_mesh_path)

print(f'scan{scan_id} point cloud aligned!')

