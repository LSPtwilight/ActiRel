import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import pytransform3d.visualizer as pv

import torch

from utils.graphics_utils import getProjectionMatrix

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def vis_camera_pose(poses, mesh_path=None):
    fig = pv.figure()
    
    # NOTE: Hard code camera intrinsic matrix
    fovy_deg = 60
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy
    w, h = 512, 512
    K = np.zeros((3, 3))
    K[0, 0] = fov2focal(fovx, w)
    K[1, 1] = fov2focal(fovy, h)
    K[0, 2] = w // 2
    K[1, 2] = h // 2
    K[2, 2] = 1
    sensor_size = (float(w), float(h))

    if mesh_path is not None:
        fig.plot_mesh(mesh_path)
    for pose in poses:
        fig.plot_transform(A2B=pose, s=0.1)
        fig.plot_camera(M=K, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix

    forward_vector = safe_normalize(target - campos)
    up_vector = np.array([0, 0, 1], dtype=np.float32)
    right_vector = safe_normalize(np.cross(forward_vector, up_vector))
    up_vector = safe_normalize(np.cross(right_vector, forward_vector))

    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


def interpolate_camera_path(poses, num_views, add_random_trans=False):
    positions = poses[:, :3, 3]  # shape: (N, 3)

    rotations = R.from_matrix(poses[:, :3, :3])
    quats = rotations.as_quat()  # shape: (N, 4)

    t = np.linspace(0, 1, len(poses))
    t_new = np.linspace(0, 1, num_views)

    if len(poses) < 3:
        kind = "linear"
    elif len(poses) < 4:
        kind = "quadratic"
    else:
        kind = "cubic"
    pos_interpolator = interp1d(t, positions, axis=0, kind=kind)
    new_positions = pos_interpolator(t_new)

    key_rots = R.from_quat(quats)
    slerp = Slerp(t, key_rots)
    new_rots = slerp(t_new)

    new_poses = np.zeros((num_views, 4, 4))
    new_poses[:, :3, :3] = new_rots.as_matrix()
    new_poses[:, :3, 3] = new_positions
    
    if add_random_trans:
        trans = np.random.uniform(0, 0.5)
        back_dir = new_poses[:, :3, 2]
        new_poses[:, :3, 3] = new_poses[:, :3, 3] + back_dir * trans
    
    new_poses[:, 3, 3] = 1.0

    return new_poses


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path(
    poses: np.ndarray,
    n_frames: int = 120,
    const_speed: bool = False,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = (-sc + offset) * 1.2
    high = (sc + offset) * 1.2
    # Optional height variation need not be symmetric
    # z_low = scale_z*np.percentile((poses[:, :3, 3]), 10, axis=0)
    # z_high = scale_z*np.percentile((poses[:, :3, 3]), 90, axis=0)
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )
    
    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix in COLMAP convention.

        Args:
            lookdir: Looking direction (will be aligned with +Z axis)
            up: Up direction (will be aligned close to +Y axis)
            position: Camera position
        Returns:
            4x4 view matrix where:
            - Z axis is the looking direction (forward)
            - Y axis is up
            - X axis is right
        """
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # center = center + np.array([0, 0, -0.5])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(2 * scale)
    sign = np.random.choice([-1, 1])
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(
            new_poses[i, :, :3], rotation_matrix
        )  # np.dot(, )

    return new_poses

def generate_control_ellipse_path(
    center: np.ndarray,             # [x_center, y_center, z_center]
    low: np.ndarray,                # [x_low, y_low, z_low]
    high: np.ndarray,               # [x_high, y_high, z_high]
    rotation_angle: float = 0.0,    # in degree, rotation angle of the ellipse along z axis
    n_frames: int = 120,
    const_speed: bool = False,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y
        # Optionally also interpolate in z to change camera height along path
        positions = np.stack(
            [
                low[0] + (high[0] - low[0]) * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high[1] - low[1]) * (np.sin(theta) * 0.5 + 0.5),
                z_variation * (low[2] + (high[2] - low[2]) * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)),
            ],
            -1,
        )
        
        if rotation_angle != 0:
            angle_rad = np.deg2rad(rotation_angle)
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            positions = np.dot(positions - center, rot_matrix.T) + center
            
        return positions

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    positions = positions[:-1]

    # NOTE: hard code up vector
    up = np.array([0, 0, 1])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(2 * scale)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)],
    ])

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(new_poses[i, :, :3], rotation_matrix)
    
    return new_poses

def generate_see3d_camera(input_c2ws, interpolate_num=10, camera_type='ellipse', ellipse_num=50, scale=5, width=512, height=512, fovy_deg=60):
    """Generate a camera path for See3D dataset."""

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy
    
    # generate novel c2w matrix
    z_variation = 1.5 - 0.20 * scale
    z_phase = np.random.random()
    poses = np.stack(input_c2ws, axis=0)

    if interpolate_num > 1:
        interpolate_poses = interpolate_camera_path(poses, interpolate_num)
    else:
        interpolate_poses = poses
    
    if camera_type == 'only_interpolate':
        random_poses = interpolate_poses
    elif camera_type == 'control_ellipse':
        random_poses = generate_control_ellipse_path(
            center=poses[:, :3, 3].mean(0),
            low=poses[:, :3, 3].min(0),
            high=poses[:, :3, 3].max(0),
        )

        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)          # c2w matrix

    elif camera_type == 'ellipse':
        random_poses = generate_ellipse_path(
            interpolate_poses[:, :3],
            ellipse_num,
            z_variation=z_variation,
            z_phase=z_phase,
            scale=scale,
        )

        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)          # c2w matrix

    else:
        raise ValueError(f'Invalid camera type: {camera_type}')

    # generate camera
    cur_cams = []
    for idx in range(len(random_poses)):
        c2w = random_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return random_poses, cur_cams


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, target=None):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target)
    T[:3, 3] = campos
    return T


def get_pose_and_cam(elevation_deg, azimuth_deg, fovx, fovy, radius=1, target=None, render_resolution=512):
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    pose = orbit_camera(elevation, azimuth, radius, target=target)
    cam = MiniCam(pose, render_resolution, render_resolution, fovy=fovy, fovx=fovx)
    return pose, cam


# generate MVDream orthogonal viewpoints
def generate_mvdream_orthogonal_viewpoints(obj_bbox, elevation_deg_list=[15, 70], azimuth_deg_list=[-180, 180], fovy_deg_list=[60, 80], radius_ratio_list=[1.0, 1.2], batch_size=2, render_resolution=512):
    # obj_bbox: [2, 3], [x_min, y_min, z_min], [x_max, y_max, z_max]

    x_min, y_min, z_min = obj_bbox[0]
    x_max, y_max, z_max = obj_bbox[1]
    obj_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    lx, ly, lz = x_max - x_min, y_max - y_min, z_max - z_min
    real_radius = np.sqrt(lx**2 + ly**2 + lz**2) / 2

    min_elevation, max_elevation = elevation_deg_list[0], elevation_deg_list[1]
    min_azimuth, max_azimuth = azimuth_deg_list[0], azimuth_deg_list[1]
    min_fovy, max_fovy = fovy_deg_list[0], fovy_deg_list[1]
    min_radius_ratio, max_radius_ratio = radius_ratio_list[0], radius_ratio_list[1]

    poses = []
    cur_cams = []

    for _ in range(batch_size):

        elevation_deg = np.random.randint(min_elevation, max_elevation)
        azimuth_deg = np.random.randint(min_azimuth, max_azimuth)
        radius = np.random.uniform(min_radius_ratio * real_radius, max_radius_ratio * real_radius)
        fovy_deg = np.random.uniform(min_fovy, max_fovy)

        # convert fovy_deg to radians
        fovy = -np.deg2rad(fovy_deg)                     # TODO: need negative fovy for vertical FOV, need check whether this is correct
        fovx = np.deg2rad(fovy_deg)

        pose, cur_cam = get_pose_and_cam(elevation_deg, azimuth_deg, fovx=fovx, fovy=fovy, radius=radius, target=obj_center, render_resolution=render_resolution)
        poses.append(pose)
        cur_cams.append(cur_cam)

        # add orthogonal viewpoints
        delta_angle = 90
        view_num = 360 // delta_angle
        for view_i in range(1, view_num):
            pose_i, cur_cam_i = get_pose_and_cam(elevation_deg, azimuth_deg + view_i * delta_angle, fovx=fovx, fovy=fovy, radius=radius, target=obj_center, render_resolution=render_resolution)
            poses.append(pose_i)
            cur_cams.append(cur_cam_i)

    return poses, cur_cams


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear=0.01, zfar=100):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.R = c2w[:3, :3]            # gs use c2w R
        w2c = np.linalg.inv(c2w)
        self.T = w2c[:3, 3]             # gs use w2c T

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.tensor(c2w[:3, 3]).cuda()                     # TODO: need check whether this is correct, camera center used to compute the SH

