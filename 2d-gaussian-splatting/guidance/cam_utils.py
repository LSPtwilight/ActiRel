import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import pytransform3d.visualizer as pv

import torch
import random

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

    rotations = Rotation.from_matrix(poses[:, :3, :3])
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

    key_rots = Rotation.from_quat(quats)
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

def generate_random_perturbed_camera_poses(
    gs_camera,
    n_poses=10,
    position_std=0.05,
    rotation_std=0.03
):
    """
    Generate N camera poses with small perturbations around a given camera pose (NumPy version).
    
    Args:
        gs_camera: gs camera views
        n_poses: Number of perturbed poses to generate
        position_std: Standard deviation for position perturbation (in same units as camera)
        rotation_std: Standard deviation for rotation perturbation (in radians)
        
    Returns:
        perturbed_poses: List of N perturbed camera poses
    """
    # Extract rotation and translation from original pose
    R = gs_camera.R             # c2w R
    temp_T = gs_camera.T        # w2c T
    T = -np.matmul(R, temp_T)   # c2w T
    
    # Convert rotation matrix to scipy rotation object
    rot = Rotation.from_matrix(R)
    
    # Initialize lists to store perturbed poses
    perturbed_poses = []
    
    for i in range(n_poses):
        # 1. Perturb camera position
        # Add Gaussian noise to position
        pos_noise = np.random.normal(0, position_std, size=3)
        perturbed_T = T + pos_noise
        
        # 2. Perturb camera rotation
        # Create small random rotation using axis-angle representation
        random_axis = np.random.normal(0, 1, size=3)
        random_axis = random_axis / np.linalg.norm(random_axis)  # normalize to unit vector
        random_angle = np.random.normal(0, rotation_std)
        
        # Create small rotation
        small_rot = Rotation.from_rotvec(random_axis * random_angle)
        
        # Apply small rotation to original rotation
        perturbed_rot = small_rot * rot
        
        # Get rotation matrix
        perturbed_R = perturbed_rot.as_matrix()
        
        # Create perturbed camera pose
        perturbed_pose = np.eye(4)
        perturbed_pose[:3, :3] = perturbed_R
        perturbed_pose[:3, 3] = perturbed_T
        perturbed_poses.append(perturbed_pose.astype(np.float32))

    # generate camera
    cur_cams = []
    for idx in range(len(perturbed_poses)):
        cur_cam = MiniCam(perturbed_poses[idx], gs_camera.image_width, gs_camera.image_height, gs_camera.FoVy, gs_camera.FoVx)
        cur_cams.append(cur_cam)
    
    return perturbed_poses, cur_cams

def generate_perturbed_camera_poses(
    gs_camera,
    horizontal_angles=[-20, -10, 10, 20],         # Horizontal angles list (degrees)
    vertical_angles=[-20, -10, 10, 20],           # Vertical angles list (degrees)
    random_translation=True,                    # Whether to perturb translation
    width=512, height=512, fovy_deg=60
):
    """
    Generate a grid of camera poses with multiple angle variations in horizontal and vertical directions
    
    Args:
        gs_camera: Original camera
        horizontal_angles: List of horizontal angles (degrees), negative values for left, positive for right
        vertical_angles: List of vertical angles (degrees), negative values for up, positive for down
        random_translation: Whether to perturb translation
    
    Returns:
        perturbed_poses: List of all generated camera poses
        cur_cams: List of all generated camera objects
    """
    # Extract rotation and translation from original camera
    R = gs_camera.R             # c2w R
    temp_T = gs_camera.T        # w2c T
    T = -np.matmul(R, temp_T)   # c2w T

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy

    # Get the three axis directions of the camera coordinate system
    x_axis = R[:, 0]  # Camera's right direction
    y_axis = R[:, 1]  # Camera's up direction
    z_axis = R[:, 2]  # Camera's forward direction (actual direction is -z)
    
    # Initialize result lists
    perturbed_poses = []
    
    # Generate camera poses for each combination of horizontal and vertical angles
    for h_angle in horizontal_angles:
        for v_angle in vertical_angles:
            # Convert angles to radians
            h_rad = np.radians(h_angle + np.random.uniform(-1.5, 1.5))
            v_rad = np.radians(v_angle + np.random.uniform(-1.5, 1.5))
            
            # Create horizontal rotation (around y-axis)
            h_rotation = Rotation.from_rotvec(y_axis / np.linalg.norm(y_axis) * h_rad)
            
            # Create vertical rotation (around x-axis)
            v_rotation = Rotation.from_rotvec(x_axis / np.linalg.norm(x_axis) * v_rad)
            
            # Apply rotation to original camera rotation
            rot = Rotation.from_matrix(R)
            # First horizontal rotation, then vertical rotation
            perturbed_rot = v_rotation * h_rotation * rot
            perturbed_R = perturbed_rot.as_matrix()
            
            # Calculate translation
            perturbed_T = T.copy()
            
            if random_translation:
                # Add Gaussian noise to position
                pos_noise = np.random.normal(0, 0.1, size=3)
                
                # Apply translation
                perturbed_T = perturbed_T + pos_noise
            
            # Create camera pose matrix after rotation
            perturbed_pose = np.eye(4)
            perturbed_pose[:3, :3] = perturbed_R
            perturbed_pose[:3, 3] = perturbed_T
            perturbed_poses.append(perturbed_pose.astype(np.float32))
    
    # Generate camera objects
    cur_cams = []
    for idx in range(len(perturbed_poses)):
        cur_cam = MiniCam(perturbed_poses[idx], width, height, fovy, fovx)
        cur_cams.append(cur_cam)
    
    return perturbed_poses, cur_cams


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

def generate_see3d_camera(input_c2ws, interpolate_num=10, camera_type='ellipse', ellipse_num=50, scale=5, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """Generate a camera path for See3D dataset."""

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)
    
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

def generate_see3d_camera_by_lookat(train_cams, train_depths, train_view_points, traj_center=None, n_frames=60, width=512, height=512, fovy_deg=60, fovx_deg=None):

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    device = train_depths.device

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    x_range = train_cam_centers[:, 0].max() - train_cam_centers[:, 0].min()
    y_range = train_cam_centers[:, 1].max() - train_cam_centers[:, 1].min()
    z_range = train_cam_centers[:, 2].max() - train_cam_centers[:, 2].min()

    # get traj center
    if traj_center is None:
        traj_center = torch.mean(train_cam_centers, dim=0)

    # NOTE: hard code for range scale
    x_range_scale = [0.4, 0.9]
    y_range_scale = [0.4, 0.9]
    z_range_scale = [0.1, 0.3]

    # generate novel camera center
    theta = torch.linspace(0, 2.0 * torch.pi, n_frames + 1, device=device)
    novel_cam_centers = torch.stack([
        x_range_scale[0] * x_range * torch.cos(theta) + traj_center[0],
        y_range_scale[0] * y_range * torch.sin(theta) + traj_center[1],
        z_range_scale[0] * z_range * torch.cos(theta) + traj_center[2],
    ], dim=-1)
    novel_cam_centers = novel_cam_centers[:-1]          # Throw away duplicated last position.

    # check valid novel camera center
    novel_cam_centers = torch.tensor(novel_cam_centers, dtype=torch.float32, device=device)
    valid_mask = check_valid_camera_center(train_cams, train_depths, novel_cam_centers)
    novel_cam_centers = novel_cam_centers[valid_mask]

    # get lookat points
    lookat_points = get_novel_cams_lookat_points(train_cam_centers, train_view_points, novel_cam_centers)

    novel_cam_centers = novel_cam_centers.cpu().numpy()
    lookat_points = lookat_points.cpu().numpy()
    
    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(p - lookat, up, p) for p, lookat in zip(novel_cam_centers, lookat_points)])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

def select_need_inpaint_views(novel_cams, gs_none_visible_rate, gaussians, select_num=10):
    """
    Select views that need inpainting
    
    Args:
        novel_cams: list of GSCamera objects
        gs_none_visible_rate: list of float, none visible rate of each view
        gaussians: GaussianModel object
        select_num: int, number of views to select
        
    Returns:
        selected_view_ids: list of int, ids of selected views
    """
    none_visible_rate_low_bound = 0.05
    none_visible_rate_high_bound = 0.5
    covisible_rate_high_bound = 0.8

    # Create pairs of (view_id, none_visible_rate)
    view_rates = [(i, rate) for i, rate in enumerate(gs_none_visible_rate)]
    
    # Step 1: shuffle the view_rates
    random.shuffle(view_rates)
    
    # Step 2: Filter views within desired none_visible_rate range
    filtered_views = [(i, rate) for i, rate in view_rates 
                     if none_visible_rate_low_bound <= rate <= none_visible_rate_high_bound]
    
    # Step 3: Select views with low co-visibility
    selected_view_ids = []
    
    # If we have filtered views, select the first one
    if filtered_views:
        first_view_id = filtered_views[0][0]
        selected_view_ids.append(first_view_id)
    
    # Try to select remaining views from filtered views
    for view_id, _ in filtered_views:
        # Skip if this view is already selected
        if view_id in selected_view_ids:
            continue
        
        # Check co-visibility with all previously selected views
        is_covisible = False
        for selected_id in selected_view_ids:
            covisible_ratio = covisibility_check_by_gs(
                novel_cams[selected_id], novel_cams[view_id], gaussians
            )
            
            if covisible_ratio > covisible_rate_high_bound:
                is_covisible = True
                break
        
        # If this view has low co-visibility with all selected views, add it
        if not is_covisible:
            selected_view_ids.append(view_id)
            
        # Stop if we have enough views
        if len(selected_view_ids) >= select_num:
            break

    # Step 4: If we still don't have enough views, relax the constraints
    if len(selected_view_ids) < select_num:
        print(f"Only found {len(selected_view_ids)} views with optimal none_visible_rate. Relaxing constraints...")
        
        # First try views with none_visible_rate < lower bound
        low_rate_views = [(i, rate) for i, rate in view_rates 
                         if rate < none_visible_rate_low_bound and i not in selected_view_ids]
        
        for view_id, _ in low_rate_views:
            # Check co-visibility with all previously selected views
            is_covisible = False
            for selected_id in selected_view_ids:
                covisible_ratio = covisibility_check_by_gs(
                    novel_cams[selected_id], novel_cams[view_id], gaussians
                )
                
                if covisible_ratio > covisible_rate_high_bound:
                    is_covisible = True
                    break
            
            # If this view has low co-visibility with all selected views, add it
            if not is_covisible:
                selected_view_ids.append(view_id)
                
            # Stop if we have enough views
            if len(selected_view_ids) >= select_num:
                break

    # Step 5: If we still don't have enough views, just add any remaining views regardless of co-visibility
    if len(selected_view_ids) < select_num:
        print(f"Only found {len(selected_view_ids)} views with optimal none_visible_rate. Adding remaining views...")
        remaining_views = [i for i in range(len(novel_cams)) if i not in selected_view_ids and gs_none_visible_rate[i] <= none_visible_rate_high_bound]
        random.shuffle(remaining_views)
        selected_view_ids.extend(remaining_views[:select_num - len(selected_view_ids)])
    
    # print(f"Selected {len(selected_view_ids)} views for inpainting")
    return selected_view_ids

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

def covisibility_check_by_gs(camera1, camera2, gaussians):
    """
    Determine co-visibility by checking the number of Gaussian points visible from both cameras
    
    Args:
        camera1, camera2: Two GSCamera objects
        gaussians: GaussianModel object
        
    Returns:
        bool: True if the ratio of shared visible points exceeds the threshold
        float: The maximum co-visibility ratio
    """
    # Get Gaussian point visibility from camera1
    visible_points1 = get_visible_points(camera1, gaussians)
    
    # Get Gaussian point visibility from camera2
    visible_points2 = get_visible_points(camera2, gaussians)
    
    # Calculate the number of points visible from both cameras
    common_visible = torch.logical_and(visible_points1, visible_points2).sum().item()
    
    # Calculate visibility ratios
    ratio1 = common_visible / visible_points1.sum().item() if visible_points1.sum().item() > 0 else 0
    ratio2 = common_visible / visible_points2.sum().item() if visible_points2.sum().item() > 0 else 0
    
    # Take the larger ratio, if it exceeds the threshold, consider the cameras to have co-visibility
    max_ratio = max(ratio1, ratio2)
    return max_ratio

def project_points_to_image(camera, points):
    """
    Project points to image plane
    
    Args:
        camera: GSCamera object where camera.R is c2w rotation and camera.T is w2c translation
        points: torch.Tensor, [N, 3], points in world coordinate
        
    Returns:
        points_depth: torch.Tensor, [N], depth of points in camera coordinate
        points_2d: torch.Tensor, [N, 2], 2D coordinates of points in image plane
        in_image: torch.Tensor, [N], boolean mask indicating which points are within the field of view
    """
    # Get camera parameters (note the special convention)
    R_c2w = camera.R  # This is already camera-to-world rotation
    T_w2c = camera.T  # This is world-to-camera translation
    
    # Calculate world-to-camera rotation (transpose of camera-to-world rotation)
    R_w2c = R_c2w.T

    # Convert to tensors
    T_w2c = torch.tensor(T_w2c, dtype=torch.float32).cuda()
    R_w2c = torch.tensor(R_w2c, dtype=torch.float32).cuda()
    
    # Get camera frustum parameters
    image_height, image_width = camera.image_height, camera.image_width
    fx = image_width / (2 * np.tan(camera.FoVx / 2))
    fy = image_height / (2 * np.tan(camera.FoVy / 2))
    
    # Transform points to camera coordinate system
    # First apply world-to-camera rotation
    points_cam = torch.matmul(R_w2c, points.T).T
    # Then apply world-to-camera translation
    points_cam = points_cam + T_w2c
    
    # Calculate depth values (z-coordinate in camera space)
    points_depth = points_cam[:, 2]

    # Project to image plane
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    points_2d[:, 0] = points_2d[:, 0] * fx + image_width / 2
    points_2d[:, 1] = points_2d[:, 1] * fy + image_height / 2

    # Check if points are within the field of view
    in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)

    return points_depth, points_2d, in_image

def get_visible_points(camera, gaussians):
    """
    Get Gaussian points visible from a camera viewpoint
    
    Args:
        camera: GSCamera object where camera.R is c2w rotation and camera.T is w2c translation
        gaussians: GaussianModel object
        
    Returns:
        torch.Tensor: Boolean mask indicating which points are visible
    """
    
    # Get Gaussian point coordinates
    points = gaussians.get_xyz

    points_depth, _, in_image = project_points_to_image(camera, points)
    
    # Filter out points behind the camera
    front_mask = points_depth > 0
    
    # Visibility mask
    visible_mask = front_mask & in_image
    
    return visible_mask

def check_valid_camera_center(train_cams, train_depths, novel_cam_centers):
    """
    Check if the novel camera center is visible from any of the training cameras
    
    Args:
        train_cams: List of GSCamera objects representing training cameras
        train_depths: List of torch.Tensor, [M, H, W], depths of points in camera coordinate
        novel_cam_centers: torch.Tensor, [N, 3], centers of novel cameras
    
    Returns:
        valid_mask: torch.Tensor, [N], boolean mask indicating which novel camera centers are visible
    """
    # Initialize valid mask (all False)
    valid_mask = torch.zeros(novel_cam_centers.shape[0], dtype=torch.bool, device=novel_cam_centers.device)

    for idx, train_cam in enumerate(train_cams):
        train_depth = train_depths[idx]  # [H, W]
        if isinstance(train_depth, np.ndarray):
            train_depth = torch.from_numpy(train_depth).cuda()
        
        # Project novel camera centers to this training camera's image plane
        points_depth, points_2d, in_image = project_points_to_image(train_cam, novel_cam_centers)
        
        # Get height and width of the training depth map
        H, W = train_depth.shape
        
        # Only process points that are within the image
        if not torch.any(in_image):
            continue
        
        # Get coordinates for points that are in the image
        valid_points_2d = points_2d[in_image]
        
        # Convert to integer coordinates and clamp to image boundaries
        u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
        v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
        
        # Get the depths of these valid points
        valid_points_depth = points_depth[in_image]
        
        # Get corresponding depths from the training depth map
        depth_at_pixels = train_depth[v, u]
        
        # Create masks for visibility conditions
        visible_points_mask = (valid_points_depth < depth_at_pixels) & (valid_points_depth > 0)
        
        # Map back to original indices and update valid_mask
        visible_indices = torch.nonzero(in_image).squeeze(-1)[visible_points_mask]
        valid_mask[visible_indices] = True
    
    return valid_mask

def farthest_point_sample(points, num_samples):
    """
    FPS sampling of points
    """
    num_points = points.shape[0]
    # If we have fewer points than requested samples, return all points
    if num_points <= num_samples:
        return points
        
    # Initialize with the first point
    selected_indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    # Distances to the selected points
    distances = torch.ones(num_points, device=points.device) * 1e10
    
    # Randomly select the first point
    selected_indices[0] = torch.randint(0, num_points, (1,), device=points.device)
    
    # Iteratively select the farthest point
    for i in range(1, num_samples):
        # Last selected point
        last_idx = selected_indices[i-1]
        # Calculate distances to the last selected point
        dist = torch.sum((points - points[last_idx].unsqueeze(0)) ** 2, dim=1)
        # Update distances (minimum distance to any selected point)
        distances = torch.min(distances, dist)
        # Select the farthest point
        selected_indices[i] = torch.argmax(distances)
        
    # Return the sampled points
    return points[selected_indices]

def get_novel_cams_lookat_points(train_cam_centers, gs_train_view_points, novel_cam_centers, fps_num=10):
    """
    Get the lookat points of valid novel cameras
    
    Args:
        train_cam_centers: torch.Tensor, [M, 3], centers of training cameras
        gs_train_view_points: torch.Tensor, [M, N, 3], points of training cameras
        novel_cam_centers: torch.Tensor, [K, 3], centers of valid novel cameras
        fps_num: int, number of points to sample from training view points
    
    Returns:
        lookat_points: torch.Tensor, [K, 3], lookat points of valid novel cameras
    """

    device = novel_cam_centers.device

    # Calculate distances between novel camera centers and training camera centers
    distances = torch.cdist(novel_cam_centers, train_cam_centers)       # [K, M]

    # Find the closest training camera indices for each novel camera
    closest_train_indices = torch.argmin(distances, dim=1)

    # only sample once for each training camera
    train_fps_points = []
    for i in range(train_cam_centers.shape[0]):
        # Get points from the training camera
        points = gs_train_view_points[i]
        # Sample points from the training camera
        sampled_points = farthest_point_sample(points, fps_num)
        train_fps_points.append(sampled_points)

    train_fps_points = torch.stack(train_fps_points, dim=0)           # [M, fps_num, 3]

    # random choose one point from train_fps_points for each novel camera
    random_ids = torch.randint(0, fps_num, (novel_cam_centers.shape[0],), device=device)
    lookat_points = train_fps_points[closest_train_indices, random_ids]

    return lookat_points

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

