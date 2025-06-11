import torch
from scene.dataset_readers import load_cameras
import os
import sys
sys.path.append(os.getcwd())
from utils.general_utils import safe_state
from utils.render_utils import save_img_f32, save_img_u8
from argparse import ArgumentParser
from arguments import ModelParams
from matcha.dm_scene.charts import depths_to_points_parallel
import trimesh
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor

def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

# Robustly compute plane normal, excluding noise interference
def compute_robust_plane_normal(normals, angle_threshold=15.0):
    """
    Robustly compute plane normal, excluding noise interference
    
    Args:
        normals: Normal vector tensor (N, 3)
        angle_threshold: Angle threshold (degrees), normals exceeding this threshold are considered noise
    
    Returns:
        plane_normal: Robustly estimated plane normal (3,)
        inlier_mask: Inlier mask (N,)
    """
    # Normalize all normal vectors
    normals_norm = torch.nn.functional.normalize(normals, dim=1)
    
    # Compute initial mean normal
    initial_mean = torch.mean(normals_norm, dim=0)
    initial_mean = torch.nn.functional.normalize(initial_mean, dim=0)
    
    # Compute angle difference between each normal and initial mean
    dot_products = torch.sum(normals_norm * initial_mean.unsqueeze(0), dim=1)
    # Clamp dot products to [-1, 1] range to avoid numerical errors
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    angles_deg = torch.acos(torch.abs(dot_products)) * 180.0 / torch.pi
    
    # Filter out normals with large angle differences
    inlier_mask = angles_deg < angle_threshold
    
    if inlier_mask.sum() < 3:  # Need at least 3 points to define a plane
        # If too few inliers, relax the threshold
        angle_threshold = 30.0
        inlier_mask = angles_deg < angle_threshold
    
    if inlier_mask.sum() > 0:
        # Recompute mean normal using inliers
        inlier_normals = normals_norm[inlier_mask]
        plane_normal = torch.mean(inlier_normals, dim=0)
        plane_normal = torch.nn.functional.normalize(plane_normal, dim=0)
    else:
        # If no sufficient inliers, use initial mean
        plane_normal = initial_mean
        inlier_mask = torch.ones(len(normals_norm), dtype=torch.bool, device=normals.device)
    
    return plane_normal, inlier_mask

def compute_robust_plane_normal_ransac(normals, num_iterations=100, angle_threshold=10.0):
    """
    Robustly compute plane normal using RANSAC
    
    Args:
        normals: Normal vector tensor (N, 3)
        num_iterations: Number of RANSAC iterations
        angle_threshold: Angle threshold (degrees)
    
    Returns:
        best_normal: Best plane normal
        best_inlier_mask: Best inlier mask
    """
    normals_norm = torch.nn.functional.normalize(normals, dim=1)
    best_inlier_count = 0
    best_normal = None
    best_inlier_mask = None
    
    for _ in range(num_iterations):
        # Randomly select 3 normals to compute candidate plane normal
        indices = torch.randperm(len(normals_norm))[:3]
        candidate_normal = torch.mean(normals_norm[indices], dim=0)
        candidate_normal = torch.nn.functional.normalize(candidate_normal, dim=0)
        
        # Compute inliers
        dot_products = torch.sum(normals_norm * candidate_normal.unsqueeze(0), dim=1)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        angles_deg = torch.acos(torch.abs(dot_products)) * 180.0 / torch.pi
        inlier_mask = angles_deg < angle_threshold
        inlier_count = inlier_mask.sum().item()
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            # Recompute normal using all inliers
            if inlier_count > 0:
                best_normal = torch.mean(normals_norm[inlier_mask], dim=0)
                best_normal = torch.nn.functional.normalize(best_normal, dim=0)
    
    return best_normal, best_inlier_mask

def normals_cluster_1d(valid_normals_1d, n_init_clusters=8, n_clusters=6, min_size_ratio=0.004):
    """
    Cluster 1D normal vectors and return 1D cluster masks
    
    Args:
        valid_normals_1d: Normal vectors from valid region (N, 3)
        n_init_clusters: Initial number of clusters for KMeans
        n_clusters: Number of clusters to keep after filtering
        min_size_ratio: Minimum size ratio for valid clusters
    
    Returns:
        cluster_masks: List of 1D cluster masks (each is boolean array of length N)
        cluster_centers: Cluster centers (normal directions)
    """
    min_cluster_size = valid_normals_1d.shape[0] * min_size_ratio

    # KMeans clustering on 1D normals
    kmeans = KMeans(n_clusters=n_init_clusters, random_state=0, n_init=1).fit(valid_normals_1d)
    pred_1d = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Select top clusters by size
    count_values = np.bincount(pred_1d)
    topk = np.argpartition(count_values, -n_clusters)[-n_clusters:]
    sorted_topk_idx = np.argsort(count_values[topk])
    sorted_topk = topk[sorted_topk_idx][::-1]
    
    cluster_masks = []
    cluster_centers = []
    
    for cluster_id in sorted_topk:
        # Create 1D mask for this cluster
        cluster_mask_1d = (pred_1d == cluster_id)
        
        # Filter by minimum size
        if cluster_mask_1d.sum() < min_cluster_size:
            continue
        
        cluster_masks.append(cluster_mask_1d)
        
        # Normalize cluster center
        center_norm = centers[cluster_id] / np.linalg.norm(centers[cluster_id])
        cluster_centers.append(center_norm)
    
    return cluster_masks, np.array(cluster_centers)

def find_best_matching_cluster_1d(cluster_masks, cluster_centers, target_normal, total_valid_area):
    """
    Find the cluster that best matches the target normal and has the largest area (1D version)
    
    Args:
        cluster_masks: List of 1D cluster masks (boolean arrays)
        cluster_centers: Cluster center normals (N, 3)
        target_normal: Target normal vector (3,)
        total_valid_area: Total valid area
    
    Returns:
        best_mask: Best matching 1D cluster mask
        best_similarity: Similarity score of the best match
    """
    if len(cluster_masks) == 0:
        return None, 0.0
    
    # Convert target_normal to numpy if it's a tensor
    if isinstance(target_normal, torch.Tensor):
        target_normal = target_normal.cpu().numpy()
    
    # Normalize target normal
    target_normal = target_normal / np.linalg.norm(target_normal)
    
    best_score = -1
    best_mask = None
    best_similarity = 0.0
    
    for i, (mask, center) in enumerate(zip(cluster_masks, cluster_centers)):
        # Calculate similarity (cosine similarity)
        similarity = np.abs(np.dot(target_normal, center))
        
        # Calculate area (number of points)
        area = mask.sum()
        area_ratio = area / total_valid_area
        
        # Combined score: similarity * log(area_ratio + 1) to balance similarity and area
        score = similarity * np.log(area_ratio + 1)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            best_similarity = similarity
    
    return best_mask, best_similarity

def compute_plane_aligned_depth(plane_normal, plane_center, camera, img_shape):
    """
    Compute depth map by intersecting camera rays with 3D plane
    
    Args:
        plane_normal: Plane normal vector (3,) in world coordinates
        plane_center: Point on the plane (3,) in world coordinates  
        camera: Camera object with intrinsics and extrinsics
        img_shape: Image shape (H, W)
    
    Returns:
        aligned_depth: Depth map aligned to the plane (H, W)
    """
    H, W = img_shape
    device = plane_normal.device
    
    # Get camera parameters
    fx = camera.focal_x
    fy = camera.focal_y
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=device).float()  # Intrinsic matrix (3, 3)
    c2w = camera.world_view_transform.inverse().T  # Camera to world transform (4, 4)
    
    # Camera center in world coordinates
    camera_center = c2w[:3, 3]  # (3,)
    
    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # (H, W, 3)
    
    # Convert to normalized camera coordinates
    K_inv = torch.inverse(K)
    ray_dirs_cam = torch.matmul(pixels, K_inv.T)  # (H, W, 3)
    
    # Transform ray directions to world coordinates
    ray_dirs_world = torch.matmul(ray_dirs_cam, c2w[:3, :3].T)  # (H, W, 3)
    ray_dirs_world = torch.nn.functional.normalize(ray_dirs_world, dim=-1)  # Normalize
    
    # Compute ray-plane intersection for each pixel
    # Plane equation: n·(P - P0) = 0
    # Ray equation: P = O + t*D
    # Solve: n·(O + t*D - P0) = 0 => t = n·(P0 - O) / (n·D)
    
    plane_normal = plane_normal.view(1, 1, 3)  # (1, 1, 3)
    plane_center = plane_center.view(1, 1, 3)  # (1, 1, 3)
    camera_center = camera_center.view(1, 1, 3)  # (1, 1, 3)
    
    # Compute dot products
    n_dot_d = torch.sum(plane_normal * ray_dirs_world, dim=-1)  # (H, W)
    n_dot_pc_minus_o = torch.sum(plane_normal * (plane_center - camera_center), dim=-1)  # (H, W)
    
    # Avoid division by zero (rays parallel to plane)
    eps = 1e-8
    n_dot_d = torch.clamp(torch.abs(n_dot_d), min=eps) * torch.sign(n_dot_d)
    
    # Compute intersection parameter t
    t = n_dot_pc_minus_o / n_dot_d  # (H, W)
    
    # Compute intersection points
    intersection_points = camera_center + t.unsqueeze(-1) * ray_dirs_world  # (H, W, 3)
    
    # Convert intersection points from world coordinates to camera coordinates
    # Transform to homogeneous coordinates
    intersection_points_homo = torch.cat([intersection_points, torch.ones_like(intersection_points[..., :1])], dim=-1)  # (H, W, 4)
    
    # World to camera transform (inverse of camera to world)
    w2c = torch.inverse(c2w)  # (4, 4)
    
    # Transform intersection points to camera coordinates
    intersection_points_cam = torch.matmul(intersection_points_homo, w2c.T)  # (H, W, 4)
    
    # Extract z-depth (distance along camera z-axis)
    aligned_depth = intersection_points_cam[..., 2]  # (H, W) - z coordinate in camera space
    
    # Handle invalid intersections (behind camera or negative depth)
    valid_mask = (t > 0) & (aligned_depth > 0)
    aligned_depth[~valid_mask] = 0

    valid_intersection_points = intersection_points[valid_mask]
    
    return aligned_depth, valid_intersection_points

def create_overlay_visualization(rgb_image, mask_obj, transparency=0.6, color=[0, 0, 255]):
    """
    Create overlay visualization of mask on RGB image
    
    Args:
        rgb_image: RGB image (H, W, 3)
        mask_obj: Binary mask (H, W)
        transparency: Mask transparency
        color: Specified color, default is blue
    
    Returns:
        blended: Overlaid image
    """
    # Ensure correct input format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask_obj] = color
    
    # Blend images
    blended = rgb_image.copy()
    mask_indices = mask_obj
    
    for i in range(3):
        blended[:, :, i] = np.where(
            mask_indices,
            rgb_image[:, :, i] * (1 - transparency) + colored_mask[:, :, i] * transparency,
            rgb_image[:, :, i]
        )
    
    return blended.astype(np.uint8)

def vis_3Dplane(plane_normal, plane_center, grid_size=1.0, num_points_per_side=21, save_path=None):
    """
    Visualize 3D plane
    
    Args:
        plane_normal: Plane normal vector (3,) in world coordinates
        plane_center: Center point on the plane (3,) in world coordinates
        grid_size: Size of the grid in meters (e.g., 1.0 for 1m x 1m)
        num_points_per_side: Number of points per side of the grid
        save_path: Path to save the plane points
    
    Returns:
        plane_points: Points on the plane (N, 3)
    """
    # Convert to numpy for easier computation
    if isinstance(plane_normal, torch.Tensor):
        plane_normal = plane_normal.cpu().numpy()
    if isinstance(plane_center, torch.Tensor):
        plane_center = plane_center.cpu().numpy()
    
    # Find two orthogonal vectors in the plane
    # Choose an arbitrary vector not parallel to the normal
    if abs(plane_normal[0]) < 0.9:
        arbitrary_vec = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    
    # First tangent vector (cross product)
    tangent1 = np.cross(plane_normal, arbitrary_vec)
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    
    # Second tangent vector (cross product of normal and first tangent)
    tangent2 = np.cross(plane_normal, tangent1)
    tangent2 = tangent2 / np.linalg.norm(tangent2)
    
    # Generate grid coordinates
    half_size = grid_size / 2.0
    coords = np.linspace(-half_size, half_size, num_points_per_side)
    u_coords, v_coords = np.meshgrid(coords, coords)
    
    # Flatten the grid
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    
    # Generate 3D points on the plane
    plane_points = []
    for u, v in zip(u_flat, v_flat):
        point = plane_center + u * tangent1 + v * tangent2
        plane_points.append(point)

    plane_points_tensor = torch.tensor(plane_points).float()
    if save_path is not None:
        save_tensor_as_pcd(plane_points_tensor, save_path)
        print(f"Saved plane points to {save_path}")

    return plane_points_tensor

def fit_plane_ransac(pnts, threshold=0.01, min_samples=3, max_trials=1000):
    """
    Fit a plane to 3D points using the RANSAC algorithm.

    Args:
        pnts: numpy array of shape [N, 3], representing N 3D points.
        threshold: Distance threshold to determine inliers.
        min_samples: Minimum number of data points to fit the model.
        max_trials: Maximum number of iterations for RANSAC.

    Returns:
        plane_normal: Normal vector of the plane [a, b, c]
        d: Offset in the plane equation ax + by + cz + d = 0
        inlier_mask: Boolean mask of inliers, shape [N,]
    """

    # Split X, Y to predict Z
    X = pnts[:, :2]  # [N, 2]
    Z = pnts[:, 2]   # [N]

    # Use RANSAC to fit the plane: Z = aX + bY + d
    ransac = RANSACRegressor(residual_threshold=threshold,
                             min_samples=min_samples,
                             max_trials=max_trials)
    ransac.fit(X, Z)

    # Retrieve plane parameters
    a, b = ransac.estimator_.coef_
    c = -1.0
    d = ransac.estimator_.intercept_

    # Normalize the plane normal vector [a, b, c]
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal /= norm                  # NOTE: normalize normal vector, d is not normalized

    return normal, d, ransac.inlier_mask_


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--data_path", required=True, type=str)
    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(False)

    train_viewpoints, _ = load_cameras(model.extract(args))

    # load depth
    data_path = args.data_path
    for i in range(len(train_viewpoints)):
        depth_path = os.path.join(data_path, f"charts_depth_frame{i:06d}.tiff")
        depth = np.array(Image.open(depth_path))
        depth = torch.from_numpy(depth).to('cuda')

        # get pnts
        view_points = depths_to_points_parallel(depth, [train_viewpoints[i]])       # [1, H*W, 3]
        view_points = view_points.squeeze(0)

        # get rend normal in world coordinate
        rend_normal_path = os.path.join(data_path, f"charts_depth_normal_frame{i:06d}.npy")
        rend_normal = np.load(rend_normal_path)
        rend_normal = torch.from_numpy(rend_normal).to('cuda')

        # load plane mask
        plane_mask_path = os.path.join(data_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)

        conf_path = os.path.join(data_path, f"vis_frequency_frame{i:06d}.npy")
        conf_map = np.load(conf_path)
        conf_map = (conf_map > 0.5)

        plane_id_list = np.unique(plane_mask)
        for plane_id in plane_id_list:
            if plane_id == 0:                       # 0 is default, not plane
                continue

            mask = (plane_mask == plane_id).astype(np.float32)
            mask = (mask > 0.5)

            valid_mask = mask & conf_map
            valid_mask = torch.from_numpy(valid_mask).to('cuda')

            # get valid points for alignment
            valid_points = view_points[valid_mask.reshape(-1)]
            valid_points_num = valid_points.shape[0]

            valid_ratio = valid_points_num / mask.sum()
            if valid_ratio < 0.15 or valid_ratio > 0.95:
                continue
            if valid_points_num < 20:        # need at least 20 valid points for fitting
                continue

            # get max valid plane regions from rend normal
            valid_rend_normal = (rend_normal[valid_mask]).reshape(-1, 3)
            cluster_masks, cluster_centers = normals_cluster_1d(valid_rend_normal.cpu().numpy())
            max_plane_mask = cluster_masks[0]           # already sorted by area
            max_plane_points = valid_points[max_plane_mask]
            max_plane_normal, max_plane_d, _ = fit_plane_ransac(max_plane_points.cpu().numpy())
            plane_normal = torch.from_numpy(max_plane_normal).to('cuda').float()
            plane_center = torch.tensor([0, 0, max_plane_d], device='cuda').float()
            print(f'max_plane_normal: {max_plane_normal}, max_plane_d: {max_plane_d}')

            # vis plane
            vis_3Dplane(plane_normal, plane_center, grid_size=3.0, save_path=os.path.join(data_path, f"plane_frame{i:06d}_plane{plane_id:06d}.ply"))
            # vis mask map
            rgb_path = os.path.join(data_path, f'rgb_frame{i:06d}.png')
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = create_overlay_visualization(rgb, mask)
            rgb_map = Image.fromarray(rgb)
            rgb_map.save(os.path.join(data_path, f"plane_frame{i:06d}_plane{plane_id:06d}_ratio_{valid_ratio:.2f}_mask.png"))

            # get plane aligned depth
            aligned_depth, valid_intersection_points = compute_plane_aligned_depth(plane_normal, plane_center, train_viewpoints[i], depth.shape)
            save_tensor_as_pcd(valid_intersection_points, os.path.join(data_path, f"plane_frame{i:06d}_plane{plane_id:06d}_intersection_points.ply"))

            # replace depth use aligned depth in mask region
            mask = torch.from_numpy(mask).to('cuda')
            depth[mask] = aligned_depth[mask]

        # save refine depth
        save_img_f32(depth.cpu().numpy(), os.path.join(data_path, f"refine_depth_frame{i:06d}.tiff"))

        # get points
        view_points = depths_to_points_parallel(depth, [train_viewpoints[i]])
        view_points = view_points.squeeze(0)

        # save points
        save_path = os.path.join(data_path, f"points_{i:06d}.ply")
        save_tensor_as_pcd(view_points, save_path)
        print(f"Saved points to {save_path}")

    print('done')
