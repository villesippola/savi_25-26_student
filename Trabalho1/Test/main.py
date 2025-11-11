#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
import numpy as np
import argparse
import open3d as o3d
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [10.0, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.87911045824568079, -0.1143707949631662, 0.46269225567601935],
                "lookat": [-14.857198715209961, 8.7558956146240234, 4.6393190026283264],
                "up": [-0.45122740480118839, 0.11291073802962912, 0.88523725316662361],
                "zoom": 0.53999999999999981
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def load_and_filter_depth(rgb_path, depth_path, depth_scale=5000.0, depth_trunc=3.0):
    """
    Load RGB and depth images using OpenCV and filter depth values.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        depth_scale: Scale factor for depth values (TUM uses 5000.0)
        depth_trunc: Maximum depth value to consider (in meters)
    
    Returns:
        rgb_o3d: Open3D RGB image
        depth_o3d: Open3D depth image (filtered)
    """
    # Load images with OpenCV
    rgb_cv = cv2.imread(rgb_path)
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Load as 16-bit
    
    # Filter depth values
    # Remove invalid depth values (0 or too far)
    depth_cv = depth_cv.astype(np.float32) / depth_scale  # Convert to meters
    depth_cv[depth_cv > depth_trunc] = 0  # Truncate far values
    depth_cv[depth_cv <= 0] = 0  # Remove invalid values
    
    # Convert to Open3D images
    rgb_o3d = o3d.geometry.Image(rgb_cv.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_cv.astype(np.float32))  # Open3D expects mm
    
    return rgb_o3d, depth_o3d


def create_point_cloud_from_rgbd(rgb_o3d, depth_o3d, intrinsic):
    """
    Create point cloud from RGB-D images.
    
    Args:
        rgb_o3d: Open3D RGB image
        depth_o3d: Open3D depth image
        intrinsic: Camera intrinsic parameters
    
    Returns:
        pcd: Open3D point cloud
    """
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, 
        depth_scale=1.0,  # We converted to mm
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.03):
    """
    Preprocess point cloud: downsampling and normal estimation.
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling
    
    Returns:
        pcd_down: Downsampled point cloud with normals
    """
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Normal estimation
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )
    
    # Orient normals consistently
    pcd_down.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd_down


def transform_params_to_matrix(params):
    """
    Convert 6-parameter vector to 4x4 transformation matrix.
    
    Args:
        params: [rx, ry, rz, tx, ty, tz] - rotation (axis-angle) and translation
    
    Returns:
        T: 4x4 transformation matrix
    """
    rotation = Rotation.from_rotvec(params[:3])
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = params[3:]
    return T


def point_to_plane_residuals(params, source_points, target_points, target_normals):
    """
    Calculate point-to-plane residuals for optimization.
    
    Args:
        params: [rx, ry, rz, tx, ty, tz] - transformation parameters
        source_points: Nx3 array of source points
        target_points: Nx3 array of corresponding target points
        target_normals: Nx3 array of target normals
    
    Returns:
        residuals: N array of point-to-plane distances
    """
    # Convert parameters to transformation matrix
    T = transform_params_to_matrix(params)
    
    # Transform source points
    source_homogeneous = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
    transformed_source = (T @ source_homogeneous.T).T[:, :3]
    
    # Calculate point-to-plane distances
    diff = transformed_source - target_points
    residuals = np.sum(diff * target_normals, axis=1)
    
    return residuals


def custom_icp(source, target, initial_transform, max_iterations=50, 
               tolerance=1e-6, max_correspondence_distance=0.05):
    """
    Custom ICP implementation using scipy.optimize.least_squares.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        initial_transform: Initial 4x4 transformation matrix
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance
        max_correspondence_distance: Maximum distance for correspondence matching
    
    Returns:
        final_transform: Final transformation matrix
        trajectory: List of transformations at each iteration
    """
    # Initialize
    current_transform = initial_transform.copy()
    trajectory = [current_transform.copy()]
    
    # Build KDTree for target
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    # Get target points and normals
    target_points = np.asarray(target.points)
    target_normals = np.asarray(target.normals)
    
    print("\n=== Starting Custom ICP ===")
    
    for iteration in range(max_iterations):
        # Transform source point cloud
        source_transformed = deepcopy(source)
        source_transformed.transform(current_transform)
        source_points = np.asarray(source_transformed.points)
        
        # Find correspondences
        correspondences = []
        source_matched = []
        target_matched = []
        target_normals_matched = []
        
        for i, point in enumerate(source_points):
            [k, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
            if dist[0] < max_correspondence_distance ** 2:
                correspondences.append((i, idx[0]))
                source_matched.append(source_points[i])
                target_matched.append(target_points[idx[0]])
                target_normals_matched.append(target_normals[idx[0]])
        
        if len(correspondences) < 10:
            print(f"Iteration {iteration}: Too few correspondences ({len(correspondences)}), stopping.")
            break
        
        source_matched = np.array(source_matched)
        target_matched = np.array(target_matched)
        target_normals_matched = np.array(target_normals_matched)
        
        # Initial parameters (identity transformation)
        initial_params = np.zeros(6)
        
        # Optimize using least squares
        result = least_squares(
            point_to_plane_residuals,
            initial_params,
            args=(source_matched, target_matched, target_normals_matched),
            method='lm',  # Levenberg-Marquardt
            max_nfev=100
        )
        
        # Get incremental transformation
        incremental_transform = transform_params_to_matrix(result.x)
        
        # Update current transformation
        previous_transform = current_transform.copy()
        current_transform = incremental_transform @ current_transform
        trajectory.append(current_transform.copy())
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(result.fun ** 2))
        
        # Check convergence
        transform_diff = np.linalg.norm(current_transform - previous_transform)
        
        print(f"Iteration {iteration}: Correspondences={len(correspondences)}, "
              f"RMSE={rmse:.6f}, Transform_diff={transform_diff:.6f}")
        
        if transform_diff < tolerance:
            print(f"Converged at iteration {iteration}")
            break
    
    print("=== ICP Finished ===\n")
    
    return current_transform, trajectory


def visualize_registration(source, target, transformation, window_name="Registration Result"):
    """
    Visualize registration result.
    """
    source_temp = deepcopy(source)
    source_temp.transform(transformation)
    
    # Color point clouds
    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 0, 1])  # Blue
    
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries(
        [source_temp, target, axes_mesh],
        window_name=window_name,
        front=view['trajectory'][0]['front'],
        lookat=view['trajectory'][0]['lookat'],
        up=view['trajectory'][0]['up'],
        zoom=view['trajectory'][0]['zoom']
    )


def main():
    # Camera intrinsic parameters (PrimeSense default)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    
    # ------------------------------------
    # Load and preprocess first point cloud
    # ------------------------------------
    print("Loading and preprocessing first point cloud...")
    rgb1, depth1 = load_and_filter_depth(
        '../tum_dataset/rgb/1.png',
        '../tum_dataset/depth/1.png'
    )
    pcd1 = create_point_cloud_from_rgbd(rgb1, depth1, intrinsic)
    pcd1_processed = preprocess_point_cloud(pcd1, voxel_size=0.03)
    print(f"Point cloud 1: {len(pcd1_processed.points)} points after preprocessing")
    
    # ------------------------------------
    # Load and preprocess second point cloud
    # ------------------------------------
    print("Loading and preprocessing second point cloud...")
    rgb2, depth2 = load_and_filter_depth(
        '../tum_dataset/rgb/2.png',
        '../tum_dataset/depth/2.png'
    )
    pcd2 = create_point_cloud_from_rgbd(rgb2, depth2, intrinsic)
    pcd2_processed = preprocess_point_cloud(pcd2, voxel_size=0.03)
    print(f"Point cloud 2: {len(pcd2_processed.points)} points after preprocessing")
    
    # ------------------------------------
    # Visualize initial alignment
    # ------------------------------------
    print("\nVisualizing initial alignment (before ICP)...")
    pcd1_vis = deepcopy(pcd1_processed)
    pcd2_vis = deepcopy(pcd2_processed)
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
    pcd2_vis.paint_uniform_color([0, 0, 1])  # Blue
    
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries(
        [pcd1_vis, pcd2_vis, axes_mesh],
        window_name="Initial Alignment (Before ICP)",
        front=view['trajectory'][0]['front'],
        lookat=view['trajectory'][0]['lookat'],
        up=view['trajectory'][0]['up'],
        zoom=view['trajectory'][0]['zoom']
    )
    
    # ------------------------------------
    # Manual initial transformation
    # ------------------------------------
    # You can adjust these values based on visual inspection
    # Format: 4x4 transformation matrix
    # initial_transform = np.array([
    #     [1.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    
    # Alternative: You can provide a rough initial guess
    # For example, if you know there's a small rotation and translation:
    initial_transform = np.array([
        [0.99, -0.1, 0.0, 0.1],
        [0.1, 0.99, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("\nInitial transformation matrix:")
    print(initial_transform)
    
    # ------------------------------------
    # Run custom ICP
    # ------------------------------------
    final_transform, trajectory = custom_icp(
        source=pcd1_processed,
        target=pcd2_processed,
        initial_transform=initial_transform,
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=0.05
    )
    
    print("\nFinal transformation matrix:")
    print(final_transform)
    
    # ------------------------------------
    # Visualize final result
    # ------------------------------------
    print("\nVisualizing final alignment (after ICP)...")
    visualize_registration(
        pcd1_processed,
        pcd2_processed,
        final_transform,
        window_name="Final Alignment (After Custom ICP)"
    )
    
    # ------------------------------------
    # Compare with Open3D's ICP
    # ------------------------------------
    print("\nRunning Open3D's ICP for comparison...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1_processed, pcd2_processed, 0.05, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    print("Open3D ICP transformation:")
    print(reg_p2p.transformation)
    print(f"Open3D ICP fitness: {reg_p2p.fitness}")
    print(f"Open3D ICP RMSE: {reg_p2p.inlier_rmse}")
    
    print("\nVisualizing Open3D ICP result...")
    visualize_registration(
        pcd1_processed,
        pcd2_processed,
        reg_p2p.transformation,
        window_name="Open3D ICP Result (for comparison)"
    )


if __name__ == '__main__':
    main()