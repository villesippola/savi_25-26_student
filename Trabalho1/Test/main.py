#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def load_and_filter_depth(rgb_path, depth_path, depth_scale=5000.0, depth_trunc=3.0):
    """
    Load RGB and depth images using OpenCV and filter depth values.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        depth_scale: Scale factor for depth (TUM uses 5000.0)
        depth_trunc: Maximum depth value in meters
    
    Returns:
        rgb_o3d: Open3D Image object (RGB)
        depth_o3d: Open3D Image object (Depth)
    """
    # Load images with OpenCV
    rgb_cv = cv2.imread(rgb_path)
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Filter depth values
    # Set invalid depth values (0 or too far) to 0
    depth_cv = depth_cv.astype(np.float32) / depth_scale
    depth_cv[depth_cv > depth_trunc] = 0
    depth_cv = (depth_cv * depth_scale).astype(np.uint16)
    
    # Convert OpenCV arrays to Open3D Image objects
    rgb_o3d = o3d.geometry.Image(rgb_cv.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_cv)
    
    return rgb_o3d, depth_o3d


def create_point_cloud(rgb_o3d, depth_o3d, intrinsic):
    """
    Create point cloud from RGB-D images.
    
    Args:
        rgb_o3d: Open3D RGB image
        depth_o3d: Open3D depth image
        intrinsic: Camera intrinsic parameters
    
    Returns:
        Point cloud object
    """
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_tum_format(rgb_o3d, depth_o3d)
    
    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.03):
    """
    Preprocess point cloud: downsampling and normal estimation.
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling
    
    Returns:
        Preprocessed point cloud with normals
    """
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Normal estimation
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    
    return pcd_down


def transformation_matrix_to_vector(T):
    """
    Convert 4x4 transformation matrix to 6D vector [tx, ty, tz, rx, ry, rz].
    Rotation is represented as Euler angles (in radians).
    """
    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    return np.concatenate([translation, euler_angles])


def vector_to_transformation_matrix(vec):
    """
    Convert 6D vector [tx, ty, tz, rx, ry, rz] to 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, 3] = vec[:3]  # Translation
    rotation = Rotation.from_euler('xyz', vec[3:], degrees=False)
    T[:3, :3] = rotation.as_matrix()
    
    return T


def point_to_plane_residuals(params, source_points, target_points, target_normals):
    """
    Calculate point-to-plane residuals for ICP optimization.
    
    Args:
        params: 6D transformation vector [tx, ty, tz, rx, ry, rz]
        source_points: Source point cloud points (Nx3)
        target_points: Target point cloud points (Nx3)
        target_normals: Target point cloud normals (Nx3)
    
    Returns:
        Array of residuals (point-to-plane distances)
    """
    # Convert parameters to transformation matrix
    T = vector_to_transformation_matrix(params)
    
    # Transform source points
    source_points_hom = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
    transformed_points = (T @ source_points_hom.T).T[:, :3]
    
    # Calculate point-to-plane distances
    diff = transformed_points - target_points
    residuals = np.sum(diff * target_normals, axis=1)
    
    return residuals


def find_correspondences(source_pcd, target_pcd, max_correspondence_distance):
    """
    Find nearest neighbor correspondences between source and target point clouds.
    
    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        max_correspondence_distance: Maximum distance for valid correspondence
    
    Returns:
        Indices and distances of correspondences
    """
    # Build KD-Tree for target point cloud
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    
    source_points = np.asarray(source_pcd.points)
    correspondences = []
    distances = []
    
    for i, point in enumerate(source_points):
        # Find nearest neighbor in target
        [k, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        
        if dist[0] < max_correspondence_distance ** 2:  # dist is squared distance
            correspondences.append((i, idx[0]))
            distances.append(np.sqrt(dist[0]))
    
    return correspondences, distances


def custom_icp(source_pcd, target_pcd, initial_transform=np.eye(4), 
               max_iterations=50, tolerance=1e-6, 
               max_correspondence_distance=0.05):
    """
    Custom ICP implementation using scipy.optimize.least_squares.
    
    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        initial_transform: Initial 4x4 transformation matrix
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance
        max_correspondence_distance: Maximum distance for correspondences
    
    Returns:
        Final transformation matrix and list of intermediate transformations
    """
    current_transform = initial_transform.copy()
    source_pcd_transformed = source_pcd
    
    transformations_history = [current_transform.copy()]
    errors_history = []
    
    print(f"\n{'='*60}")
    print("Starting Custom ICP Algorithm")
    print(f"{'='*60}")
    
    for iteration in range(max_iterations):
        # Step 1: Find correspondences
        correspondences, distances = find_correspondences(
            source_pcd_transformed, target_pcd, max_correspondence_distance
        )
        
        if len(correspondences) < 10:
            print(f"\nIteration {iteration}: Too few correspondences ({len(correspondences)}). Stopping.")
            break
        
        # Extract corresponding points and normals
        source_indices = [c[0] for c in correspondences]
        target_indices = [c[1] for c in correspondences]
        
        source_points = np.asarray(source_pcd_transformed.points)[source_indices]
        target_points = np.asarray(target_pcd.points)[target_indices]
        target_normals = np.asarray(target_pcd.normals)[target_indices]
        
        # Calculate current error
        current_error = np.mean(distances)
        errors_history.append(current_error)
        
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        print(f"  Correspondences: {len(correspondences)}")
        print(f"  Mean error: {current_error:.6f} m")
        
        # Step 2: Optimize transformation using least squares
        initial_params = transformation_matrix_to_vector(np.eye(4))
        
        result = least_squares(
            point_to_plane_residuals,
            initial_params,
            args=(source_points, target_points, target_normals),
            method='lm',  # Levenberg-Marquardt
            verbose=0
        )
        
        # Get incremental transformation
        incremental_transform = vector_to_transformation_matrix(result.x)
        
        # Step 3: Update transformation
        current_transform = incremental_transform @ current_transform
        
        # Apply transformation to source point cloud
        source_pcd_transformed = source_pcd.transform(incremental_transform)
        
        transformations_history.append(current_transform.copy())
        
        # Check convergence
        if iteration > 0 and abs(errors_history[-1] - errors_history[-2]) < tolerance:
            print(f"\nConverged at iteration {iteration + 1}")
            print(f"Error change: {abs(errors_history[-1] - errors_history[-2]):.8f} < {tolerance}")
            break
    
    print(f"\n{'='*60}")
    print("ICP Algorithm Completed")
    print(f"{'='*60}")
    print(f"Final mean error: {errors_history[-1]:.6f} m")
    print(f"\nFinal Transformation Matrix:")
    print(current_transform)
    
    return current_transform, transformations_history, errors_history


def visualize_registration(source, target, transformation, window_name="Registration Result"):
    """
    Visualize the registration result.
    """
    source_temp = source.transform(transformation)
    
    # Color the point clouds
    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries(
        [source_temp, target, axes],
        window_name=window_name,
        width=1280,
        height=720
    )


def main():
    print("Custom ICP Implementation with Least-Squares Optimization")
    print("="*60)
    
    # Camera intrinsic parameters (PrimeSense default)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    
    # Step 1: Load and filter depth images
    print("\n1. Loading RGB-D images...")
    rgb1_o3d, depth1_o3d = load_and_filter_depth(
        '../tum_dataset/rgb/1.png',
        '../tum_dataset/depth/1.png'
    )
    
    rgb2_o3d, depth2_o3d = load_and_filter_depth(
        '../tum_dataset/rgb/2.png',
        '../tum_dataset/depth/2.png'
    )
    print("   ✓ Images loaded successfully")
    
    # Step 2: Create point clouds
    print("\n2. Creating point clouds...")
    pcd1 = create_point_cloud(rgb1_o3d, depth1_o3d, intrinsic)
    pcd2 = create_point_cloud(rgb2_o3d, depth2_o3d, intrinsic)
    print(f"   ✓ Point cloud 1: {len(pcd1.points)} points")
    print(f"   ✓ Point cloud 2: {len(pcd2.points)} points")
    
    # Step 3: Preprocess point clouds
    print("\n3. Preprocessing point clouds...")
    voxel_size = 0.03
    pcd1_processed = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_processed = preprocess_point_cloud(pcd2, voxel_size)
    print(f"   ✓ Downsampled point cloud 1: {len(pcd1_processed.points)} points")
    print(f"   ✓ Downsampled point cloud 2: {len(pcd2_processed.points)} points")
    print(f"   ✓ Normals estimated")
    
    # Manual initial transformation (example: small translation and rotation)
    # You can adjust these values based on your data
    print("\n4. Setting initial transformation...")
    initial_transform = np.eye(4)
    # Example: translate by [0.1, 0.05, 0.2] and rotate slightly
    initial_transform[:3, 3] = [0.1, 0.05, 0.2]
    rotation = Rotation.from_euler('xyz', [5, 10, 5], degrees=True)
    initial_transform[:3, :3] = rotation.as_matrix()
    
    print("   Initial transformation matrix:")
    print(initial_transform)
    
    # Visualize before ICP
    print("\n5. Visualizing initial alignment...")
    visualize_registration(pcd1_processed, pcd2_processed, 
                          initial_transform, 
                          "Before ICP - Initial Alignment")
    
    # Step 4: Run custom ICP
    print("\n6. Running Custom ICP...")
    final_transform, transform_history, errors = custom_icp(
        pcd1_processed,
        pcd2_processed,
        initial_transform=initial_transform,
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=0.05
    )
    
    # Visualize after ICP
    print("\n7. Visualizing final alignment...")
    visualize_registration(pcd1_processed, pcd2_processed, 
                          final_transform, 
                          "After ICP - Final Alignment")
    
    # Plot error evolution
    print("\n8. Generating error plot...")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Error (m)', fontsize=12)
    plt.title('ICP Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()