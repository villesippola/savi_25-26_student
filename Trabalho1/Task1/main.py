#!/usr/bin/env python3
# shebang line for linux / mac

import copy
from functools import partial
import glob
from random import randint
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d


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


def draw_registration_result(source, target, transformation, title="Registration Result"):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    
    print(f"\n{title}")
    o3d.visualization.draw_geometries([source_temp, target_temp, axes_mesh],
                                      window_name=title,
                                      zoom=0.53999999999999981,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])


def main():

    # ------------------------------------
    # Load and create point clouds
    # ------------------------------------
    filename_rgb1 = '../tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = '../tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Create the rgbd image
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    print("RGBD Image 1:")
    print(rgbd1)

    filename_rgb2 = '../tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = '../tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)

    # Create the rgbd image
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    print("RGBD Image 2:")
    print(rgbd2)

    # Obtain the point clouds from the rgbd images
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # ------------------------------------
    # Visualize before registration
    # ------------------------------------
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)

    # Paint points to get a better visualization
    pcd1.paint_uniform_color([1, 0, 0])  # red
    pcd2.paint_uniform_color([0, 0, 1])  # blue
    entities = [pcd1, pcd2, axes_mesh]

    print("\n=== Before ICP Registration ===")
    o3d.visualization.draw_geometries(entities,
                                      window_name="Before ICP Registration",
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      zoom=view['trajectory'][0]['zoom'])

    # ------------------------------------
    # ICP Registration
    # ------------------------------------
    
    # Set source and target (pcd2 will be aligned to pcd1)
    source = pcd2
    target = pcd1
    
    # ICP parameters
    threshold = 0.02  # 2cm maximum correspondence distance
    trans_init = np.identity(4)  # Start with identity matrix (no transformation)
    
    # Estimate normals for point-to-plane ICP
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Visualize initial alignment (identity transformation)
    print("\n=== Initial Alignment (Identity) ===")
    draw_registration_result(source, target, trans_init, "Initial Alignment")
    
    # Evaluate initial alignment
    print("\nInitial alignment evaluation:")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(f"Fitness: {evaluation.fitness:.4f}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse:.4f}")
    
    # Apply point-to-plane ICP
    print("\n=== Applying Point-to-Plane ICP ===")
    reg_p2plane = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    print("\nICP Registration Result:")
    print(f"Fitness: {reg_p2plane.fitness:.4f}")
    print(f"Inlier RMSE: {reg_p2plane.inlier_rmse:.4f}")
    print("\nTransformation matrix:")
    print(reg_p2plane.transformation)
    
    # Visualize final alignment
    draw_registration_result(source, target, reg_p2plane.transformation, 
                            "After ICP Registration")
    
    # ------------------------------------
    # Show aligned point clouds with original colors
    # ------------------------------------
    pcd2_aligned = copy.deepcopy(pcd2)
    pcd2_aligned.transform(reg_p2plane.transformation)
    pcd2_aligned.paint_uniform_color([0, 0, 1])  # blue
    pcd1.paint_uniform_color([1, 0, 0])  # red
    
    print("\n=== Final Aligned Point Clouds ===")
    o3d.visualization.draw_geometries([pcd1, pcd2_aligned, axes_mesh],
                                      window_name="Final Aligned Point Clouds",
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      zoom=view['trajectory'][0]['zoom'])


if __name__ == '__main__':
    main()