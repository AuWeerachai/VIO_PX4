#!/usr/bin/env python3
"""Experimental staged RealSense D456 visual-inertial launch."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    realsense_config = os.path.join(
        get_package_share_directory('isaac_ros_realsense'),
        'config',
        'realsense_stereo.yaml',
    )

    nodes = [
        ComposableNode(
            package='realsense2_camera',
            plugin='realsense2_camera::RealSenseNodeFactory',
            name='realsense2_camera',
            namespace='',
            # Configure the motion module atomically during node construction.
            # Runtime enable calls restart the D456 sensors independently and
            # can overwhelm its USB control endpoint.
            parameters=[realsense_config, {
                'enable_gyro': True,
                'enable_accel': True,
                'depth_module.profile': '640x360x30',
                'gyro_fps': 200,
                'accel_fps': 200,
                'unite_imu_method': 2,
            }],
            remappings=[
                ('infra1/image_rect_raw', 'infra1/image_rect_raw_mono'),
                ('infra1/camera_info', 'left/camera_info_rect'),
                ('infra2/image_rect_raw', 'infra2/image_rect_raw_mono'),
                ('infra2/camera_info', 'right/camera_info_rect'),
                ('imu', 'visual_slam/imu'),
            ],
        ),
        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_left',
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': 1280,
                'image_height': 720,
            }],
            remappings=[
                ('image_raw', 'infra1/image_rect_raw_mono'),
                ('image', 'left/image_rect'),
            ],
        ),
        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_right',
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': 1280,
                'image_height': 720,
            }],
            remappings=[
                ('image_raw', 'infra2/image_rect_raw_mono'),
                ('image', 'right/image_rect'),
            ],
        ),
        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_left',
            parameters=[{
                'encoding_desired': 'mono8',
                'image_width': 1920,
                'image_height': 1200,
            }],
            remappings=[
                ('image_raw', 'left/image_rect'),
                ('image', 'left/image_rect_mono'),
            ],
        ),
        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_right',
            parameters=[{
                'encoding_desired': 'mono8',
                'image_width': 1920,
                'image_height': 1200,
            }],
            remappings=[
                ('image_raw', 'right/image_rect'),
                ('image', 'right/image_rect_mono'),
            ],
        ),
        ComposableNode(
            package='isaac_ros_visual_slam',
            plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
            name='visual_slam_node',
            parameters=[{
                'enable_image_denoising': False,
                'rectified_images': True,
                'enable_slam_visualization': True,
                'enable_landmarks_view': True,
                'enable_observations_view': True,
                'camera_optical_frames': [
                    'camera_infra1_optical_frame',
                    'camera_infra2_optical_frame',
                ],
                'base_frame': 'drone_link',
                'num_cameras': 2,
                'enable_imu_fusion': True,
                'imu_frame': 'camera_gyro_optical_frame',
                'gyro_noise_density': 0.000244,
                'gyro_random_walk': 0.000019393,
                'accel_noise_density': 0.001862,
                'accel_random_walk': 0.003,
                'calibration_frequency': 200.0,
                'image_jitter_threshold_ms': 60.0,
            }],
            remappings=[
                ('/visual_slam/image_0', 'left/image_rect_mono'),
                ('/visual_slam/camera_info_0', 'left/camera_info_rect'),
                ('/visual_slam/image_1', 'right/image_rect_mono'),
                ('/visual_slam/camera_info_1', 'right/camera_info_rect'),
            ],
        ),
    ]

    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        name='container',
        namespace='isaac_ros_examples',
        composable_node_descriptions=nodes,
        output='screen',
    )

    return LaunchDescription([container])
