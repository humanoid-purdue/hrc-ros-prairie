from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, SetLaunchConfiguration)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    ros_gz_bridge_config_file_path = 'config/config_gzbridge.yaml'
    bridge_config = os.path.join(
        get_package_share_directory('gz_handler'),
        ros_gz_bridge_config_file_path)
    sdf = os.path.join(
        get_package_share_directory('gz_handler'),
        'sdf/walk_plane.sdf')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_launch_path = PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
    gz_launch_desc = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': [sdf],
                'on_exit_shutdown': 'True'
            }.items(),
        )

    return LaunchDescription([
        gz_launch_desc
    ])