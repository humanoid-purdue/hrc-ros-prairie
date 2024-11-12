import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, SetLaunchConfiguration)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    default_rviz_config_path = os.path.join(get_package_share_directory('hrc_handler'), 'rviz/mpc_viewer.rviz')

    urdf_file_name = 'urdf/g1.urdf'
    urdf = os.path.join(
        get_package_share_directory('hrc_handler'),
        urdf_file_name)


    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='hrc_handler',
            executable='mpc_visualizer_pub',
            name='mpc_viz_bag_pub',
            output='screen'),
        Node(
            package='hrc_handler',
            executable='zmp_preview_controller',
            name='zmp_preview_controller',
            output='screen'),
    ])