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
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf]),
        Node(
            package='hrc_handler',
            executable='continuous_mpc_visualizer',
            name='continuous_mpc_visualizer',
            output='screen'),
        Node(
            package='hrc_handler',
            executable='walking_command_pub',
            name='walking_command_pub',
            output='screen'),
        Node(
            package='hrc_handler',
            executable='zmp_preview_controller',
            name='zmp_preview_controller',
            output='screen'),
        Node(
            package='hrc_handler',
            executable='logging_node',
            name='logging_node',
            output='screen'),
        Node(
            package='rviz2',
            executable='rviz2',
            name = 'rviz2',
            arguments = ['-d' + default_rviz_config_path]
        )
    ])