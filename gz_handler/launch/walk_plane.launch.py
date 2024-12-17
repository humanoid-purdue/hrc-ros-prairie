from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, SetLaunchConfiguration)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
import sys
print(sys.executable)

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
    pkg_hrc_handler = get_package_share_directory('hrc_handler')
    pkg_hrc_handler = pkg_hrc_handler[:pkg_hrc_handler.rindex("/")]
    os.environ['GZ_SIM_RESOURCE_PATH'] = pkg_hrc_handler
    gz_launch_desc = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': [sdf],
                'on_exit_shutdown': 'True'
            }.items(),
        )
    start_gazebo_ros_spawner_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'g1',
            '-topic', 'robot_description',
            '-x', ['0.0'],
            '-y', ['0.0'],
            '-z', ['0.75'],
            '-R', ['0.0'],
            '-P', ['0.0'],
            '-Y', ['0.0']
        ],
        output='screen')

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config,
        }],
        output='screen'
    )

    return LaunchDescription([
        gz_launch_desc,
        start_gazebo_ros_spawner_cmd,
        start_gazebo_ros_bridge_cmd,
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[],
            remappings=[],
            output='screen'
        ),
        Node(
            package='gz_handler',
            executable='gz_state_estimator',
            name='gz_state_estimator',
            output='screen'
        ),
    ])