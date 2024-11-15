from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hrc_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/meshes/g1', glob('meshes/g1/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/helpers', glob('helpers/*')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aurum',
    maintainer_email='ludwigtaycheeying@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'default_state = hrc_handler.default_state:main',
            'mpc_visualizer_pub = hrc_handler.mpc_visualizer_pub:main',
            'mpc_visualizer_sub = hrc_handler.mpc_visualizer_sub:main',
            'bipedal_command_pub = hrc_handler.bipedal_command_pub:main',
            'fullbody_inv_ddp = hrc_handler.fullbody_inv_ddp:main',
            'fullbody_fwdinv_controller = hrc_handler.fullbody_fwdinv_controller:main',
            'mpc_jtst_pub = hrc_handler.mpc_jtst_pub:main',
            'continuous_mpc_visualizer = hrc_handler.continuous_mpc_visualizer:main',
            'walking_command_pub = hrc_handler.walking_command_pub:main',
            'milestone_walking_command_pub = hrc_handler.milestone_walking_command_pub:main',
            'zmp_preview_controller = hrc_handler.zmp_preview_controller:main',
            'logging_node = hrc_handler.logging_node:main'
        ],
    },
)
