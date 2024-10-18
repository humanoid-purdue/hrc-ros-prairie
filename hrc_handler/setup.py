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
            'joint_trajectory_pd_controller = hrc_handler.joint_trajectory_pd_controller:main',
            'zero_jtst_pub = hrc_handler.zero_jtst_pub:main',
            'fullbody_dual_ddf_gz = hrc_handler.fullbody_dual_ddf:ddf_gz',
            'dummy_controller = hrc_handler.dummy_controller:main',
            'joint_trajectory_pos_controller = hrc_handler.joint_trajectory_pos_controller:main',
            'mpc_test_service = hrc_handler.mpc_test_service:main',
            'mpc_test_eclient = hrc_handler.mpc_test_service:eclient',
            'mpc_visualizer_pub = hrc_handler.mpc_visualizer_pub:main',
            'mpc_visualizer_sub = hrc_handler.mpc_visualizer_sub:main'
        ],
    },
)
