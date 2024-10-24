# Purdue Humanoid Robotics Club RO2 Bipedal Controller
Bipedal robotics leg controller for stability oriented tasks. Control algorithms implemented in ROS2 with hooks into gazebo for simulation capability. 

## Controller:
Two layer model predictive controller used to control the robot. Long horizon Center-Of-Mass controller using Linear Inverted Pendulum Model with ZMP criteria and short horizon full-body controller using dynamic differential programming [https://github.com/loco-3d/crocoddyl/tree/master]. More detailed WIP doccumentation can be found at: [https://docs.humanoidrobot.club/wiki]

## Simulation:
Hardware independent testing done on commercial-of-the-shelf bipedal robots. Testing with unitree g1 URDF description: [https://github.com/unitreerobotics/unitree_ros/tree/master]

## Dependencies
Recommended to install in linux environment. WSL and VMs both seem to work.
```
ros-jazzy [see installation guide]
gz-harmonic [https://staging.gazebosim.org/docs/harmonic/install_ubuntu]
```
CLI install For ROS
```
sudo apt-get install ros-jazzy-robot-state-publisher
sudo apt-get install ros-jazzy-rviz
sudo apt-get install ros-jazzy-ros-gz-bridge
```
Python Packages. Recommended to setup virtual environment for python which needs special configuration to work with ROS (https://medium.com/ros2-tips-and-tricks/running-ros2-nodes-in-a-python-virtual-environment-b31c1b863cdb)
```
pip install numpy
pip install crocoddyl
pip install scipy
pip install pyYAML
pip install empy==3.3.4
pip install -U catkin_pkg
pip install lark
```
