# Purdue Humanoid Robotics Club Ros Controller
Bipedal robotics leg controller for stability oriented tasks. Control algorithms implemented in ROS with hooks into moveit2 for IK solvers and gz for simulation capability. 

## Controller:
COM Control based off of Linear Inverted Pendulum with MPC, similar implementation as : [https://inria.hal.science/inria-00390462v1/document]

## Simulation:
Hardware independent testing done on commercial-of-the-shelf bipedal robots. Testing with unitree h1 URDF description: [https://github.com/unitreerobotics/unitree_ros/tree/master]

## Dependencies
Look up how to install
```
ros-jazzy [see installation guide]
gz-harmonic [https://gazebosim.org/docs/latest/ros_installation/, make sure to install ros side pkg]
```
CLI install
```
sudo apt-get install ros-jazzy-robot-state-publisher
sudo apt-get install ros-jazzy-moveit
sudo apt-get install ros-jazzy-rviz
sudo apt-get install ros-jazzy-ros-gzharmonic
pip install numpy
pip install qpsolvers
```
