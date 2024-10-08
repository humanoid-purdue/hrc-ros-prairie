from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped
from hrc_msgs.msg import StateVector
import numpy as np
import time

JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

class gz_state_estimator(Node):

    def __init__(self):
        super().__init__('gz_state_estimator')

        qos_profile = QoSProfile(depth=10)

        self.odom_pos = [0., 0., 1.]
        self.odom_rot = np.array([0., 0., 0., 1.])

        self.joint_traj_pub = self.create_publisher(StateVector, 'state_vector', qos_profile)

        self.subscription_1 = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
        self.subscription_2 = self.create_subscription(
            Odometry,
            'robot_odometry',
            self.odometry_callback,
            10)

    def odometry_callback(self, msg):
        pose = msg.pose.pose
        self.odom_pos = [pose.position.x, pose.position.y, pose.position.z]
        self.odom_rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    def joint_state_callback(self, msg):
        sv = StateVector()
        sv.joint_name = msg.name
        sv.joint_pos = msg.position
        sv.pos_vec = self.odom_pos
        sv.orien_quat = self.odom_rot

def main():
    rclpy.init(args=None)
    node = gz_state_estimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()