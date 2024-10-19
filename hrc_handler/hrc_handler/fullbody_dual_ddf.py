
import time
import numpy as np
import crocoddyl
import pinocchio
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped
from tf2_msgs.msg import TFMessage
from hrc_msgs.msg import JointTrajectoryST
from hrc_msgs.msg import StateVector
import os
from geometry_msgs.msg import Wrench
from pinocchio.shortcuts import buildModelsFromUrdf, createDatas
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
import scipy
import sys
from ament_index_python.packages import get_package_share_directory

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
from helpers import BipedalPoser, SquatSM

JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']



class fullbody_dual_ddf_gz(Node):
    def __init__(self):
        super().__init__('fullbody_dual_ddf')


        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)


        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        #urdf_config_path = '/home/aurum/PycharmProjects/cropSandbox/urdf/g1.urdf'
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST, LEG_JOINTS, "left_ankle_roll_link", "right_ankle_roll_link")

        self.ji = helpers.JointInterpolation(len(LEG_JOINTS), 0.1, 0.1)

        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.55]))

        self.get_logger().info("Start sub")
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.state_time = None

        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_callback,
            10)

    def state_callback(self, msg):
        names = msg.joint_name
        j_pos = msg.joint_pos
        self.state_time = msg.time
        j_pos_config = dict(zip(names, j_pos))
        pos = msg.pos
        orien = msg.orien_quat
        ang_vel = msg.ang_vel
        self.poser.x[7 + len(LEG_JOINTS):] = 0
        #self.poser.setState(pos, j_pos_config, orien = orien, vel = msg.vel ,config_vel = dict(zip(names, msg.joint_vel)), ang_vel = ang_vel)
        self.poser.setState(pos, j_pos_config)
        self.squat_sm.com_pos = np.array([0., 0., 0.55 + 0.05 * np.cos(self.state_time)])

    def timer_callback(self):
        if self.state_time is None:
            self.get_logger().info("No sim time")
            return
        timeseries = np.arange(10)* 0.02 + self.state_time
        if self.ji.hasHistory():
            x = self.ji.getSeedX(timeseries)
            x[0] = self.poser.x.copy()
        else:
            x = None

        y = self.squat_sm.simpleNextMPC(x)
        self.ji.updateX(timeseries, y)

        self.joint_trajst_publish(timeseries, y, self.squat_sm.us)

    def r2whole(self, joint_dict):
        pos_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in joint_dict.keys():
                pos_list[c] = joint_dict[name]
        return pos_list

    def joint_trajst_publish(self, timestamps, y, efforts):
        js_list = []
        joint_traj_desired = JointTrajectoryST()
        joint_traj_desired.timestamps = timestamps
        for c in range(len(y)):
            x0 = y[c]
            if c == 0:
                tau = efforts[0]
            else:
                tau = efforts[c - 1]
            pos, joint_dict, joint_vels, joint_efforts = self.poser.getJointConfig(x0, efforts = tau)
            js = JointState()
            js.name = JOINT_LIST
            pos_list = self.r2whole(joint_dict )
            vel_list = self.r2whole(joint_vels)
            effort_list = self.r2whole(joint_efforts)

            js0 = JointState()
            js0.name = JOINT_LIST
            js0.position = pos_list
            js0.velocity = vel_list
            js0.effort = effort_list
            js_list += [js0]
        joint_traj_desired.jointstates = js_list
        self.joint_traj_pub.publish(joint_traj_desired)

def ddf_gz():
    rclpy.init(args=None)

    fb = fullbody_dual_ddf_gz()

    rclpy.spin(fb)

    fb.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    ddf_gz()