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
from ament_index_python.packages import get_package_share_directory
from yaml import load, dump
from yaml import Loader, Dumper
from hrc_msgs.msg import StateVector
import numpy as np
import scipy
import time
import os

JOINT_LIST = ['left_hip_pitch_joint',
              'left_hip_roll_joint',
              'left_hip_yaw_joint',
              'left_knee_joint',
              'left_ankle_pitch_joint',
              'left_ankle_roll_joint',
              'right_hip_pitch_joint',
              'right_hip_roll_joint',
              'right_hip_yaw_joint',
              'right_knee_joint',
              'right_ankle_pitch_joint',
              'right_ankle_roll_joint',
              'torso_joint',
              'left_shoulder_pitch_joint',
              'left_shoulder_roll_joint',
              'left_shoulder_yaw_joint',
              'left_elbow_pitch_joint',
              'left_elbow_roll_joint',
              'right_shoulder_pitch_joint',
              'right_shoulder_roll_joint',
              'right_shoulder_yaw_joint',
              'right_elbow_pitch_joint',
              'right_elbow_roll_joint',
              'left_zero_joint',
              'left_one_joint',
              'left_two_joint',
              'left_three_joint',
              'left_four_joint',
              'left_five_joint',
              'left_six_joint',
              'right_zero_joint',
              'right_one_joint',
              'right_two_joint',
              'right_three_joint',
              'right_four_joint',
              'right_five_joint',
              'right_six_joint']

JOINT_LIST2 = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

class dummy_controller(Node):

    def __init__(self):
        super().__init__('dummy_controller')

        self.cs = None
        self.cs_vel = None
        self.joint_list = None
        self.joint_state = None
        self.js_time = 0
        self.prev_pos = np.zeros([len(JOINT_LIST)])
        self.prev_time = time.time()
        pid_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/pid_config.yaml")
        with open(pid_config_path, 'r') as infp:
            pid_txt = infp.read()
        self.pd = load(pid_txt, Loader = Loader)['g1_gazebo']

        qos_profile = QoSProfile(depth=10)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.subscription_pos = self.create_subscription(
            JointTrajectoryST,
            'joint_trajectory_desired',
            self.joint_traj_callback,
            10
        )
        self.state_vec_pub = self.create_publisher(StateVector, 'state_vector', qos_profile)

        self.timer = self.create_timer(0.001, self.publish_state_vec)

    def publish_state_vec(self):
        state_vec = StateVector()
        state_vec.pos_vec = np.array([0., 0., 0.])
        state_vec.orien_quat = np.array([0., 0., 0., 1.])
        if self.joint_state is not None:
            state_vec.joint_name = self.joint_state.name
            state_vec.joint_pos = self.joint_state.position

        else:
            state_vec.joint_name = JOINT_LIST
            state_vec.joint_pos = [0.] * len(JOINT_LIST)
        self.state_vec_pub.publish(state_vec)


    def joint_traj_callback(self, msg):
        x = np.array(msg.timestamps)
        yr = None
        yv = None
        for jointstate in msg.jointstates:
            if yr is None:
                yr = np.array(jointstate.position)[None, :]
                yv = np.array(jointstate.velocity)[None, :]
            else:
                yr = np.concatenate([yr, np.array(jointstate.position)[None, :]], axis = 0)
                yv = np.concatenate([yv, np.array(jointstate.velocity)[None, :]], axis = 0)
            name_list = jointstate.name

        self.joint_list = msg.jointstates[0].name

        self.cs = scipy.interpolate.CubicSpline(x, yr, axis = 0)
        self.cs_vel = scipy.interpolate.CubicSpline(x, yv, axis = 0)

        st = time.time()
        set_points = self.cs(st)
        set_vel = self.cs_vel(st)

        joint_state = JointState()
        joint_state.name = JOINT_LIST2


        pos_list = np.zeros([len(JOINT_LIST2)])
        for c in range(len(name_list)):
            name = name_list[c]
            pos_list[JOINT_LIST2.index(name)] = set_points[c]
        joint_state.position = pos_list
        self.joint_state = joint_state
        now = self.get_clock().now()
        joint_state.header.stamp = now.to_msg()
        self.joint_state_pub.publish(joint_state)





def main(args=None):
    rclpy.init(args=args)

    hrpid = dummy_controller()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()