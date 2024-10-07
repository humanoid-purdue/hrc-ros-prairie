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
              'right_one_joint',
              'right_two_joint',
              'right_three_joint',
              'right_four_joint',
              'right_five_joint',
              'right_six_joint']



class joint_trajectory_pd_controller(Node):

    def __init__(self):
        super().__init__('joint_trajectory_pd_controller')

        self.cs = None
        self.joint_list = None
        self.joint_state = None
        self.js_time = 0
        self.prev_delta = np.zeros([len(JOINT_LIST)])
        self.prev_time = time.time()
        pid_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/pid_config.yaml")
        with open(pid_config_path, 'r') as infp:
            pid_txt = infp.read()
        self.pd = load(pid_txt, Loader = Loader)['g1_gazebo']

        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.pd_callback,
            10)
        self.subscription_pos = self.create_subscription(
            JointTrajectoryST,
            'joint_trajectory_desired',
            self.joint_traj_callback,
            10
        )


        self.freq = 1000


    def joint_traj_callback(self, msg):
        x = np.array(msg.timestamps)
        y = None
        for jointstate in msg.jointstates:
            if y is None:
                y = np.array(jointstate.position)[None, :]
            else:
                y = np.concatenate([y, np.array(jointstate.position)[None, :]], axis = 0)
        self.joint_list = msg.jointstates[0].name

        self.cs = scipy.interpolate.CubicSpline(x, y, axis = 0)



    def pd_callback(self, msg):
        joint_traj = JointTrajectory()
        jtp = JointTrajectoryPoint()
        duration = Duration()
        st = time.time()
        now = self.get_clock().now()

        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST

        if self.cs is not None:
            set_points = self.cs(st)
        p = 200
        d = 0.01
        delta = np.zeros([len(JOINT_LIST)])

        for c in range(len(JOINT_LIST)):
            index = msg.name.index(JOINT_LIST[c])
            cp = msg.position[index]
            if self.cs is not None and self.joint_list is not None and JOINT_LIST[c] in self.joint_list:
                tpos = set_points[self.joint_list.index(JOINT_LIST[c])]
                delta[c] = tpos - cp
            else:
                delta[c] = -1 * cp
        if (st - self.prev_time) > 0:
            deriv = (delta - self.prev_delta) / (st - self.prev_time)
        else:
            deriv = (delta - self.prev_delta)
        self.prev_delta = delta.copy()
        efforts = np.zeros([len(JOINT_LIST)])
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c][:-5] + "controller"
            if name in self.pd.keys():
                p = self.pd[name]['pid']['p']
                d = self.pd[name]['pid']['d']
            else:
                p = 100
                d = 5
            efforts[c] = p * delta[c] + deriv[c] * d


        jtp.effort = efforts
        duration.sec = 9999
        duration.nanosec = 0
        jtp.time_from_start = duration

        joint_traj.points = [jtp]

        self.joint_traj_pub.publish(joint_traj)
        self.prev_time = st



def main(args=None):
    rclpy.init(args=args)

    hrpid = joint_trajectory_pd_controller()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()