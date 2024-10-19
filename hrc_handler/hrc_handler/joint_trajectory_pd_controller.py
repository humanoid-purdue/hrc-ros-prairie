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
from ament_index_python.packages import get_package_share_directory
from yaml import load, dump
from yaml import Loader, Dumper
import numpy as np
import scipy
import time
import os, sys

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

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

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

class joint_trajectory_pd_controller(Node):

    def __init__(self):
        super().__init__('joint_trajectory_pd_controller')
        pid_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/pid_config.yaml")
        with open(pid_config_path, 'r') as infp:
            pid_txt = infp.read()
        self.pd = load(pid_txt, Loader = Loader)['g1_gazebo']
        self.prev_time = 0

        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)

        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.pd_callback,
            10)
        self.subscription_pos = self.create_subscription(
            JointTrajectoryST,
            'joint_trajectory_desired',
            self.joint_traj_callback,
            10
        )

        self.ji = helpers.JointInterpolation(len(JOINT_LIST), 0.1, 0.1)
        self.pos_t_filt = helpers.SignalFilter(len(JOINT_LIST), 2000, 200)
        self.vel_t_filt = helpers.SignalFilter(len(JOINT_LIST), 2000, 200)
        self.tau_t_filt = helpers.SignalFilter(len(JOINT_LIST), 2000, 100)

        self.efforts_t_filt = helpers.SignalFilter(len(JOINT_LIST), 2000, 200)

        self.csv_dump = helpers.CSVDump(6, ["apos","avel","dpos","dvel", "tau", "efforts"])

        self.timer = self.create_timer(1, self.timer_callback)

        self.dip = helpers.discreteIntegral(len(JOINT_LIST))
        self.div = helpers.discreteIntegral(len(JOINT_LIST))

    def timer_callback(self):
        self.csv_dump.save()
        self.get_logger().info("Saved txt")

    def joint_traj_callback(self, msg):
        x = np.array(msg.timestamps)
        yr = None
        yv = None
        yt = None
        for jointstate in msg.jointstates:
            if yr is None:
                yr = np.array(jointstate.position)[None, :]
                yv = np.array(jointstate.velocity)[None, :]
                yt = np.array(jointstate.effort)[None, :]
            else:
                yr = np.concatenate([yr, np.array(jointstate.position)[None, :]], axis = 0)
                yv = np.concatenate([yv, np.array(jointstate.velocity)[None, :]], axis = 0)
                yt = np.concatenate([yt, np.array(jointstate.effort)[None, :]], axis = 0)
        joint_list = msg.jointstates[0].name
        new_pos = np.zeros([x.shape[0], len(JOINT_LIST)])
        new_vel = np.zeros([x.shape[0], len(JOINT_LIST)])
        new_tau = np.zeros([x.shape[0], len(JOINT_LIST)])
        for c in range(len(joint_list)):
            if joint_list[c] in JOINT_LIST:
                new_pos[:, JOINT_LIST.index(joint_list[c])] = yr[:, c]
                new_vel[:, JOINT_LIST.index(joint_list[c])] = yv[:, c]
                new_tau[:, JOINT_LIST.index(joint_list[c])] = yt[:, c]
        self.ji.forceUpdateState(x, new_pos, new_vel, new_tau)

    def reorder(self, name_arr, vec):
        actual = np.zeros([len(JOINT_LIST)])
        for c in range(len(name_arr)):
            if name_arr[c] in JOINT_LIST:
                actual[JOINT_LIST.index(name_arr[c])] = vec[c]
        return actual

    def pd_callback(self, msg):
        name_arr = msg.joint_name
        pos_arr = msg.joint_pos
        vel_arr = msg.joint_vel
        pos_arr = self.reorder(name_arr, np.array(pos_arr))
        vel_arr = self.reorder(name_arr, np.array(vel_arr))
        sim_time = msg.time

        #self.vel_filt.update(np.array(vel_arr))


        if self.ji.hasHistory():
            pos_t, vel_t, tau_t = self.ji.getInterpolation(sim_time, pos_delta = 0.0)
            self.pos_t_filt.update(pos_t)
            self.vel_t_filt.update(vel_t)
            self.tau_t_filt.update(tau_t)
            pos_tf = self.pos_t_filt.get()
            vel_tf = self.vel_t_filt.get()
            tau_tf = self.tau_t_filt.get()
        else:
            pos_tf = np.zeros([len(JOINT_LIST)])
            vel_tf = np.zeros([len(JOINT_LIST)])
            tau_tf = np.zeros([len(JOINT_LIST)])


        efforts = np.zeros([len(JOINT_LIST)])

        delta_r = pos_tf - pos_arr
        delta_v = vel_tf - vel_arr
        pi = self.dip.update(sim_time, delta_r)
        di = self.div.update(sim_time, delta_v)

        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c][:-5] + "controller"
            if name in self.pd.keys():
                p = self.pd[name]['pid']['p']
                d = self.pd[name]['pid']['d']
            else:
                p = 100
                d = 5
            if JOINT_LIST[c] in LEG_JOINTS:
                p = 10000
                d = 5
            pd_delta = p * delta_r[c] + d * delta_v[c] + tau_tf[c] * 0.5
            control = pd_delta
            control = min(max(control, -90), 90)
            efforts[c] = control


        self.efforts_t_filt.update(efforts)
        jtp = JointTrajectoryPoint()
        duration = Duration()
        effort_tf = self.efforts_t_filt.get()
        jtp.effort = effort_tf
        duration.sec = 0
        duration.nanosec = 0
        jtp.time_from_start = duration

        joint_traj = JointTrajectory()
        now = self.get_clock().now()
        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST

        joint_traj.points = [jtp]

        self.joint_traj_pub.publish(joint_traj)
        self.prev_time = sim_time

        if sim_time != 0:
            self.csv_dump.update([pos_arr[0:6], vel_arr[0:6], pos_tf[0:6], vel_tf[0:6], tau_tf[0:6], effort_tf[0:6]])




def main(args=None):
    rclpy.init(args=args)

    hrpid = joint_trajectory_pd_controller()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()