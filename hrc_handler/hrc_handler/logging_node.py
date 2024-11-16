import rclpy
from std_msgs.msg import String
from numpy.f2py.cfuncs import callbacks
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import StateVector, BipedalCommand, JointTrajectoryST, CentroidalTrajectory
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import scipy
import time
import os, sys
from threading import Thread

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
class logging_node(Node):
    def __init__(self):
        super().__init__('logging_node')
        
        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10
        )

        self.subscription2 = self.create_subscription(
            CentroidalTrajectory,
            'centroidal_trajectory',
            self.zmp_callback,
            10
        )

        self.joint_list, self.joint_movable, self.leg_joints = helpers.makeJointList()

        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=None)
        self.timer2 = self.create_timer(1.5, self.save_callback, callback_group=None)

        self.max_size = 1000

        self.history_dict = {
            "sv_pos": np.zeros([0, 3]),
            "sv_orien": np.zeros([0, 4]),
            "sv_vel": np.zeros([0, 3]),
            "sv_angvel": np.zeros([0, 3]),
            "sv_com_pos": np.zeros([0, 3]),
            "sv_com_vel": np.zeros([0, 3]),
            "sv_com_acc": np.zeros([0, 3]),
            "sv_l_foot_pos": np.zeros([0, 3]),
            "sv_r_foot_pos": np.zeros([0, 3]),
            "sv_timestamps": np.zeros([0]),
            "zmp_com_pos": np.zeros([0, 100, 3]),
            "zmp_timestamps": np.zeros([0, 100])
        }
        for name in self.joint_movable:
            self.history_dict["sv_" + name + "_pos"] = np.zeros([0])
            self.history_dict["sv_" + name + "_vel"] = np.zeros([0])

        self.prev_time = 0
        self.state_time = 0
        self.state_dict = None
        self.zmp_dict = None

        path = helper_path.split('/')[0:-5]
        self.npath = ""
        for str in path:
            self.npath += str + "/"
        self.npath += "datalog/logging_node.npz"


    def concatHistory(self, hist_dict_key, new_value_vec):
        base_arr = self.history_dict[hist_dict_key]
        vec = np.array(new_value_vec)
        if len(base_arr.shape) == 1:
            base_arr = np.concatenate([[vec], base_arr], axis = 0)
            if base_arr.shape[0] > self.max_size:
                base_arr = base_arr[:self.max_size]
        elif len(base_arr.shape) == 2:
            base_arr = np.concatenate([vec[None, :], base_arr], axis = 0)
            if base_arr.shape[0] > self.max_size:
                base_arr = base_arr[:self.max_size, :]
        else:
            base_arr = np.concatenate([vec[None, :, :], base_arr], axis=0)
            if base_arr.shape[0] > self.max_size:
                base_arr = base_arr[:self.max_size, :, :]
        self.history_dict[hist_dict_key] = base_arr

    def timer_callback(self):
        #fill sv items
        dt = self.state_time - self.prev_time
        if dt != 0 and self.state_dict is not None and self.zmp_dict is not None:
            self.concatHistory("sv_pos", self.state_dict["pos"])
            self.concatHistory("sv_vel", self.state_dict["vel"])
            self.concatHistory("sv_orien", self.state_dict["orien"])
            self.concatHistory("sv_angvel", self.state_dict["ang_vel"])
            self.concatHistory("sv_com_pos", self.state_dict["com_pos"])
            self.concatHistory("sv_com_vel", self.state_dict["com_vel"])
            self.concatHistory("sv_com_acc", self.state_dict["com_acc"])
            self.concatHistory("sv_timestamps", self.state_time)

            for name in list(self.state_dict["joint_pos"].keys()):
                self.concatHistory("sv_" + name + "_pos", self.state_dict["joint_pos"][name])
                self.concatHistory("sv_" + name + "_vel", self.state_dict["joint_vel"][name])

            self.concatHistory("zmp_com_pos", self.zmp_dict["com"])
            self.concatHistory("zmp_timestamps", self.zmp_dict["ts"])
        self.prev_time = self.state_time

    def save_callback(self):
        np.savez(self.npath, **self.history_dict)
        self.get_logger().info("saving npz")

    def zmp_callback(self, msg):
        timestamps = msg.timestamps
        com_xyz = np.zeros([len(msg.com_pos), 3])
        for point, i in zip(msg.com_pos, np.arange(len(msg.com_pos))):
            com_xyz[i, 0] = point.x
            com_xyz[i, 1] = point.y
            com_xyz[i, 2] = point.z
        self.zmp_dict = {"ts": timestamps[0:100], "com": com_xyz[0:100, :]}


    def state_vector_callback(self, msg):
        names = msg.joint_name
        j_pos = msg.joint_pos
        j_vel = msg.joint_vel
        self.state_time = msg.time
        j_pos_config = dict(zip(names, j_pos))
        j_vel_config = dict(zip(names, j_vel))
        pos = msg.pos
        orien = msg.orien_quat
        ang_vel = msg.ang_vel
        vel = msg.vel
        self.state_dict = {"pos": pos, "orien": orien, "vel": vel, "ang_vel": ang_vel, "joint_pos": j_pos_config,
                           "joint_vel": j_vel_config,
                           "com_pos": msg.com_pos, "com_vel": msg.com_vel, "com_acc": msg.com_acc,
                           "l_foot_pos": msg.l_foot_pos, "r_foot_pos": msg.r_foot_pos}


def main(args=None):
    rclpy.init(args=args)
    subscriber = logging_node()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()