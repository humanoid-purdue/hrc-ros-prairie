import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Duration, Time
import time
helper_path = os.path.join(
    get_package_share_directory('hrc_handler'),
    "helpers")
from hrc_msgs.msg import StateVector

sys.path.append(helper_path)
import helpers
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

class ik_pd(Node):
    def __init__(self):
        super().__init__("ik_pd")
        qos_profile = QoSProfile(depth=10)
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        self.state_time = 0.0
        self.state_dict = None
        self.fwd_poser = helpers.ForwardPoser(urdf_config_path, JOINT_LIST, leg_joints=LEG_JOINTS)

        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group=None)

        self.left_se3 = None
        self.right_se3 = None
        self.top_com = None

        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)

        self.timer = self.create_timer(0.0001, self.timer_callback)

        self.dt = 0.01

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

        self.state_dict = {
            "pos": pos,
            "orien": orien,
            "vel": vel,
            "ang_vel": ang_vel,
            "joint_pos": j_pos_config,
            "joint_vel": j_vel_config,
            "com_pos": np.array(msg.com_pos),
            "com_vel": np.array(msg.com_vel)
        }


    def timer_callback(self):

        def stepCOM(com_pos, com_vel, com_target):
            acc_vec = com_target - com_pos
            acc_vec = 6.0 * acc_vec / np.linalg.norm(acc_vec)

            new_pos = com_pos + com_vel * self.dt + 0.5 * self.dt ** 2 * acc_vec
            new_vel = com_vel + self.dt * acc_vec
            return new_pos, new_vel


        joint_traj = JointTrajectory()
        stamp = Time()
        stamp.sec = 0
        stamp.nanosec = 1
        joint_traj.header.stamp = stamp
        joint_traj.joint_names = JOINT_LIST


        if self.state_dict is not None:
            self.fwd_poser.updateReducedQ(self.state_dict["pos"], self.state_dict["orien"], self.state_dict["joint_pos"])
        else:
            return


        if (self.state_time < 0.2):
            self.fwd_poser.updateDataR()
            self.left_se3 = self.fwd_poser.getRLinkSE3("left_ankle_roll_link")
            self.right_se3 = self.fwd_poser.getRLinkSE3("right_ankle_roll_link")
            self.top_com = self.state_dict["com_pos"].copy() - np.array([0., 0., -0.04])

            jtp = JointTrajectoryPoint()
            duration = Duration()
            duration.sec = 0
            duration.nanosec = 0

            jtp.positions = [0] * len(JOINT_LIST)
            jtp.velocities = [0] * len(JOINT_LIST)
            jtp.time_from_start = duration
            joint_traj.points = [jtp]

            self.joint_traj_pub.publish(joint_traj)

        else:
            com_target = self.top_com
            x1, xd1 = stepCOM(self.state_dict["com_pos"], self.state_dict["com_vel"], com_target)
            x2, xd2 = stepCOM(x1, xd1, com_target)
            link_target_dict = {"left_ankle_roll_link": self.left_se3,
                                "right_ankle_roll_link": self.right_se3}

            pos, quaternion, joint_dict_1, q = self.fwd_poser.ikSolver(x1, link_target_dict)
            self.fwd_poser.q_r = q
            _, _, joint_dict_2, _ = self.fwd_poser.ikSolver(x2, link_target_dict)

            jtp = JointTrajectoryPoint()
            duration = Duration()
            duration.sec = 0
            duration.nanosec = 0

            pos_list = [0] * len(JOINT_LIST)
            vel_list = [0] * len(JOINT_LIST)
            for name in joint_dict_1.keys():
                pos_list[JOINT_LIST.index(name)] = joint_dict_1[name]

                vel_list[JOINT_LIST.index(name)] = ( joint_dict_2[name] - joint_dict_1[name] ) / self.dt

            jtp.positions = pos_list
            jtp.velocities = vel_list
            jtp.time_from_start = duration

            joint_traj.points = [jtp]

            self.joint_traj_pub.publish(joint_traj)

def main(args=None):
    rclpy.init(args=args)

    ikt = ik_pd()

    rclpy.spin(ikt)
    ikt.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()