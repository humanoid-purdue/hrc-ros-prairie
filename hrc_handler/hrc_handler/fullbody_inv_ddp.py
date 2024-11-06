import rclpy
from numpy.f2py.cfuncs import callbacks
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import StateVector, BipedalCommand, JointTrajectoryST
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import scipy
import time
import os, sys
from sensor_msgs.msg import JointState

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers


class fullbody_inv_ddp(Node):
    def __init__(self):
        super().__init__("fullbody_inv_ddp")
        qos_profile = QoSProfile(depth=10)

        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'inv_joint_traj', qos_profile)
        self.joint_list, self.joint_movable, self.leg_joints = helpers.makeJointList()

        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group = None)

        self.subscription_2 = self.create_subscription(
            BipedalCommand,
            'bipedal_command',
            self.bpc_callback,
            10, callback_group = None)

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.poser = helpers.BipedalPoser(urdf_config_path, self.joint_list, self.leg_joints, "left_ankle_roll_link",
                                  "right_ankle_roll_link")
        self.sm = helpers.SimpleFwdInvSM(self.poser)

        self.state_dict = None
        self.state_time = None
        self.bipedal_command = None

        self.ji = None
        self.ji_joints = None
        self.inverse_joints = None
        self.ji_joint_name = None

        self.inverse_joint_states = None

        self.prev_time = time.time()
        self.prev_time2 = time.time()

        self.timer1 = self.create_timer(0.001, self.inverse_callback, callback_group = None)

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
        self.state_dict = {"pos": pos, "orien": orien, "vel": vel, "ang_vel": ang_vel, "joint_pos": j_pos_config, "joint_vel": j_vel_config}

    def bpc_callback(self, msg):
        if self.inverse_joints is None:
            self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
        else:
            if msg.inverse_joints != self.inverse_joints:
                self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
        self.inverse_joints = msg.inverse_joints
        self.bipedal_command = msg

    def inverse_callback(self):
        if self.state_dict is not None and self.bipedal_command is not None:
            state_dict = self.state_dict.copy()
            self.poser.updateReducedModel(self.inverse_joints, state_dict["joint_pos"])
            self.poser.setState(state_dict["pos"], state_dict["joint_pos"],
                                orien = state_dict["orien"],
                                vel = state_dict["vel"],
                                ang_vel = None,
                                config_vel = state_dict["joint_vel"]) #state_dict["joint_vel"]
            x = None
            timestamps = [self.state_time ] + list(np.array(self.bipedal_command.inverse_timestamps) + self.state_time)
            if self.ji is not None and self.ji.hasHistory():
                x = np.array(self.ji.getSeedX(timestamps))
                x[0, :] = self.poser.x.copy()
            y, tau = self.sm.nextMPC(self.bipedal_command.inverse_timestamps, self.bipedal_command.inverse_commands, x)
            self.get_logger().info("uninterp y {}".format(y[1][6 + len(self.leg_joints):]))
            b, pos_e, vel_e = self.ji.updateX(timestamps, y)
            joint_pos = np.zeros([len(y), len(self.inverse_joints)])
            joint_vels = np.zeros([len(y), len(self.inverse_joints)])
            joint_taus = np.zeros([len(y), len(self.inverse_joints)])
            #self.get_logger().info("{}".format(tau[0,:]))
            pos_arr = np.zeros([len(y), 3])
            orien_arr = np.zeros([len(y), 4])

            for xi, i in zip(y, range(len(y))):

                if i == 0:
                    t2 = tau[0, :]
                else:
                    t2 = tau[i - 1, :]
                pos, orien, joint_dict, joint_vel, joint_efforts = self.poser.getJointConfig(xi, efforts = t2)
                self.ji_joint_name = list(joint_dict.keys())
                pos_arr[i, :] = np.array(pos)
                orien_arr[i, :] = np.array(orien)
                for name, j in zip(self.ji_joint_name, range(len(self.inverse_joints))):
                    joint_pos[i, j] = joint_dict[name]
                    joint_vels[i, j] = joint_vel[name]
                    joint_taus[i, j] = joint_efforts[name]
            #self.get_logger().info("timestamps inv: {}".format(time.time() - self.prev_time2))
            self.prev_time2 = time.time()

            jts = JointTrajectoryST()
            jts.timestamps = list(timestamps[:])
            js_list = []
            pose_list = []
            for c in range(joint_pos.shape[0]):
                js = JointState()
                js.name = self.ji_joint_name
                js.position = list(joint_pos[c, :])
                js.velocity = list(joint_vels[c, :])
                js.effort = list(joint_taus[c, :])
                js_list += [js]

                pose = Pose()
                point = Point()
                quat = Quaternion()
                point.x = pos_arr[c, 0]
                point.y = pos_arr[c, 1]
                point.z = pos_arr[c, 2]
                quat.x = orien_arr[c, 0]
                quat.y = orien_arr[c, 1]
                quat.z = orien_arr[c, 2]
                quat.w = orien_arr[c, 3]

                pose.position = point
                pose.orientation = quat
                pose_list += [pose]

            jts.jointstates = js_list
            jts.rootpose = pose_list
            self.joint_traj_pub.publish(jts)

            lpos, rpos, com_pos = self.poser.getPos(None)

            #self.get_logger().info("{} {} {}".format(lpos, rpos, com_pos))

def main():
    rclpy.init(args=None)

    fb = fullbody_inv_ddp()
    executor = SingleThreadedExecutor()
    executor.add_node(fb)

    executor.spin()

    fb.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()