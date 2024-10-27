import rclpy
from numpy.f2py.cfuncs import callbacks
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import StateVector, BipedalCommand, JointTrajectoryST
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


class fullbody_fwdinv_controller(Node):
    def __init__(self):
        super().__init__("fullbody_fwdinv_controller")
        qos_profile = QoSProfile(depth=10)

        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.joint_list, self.joint_movable, self.leg_joints = helpers.makeJointList()

        p0 = MutuallyExclusiveCallbackGroup()

        p1 = MutuallyExclusiveCallbackGroup()
        p2 = MutuallyExclusiveCallbackGroup()
        p4 = MutuallyExclusiveCallbackGroup()

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

        self.subscription_3 = self.create_subscription(
            JointTrajectoryST,
            'inv_joint_traj',
            self.inv_callback,
            10, callback_group=None)

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.poser = helpers.BipedalPoser(urdf_config_path, self.joint_list, self.leg_joints, "left_ankle_roll_link",
                                  "right_ankle_roll_link")
        self.sm = helpers.SimpleFwdInvSM(self.poser)

        self.state_dict = None
        self.state_time = None
        self.bipedal_command = None
        self.torque_filter = None

        self.ji = None
        self.ji_joints = None
        self.inverse_joints = None
        self.ji_joint_name = None


        self.inverse_joint_states = None

        self.prev_time = time.time()
        self.prev_time2 = time.time()

        self.timer2 = self.create_timer(0.001, self.joint_trajectory_publisher, callback_group = p0)
        self.timer3 = self.create_timer(1, self.save_callback, callback_group = None)

        self.csv_dump = helpers.CSVDump(10, ["time_traj", "pos_traj", "vel_traj", "tau_traj", "tau_raw"])
        self.csv_dump2 = helpers.CSVDump(2, ["timepos_r", "timevel_r"])


    def save_callback(self):
        self.get_logger().info("Savetxt")
        self.csv_dump.save()
        self.csv_dump2.save()

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
        self.csv_dump2.update([np.array([self.state_time, j_pos_config["left_ankle_pitch_joint"]]), np.array([self.state_time, j_vel_config["left_ankle_pitch_joint"]])])

    def bpc_callback(self, msg):
        if self.inverse_joints is None:
            self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
            self.ji_joints = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
            self.torque_filter = helpers.SignalFilter(len(msg.inverse_joints), 1000, 5)
        else:
            if msg.inverse_joints != self.inverse_joints:
                self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
                self.ji_joints = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
                self.torque_filter = helpers.SignalFilter(len(msg.inverse_joints), 1000, 5)
        self.inverse_joints = msg.inverse_joints
        self.bipedal_command = msg

    def inv_callback(self, msg):
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
                yr = np.concatenate([yr, np.array(jointstate.position)[None, :]], axis=0)
                yv = np.concatenate([yv, np.array(jointstate.velocity)[None, :]], axis=0)
                yt = np.concatenate([yt, np.array(jointstate.effort)[None, :]], axis=0)
            self.ji_joint_name = jointstate.name
        yt[ np.abs(yt) > 10] = 0
        if self.ji_joints is not None:
            #self.ji_joints.forceUpdateState(x, yr, yv, yt)
            self.ji_joints.updateMixState(self.state_time, x, yr, yv, yt)


    def forward_cmds(self, new_timestamps):
        forward_names = []
        timestamps = self.bipedal_command.forward_timestamps
        if self.bipedal_command is None:
            inv_joints = []
        else:
            inv_joints = self.bipedal_command.inverse_joints
        js_list = self.bipedal_command.forward_commands
        if len(js_list) == 0:
            js_names = []
        else:
            js_names = js_list[0].name
        #make mapping between forward names and js_list
        mapping_js = []
        mapping_fwd = []
        c = 0
        for joint in self.joint_movable:
            if joint not in inv_joints:
                forward_names += [joint]
                if joint in js_names:
                    mapping_js += [js_names.index(joint)]
                    mapping_fwd += [c]
                c += 1

        defined_pos_traj = np.zeros([len(js_list), len(js_names)])
        defined_vel_traj = np.zeros([len(js_list), len(js_names)])
        for c in range(len(js_list)):
            defined_pos_traj[c, :] = np.array(js_list[c].position)
            defined_vel_traj[c, :] = np.array(js_list[c].velocity)

        if len(js_list) != 0:
            cs_pos = scipy.interpolate.CubicSpline(timestamps, defined_pos_traj, axis=0)
            cs_vel = scipy.interpolate.CubicSpline(timestamps, defined_vel_traj, axis=0)
            defined_pos = cs_pos(new_timestamps)
            defined_vel = cs_vel(new_timestamps)
        else:
            defined_pos = np.zeros([100, 0])
            defined_vel = np.zeros([100, 0])

        forward_pos_traj = np.zeros([100, len(forward_names)])
        forward_vel_traj = np.zeros([100, len(forward_names)])

        forward_pos_traj[:, mapping_fwd] = defined_pos[:, mapping_js]
        forward_vel_traj[:, mapping_fwd] = defined_vel[:, mapping_js]

        #define vels for all
        for name, i in zip(forward_names, range(len(forward_names))):
            if name not in js_names:
                if self.state_dict is not None:
                    pos, vel = helpers.makeFwdTraj(self.state_dict["joint_pos"][name], 0.0)
                    forward_pos_traj[:, i] = pos
                    forward_vel_traj[:, i] = vel

        return forward_names, forward_pos_traj, forward_vel_traj

    def joint_trajectory_publisher(self):
        if self.state_time == None:
            return
        timestamps = self.state_time + np.arange(10) * 0.005
        forward_names, forward_pos_traj, forward_vel_traj = self.forward_cmds(timestamps)
        if self.ji_joints is not None and self.ji_joints.hasHistory():
            pos_t, vel_t, tau_t = self.ji_joints.getInterpolation(timestamps)
            inv_names = self.ji_joint_name
            if self.torque_filter is not None:
                self.torque_filter.update(tau_t[0, :])
                tau2 = tau_t.copy()
                tau_filt = self.torque_filter.get()
                #tau_t = np.tile(tau_filt[None, :], [pos_t.shape[0], 1])
            self.csv_dump.update([timestamps, pos_t[:, 5], vel_t[:, 5], tau_t[:, 5], tau_t[:, 5]])
        else:
            pos_t = np.zeros([10,0])
            vel_t = np.zeros([10,0])
            tau_t = np.zeros([10,0])
            inv_names = []


        joint_traj = JointTrajectory()
        stamp = Time()
        stamp.sec = 0
        stamp.nanosec = 1
        joint_traj.header.stamp = stamp
        joint_traj.joint_names = inv_names + forward_names
        jtps = []
        for c in range(timestamps.shape[0]):
            jtp = JointTrajectoryPoint()
            duration = Duration()
            jtp.positions = list(pos_t[c, :]) + list(forward_pos_traj[c,:])
            jtp.velocities = list(vel_t[c, :]) + list(forward_vel_traj[c,:])
            jtp.effort = list(tau_t[c, :] * 1.05) + list(np.zeros(forward_vel_traj[c,:].shape))
            # jtp.effort = tau_tf
            secs, nsecs = divmod(timestamps[c] - self.state_time, 1)
            duration.sec = int(secs)
            duration.nanosec = int(nsecs * 10 ** 9)
            jtp.time_from_start = duration
            jtps += [jtp]
        joint_traj.points = jtps

        self.get_logger().info("timestamps 1: {}".format(time.time() - self.prev_time))
        self.prev_time = time.time()
        self.joint_traj_pub.publish(joint_traj)

def main():
    rclpy.init(args=None)

    fb = fullbody_fwdinv_controller()
    executor = SingleThreadedExecutor()
    executor.add_node(fb)

    executor.spin()

    fb.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()