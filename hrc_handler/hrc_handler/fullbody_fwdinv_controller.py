import rclpy
from numpy.f2py.cfuncs import callbacks
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import StateVector, BipedalCommand
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import scipy
import time
import os, sys

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

        thread_safe = MutuallyExclusiveCallbackGroup()

        p1 = MutuallyExclusiveCallbackGroup()
        p2 = MutuallyExclusiveCallbackGroup()

        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group = thread_safe)

        self.subscription_2 = self.create_subscription(
            BipedalCommand,
            'bipedal_command',
            self.bpc_callback,
            10, callback_group = thread_safe)

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

        self.timer1 = self.create_timer(0.00, self.inverse_callback, callback_group = p1)
        self.timer2 = self.create_timer(0.00, self.joint_trajectory_publisher, callback_group = p2)
        self.timer3 = self.create_timer(1, self.save_callback, callback_group = thread_safe)

        self.csv_dump = helpers.CSVDump(100, ["time_traj", "pos_traj", "vel_traj"])
        self.csv_dump2 = helpers.CSVDump(2, ["timepos_r"])

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
        self.csv_dump2.update([np.array([self.state_time, j_pos_config["left_hip_pitch_joint"]])])

    def bpc_callback(self, msg):
        if self.inverse_joints is None:
            self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
            self.ji_joints = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
        else:
            if msg.inverse_joints != self.inverse_joints:
                self.ji = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
                self.ji_joints = helpers.JointInterpolation(len(msg.inverse_joints), 0.05, 0.5)
        self.inverse_joints = msg.inverse_joints
        self.bipedal_command = msg

    def inverse_callback(self):
        if self.state_dict is not None and self.bipedal_command is not None:
            state_dict = self.state_dict.copy()
            self.poser.updateReducedModel(self.inverse_joints, state_dict["joint_pos"])
            self.poser.setState(state_dict["pos"], state_dict["joint_pos"],
                                orien = state_dict["orien"],
                                vel = state_dict["vel"],
                                ang_vel = state_dict["ang_vel"],
                                config_vel = None) #state_dict["joint_vel"]
            x = None
            timestamps = [self.state_time ] + list(np.array(self.bipedal_command.inverse_timestamps) + self.state_time)
            if self.ji is not None and self.ji.hasHistory():
                x = np.array(self.ji.getSeedX(timestamps))
            y = self.sm.nextMPC(self.bipedal_command.inverse_timestamps, self.bipedal_command.inverse_commands, x)
            b, pos_e, vel_e = self.ji.updateX(timestamps, y)
            joint_pos = np.zeros([len(y), len(self.inverse_joints)])
            joint_vels = np.zeros([len(y), len(self.inverse_joints)])

            for xi, i in zip(y, range(len(y))):
                pos, orien, joint_dict, joint_vel, joint_efforts = self.poser.getJointConfig(xi)
                joint_pos[i, :] = np.array(list(joint_dict.values()))
                joint_vels[i, :] = np.array(list(joint_vel.values()))
                self.ji_joint_name = list(joint_dict.keys())


            state_pos = np.array(list(state_dict["joint_pos"].values()))

            if np.mean(np.abs((joint_pos[:, 0] - state_pos[0]))) < 0.5:
                #self.ji_joints.updateMixState(self.state_time, timestamps, joint_pos, joint_vels)
                self.ji_joints.forceUpdateState(timestamps[1:], joint_pos[1:,:], joint_vels[1:,:])



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

        self.get_logger().info("timestamps 2: {}".format(time.time() - self.prev_time))
        self.prev_time = time.time()
        return forward_names, forward_pos_traj, forward_vel_traj

    def joint_trajectory_publisher(self):
        if self.state_time == None:
            return
        timestamps = self.state_time + np.arange(100) * 0.001
        forward_names, forward_pos_traj, forward_vel_traj = self.forward_cmds(timestamps)
        if self.ji_joints is not None and self.ji_joints.hasHistory():
            pos_t, vel_t = self.ji_joints.getInterpolation(timestamps)
            inv_names = self.ji_joint_name
            self.csv_dump.update([timestamps, pos_t[:, 0], vel_t[:, 0]])
        else:
            pos_t = np.zeros([100,0])
            vel_t = np.zeros([100,0])
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
    executor = MultiThreadedExecutor()
    executor.add_node(fb)

    executor.spin()

    fb.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()