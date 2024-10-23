import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import JointTrajectoryST
from hrc_msgs.msg import StateVector
from ament_index_python.packages import get_package_share_directory

import numpy as np
import scipy
import time
import os, sys

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

_, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

class joint_trajectory_pd_controller(Node):

    def __init__(self):
        super().__init__('joint_trajectory_pd_controller')

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
        self.pos_t_filt = helpers.SignalFilter(len(JOINT_LIST), 1000, 100)
        self.vel_t_filt = helpers.SignalFilter(len(JOINT_LIST), 1000, 100)
        self.tau_t_filt = helpers.SignalFilter(len(JOINT_LIST), 1000, 70)

        self.efforts_t_filt = helpers.SignalFilter(len(JOINT_LIST), 1000, 200)

        self.csv_dump = helpers.CSVDump(6, ["apos","avel","dpos","dvel"])

        self.csv_dump2 = helpers.CSVDump(20, ['times', 'mpos'])

        self.csv_dump3 = helpers.CSVDump(2, ['tap'])

        self.csv_dump4 = helpers.CSVDump(10, ['t2', 'p2'])
        self.pos2 = None


        self.timer = self.create_timer(1, self.timer_callback)

        self.dip = helpers.discreteIntegral(len(JOINT_LIST))
        self.div = helpers.discreteIntegral(len(JOINT_LIST))

    def timer_callback(self):
        self.csv_dump.save()
        self.csv_dump2.save()
        self.csv_dump3.save()
        self.csv_dump4.save()
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

        if np.mean(np.abs((new_pos[:, 0] - self.pos2[0]))) < 0.5:
            self.ji.forceUpdateState(x, new_pos, new_vel, new_tau)

        self.csv_dump4.update([x, new_pos[:, 0]])

        #make joint trajectory
        dt = 0.003
        initial_time = self.sim_time

        #self.get_logger().info("{} {} {}".format(x[0], x[-1], initial_time))

        joint_traj = JointTrajectory()
        stamp = Time()
        stamp.sec = 0
        stamp.nanosec = 1
        joint_traj.header.stamp = stamp
        joint_traj.joint_names = JOINT_LIST
        jtps = []
        mpos = np.zeros([20])
        times = np.arange(20) * dt + initial_time
        for c in range(20):
            jtp, pos_tf, vel_tf = self.make_jtp(initial_time, c * dt)
            jtps += [jtp]
            mpos[c] = pos_tf[0]

        self.csv_dump2.update([times, mpos])

        if self.pos2 is not None:
            self.csv_dump3.update([np.array([initial_time, self.pos2[0]])])

        joint_traj.points = jtps
        self.joint_traj_pub.publish(joint_traj)


    def reorder(self, name_arr, vec):
        actual = np.zeros([len(JOINT_LIST)])
        for c in range(len(name_arr)):
            if name_arr[c] in JOINT_LIST:
                actual[JOINT_LIST.index(name_arr[c])] = vec[c]
        return actual

    def make_jtp(self, initial_sim_time, delta_t):
        desired_sim_time = initial_sim_time + delta_t
        if self.ji.hasHistory():
            pos_t, vel_t, tau_t = self.ji.getInterpolation(desired_sim_time, pos_delta = 0.0)
            self.pos_t_filt.update(pos_t)
            self.vel_t_filt.update(vel_t)
            self.tau_t_filt.update(tau_t)
            vel_tf = vel_t
            pos_tf = pos_t
        else:
            pos_tf = np.zeros([len(JOINT_LIST)])
            vel_tf = np.zeros([len(JOINT_LIST)])
            tau_tf = np.zeros([len(JOINT_LIST)])
        jtp = JointTrajectoryPoint()
        duration = Duration()
        jtp.positions = pos_tf
        jtp.velocities = vel_tf
        #jtp.effort = tau_tf
        secs, nsecs = divmod(delta_t, 1)
        duration.sec = int(secs)
        duration.nanosec = int(nsecs * 10 ** 9)
        jtp.time_from_start = duration
        return jtp, pos_tf, vel_tf

    def pd_callback(self, msg):
        self.sim_time = msg.time
        #get joint pos and vel and desired vel and save to txt
        pos_arr = msg.joint_pos
        self.pos2 = msg.joint_pos
        vel_arr = msg.joint_vel
        _, pos_tf, vel_tf = self.make_jtp(self.sim_time, 0.0)
        self.csv_dump.update([pos_arr[0:6], vel_arr[0:6], pos_tf[0:6], vel_tf[0:6]])

def main(args=None):
    rclpy.init(args=args)

    hrpid = joint_trajectory_pd_controller()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()