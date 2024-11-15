from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion, Wrench
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped

from hrc_msgs.msg import StateVector
import numpy as np
import time
from trajectory_msgs.msg import JointTrajectory
import os, sys
from ament_index_python.packages import get_package_share_directory
helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

JOINT_LIST = helpers.makeJointList()[0]

class gz_state_estimator(Node):

    def __init__(self):
        super().__init__('gz_state_estimator')

        qos_profile = QoSProfile(depth=10)

        self.efforts = None
        self.sim_time = 0
        self.prev_time = 0
        self.prev_vel = None
        self.prev_acc = None
        self.prev_pos = None


        self.jvel_filt = helpers.SignalFilter(len(JOINT_LIST) - 6, 1000, 20)
        self.vel_filt = helpers.SignalFilter(3, 1000, 20)
        self.angvel_filt = helpers.SignalFilter(3, 1000, 10)
        self.jpos_filt = helpers.SignalFilter(len(JOINT_LIST) - 6, 1000, 20)


        self.odom_pos = [0., 0., 0.743]
        self.odom_rot = np.array([0., 0., 0., 1.])
        self.ang_vel = np.array([0., 0., 0.])
        self.left_force = np.zeros([3])
        self.right_force = np.zeros([3])

        self.sv_pub = self.create_publisher(StateVector, 'state_vector', qos_profile)

        self.sv_fwd = helpers.SVFwdKinematics()

        self.subscription_1 = self.create_subscription(
            JointState,
            '/joint_states_gz',
            self.joint_state_callback,
            10)
        self.subscription_2 = self.create_subscription(
            Odometry,
            '/robot_odometry',
            self.odometry_callback,
            10)
        self.subscription_3 = self.create_subscription(
            Imu,
            '/pelvis_imu',
            self.imu_callback,
            10)
        self.subscription_4 = self.create_subscription(
            Clock,
            '/sim_clock',
            self.clock_callback, 10
        )
        self.subscription_5 = self.create_subscription(
            JointTrajectory,
            '/joint_trajectories',
            self.effort_callback, 10
        )
        self.subscription_6 = self.create_subscription(
            Wrench,
            '/left_foot_force',
            self.left_foot_callback, 10
        )
        self.subscription_7 = self.create_subscription(
            Wrench,
            '/right_foot_force',
            self.right_foot_callback, 10
        )

    def right_foot_callback(self, msg):
        force = msg.force
        force = np.array([force.x, force.y, force.z])
        self.right_force = force

    def left_foot_callback(self, msg):
        force = msg.force
        force = np.array([force.x, force.y, force.z])
        self.left_force = force

    def effort_callback(self, msg):
        point = msg.points[0]
        self.efforts = dict(zip(msg.joint_names, point.effort))

    def clock_callback(self, msg):
        secs = msg.clock.sec
        nsecs = msg.clock.nanosec
        self.sim_time = secs + nsecs * (10 ** -9)


    def imu_callback(self, msg):
        self.ang_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def odometry_callback(self, msg):
        pose = msg.pose.pose
        self.odom_pos = [pose.position.x, pose.position.y, pose.position.z]
        self.odom_rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


    def joint_state_callback(self, msg):
        sim_time = self.sim_time
        sv = StateVector()
        sv.joint_name = msg.name
        dt = sim_time - self.prev_time

        self.get_logger().info("L{} R{}".format(self.left_force, self.right_force))

        sv.pos = self.odom_pos
        sv.orien_quat = self.odom_rot



        if self.prev_vel is None:
            self.prev_vel = np.array(msg.velocity)
        if self.prev_acc is None:
            self.prev_acc = np.zeros([len(msg.name)])
        if self.prev_pos is None:
            self.prev_pos = self.odom_pos

        if dt == 0:
            acc = self.prev_acc
            vel = np.zeros([3])
        else:
            acc = (np.array(msg.velocity) - self.prev_vel) / dt
            vel = (np.array(self.odom_pos) - np.array(self.prev_pos)) / dt
            self.prev_vel = np.array(msg.velocity)
            self.prev_acc = acc
        sv.joint_acc = acc

        self.vel_filt.update(vel)

        self.jpos_filt.update(np.array(msg.position))
        self.jvel_filt.update(np.array(msg.velocity))
        self.angvel_filt.update(np.array(self.ang_vel))
        if dt != 0 and self.sim_time > 0.1:
            sv.joint_pos = msg.position
            sv.joint_vel = self.jvel_filt.get()
            sv.vel = self.vel_filt.get()
            sv.ang_vel = self.angvel_filt.get()
        elif self.sim_time < 0.05:
            sv.vel = np.zeros([3])
            sv.ang_vel = np.zeros([3])
            sv.joint_pos = msg.position
            sv.joint_vel = np.zeros([len(msg.velocity)])
        else:
            sv.joint_pos = msg.position
            sv.joint_vel = msg.velocity
            sv.vel = vel
            sv.ang_vel = self.ang_vel

        sv = self.sv_fwd.update(sv)


        new_efforts = np.zeros([len(msg.name)])
        if self.efforts is not None:
            for c in range(len(msg.name)):
                jn = msg.name[c]
                if jn == self.efforts.keys():
                    new_efforts[c] = self.efforts[jn]
        sv.efforts = new_efforts
        sv.time = sim_time
        self.prev_time = sim_time
        self.prev_pos = self.odom_pos
        self.sv_pub.publish(sv)

def main():
    rclpy.init(args=None)
    node = gz_state_estimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()