import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64
from hrc_msgs.msg import JointTrajectoryST
from ament_index_python.packages import get_package_share_directory
from yaml import load, dump
from yaml import Loader, Dumper
import numpy as np
import scipy
import time
import os
import os, sys
helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

_, JOINT_LIST, _ = helpers.makeJointList()



class joint_traj_pos(Node):

    def __init__(self):
        super().__init__('joint_traj_pos')

        self.cs = None
        self.cs_vel = None
        self.joint_list = None
        self.joint_state = None
        self.js_time = 0
        self.prev_vel = np.zeros([len(JOINT_LIST)])
        self.prev_time = time.time()
        pid_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/pid_config.yaml")
        with open(pid_config_path, 'r') as infp:
            pid_txt = infp.read()
        self.pd = load(pid_txt, Loader = Loader)['g1_gazebo']

        self.integral = np.zeros([len(JOINT_LIST)])

        qos_profile = QoSProfile(depth=10)
        #self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.joint_pos_dict = {}
        for joint in JOINT_LIST:
            name = joint + "_cmd_pos"
            self.joint_pos_dict[joint] = self.create_publisher(Float64, name, qos_profile)


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
        #self.anti_torque_factor = 0.005
        self.anti_torque_factor = 0.0

        self.freq = 1000


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
        self.joint_list = msg.jointstates[0].name

        def cs_dummy(t):
            return yr[0,:], x[0]
        def cs_v_dummy(t):
            return yv[0,:]

        #self.cs = scipy.interpolate.CubicSpline(x, yr, axis = 0)
        #self.cs_vel = scipy.interpolate.CubicSpline(x, yv, axis = 0)

        self.cs = cs_dummy
        self.cs_vel = cs_v_dummy



    def pd_callback(self, msg):
        joint_traj = JointTrajectory()

        st = time.time()
        now = self.get_clock().now()

        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST

        if self.cs is not None:
            set_points, t = self.cs(st)
            set_vel = self.cs_vel(st)

        delta = np.zeros([len(JOINT_LIST)])
        efforts = np.zeros([len(JOINT_LIST)])

        positions = np.zeros([len(JOINT_LIST)])
        velocity = np.zeros([len(JOINT_LIST)])

        for c in range(len(JOINT_LIST)):
            index = msg.name.index(JOINT_LIST[c])
            cp = msg.position[index]
            dt = st - self.prev_time
            if dt == 0:
                dt = 1
            acc = (msg.velocity[index] - self.prev_vel[index]) / dt
            vel = msg.velocity[index]

            if self.cs is not None and self.joint_list is not None and JOINT_LIST[c] in self.joint_list:
                tpos = set_points[self.joint_list.index(JOINT_LIST[c])]
                tvel = set_vel[self.joint_list.index(JOINT_LIST[c])]
                future = t - time.time()
                actual = 1/1000
                delta = tpos - cp
                if future != 0:
                    delta = delta * (actual / future)
                else:
                    delta = delta * 0.01
                tpos = cp + delta
            else:
                tpos = 0
                tvel = 0

            pos_msg = Float64()
            pos_msg.data = tpos

            self.joint_pos_dict[JOINT_LIST[c]].publish(pos_msg)

        self.prev_vel = np.array(msg.velocity)
        self.prev_time = st



def main(args=None):
    rclpy.init(args=args)

    hrpid = joint_traj_pos()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()