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
              'right_zero_joint',
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
        self.cs_vel = None
        self.joint_list = None
        self.joint_state = None
        self.js_time = 0
        self.prev_vel = np.zeros([len(JOINT_LIST)])
        self.prev_time = time.time()
        self.grav_comp = np.zeros([len(JOINT_LIST)])
        pid_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/pid_config.yaml")
        with open(pid_config_path, 'r') as infp:
            pid_txt = infp.read()
        self.pd = load(pid_txt, Loader = Loader)['g1_gazebo']

        self.integral = np.zeros([len(JOINT_LIST)])

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
        self.subscription_grav = self.create_subscription(
            JointState,
            'joint_grav',
            self.grav_callback,
            10
        )
        #self.anti_torque_factor = 0.005
        self.anti_torque_factor = 0.0

        self.freq = 1000

    def grav_callback(self, msg):
        names = msg.name
        efforts = msg.effort
        for c in range(len(JOINT_LIST)):
            index = names.index(JOINT_LIST[c])
            self.grav_comp[c] = efforts[index]


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
            return yr[0,:]
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
            set_points = self.cs(st)
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
            else:
                tpos = 0
                tvel = 0
            #tpos = 0
            #tvel = 0

            delta_r = tpos - cp
            delta_v = tvel - vel



            name = JOINT_LIST[c][:-5] + "controller"
            i = 30
            if name in self.pd.keys():
                p = self.pd[name]['pid']['p']
                d = self.pd[name]['pid']['d']
            else:
                p = 100
                d = 5
            pd_delta = p * delta_r + d * delta_v
            self.integral[c] = self.integral[c] + (time.time() - st) * pd_delta
            #control = (pd_delta + i * self.integral[c])
            #if JOINT_LIST[c] == "left_hip_pitch_joint":
             #   print(pd_delta, self.grav_comp[c])
            #control = pd_delta + self.grav_comp[c] * -0.5
            control = self.grav_comp[c] * 1
            control = min(max(control, -90), 90)
            efforts[c] = control
            positions[c] = tpos
            velocity[c] = tvel

        self.prev_vel = np.array(msg.velocity)
        jtp = JointTrajectoryPoint()
        duration = Duration()
        jtp.effort = efforts
        duration.sec = 0
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