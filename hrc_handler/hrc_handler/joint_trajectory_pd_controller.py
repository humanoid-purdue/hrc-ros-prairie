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

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

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
        self.grav_comp = np.zeros([300, len(JOINT_LIST)])
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
        self.subscription_grav = self.create_subscription(
            JointState,
            'joint_grav',
            self.grav_callback,
            10
        )
        #self.anti_torque_factor = 0.005
        self.anti_torque_factor = 0.0

        self.freq = 1000

        self.prev_effort = np.zeros([len(JOINT_LIST)])

        self.plot_bag = np.zeros([20000, len(LEG_JOINTS), 4])

        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        desired_pos = self.plot_bag[:,:,0]
        desired_vel = self.plot_bag[:,:,1]
        actual_pos = self.plot_bag[:,:,2]
        actual_vel = self.plot_bag[:,:,3]
        np.savetxt("/home/aurum/RosProjects/prairie/datadump/desired_pos.csv", desired_pos, delimiter = ',')
        np.savetxt("/home/aurum/RosProjects/prairie/datadump/desired_vel.csv", desired_vel, delimiter=',')
        np.savetxt("/home/aurum/RosProjects/prairie/datadump/actual_pos.csv", actual_pos, delimiter=',')
        np.savetxt("/home/aurum/RosProjects/prairie/datadump/actual_vel.csv", actual_vel, delimiter=',')
        self.get_logger().info("Saved to csv")

    def bagControlData(self, name_list, desired_pos, desired_vel, actual_pos, actual_vel):
        v1 = np.zeros([len(LEG_JOINTS), 4])
        for c in range(len(LEG_JOINTS)):
            name = LEG_JOINTS[c]
            i = name_list.index(name)
            v1[c, 0] = desired_pos[i]
            v1[c, 1] = desired_vel[i]
            v1[c, 2] = actual_pos[i]
            v1[c, 3] = actual_vel[i]
        self.plot_bag = np.roll(self.plot_bag, 1, axis = 0)
        self.plot_bag[0, :, :] = v1



    def grav_callback(self, msg):
        names = msg.name
        efforts = msg.effort
        self.grav_comp = np.roll(self.grav_comp, 1, axis = 0)
        for c in range(len(JOINT_LIST)):
            index = names.index(JOINT_LIST[c])
            self.grav_comp[0,c] = efforts[index]



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

    def velFilter(self, series):
        b, a = scipy.signal.butter(2, 0.01, btype='low', analog=False)
        y = scipy.signal.filtfilt(b, a, series)
        return y



    def pd_callback(self, msg):
        name_arr = msg.joint_name
        pos_arr = msg.joint_pos
        vel_arr = msg.joint_vel
        sim_time = msg.time

        joint_traj = JointTrajectory()

        now = self.get_clock().now()

        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST

        dt = sim_time - self.prev_time
        if dt != 0:
            self.get_logger().info("{} Sim freq".format(1/dt))

        if self.cs is not None:
            set_points = self.cs(time.time())
            set_vel = self.cs_vel(time.time())

        efforts = np.zeros([len(JOINT_LIST)])

        positions = np.zeros([len(JOINT_LIST)])
        velocity = np.zeros([len(JOINT_LIST)])

        actual_pos = np.zeros([len(JOINT_LIST)])
        actual_vel = np.zeros([len(JOINT_LIST)])

        delta = np.mean(self.grav_comp, axis = 0)


        for c in range(len(JOINT_LIST)):
            index = name_arr.index(JOINT_LIST[c])
            cp = pos_arr[index]
            vel = vel_arr[index]

            if self.cs is not None and self.joint_list is not None and JOINT_LIST[c] in self.joint_list:
                tpos = set_points[self.joint_list.index(JOINT_LIST[c])]
                tvel = set_vel[self.joint_list.index(JOINT_LIST[c])]
            else:
                tpos = 0
                tvel = 0

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
            #control = (pd_delta + i * self.integral[c])
            #if JOINT_LIST[c] == "left_hip_pitch_joint":
             #   print(pd_delta, self.grav_comp[c])
            control = pd_delta
            #control = delta[c] * -1
            control = min(max(control, -90), 90)
            efforts[c] = control
            positions[c] = tpos
            velocity[c] = tvel
            actual_pos[c] = cp
            actual_vel[c] = vel

        self.bagControlData(JOINT_LIST, positions, velocity, actual_pos, actual_vel)

        if dt != 0:
            self.prev_effort = np.array(efforts)
        jtp = JointTrajectoryPoint()
        duration = Duration()
        jtp.effort = efforts
        duration.sec = 0
        duration.nanosec = 0
        jtp.time_from_start = duration

        joint_traj.points = [jtp]

        self.joint_traj_pub.publish(joint_traj)
        self.prev_time = sim_time




def main(args=None):
    rclpy.init(args=args)

    hrpid = joint_trajectory_pd_controller()

    rclpy.spin(hrpid)

    hrpid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()