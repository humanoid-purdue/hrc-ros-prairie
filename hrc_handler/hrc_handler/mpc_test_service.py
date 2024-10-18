import time
import numpy as np
import crocoddyl
import pinocchio
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
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
from hrc_msgs.srv import MPCTest
import os
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
import scipy
import sys
from ament_index_python.packages import get_package_share_directory

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
from helpers import BipedalPoser, SquatSM



JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']



class MPCTestSrv(Node):

    def __init__(self):
        super().__init__('minimal_service')

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST, LEG_JOINTS, "left_ankle_roll_link",
                                  "right_ankle_roll_link")
        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))
        self.srv = self.create_service(MPCTest, 'mpc_test', self.mpc_test_callback)

    def mpc_test_callback(self, request, response):
        state_vector = request.state_vector

        names = state_vector.joint_name
        j_pos = state_vector.joint_pos
        state_time = state_vector.time
        j_pos_config = dict(zip(names, j_pos))
        pos = state_vector.pos
        orien = state_vector.orien_quat
        ang_vel = state_vector.ang_vel
        #self.poser.x[7 + len(LEG_JOINTS):] = 0
        timeseries = np.arange(10) * 0.02 + self.state_time
        self.poser.setState(pos, j_pos_config, orien = orien, vel = state_vector.vel ,config_vel = dict(zip(names, state_vector.joint_vel)), ang_vel = ang_vel)
        self.squat_sm.com_pos = np.array([0., 0., 0.65])

        y = self.squat_sm.simpleNextMPC(None)

        #(10, 7 + number of joints)
        self.joint_trajst_publish(response, timeseries, y)


        return response

    def r2whole(self, joint_dict):
        pos_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in joint_dict.keys():
                pos_list[c] = joint_dict[name]
        return pos_list

    def joint_trajst(self, joint_traj_desired, timestamps, y):
        js_list = []
        joint_traj_desired.timestamps = timestamps
        for x0 in y:
            pos, joint_dict, joint_vels = self.poser.getJointConfig(x0)
            js = JointState()
            js.name = JOINT_LIST
            pos_list = self.r2whole(joint_dict)
            vel_list = self.r2whole(joint_vels)

            js0 = JointState()
            js0.name = JOINT_LIST
            js0.position = pos_list
            js0.velocity = vel_list
            js_list += [js0]
        joint_traj_desired.jointstates = js_list

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(MPCTest, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = MPCTest.Request()

    def send_request(self):
        self.req.joint_name = None
        self.req.joint_pos = None
        self.req.joint_vel = None

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    mpct = MPCTestSrv()

    rclpy.spin(mpct)

    rclpy.shutdown()

def eclient(args = None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request()


    minimal_client.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()