from email.utils import encode_rfc2231

import rclpy
from rclpy.node import Node
import numpy as np
from hrc_msgs.msg import StateVector, BipedalCommand, InverseCommand
from geometry_msgs.msg import Point, Pose, Quaternion
from ament_index_python.packages import get_package_share_directory
import os, sys
import time
helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

JOINT_LIST_FULL, JOINT_LIST, _ = helpers.makeJointList()


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(StateVector, 'InitialStateVector', 10)
        self.publisher2 = self.create_publisher(BipedalCommand, 'bipedal_command', 10)
        timer_period = 0.001  # seconds
        self.timer_callback()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = StateVector()
        msg.joint_name = JOINT_LIST
        msg.joint_pos = [0 for _ in range(len(JOINT_LIST))]
        msg.joint_vel = [0 for _ in range(len(JOINT_LIST))]
        msg.joint_acc = [0 for _ in range(len(JOINT_LIST))]
        msg.pos = [0, 0, 0.75]
        msg.orien_quat = [0, 0, 0, 1]
        msg.efforts = [0 for _ in range(len(JOINT_LIST))]
        msg.ang_vel = [0 for _ in range(len(JOINT_LIST))]

        #Make bipedal_command
        bpc = BipedalCommand()
        bpc.inverse_timestamps = np.arange(40) * 0.02

        ics = []

        for c in range(40):
            ic = InverseCommand()

            ic.state_cost = 1e3
            ic.torque_cost = 1e-4
            pose = Pose()
            point = Point()
            point.x = 0.0
            point.y = 0.1
            point.z = 0.5
            orien = Quaternion()
            orien.x = 0.
            orien.y = 0.0
            orien.z = - 0.096
            orien.w =   0.995
            pose.position = point
            pose.orientation = orien
            ic.link_poses = [pose]
            ic.link_pose_names = ["pelvis"]
            ic.link_costs = [float(1e9)]
            ic.link_orien_weight = [float(100000)]
            ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
            ic.friction_contact_costs = [float(1e3), float(1e3)]
            com_pos = Point()
            com_pos.x = 0.0
            com_pos.y = 0.1
            com_pos.z = 0.6
            ic.com_pos = com_pos
            ic.com_cost = float(1e6)

            ics += [ic]

        bpc.inverse_commands = ics

        self.publisher2.publish(bpc)
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()