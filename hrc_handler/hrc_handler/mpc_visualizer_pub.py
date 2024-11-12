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
        self.publisher_ = self.create_publisher(StateVector, 'state_vector', 10)
        timer_period = 0.001
        self.bpg = helpers.BipedalGait(0.25, 0.1)# seconds
        self.timer_callback()
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = StateVector()
        msg.joint_name = JOINT_LIST
        msg.joint_pos = [0. for _ in range(len(JOINT_LIST))]
        msg.joint_vel = [0. for _ in range(len(JOINT_LIST))]
        msg.joint_acc = [0. for _ in range(len(JOINT_LIST))]
        msg.vel = [0., 0., 0.]
        msg.pos = [0., 0., 0.75]
        msg.orien_quat = [0., 0., 0., 1]
        msg.efforts = [0. for _ in range(len(JOINT_LIST))]
        msg.ang_vel = [0. for _ in range(len(JOINT_LIST))]

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