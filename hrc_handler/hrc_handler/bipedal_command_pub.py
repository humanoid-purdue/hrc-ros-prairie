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

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

def makePose(pos, rot):
    pose = Pose()
    point = Point()
    point.x = float(pos[0])
    point.y = float(pos[1])
    point.z = float(pos[2])
    orien = Quaternion()
    orien.x = float(rot[0])
    orien.y = float(rot[1])
    orien.z = float(rot[2])
    orien.w = float(rot[3])
    pose.position = point
    pose.orientation = orien
    return pose

class bipedal_command_pub(Node):

    def __init__(self):
        super().__init__('bipedal_command_pub')
        self.start_time = time.time()
        self.publisher2 = self.create_publisher(BipedalCommand, 'bipedal_command', 10)
        timer_period = 0.001  # seconds
        self.timer_callback()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        bpc = BipedalCommand()
        bpc.inverse_timestamps = [0.01, 0.02, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

        ics = []

        for c in range(8):
            ic = InverseCommand()

            ic.state_cost = float(2e2)
            ic.torque_cost = float(1e-2)
            pelvis_pose = makePose([0,0,0.7], [0,0,0,1])
            ic.link_poses = [pelvis_pose]
            ic.link_pose_names = ["pelvis"]
            ic.link_costs = [float(1e6)]
            ic.link_orien_weight = [float(100000), float(1), float(1)]
            ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
            ic.friction_contact_costs = [float(1e3), float(1e3)]
            ic.contact_lock_costs = [float(0), float(0)]
            com_pos = Point()
            com_pos.x = 0.03
            if time.time() - self.start_time > 20:
                com_pos.y = 0.05
            else:
                com_pos.y = 0.0
            com_pos.z = 0.63
            ic.com_pos = com_pos
            ic.com_cost = float(2e7)

            ics += [ic]

        bpc.inverse_commands = ics
        bpc.inverse_joints = LEG_JOINTS

        self.publisher2.publish(bpc)


def main(args=None):
    rclpy.init(args=args)

    bpcp = bipedal_command_pub()

    rclpy.spin(bpcp)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bipedal_command_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()