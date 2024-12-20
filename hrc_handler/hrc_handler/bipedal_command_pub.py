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
        bpc.inverse_timestamps = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

        ics = []

        for c in range(6):
            ic = InverseCommand()

            ic.state_cost = float(2e2)
            ic.torque_cost = float(1e-1)
            pelvis_pose = makePose([0,0,0.7], [0,0,0,1])
            ic.link_poses = [pelvis_pose]
            ic.link_pose_names = ["pelvis"]
            ic.link_costs = [float(1e3)]
            ic.link_orien_weight = [float(100000)]
            ic.link_vel_costs = [float(1e3)]
            ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
            ic.friction_contact_costs = [float(1e3), float(1e3)]
            ic.force_limit_costs = [float(1e4), float(1e4)]
            ic.cop_costs = [0., 0.]
            ic.max_linear_vel = 0.4
            ic.max_ang_vel = 0.4
            ic.state_limit_cost = 1e1
            ic.centroid_vel_cost = 1e9
            com_pos = Point()

            if np.sin(time.time() - self.start_time) > 0:
                com_pos.x = 0.03
                com_pos.y = 0.04
                com_pos.z = 0.55
            else:
                com_pos.x = 0.03
                com_pos.y = -0.04
                com_pos.z = 0.60

            ic.com_pos = com_pos
            ic.com_cost = float(1e6)

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