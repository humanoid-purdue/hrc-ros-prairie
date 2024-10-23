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
import numpy as np
import time

import os, sys
helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
_, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

ELBOW_JOINTS = ['right_elbow_pitch_joint', 'left_elbow_pitch_joint']

class zero_jtst_pub(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('zero_jtst_pub')

        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        loop_rate = self.create_rate(30)
        joint_traj_desired = JointTrajectoryST()
        try:
            while rclpy.ok():
                rclpy.spin_once(self)
                joint_traj_desired.timestamps = np.arange(10) * 0.01 + time.time()
                js = JointState()
                js.name = LEG_JOINTS + ELBOW_JOINTS
                js.position = [0.0] * len(LEG_JOINTS) + [np.pi/2, np.pi /2]
                js.velocity = [0.0] * len(LEG_JOINTS) + [0.0] * len(ELBOW_JOINTS)
                joint_traj_desired.jointstates = [js] * 10
                self.joint_traj_pub.publish(joint_traj_desired)
                loop_rate.sleep()
        except KeyboardInterrupt:
            pass

def main():
    node = zero_jtst_pub()

if __name__ == '__main__':
    main()