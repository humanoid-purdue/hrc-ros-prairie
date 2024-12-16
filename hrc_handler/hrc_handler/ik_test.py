import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
import time
from geometry_msgs.msg import TransformStamped
helper_path = os.path.join(
    get_package_share_directory('hrc_handler'),
    "helpers")
from tf2_ros import TransformBroadcaster

sys.path.append(helper_path)
import helpers

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

class ik_test(Node):
    def __init__(self):
        super().__init__("ik_test")
        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.fwd_poser = helpers.ForwardPoser(urdf_config_path, JOINT_LIST, leg_joints = LEG_JOINTS)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.q = self.fwd_poser.q_r.copy()
        self.com = np.array([ 0.03126703,  0.00053525, -0.09025822])
        self.st = time.time()

    def r2whole(self, x):
        full_list = [0 for _ in range(len(JOINT_LIST_FULL))]
        for i in range(len(JOINT_LIST_FULL)):
            if JOINT_LIST_FULL[i] in x.keys():
                full_list[i] = x[JOINT_LIST_FULL[i]]
        return full_list

    def timer_callback(self):
        t2 = time.time()
        # ik problem posed as two fixed position feet at (0, pm 0.07, -0.4) with orbiting com
        link_target_dict = {"left_ankle_roll_link": np.array([ 8.06828946e-07, 1.17871300e-01, -7.19675917e-01]),
                            "right_ankle_roll_link": np.array([ 8.06828946e-07, -1.17871300e-01, -7.19675917e-01])}

        self.com[2] = -0.09025822 + np.cos(t2 - self.st) * 0.05 - 0.07
        self.com[1] = np.sin( (t2 - self.st ) * 0.3) * 0.1
        self.fwd_poser.q_r = self.q.copy()
        pos, quaternion, joint_dict, q = self.fwd_poser.ikSolver(self.com, link_target_dict)
        pos_list = self.r2whole(joint_dict)
        self.q = q + np.random.normal(size = q.shape) * 0.0

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'pelvis'

        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        js0 = JointState()
        now = self.get_clock().now()
        js0.header.stamp = now.to_msg()
        js0.name = JOINT_LIST_FULL
        js0.position = pos_list

        self.joint_pub.publish(js0)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)

    ikt = ik_test()

    rclpy.spin(ikt)
    ikt.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()