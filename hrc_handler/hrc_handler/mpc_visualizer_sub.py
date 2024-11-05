import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from hrc_msgs.msg import StateVector, BipedalCommand
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
import time
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros import TransformBroadcaster

helper_path = os.path.join(
    get_package_share_directory('hrc_handler'),
    "helpers")

sys.path.append(helper_path)
import helpers
from helpers import BipedalPoser, SimpleFwdInvSM

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            StateVector,
            'InitialStateVector',
            self.listener_callback,
            10)

        self.subscription = self.create_subscription(
            BipedalCommand,
            'bipedal_command',
            self.bipedc_callback,
            10)

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST_FULL, LEG_JOINTS, "left_ankle_roll_link",
                                  "right_ankle_roll_link")
        # self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))

        self.timestamps = None
        self.inverse_commands = None
        self.state_vector = None

        self.simple_sm = SimpleFwdInvSM(self.poser)

        qos_profile = QoSProfile(depth=10)

        self.tf_broadcaster = TransformBroadcaster(self)
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self, qos=qos_profile)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.timer = self.create_timer(1, self.timer_callback)

    def bipedc_callback(self, msg):
        self.timestamps = msg.inverse_timestamps
        self.inverse_commands = msg.inverse_commands

    def listener_callback(self, msg):
        self.state_vector = msg

    def timer_callback(self):
        state_vector = self.state_vector

        names = state_vector.joint_name
        j_pos = state_vector.joint_pos
        state_time = state_vector.time
        j_pos_config = dict(zip(names, j_pos))
        pos = state_vector.pos
        orien = state_vector.orien_quat
        ang_vel = state_vector.ang_vel
        # self.poser.x[7 + len(LEG_JOINTS):] = 0
        # self.poser.setState(pos, j_pos_config, orien = orien, vel = state_vector.vel ,config_vel = dict(zip(names, state_vector.joint_vel)), ang_vel = ang_vel)
        self.poser.setState(pos, j_pos_config)
        y,_ = self.simple_sm.nextMPC(self.timestamps, self.inverse_commands, None)

        self.joint_trajst(y)

    def joint_trajst(self, y):
        js_list = []
        for x0 in y:
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = 'pelvis'
            pos, quaternion, joint_dict, joint_vels, _ = self.poser.getJointConfig(x0)

            pos_list = self.r2whole(joint_dict)
            vel_list = self.r2whole(joint_vels)


            js0 = JointState()
            now = self.get_clock().now()
            js0.header.stamp = now.to_msg()
            js0.name = JOINT_LIST_FULL
            js0.position = pos_list
            js0.velocity = vel_list

            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]

            # transformation.header.stamp = now.to_msg()
            # transformation.transform.translation.x = pos[0]
            # transformation.transform.translation.y = pos[1]
            # transformation.transform.translation.z = pos[2]
            # transformation.transform.rotation.x = quaternion[0]
            # transformation.transform.rotation.y = quaternion[1]
            # transformation.transform.rotation.z = quaternion[2]
            # transformation.transform.rotation.w = quaternion[3]

            self.get_logger().info("{}".format(pos_list))

            self.joint_pub.publish(js0)
            self.tf_broadcaster.sendTransform(t)
            # self.pose_pub.publish(pose)
            # self.tf_static_broadcaster.sendTransform(transformation)
            time.sleep(0.02)

    def r2whole(self, x):
        full_list = [0 for _ in range(len(JOINT_LIST_FULL))]
        for i in range(len(JOINT_LIST_FULL)):
            if JOINT_LIST_FULL[i] in x.keys():
                full_list[i] = x[JOINT_LIST_FULL[i]]
        return full_list


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()