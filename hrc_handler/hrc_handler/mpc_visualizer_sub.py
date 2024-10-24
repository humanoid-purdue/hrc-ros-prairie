import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from hrc_msgs.msg import StateVector
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
import time

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
from helpers import BipedalPoser, SquatSM

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            StateVector,
            'InitialStateVector',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST_FULL, LEG_JOINTS, "left_ankle_roll_link",
                                  "right_ankle_roll_link")
        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))

        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)


    def listener_callback(self, msg):
        state_vector = msg

        names = state_vector.joint_name
        j_pos = state_vector.joint_pos
        state_time = state_vector.time
        j_pos_config = dict(zip(names, j_pos))
        pos = state_vector.pos
        orien = state_vector.orien_quat
        ang_vel = state_vector.ang_vel
        #self.poser.x[7 + len(LEG_JOINTS):] = 0
        #self.poser.setState(pos, j_pos_config, orien = orien, vel = state_vector.vel ,config_vel = dict(zip(names, state_vector.joint_vel)), ang_vel = ang_vel)
        self.poser.setState(pos, j_pos_config)
        self.squat_sm.com_pos = np.array([0., 0., 0.55])
        y = self.squat_sm.simpleNextMPC(None)
        us = self.squat_sm.us
        self.get_logger().info("{}".format(us[:,0]))

        self.joint_trajst(y)




        return msg

    def joint_trajst(self, y):
        js_list = []
        for x0 in y:
            pos, joint_dict, joint_vels, _ = self.poser.getJointConfig(x0)


            pos_list = self.r2whole(joint_dict)
            vel_list = self.r2whole(joint_vels)

            js0 = JointState()
            now = self.get_clock().now()
            js0.header.stamp = now.to_msg()
            js0.name = JOINT_LIST_FULL
            js0.position = pos_list
            js0.velocity = vel_list

            self.joint_pub.publish(js0)
            time.sleep(0.3)

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