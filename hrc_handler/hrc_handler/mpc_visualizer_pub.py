import rclpy
from rclpy.node import Node

from hrc_msgs.msg import StateVector
import os, sys
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
        timer_period = 7  # seconds
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
        


        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % self.i)
        self.i += 1


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