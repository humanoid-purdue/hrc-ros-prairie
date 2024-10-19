import rclpy
from rclpy.node import Node

from hrc_msgs.msg import StateVector


class BagPub(Node):

    def __init__(self):
        super().__init__('BagPub')
        self.publisher_ = self.create_publisher(StateVector, 'InitialStateVector', 10)
        timer_period = 4  # seconds
        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.listener_callback,
            10)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.state_vec = None

    def listener_callback(self, msg):
        self.state_vec = msg

    def timer_callback(self):
        if self.state_vec is not None:
            self.publisher_.publish(self.state_vec)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = BagPub()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()