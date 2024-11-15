import rclpy
from std_msgs.msg import String
from numpy.f2py.cfuncs import callbacks
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from hrc_msgs.msg import StateVector, BipedalCommand, JointTrajectoryST
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import scipy
import time
import os, sys
from threading import Thread

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
class StateSubscriber(Node):
    def __init__(self):
        super().__init__('state_subscriber')
        
        self.subscription = self.create_subscription(
            RobotState,
            'state_vector',
            self.listener_callback,
            10
        )
        
        # Initialize storage for the numpy arrays if needed
        self.data_arrays = []

    def listener_callback(self, msg):
        # Reconstruct dictionaries
        joint_pos_dict = dict(zip(msg.joint_pos_names, msg.joint_pos_values))
        joint_vel_dict = dict(zip(msg.joint_vel_names, msg.joint_vel_values))
        
        # Reconstruct the numpy arrays in the original format
        pos_array = np.array([msg.time, msg.left_knee_pos])
        vel_array = np.array([msg.time, msg.left_knee_vel])
        
        # Store the arrays in the same format as the original csv_dump2.update
        self.data_arrays = [pos_array, vel_array]
        
        # Example of using the reconstructed data
        print("Position array:", self.data_arrays[0])
        print("Velocity array:", self.data_arrays[1])
        
        # If you need to process these arrays further:
        self.process_arrays(self.data_arrays)
    
    def process_arrays(self, arrays):
        # Add your processing logic here
        # This is equivalent to what csv_dump2.update would have done
        pass

def main(args=None):
    rclpy.init(args=args)
    subscriber = StateSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
