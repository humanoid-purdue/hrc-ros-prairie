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

JOINT_LIST = ['left_hip_pitch_joint',
              'left_hip_roll_joint',
              'left_hip_yaw_joint',
              'left_knee_joint',
              'left_ankle_pitch_joint',
              'left_ankle_roll_joint',
              'right_hip_pitch_joint',
              'right_hip_roll_joint',
              'right_hip_yaw_joint',
              'right_knee_joint',
              'right_ankle_pitch_joint',
              'right_ankle_roll_joint',
              'torso_joint',
              'left_shoulder_pitch_joint',
              'left_shoulder_roll_joint',
              'left_shoulder_yaw_joint',
              'left_elbow_pitch_joint',
              'left_elbow_roll_joint',
              'right_shoulder_pitch_joint',
              'right_shoulder_roll_joint',
              'right_shoulder_yaw_joint',
              'right_elbow_pitch_joint',
              'right_elbow_roll_joint',
              'left_zero_joint',
              'left_one_joint',
              'left_two_joint',
              'left_three_joint',
              'left_four_joint',
              'left_five_joint',
              'left_six_joint',
              'right_zero_joint',
              'right_one_joint',
              'right_two_joint',
              'right_three_joint',
              'right_four_joint',
              'right_five_joint',
              'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

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
                joint_traj_desired.timestamps = (np.arange(10) * 0.01 + time.time()).tolist()
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