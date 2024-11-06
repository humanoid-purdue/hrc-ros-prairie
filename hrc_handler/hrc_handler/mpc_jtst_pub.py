import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from hrc_msgs.msg import StateVector, BipedalCommand, JointTrajectoryST
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

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

class mpc_jtst_pub(Node):

    def __init__(self):
        super().__init__('mpc_jtst_pub')

        qos_profile = QoSProfile(depth=10)

        self.tf_broadcaster = TransformBroadcaster(self)
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self, qos=qos_profile)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group=None)

        self.subscription_3 = self.create_subscription(
            JointTrajectoryST,
            'inv_joint_traj',
            self.inv_callback,
            10, callback_group=None)

        self.state_time = 0
        self.state_dict = None
        self.ji_joint_name = None
        self.yr = None
        self.pos = None
        self.orien = None

        self.timer = self.create_timer(0.001, self.rviz_callback, callback_group = None)

    def state_vector_callback(self, msg):
        names = msg.joint_name
        j_pos = msg.joint_pos
        j_vel = msg.joint_vel
        self.state_time = msg.time
        j_pos_config = dict(zip(names, j_pos))
        j_vel_config = dict(zip(names, j_vel))
        pos = msg.pos
        orien = msg.orien_quat
        ang_vel = msg.ang_vel
        vel = msg.vel
        self.state_dict = {"pos": pos, "orien": orien, "vel": vel, "ang_vel": ang_vel, "joint_pos": j_pos_config, "joint_vel": j_vel_config}

    def inv_callback(self, msg):

        x = np.array(msg.timestamps)
        yr = None
        yv = None
        yt = None
        for jointstate in msg.jointstates:
            if yr is None:
                yr = np.array(jointstate.position)[None, :]
                yv = np.array(jointstate.velocity)[None, :]
                yt = np.array(jointstate.effort)[None, :]
            else:
                yr = np.concatenate([yr, np.array(jointstate.position)[None, :]], axis=0)
                yv = np.concatenate([yv, np.array(jointstate.velocity)[None, :]], axis=0)
                yt = np.concatenate([yt, np.array(jointstate.effort)[None, :]], axis=0)
            self.ji_joint_name = jointstate.name

        self.pos = np.zeros([len(msg.rootpose), 3])
        self.orien = np.zeros([len(msg.rootpose), 4])
        for pose, i in zip(msg.rootpose, range(len(msg.rootpose))):
            pos = [pose.position.x, pose.position.y, pose.position.z]
            orien = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
                     pose.orientation.w]
            self.pos[i, :] = np.array(pos)
            self.orien[i, :] = np.array(orien)
        self.yr = yr

    def r2whole(self, x):
        full_list = [0 for _ in range(len(JOINT_LIST_FULL))]
        for i in range(len(JOINT_LIST_FULL)):
            if JOINT_LIST_FULL[i] in x.keys():
                full_list[i] = x[JOINT_LIST_FULL[i]]
        return full_list

    def rviz_callback(self):
        if self.yr is not None:
            for c in range(3):
                t = TransformStamped()

                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'world'
                t.child_frame_id = 'pelvis'

                full_list = self.r2whole(dict(zip(self.ji_joint_name, list(self.yr[c, :]))))

                js0 = JointState()
                now = self.get_clock().now()
                js0.header.stamp = now.to_msg()
                js0.name = JOINT_LIST_FULL
                js0.position = full_list

                t.transform.translation.x = self.pos[c,0]
                t.transform.translation.y = self.pos[c,1]
                t.transform.translation.z = self.pos[c,2]
                t.transform.rotation.x = self.orien[c,0]
                t.transform.rotation.y = self.orien[c,1]
                t.transform.rotation.z = self.orien[c,2]
                t.transform.rotation.w = self.orien[c,3]

                self.joint_pub.publish(js0)
                self.tf_broadcaster.sendTransform(t)
                time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = mpc_jtst_pub()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()