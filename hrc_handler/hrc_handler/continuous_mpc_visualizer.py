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

#At each time step, publish state vector to walking command pub of the first time step from mpc and iterate state time accordingly
def initialStateVector():
    msg = StateVector()
    msg.joint_name = JOINT_LIST
    msg.joint_pos = [0. for _ in range(len(JOINT_LIST))]
    msg.joint_vel = [0. for _ in range(len(JOINT_LIST))]
    msg.joint_acc = [0. for _ in range(len(JOINT_LIST))]
    msg.pos = [0., 0., 0.75]
    msg.orien_quat = [0., 0., 0., 1.]
    msg.efforts = [0. for _ in range(len(JOINT_LIST))]
    msg.ang_vel = [0., 0., 0.]
    msg.vel = [0., 0., 0.]
    msg.time = 0.
    return msg

class ContinuousMPCViz(Node):

    def __init__(self):
        super().__init__('continuous_mpc_viz')

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
        self.fwd_poser = helpers.ForwardPoser(urdf_config_path, JOINT_LIST)
        # self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))

        self.timestamps = None
        self.inverse_commands = None
        self.state_vector = initialStateVector()

        self.simple_sm = SimpleFwdInvSM(self.poser)

        qos_profile = QoSProfile(depth=10)

        self.tf_broadcaster = TransformBroadcaster(self)
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self, qos=qos_profile)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)
        self.state_pub = self.create_publisher(StateVector, 'state_vector', qos_profile)

        self.timer = self.create_timer(0.00, self.timer_callback)

        self.ji = helpers.JointInterpolation(len(LEG_JOINTS), 0.05, 0.5)

        self.sv_fwd = helpers.SVFwdKinematics()

    def bipedc_callback(self, msg):
        self.timestamps = msg.inverse_timestamps
        self.inverse_commands = msg.inverse_commands

    def timer_callback(self):
        state_vector = self.state_vector

        if self.inverse_commands is not None:
            st = time.time()
            names = state_vector.joint_name
            j_pos = state_vector.joint_pos
            state_time = state_vector.time
            j_pos_config = dict(zip(names, j_pos))
            j_vel_config = dict(zip(names, state_vector.joint_vel))
            pos = state_vector.pos
            orien = state_vector.orien_quat
            ang_vel = state_vector.ang_vel
            # self.poser.x[7 + len(LEG_JOINTS):] = 0
            # self.poser.setState(pos, j_pos_config, orien = orien, vel = state_vector.vel ,config_vel = dict(zip(names, state_vector.joint_vel)), ang_vel = ang_vel)

            self.poser.updateReducedModel(LEG_JOINTS, j_pos_config)

            self.fwd_poser.updateData(pos, orien, j_pos_config)

            self.poser.setState(pos, j_pos_config, orien = orien, config_vel = j_vel_config, ang_vel = ang_vel, vel = state_vector.vel)


            x = None
            timestamps = [state_time] + list(np.array(self.timestamps) + state_time)
            if self.ji is not None and self.ji.hasHistory():
                x = np.array(self.ji.getSeedX(timestamps))
                x[0:] = self.poser.x.copy()
                x[7 + len(LEG_JOINTS):] = 0
            y, u = self.simple_sm.nextMPC(self.timestamps, self.inverse_commands, None)
            self.get_logger().info("{}".format(self.simple_sm.solved))


            #self.get_logger().info("Inter traj acc {}".format((y[1][7 + len(LEG_JOINTS):] - y[0][7 + len(LEG_JOINTS):])/ 0.01))
            b, pos_e, vel_e = self.ji.updateX(timestamps, y)
            pos, orien, vel, ang_vel, pos_list, vel_list = self.joint_trajst(y, u)
            pos_dict = dict(zip(JOINT_LIST_FULL, pos_list))
            vel_dict = dict(zip(JOINT_LIST_FULL, vel_list))

            msg = StateVector()
            msg.joint_name = JOINT_LIST
            msg.joint_pos = self.full2Movable(pos_dict)
            msg.joint_vel = self.full2Movable(vel_dict)
            msg.pos = pos
            msg.orien_quat = orien
            msg.ang_vel = ang_vel
            msg.vel = vel
            msg.time = state_time + self.timestamps[0]
            self.get_logger().info("{}".format(time.time() - st))
            t_elapse = time.time() - st
            if t_elapse < self.timestamps[0]:
                time.sleep(self.timestamps[0] - t_elapse)
            self.state_vector = msg

        self.state_vector = self.sv_fwd.update(self.state_vector)
        self.state_pub.publish(self.state_vector)

    def full2Movable(self, val_dict):
        y = []
        for name in JOINT_LIST:
            y += [val_dict[name]]
        return y


    def joint_trajst(self, y, u):
        #iters = len(y) - 1
        iters = 1
        for c in range(iters ):
            x0 = y[c + 1]
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = 'pelvis'
            pos, quaternion, joint_dict, joint_vels, torques = self.poser.getJointConfig(x0, efforts = u[0])

            force_list = self.fwd_poser.jacobianTorqueForce(torques,
                                                            ["left_ankle_roll_link", "right_ankle_roll_link"])
            #self.get_logger().info("{} {}".format(u[0], force_list))

            vel = x0[7 + len(LEG_JOINTS):10 + len(LEG_JOINTS)]
            ang_vel = x0[10 + len(LEG_JOINTS):13 + len(LEG_JOINTS)]

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

            self.joint_pub.publish(js0)
            self.tf_broadcaster.sendTransform(t)
            if c > 0:
                time.sleep(0.1)
            if c == 0:
                y_tuple = (pos, quaternion, vel, ang_vel, pos_list, vel_list)
        return y_tuple



    def r2whole(self, x):
        full_list = [0 for _ in range(len(JOINT_LIST_FULL))]
        for i in range(len(JOINT_LIST_FULL)):
            if JOINT_LIST_FULL[i] in x.keys():
                full_list[i] = x[JOINT_LIST_FULL[i]]
        return full_list


def main(args=None):
    rclpy.init(args=args)

    continuous_mpc_viz = ContinuousMPCViz()

    rclpy.spin(continuous_mpc_viz)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    continuous_mpc_viz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()