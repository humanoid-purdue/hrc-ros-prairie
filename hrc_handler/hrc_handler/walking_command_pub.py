import rclpy
from rclpy.node import Node
import numpy as np
from hrc_msgs.msg import StateVector, BipedalCommand, InverseCommand, CentroidalTrajectory
from geometry_msgs.msg import Point, Pose, Quaternion
from ament_index_python.packages import get_package_share_directory
import os, sys
import time
from enum import Enum
helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers
import scipy
from helpers import WalkingSM

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()


class walking_command_pub(Node):

    def __init__(self):
        super().__init__('walking_command_pub')
        self.start_time = time.time()
        self.publisher2 = self.create_publisher(BipedalCommand, 'bipedal_command', 10)
        timer_period = 0.00  # seconds

        self.com_cs = None


        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group=None)

        self.subscription_2 = self.create_subscription(
            CentroidalTrajectory,
            'centroidal_trajectory',
            self.centroid_traj_callback,
            10, callback_group=None)
        self.state_time = 0
        self.state_dict = None

        #footstep plan: expressed as list of tuples each tuple is pair of swing foot and footstep pos.
        # At the completion of a swing phase pop a copy of the list [("L", [target pos xy], [initial pos xy], [orien xyzw]

        self.lock_pos = np.zeros([3])

        self.simple_plan = helpers.SimpleFootstepPlan()
        self.walking_sm = WalkingSM()
        self.gait = helpers.BipedalGait(self.simple_plan.step_length, self.simple_plan.step_height)

        # up to 8 timesteps
        # first 2 reserved for 0.01s and 0.02s (Either 2 ds steps or 2 ss steps)
        # ss leg start, ss leg midpoint, ss leg end, ds one
        #self.horizon_ts = np.array([0.01, 0.02] + list(0.05 + np.arange(30) * 0.03))
        self.prev_state = "0000"
        self.timer = self.create_timer(timer_period, self.timer_callback)

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
        self.state_dict = {"pos": pos, "orien": orien, "vel": vel, "ang_vel": ang_vel, "joint_pos": j_pos_config,
                           "joint_vel": j_vel_config,
                           "com_pos": msg.com_pos, "com_vel": msg.com_vel, "com_acc": msg.com_acc,
                           "l_foot_pos": msg.l_foot_pos, "r_foot_pos": msg.r_foot_pos}


    def centroid_traj_callback(self, msg):
        timestamps = msg.timestamps
        com_xyz = np.zeros([len(msg.com_pos), 3])
        for point, i in zip(msg.com_pos, np.arange(len(msg.com_pos))):
            com_xyz[i, 0] = point.x
            com_xyz[i, 1] = point.y
            com_xyz[i, 2] = point.z

        #self.get_logger().info("com xyz {} {}".format(com_xyz[0:3, :], timestamps[0:3]))
        self.com_cs = scipy.interpolate.CubicSpline(timestamps, com_xyz, axis=0)


    def setCoMZ(self, com):
        if com[2] > self.simple_plan.z_height:
            z = com[2] - self.simple_plan.com_speed * 0.001
            return max(self.simple_plan.z_height, z)
        else:
            z = com[2] + self.simple_plan.com_speed * 0.001
            return min(self.simple_plan.z_height, z)


    def getCOMTrajDeriv(self, horizon_ts, com):
        dt = 0.001
        pos = []
        vel = []
        acc = []
        z = com[2]
        com2 = com.copy()
        prev_pos = self.com_cs(self.state_time)
        for ts in horizon_ts:
            if z > self.simple_plan.z_height:
                z = com[2] - self.simple_plan.com_speed * ts / 5

            else:
                z = com[2] + self.simple_plan.com_speed * ts / 5

            pos_c = self.com_cs(ts + self.state_time) - prev_pos + com
            pos_c[2] = self.simple_plan.z_height

            vel += [(self.com_cs(ts + dt + self.state_time) - self.com_cs(ts + self.state_time)) / dt]

            com2 = com2 + vel[-1] * ts
            #com2 = pos_c
            com2[2] = self.simple_plan.z_height
            pos += [com2.copy()]
            acc += [(self.com_cs(ts + 2 * dt + self.state_time) + self.com_cs(ts + self.state_time) - 2 * self.com_cs(ts + dt + self.state_time)) / dt**2]
        return pos, vel, acc



    def timer_callback(self):
        if len(self.simple_plan.plan) == 0:
            bpc = helpers.idleInvCmd()
            self.publisher2.publish(bpc)
            return
        if self.state_dict is not None and self.com_cs is not None:
            initial_pos = self.simple_plan.plan[0][2]
            swing_target = self.simple_plan.plan[0][1]
            self.walking_sm.updateState(self.state_dict, self.state_time, initial_pos, swing_target)
            current_state = self.walking_sm.current_state

            #pos_l = self.walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            #pos_r = self.walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            #com = self.walking_sm.fwd_poser.getCOMPos()

            pos_l = np.array(self.state_dict["l_foot_pos"])
            pos_r = np.array(self.state_dict["r_foot_pos"])
            com = np.array(self.state_dict["com_pos"])

            if self.prev_state[0:4] == "DS_C" and current_state[0] == "SL":
                self.lock_pos = pos_r
            elif self.prev_state[0:4] == "DS_C" and current_state[0] == "SR":
                self.lock_pos = pos_l

            if current_state == "SR":
                pos_c = pos_r
                support_link = "left_ankle_roll_link"
                swing_link = "right_ankle_roll_link"
            else:
                pos_c  = pos_l
                support_link = "right_ankle_roll_link"
                swing_link = "left_ankle_roll_link"

            ics = []

            if current_state[0] == "S":
                horizon_ts, link_pos = self.simple_plan.swingFootPoints(initial_pos, swing_target, pos_c)
                pos, vel, acc = self.getCOMTrajDeriv(horizon_ts, com)

                for pos, x0, x1, x2 in zip(link_pos, pos, vel, acc):
                    ic = self.gait.singleSupport(support_link, swing_link, pos,
                                                  np.array([0, 0, 0, 1]), x0, x1, x2)
                    ics += [ic]

            if current_state[0:2] == "DS" and abs(com[2] - self.simple_plan.z_height) < 0.03:
                if current_state[-1] == "L":
                    support_link = "right_ankle_roll_link"
                else:
                    support_link = "left_ankle_roll_link"

                horizon_ts = self.simple_plan.horizon_ts
                pos, vel, acc = self.getCOMTrajDeriv(horizon_ts, com)


                for x0, x1, x2 in zip(pos, vel, acc):
                    ic = self.gait.dualSupport(x0, x1, x2, support_link)
                    self.get_logger().info("{} {} {} {}".format(x1, np.array(self.state_dict["com_vel"]), x0, com))
                    ics += [ic]

            elif current_state[0:2] == "DS" and abs(com[2] - self.simple_plan.z_height) > 0.03:
                if current_state[-1] == "L":
                    support_link = "right_ankle_roll_link"
                else:
                    support_link = "left_ankle_roll_link"

                horizon_ts = self.simple_plan.horizon_ts

                for ts in horizon_ts:
                    if com[2] > self.simple_plan.z_height:
                        com_p = com.copy() - np.array([0, 0, self.simple_plan.com_speed * 0.001])
                        com_p[2] = max(self.simple_plan.z_height, com_p[2])
                    else:
                        com_p = com.copy() + np.array([0, 0, self.simple_plan.com_speed * 0.001])
                        com_p[2] = min(self.simple_plan.z_height, com_p[2])
                    ic = self.gait.dualSupport(com_p, np.zeros([3]), np.zeros([3]), support_link)
                    ics += [ic]


            self.get_logger().info("{} {}".format(current_state, self.state_time))
            bpc = BipedalCommand()
            bpc.inverse_timestamps = horizon_ts
            bpc.inverse_commands = ics
            bpc.inverse_joints = LEG_JOINTS
            self.publisher2.publish(bpc)

            if self.prev_state[0] == "S" and current_state[0:2] == "DS":
                self.simple_plan.plan.pop(0)

            self.prev_state = current_state

            #make timesteps up to end of Dual support phase

def main(args=None):
    rclpy.init(args=args)

    bpcp = walking_command_pub()

    rclpy.spin(bpcp)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bpcp.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()