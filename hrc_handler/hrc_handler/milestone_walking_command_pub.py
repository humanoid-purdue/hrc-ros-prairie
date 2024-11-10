import rclpy
from rclpy.node import Node
import numpy as np
from hrc_msgs.msg import StateVector, BipedalCommand, InverseCommand
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
from helpers import WalkingSM

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

step_length = 0.15
step_height = 0.05
step_velocity = 0.15/0.4
com_velocity = 0.05/0.2
com_duration = 0.2

def makeFootStepPlan():
    step_no = 10
    com_y_prop = 0.3
    left_pos = np.array([-0.003, 0.12, 0.01]) + np.array([step_length * 1, 0, 0])
    right_pos = np.array([-0.003, -0.12, 0.01]) + np.array([step_length * 0.5, 0, 0])
    ref_plan = []
    ref_plan += [("R", right_pos.copy(), np.array([-0.003, -0.12, 0.01]), np.array([0, 0, 0, 1])),
                      ("L", left_pos.copy(), np.array([-0.003, 0.12, 0.01]), np.array([0, 0, 0, 1]))]
    for c in range(step_no):
        left_pos += np.array([step_length, 0, 0])
        right_pos += np.array([step_length, 0, 0])
        ref_plan += [
            ("R", right_pos.copy(), right_pos.copy() - np.array([step_length, 0, 0]), np.array([0, 0, 0, 1])),
            ("L", left_pos.copy(), left_pos.copy() - np.array([step_length, 0, 0]), np.array([0, 0, 0, 1]))]

    return ref_plan

def iterateMilestoneState(current_state):
    if current_state[:-1] == "Half_":
        return "End_" + current_state[-1]
    if current_state == "End_L":
        return "DS_R"
    if current_state == "End_R":
        return "DS_L"
    if current_state[:-1] == "DS_":
        return "Half_" + current_state[-1]

def swingTarget(current, target, dt, increasing = True):
    if increasing:
        target = target + np.array([0, 0, step_height])

    delta = ( target - current ) * 0.4

    if np.linalg.norm(delta[:2]) / dt > step_velocity:
        delta = delta * step_velocity / (np.linalg.norm(delta[:2]) / dt)

    return current + delta

def completeSwingPhase(initial_pos, swing_target, pos_c):
    time_remaining = np.linalg.norm(swing_target[:2] - pos_c[:2]) / step_velocity
    time_remaining = min(step_length / step_velocity, max(time_remaining, 0))
    if time_remaining > 0.08:
        splits = np.ceil(( time_remaining - 0.02 ) / 0.06)
        intervals = ( time_remaining - 0.02 ) / splits
        ts = (np.arange(splits) + 1) * intervals + 0.02
        horizon_ts = np.array([0.01, 0.02])
        horizon_ts = np.concatenate([horizon_ts, ts], axis = 0)
    else:
        horizon_ts = (np.arange(4) + 1) * time_remaining / 4

    link_pos = []

    for c in range(horizon_ts.shape[0]):
        prop = horizon_ts[c] / time_remaining
        xy_pos = swing_target * prop + pos_c * (1 - prop)
        xy_i = np.linalg.norm(xy_pos - initial_pos)
        xy_f = np.linalg.norm(xy_pos - swing_target)
        length = np.linalg.norm(initial_pos - swing_target)

        blind_height = 1 - ((xy_f - xy_i) ** 2 / (length ** 2))
        blind_height = blind_height * step_height
        blind_height = min(max(0, blind_height), step_height)
        xy_pos[2] = pos_c[2] * (1 - prop) + (blind_height + initial_pos[2]) * prop
        if xy_f > xy_i:
            xy_pos[2] = max(xy_pos[2], 0.04)
        link_pos += [xy_pos]
    horizon_ts = list(horizon_ts)
    return horizon_ts, link_pos

class milestone_walking_command_pub(Node):

    def __init__(self):
        super().__init__('milestone_walking_command_pub')
        self.start_time = time.time()
        self.publisher2 = self.create_publisher(BipedalCommand, 'bipedal_command', 10)
        timer_period = 0.001  # seconds


        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group=None)
        self.state_time = 0
        self.state_dict = None

        #footstep plan: expressed as list of tuples each tuple is pair of swing foot and footstep pos.
        # At the completion of a swing phase pop a copy of the list [("L", [target pos xy], [initial pos xy], [orien xyzw]

        self.plan = makeFootStepPlan()
        self.walking_sm = WalkingSM()
        self.com_y_prop = 0.4
        self.gait = helpers.BipedalGait(step_length, step_height)

        # up to 8 timesteps
        # first 2 reserved for 0.01s and 0.02s (Either 2 ds steps or 2 ss steps)
        # ss leg start, ss leg midpoint, ss leg end, ds one
        #self.horizon_ts = np.array([0.01, 0.02] + list(0.05 + np.arange(30) * 0.03))
        self.ds_expected_duration = 0.3
        self.swing_expected_duration = 0.7
        self.swing_linear_vel = step_length / self.swing_expected_duration
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
        self.state_dict = {"pos": pos, "orien": orien, "vel": vel, "ang_vel": ang_vel, "joint_pos": j_pos_config, "joint_vel": j_vel_config}



    def timer_callback(self):
        if len(self.plan) == 0:
            bpc = helpers.idleInvCmd()
            self.publisher2.publish(bpc)
            return
        if self.state_dict is not None:
            initial_pos = self.plan[0][2]
            swing_target = self.plan[0][1]
            self.walking_sm.updateState(self.state_dict, self.state_time, swing_target)
            current_state = self.walking_sm.current_state

            pos_l = self.walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            pos_r = self.walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            com = self.walking_sm.fwd_poser.getCOMPos()

            if current_state == "SR":
                pos_c = pos_r
                support_link = "left_ankle_roll_link"
                swing_link = "right_ankle_roll_link"
            else:
                pos_c = pos_l
                support_link = "right_ankle_roll_link"
                swing_link = "left_ankle_roll_link"

            ics = []

            if current_state[0] == "S":
                horizon_ts, link_pos = completeSwingPhase(initial_pos, swing_target, pos_c)
                #self.get_logger().info("{}".format(link_pos))
                for pos in link_pos:
                    com_pos = pos * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    ic = self.gait.singleSupport(support_link, swing_link, pos,
                                                  np.array([0, 0, 0, 1]), com_pos)
                    ics += [ic]
                #self.get_logger().info("{} {}".format(link_pos, pos_c))
                horizon_ts += [horizon_ts[-1] + com_duration * 0.5, horizon_ts[-1] + com_duration]
                com_pos = swing_target * np.array([1, self.com_y_prop, 1])
                com_pos[2] = 0.6
                ic = self.gait.dualSupport(com_pos, None, swing_link)
                ics += [ic, ic]
                pop_ind = 1
            if current_state[0:2] == "DS":
                if current_state[-1] == "L":
                    support_link = "right_ankle_roll_link"
                    com_pos = pos_r * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                else:
                    support_link = "left_ankle_roll_link"
                    com_pos = pos_l * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                time_remaining = np.linalg.norm(com_pos[:2] - com[:2]) / com_velocity
                if time_remaining < 0.02:
                    horizon_ts = [0.01, 0.02]
                else:
                    horizon_ts = [0.01, 0.02, time_remaining * 0.5, time_remaining]
                ic = self.gait.dualSupport(com_pos, None, support_link)
                ics += [ic] * len(horizon_ts)

                pop_ind = 0
            if current_state[0] == "S" and self.plan[0][0] == "L":
                steps = 1
            else:
                steps = 1

            for c in range(steps):
                plan_tuple = self.plan[pop_ind]
                if plan_tuple[0] == "L":
                    support_link = "right_ankle_roll_link"
                    swing_link = "left_ankle_roll_link"
                else:
                    support_link = "left_ankle_roll_link"
                    swing_link = "right_ankle_roll_link"
                cmya = np.array([1, self.com_y_prop, 1])
                cz = np.array([0, 0, 0.6])
                step_duration = step_length / step_velocity
                ic = self.gait.singleSupport(support_link, swing_link, plan_tuple[2], np.array([0, 0, 0, 1]), plan_tuple[2] * cmya + cz)
                horizon_ts += [horizon_ts[-1] + 0.05]
                ics += [ic]

                link_pos = ( plan_tuple[2] * 0.66 + plan_tuple[1] * 0.33 )
                link_pos = link_pos + np.array([0, 0, step_height])
                ic = self.gait.singleSupport(support_link, swing_link, link_pos, np.array([0,0,0,1]), link_pos * cmya + cz)
                horizon_ts += [horizon_ts[-1] + step_duration * 0.333 - 0.05]
                ics += [ic]

                link_pos = (plan_tuple[2] * 0.33 + plan_tuple[1] * 0.66)
                link_pos = link_pos + np.array([0, 0, step_height])
                ic = self.gait.singleSupport(support_link, swing_link, link_pos, np.array([0, 0, 0, 1]),
                                             link_pos * cmya)
                horizon_ts += [horizon_ts[-1] + step_duration * 0.333]
                ics += [ic]

                ic = self.gait.singleSupport(support_link, swing_link, plan_tuple[1], np.array([0,0,0,1]), plan_tuple[1] * cmya + cz)
                horizon_ts += [horizon_ts[-1] + step_duration * 0.333]
                ics += [ic]

                ic = self.gait.dualSupport(plan_tuple[1] * cmya + cz, None, swing_link)
                horizon_ts += [horizon_ts[-1] + com_duration]
                ics += [ic]
                pop_ind += 1

            bpc = BipedalCommand()
            bpc.inverse_timestamps = horizon_ts
            bpc.inverse_commands = ics
            bpc.inverse_joints = LEG_JOINTS
            self.publisher2.publish(bpc)

            if self.prev_state[0] == "S" and current_state[0:2] == "DS":
                self.plan.pop(0)
            self.prev_state = current_state

            #make timesteps up to end of Dual support phase

def main(args=None):
    rclpy.init(args=args)

    bpcp = milestone_walking_command_pub()

    rclpy.spin(bpcp)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bpcp.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()