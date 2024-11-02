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

JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = helpers.makeJointList()

DS_COUNTDOWN = 0.05

class WalkingSM:
    def __init__(self):
        #4 states DS_SR, DS_CR, SR, DS_SL, DS_CL, SL
        #DS_SR: Dual support state end cond, prep for right swing
        #DS_CR: Dual support countdown end cond, prep for right swing
        #SR: right swing foot
        #DS_SL: Dual support state end cond, prep for left swing
        #DS_CL: Dual support countdown end cond, prep for left swing
        #SL: left swing foot
        self.current_state = "DS_SR"
        self.countdown_start = 0
        self.countdown_duration = DS_COUNTDOWN
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.fwd_poser = helpers.ForwardPoser(urdf_config_path, JOINT_LIST)

    def updateState(self, state_dict, state_time, current_swing_target):
        self.fwd_poser.updateData(state_dict["pos"], state_dict["orien"], state_dict["joint_pos"])
        #SL SR determined by xy being within 0.05m of the target and z being 0.01 off
        #DS_SL/R determined by COM xy being withing 0.05m of support foot
        if self.current_state[:4] == "DS_C":
            if state_time - self.countdown_start > self.countdown_duration:
                if self.current_state == "DS_CR":
                    self.current_state = "SR"
                else:
                    self.current_state = "SL"
                return True
            else:
                return False
        if self.current_state[0] == "S":
            if self.current_state == "SL":
                swing_link = "left_ankle_roll_link"
            else:
                swing_link = "right_ankle_roll_link"
            link_pos = self.fwd_poser.getLinkPose(swing_link)
            xy = np.linalg.norm(link_pos[0:2] - current_swing_target[0:2])
            z = abs(link_pos[2] - current_swing_target[2])
            if xy < 0.07 and z < 0.02:
                if self.current_state == "SL":
                    self.current_state = "DS_SR"
                else:
                    self.current_state = "DS_SL"
                return True
            return False
        if self.current_state[:4] == "DS_S":
            com_pos = self.fwd_poser.getCOMPos()
            if self.current_state == "DS_SL":
                support_pos = self.fwd_poser.getLinkPose("right_ankle_roll_link")
            else:
                support_pos = self.fwd_poser.getLinkPose("left_ankle_roll_link")
            xy = np.linalg.norm(com_pos[0:2] - support_pos[0:2])
            if xy < 0.07:
                self.countdown_start = state_time
                if self.current_state == "DS_SL":
                    self.current_state = "DS_CL"
                    return True
                else:
                    self.current_state = "DS_CR"
                    return True
            else:
                return False
        return False

class BlindWalkSM:
    def __init__(self, current_state, ds_expected_duration, swing_expected_duration):
        self.current_state = current_state
        self.ds_c_expected_duration = DS_COUNTDOWN
        self.ds_s_expected_duration = ds_expected_duration
        self.swing_expected_durations = swing_expected_duration
        self.state_start_time = 0
        self.transitions = 0

    def initializeStartTime(self, walking_sm, lin_vel, state_time, swing_target):
        if walking_sm.current_state[:4] == "DS_C":
            remaining = (walking_sm.countdown_start + walking_sm.countdown_duration) - state_time
            self.state_start_time = remaining - self.ds_c_expected_duration
        elif walking_sm.current_state[0] == "S":
            if walking_sm.current_state == "SL":
                swing_pos = walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            else:
                swing_pos = walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            t2 = np.linalg.norm(swing_pos[0:2] - swing_target[0:2]) / lin_vel
            self.state_start_time = t2 - self.swing_expected_durations
        elif walking_sm.current_state[:4] == "DS_S":
            #For DS_S measure distance as proportion of lr foot distance and
            left = walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            right = walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            max_d = np.linalg.norm(left[0:2] - right[0:2])
            if walking_sm == "DS_SL":
                remaining_time = (self.ds_s_expected_duration *
                                  np.linalg.norm(right[0:2] - walking_sm.fwd_poser.getCOMPos()[0:2]) / max_d)
            else:
                remaining_time = (self.ds_s_expected_duration *
                                  np.linalg.norm(left[0:2] - walking_sm.fwd_poser.getCOMPos()[0:2]) / max_d)
            remaining_time = max(min(remaining_time, self.ds_s_expected_duration), 0)
            self.state_start_time = remaining_time - self.ds_s_expected_duration

    def updateState(self, time_elapsed):
        if self.current_state[:4] == "DS_S":
            if time_elapsed - self.state_start_time > self.ds_s_expected_duration:
                self.current_state = "DS_C" + self.current_state[4]
                self.state_start_time = time_elapsed
                self.transitions += 1
        elif self.current_state[:4] == "DS_C":
            if time_elapsed - self.state_start_time > self.ds_c_expected_duration:
                self.current_state = "S" + self.current_state[4]
                self.state_start_time = time_elapsed
                self.transitions += 1
        elif self.current_state[0] == "S":
            if time_elapsed - self.state_start_time > self.swing_expected_durations:
                self.current_state = "DS_S" + self.current_state[1]
                self.state_start_time = time_elapsed
                self.transitions += 1
        return self.current_state, self.transitions

class walking_command_pub(Node):

    def __init__(self):
        super().__init__('walking_command_pub')
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
        step_length = 0.25
        step_height = 0.15
        step_no = 10
        self.com_y_prop = 0.5
        left_pos = np.array([-0.003, 0.12, 0.01]) + np.array([step_length * 1, 0, 0])
        right_pos = np.array([-0.003, -0.12, 0.01]) + np.array([step_length * 0.5, 0, 0])
        self.ref_plan = []
        self.ref_plan += [("R", right_pos.copy(), np.array([-0.003, -0.12, 0.01]), np.array([0, 0, 0, 1])), ("L", left_pos.copy(), np.array([-0.003, 0.12, 0.01]), np.array([0, 0, 0, 1]))]
        for c in range(step_no):
            left_pos += np.array([step_length, 0, 0])
            right_pos += np.array([step_length, 0, 0])
            self.ref_plan += [("R", right_pos.copy(), right_pos.copy() - np.array([step_length, 0, 0]), np.array([0, 0, 0, 1])),
                              ("L", left_pos.copy(), left_pos.copy() - np.array([step_length, 0, 0]), np.array([0, 0, 0, 1]))]
        self.plan = self.ref_plan.copy()
        self.walking_sm = WalkingSM()
        self.gait = helpers.BipedalGait(step_length, step_height)

        #self.horizon_ts = 0.01 + np.arange(10) * 0.01
        self.horizon_ts = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
        self.ds_expected_duration = 0.2
        self.swing_expected_duration = 0.4
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

            swing_target = self.plan[0][1]
            self.walking_sm.updateState(self.state_dict, self.state_time, swing_target)
            current_state = self.walking_sm.current_state

            if self.prev_state[0] == "S" and current_state[0:2] == "DS":
                self.plan.pop(0)
            self.prev_state = current_state
            blind_sm = BlindWalkSM(current_state, self.ds_expected_duration, self.swing_expected_duration)

            if len(self.plan) == len(self.ref_plan):
                blind_sm.initializeStartTime(self.walking_sm, self.swing_linear_vel / 2, self.state_time, swing_target)
            else:
                blind_sm.initializeStartTime(self.walking_sm, self.swing_linear_vel, self.state_time, swing_target)

            pos_l = self.walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            pos_r = self.walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            ics = []


            t = 0
            for c in range(self.horizon_ts.shape[0]):
                state, trans = blind_sm.updateState(t)
                prop = (t - blind_sm.state_start_time) / blind_sm.swing_expected_durations
                prop = max(min(prop, 1), 0)
                t += self.horizon_ts[c]

                completion_time = blind_sm.state_start_time + blind_sm.swing_expected_durations
                prop2 = t / completion_time
                if state == "SR":
                    pos_c = pos_r
                else:
                    pos_c = pos_l

                if trans == 0 and np.linalg.norm(pos_c - swing_target) < 0.1:
                    link_pos = swing_target * 0.5 + pos_c * 0.5
                elif trans == 0 and state[0] == "S":
                    xy = (pos_c * (1 - prop2) + prop2 * swing_target)[0:2]
                    z = self.gait.swingTrajectory(self.plan[0][2], swing_target, prop)[2]
                    link_pos = np.array([xy[0], xy[1], z * prop2 + pos_c[2] * (1 - prop2)])
                else:

                    link_pos = self.gait.swingTrajectory(self.plan[0][2], swing_target, prop)

                if c == 0:
                    self.get_logger().info("{} {}".format(prop, link_pos))

                if state == "DS_SL" or state == "DS_CL":
                    com_pos = pos_r * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    ic = self.gait.dualSupport(com_pos, None)
                elif state == "DS_SR" or state == "DS_CR":
                    com_pos = pos_l * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    ic = self.gait.dualSupport(com_pos, None)
                elif state == "SL":
                    com_pos = swing_target * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6

                    ic = self.gait.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", link_pos,
                                                 np.array([0,0,0,1]), com_pos)
                else:
                    com_pos = swing_target * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6

                    ic = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", link_pos,
                                                 np.array([0, 0, 0, 1]), com_pos)
                ics += [ic]


            self.get_logger().info("{} {} {}".format(current_state, pos_l, pos_r))
            bpc = BipedalCommand()
            bpc.inverse_timestamps = self.horizon_ts
            bpc.inverse_commands = ics
            bpc.inverse_joints = LEG_JOINTS
            self.publisher2.publish(bpc)


def main(args=None):
    rclpy.init(args=args)

    bpcp = walking_command_pub()

    rclpy.spin(bpcp)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    walking_command_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()