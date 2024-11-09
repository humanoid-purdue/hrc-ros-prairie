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

step_length = 0.25
step_height = 0.1
step_velocity = 0.25/0.4
com_velocity = 0.1/0.2
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
            #self.get_logger().info("{}".format(current_state))

            pos_l = self.walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            pos_r = self.walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            com = self.walking_sm.fwd_poser.getCOMPos()

            if current_state == "SR":
                pos_c = pos_r
            else:
                pos_c = pos_l

            ics = []
            horizon_ts = [0.01, 0.02]



            if current_state[0] == "S":
                xy_i = np.linalg.norm(pos_c[0:2] - initial_pos[0:2])
                xy_f = np.linalg.norm(pos_c[0:2] - swing_target[0:2])
                if xy_i > xy_f:
                    milestone_state = "End"
                    increasing = False
                else:
                    milestone_state = "Half"
                    increasing = True

                target_pos1 = swingTarget(pos_c, swing_target, 0.01, increasing = increasing)
                target_pos2 = swingTarget(target_pos1, swing_target, 0.01, increasing=increasing)

                com_pos1 = target_pos1 * np.array([1, self.com_y_prop, 1])
                com_pos1[2] = 0.6

                com_pos2 = target_pos2 * np.array([1, self.com_y_prop, 1])
                com_pos2[2] = 0.6

                #self.get_logger().info("{}".format(target_pos))

                if current_state == "SR":
                    ic1 = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos1,
                                                 np.array([0, 0, 0, 1]), com_pos1)
                    ic2 = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos2,
                                                 np.array([0, 0, 0, 1]), com_pos2)
                    milestone_state += "_R"
                    ics += [ic1, ic2]
                else:
                    ic1 = self.gait.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", target_pos1,
                                                 np.array([0, 0, 0, 1]), com_pos1)
                    ic2 = self.gait.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", target_pos2,
                                                 np.array([0, 0, 0, 1]), com_pos2)
                    ics += [ic1, ic2]
                    milestone_state += "_L"


            elif current_state[0:2] == "DS" and current_state[-1] == "L":
                com_pos = pos_r * np.array([1, self.com_y_prop, 1])
                com_pos[2] = 0.6
                ic = self.gait.dualSupport(com_pos, None, "right_ankle_roll_link")
                ics += [ic, ic]
                milestone_state = "DS_L"
            elif current_state[0:2] == "DS" and current_state[-1] == "R":
                com_pos = pos_l * np.array([1, self.com_y_prop, 1])
                com_pos[2] = 0.6
                ic = self.gait.dualSupport(com_pos, None, "left_ankle_roll_link")
                milestone_state = "DS_R"
                ics += [ic, ic]




            if milestone_state[0:-1] == "Half_":
                target_pos = swing_target + np.array([0, 0, step_height])
                com_pos = target_pos * np.array([1, self.com_y_prop, 1])
                com_pos[2] = 0.6
                if milestone_state == "Half_L":
                    ic = self.gait.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", target_pos,
                                                 np.array([0, 0, 0, 1]), com_pos)
                else:
                    ic = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                 np.array([0, 0, 0, 1]), com_pos)
                timesize = 0.6 * xy_i / step_velocity
            elif milestone_state[0:-1]  == "End_":
                target_pos = swing_target + np.array([0, 0, 0])
                com_pos = target_pos * np.array([1, self.com_y_prop, 1])
                com_pos[2] = 0.6
                if milestone_state == "End_L":
                    ic = self.gait.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", target_pos,
                                                 np.array([0, 0, 0, 1]), com_pos)
                else:
                    ic = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                 np.array([0, 0, 0, 1]), com_pos)
                timesize = 0.4 * xy_i / step_velocity
            elif milestone_state[0:-1] == "DS_":
                dist = np.linalg.norm(com[0:2] - com_pos[0:2])
                timesize = dist / com_velocity

            ics += [ic, ic]
            horizon_ts += [timesize * 0.2 + horizon_ts[-1], timesize * 1 + horizon_ts[-1]]

            pop_counter = 0
            for c in range(2):
                milestone_state = iterateMilestoneState(milestone_state)
                if milestone_state[:-1] == "DS_":
                    pop_counter += 1
                initial_pos = self.plan[pop_counter][2]
                swing_target = self.plan[pop_counter][1]
                if milestone_state[:-1] == "Half_":
                    target_pos = swing_target + np.array([0, 0, step_height])
                    com_pos = target_pos * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    if milestone_state == "Half_L":
                        ic = self.gait.singleSupport("right_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                     np.array([0, 0, 0, 1]), com_pos)
                    else:
                        ic = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                     np.array([0, 0, 0, 1]), com_pos)
                    timesize = 0.6 * step_length / step_velocity
                elif milestone_state[:-1] == "Full_":
                    target_pos = swing_target + np.array([0, 0, 0])
                    com_pos = target_pos * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    if milestone_state == "Half_L":
                        ic = self.gait.singleSupport("right_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                     np.array([0, 0, 0, 1]), com_pos)
                    else:
                        ic = self.gait.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", target_pos,
                                                     np.array([0, 0, 0, 1]), com_pos)
                    timesize = 0.4 * step_length / step_velocity
                elif milestone_state[:-1] == "DS_":
                    com_pos = self.plan[pop_counter - 1][1] * np.array([1, self.com_y_prop, 1])
                    com_pos[2] = 0.6
                    timesize = com_duration
                    if milestone_state == "DS_L":
                        ic = self.gait.dualSupport(com_pos, None, "right_ankle_roll_link")
                    if milestone_state == "DS_L":
                        ic = self.gait.dualSupport(com_pos, None, "left_ankle_roll_link")
                ics += [ic, ic]
                horizon_ts += [timesize * 0.5 + horizon_ts[-1], timesize * 1 + horizon_ts[-1]]

            bpc = BipedalCommand()
            bpc.inverse_timestamps = horizon_ts
            bpc.inverse_commands = ics
            bpc.inverse_joints = LEG_JOINTS
            self.publisher2.publish(bpc)


            if self.prev_state[0] == "S" and current_state[0:2] == "DS":
                self.plan.pop(0)
            self.prev_state = current_state


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