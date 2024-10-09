
import time
import numpy as np
import crocoddyl
import pinocchio
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
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
from hrc_msgs.msg import StateVector
import os
from pinocchio.shortcuts import buildModelsFromUrdf, createDatas
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
import scipy
JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

class SuperRobot:
    def __init__(self, urdf_path, joint_list, leg_joints, default_pos, model, logger = None):
        self.path = urdf_path
        self.joint_list = joint_list
        self.legs_joints = leg_joints
        self.default_pos = default_pos
        builder = RobotWrapper.BuildFromURDF
        logger("Reached building")

        self.robot = RobotWrapper(model = model ,visual_model= None, collision_model=None)
        logger("{}".format(model))

        self.initial_joints_config = np.zeros([len(self.joint_list) + 1])
        model = self.robot.model.copy()
        self.get_lock_joint()
        model_reduced = pin.buildReducedModel(model,
                                              list_of_joints_to_lock = self.lock_joints,
                                              reference_configuration = self.initial_joints_config)
        self.robot_reduced = RobotWrapper(model = model_reduced, visual_model = None, collision_model= None)
        self.add_freeflyer_limits(self.robot)
        self.add_freeflyer_limits(self.robot_reduced)
        self.robot.q0 = self.overwrite_floating_pose(self.robot)
        self.robot_reduced.q0 = self.overwrite_floating_pose(self.robot_reduced)
        logger("12345")


    def add_freeflyer_limits(self, robot):
        ub = robot.model.upperPositionLimit
        ub[:7] = 10
        robot.model.upperPositionLimit = ub
        lb = robot.model.lowerPositionLimit
        lb[:7] = -10
        robot.model.lowerPositionLimit = lb

    def get_lock_joint(self):
        self.lock_joints = []
        for joint in self.joint_list:
            if not(joint in self.legs_joints):
                num = self.robot.model.getJointId(joint)
                if num not in self.lock_joints:
                    self.lock_joints += [num]

    def reorder(self, joint_config, joint_names, reduced = True):
        new_array = np.zeros([len(JOINT_LIST) + 1])
        for c in range(len(joint_names)):
            name = joint_names[c]

            if reduced:
                index = self.robot.model.getJointId(name)
            else:
                index = self.robot_reduced.model.getJointId(name)
            new_array[index - 2] = joint_config[c]
        return new_array

    def setJoints(self, joint_config, joint_names):
        new_config = self.reorder(joint_config, joint_names, reduced = False)
        model_reduced = pin.buildReducedModel(self.robot.model,
                                                    list_of_joints_to_lock=self.lock_joints,
                                                    reference_configuration=new_config)
        self.robot_reduced = RobotWrapper(model=model_reduced, visual_model=None, collision_model=None)


    def overwrite_floating_pose(self, robot):
        q0 = np.array(robot.q0.copy())
        q0[0:7] = np.concatenate([self.default_pos, np.array([0, 0., 0., 1.])])
        return q0

class BipedalPoser():
    def __init__(self, super_robot, left_foot_link, right_foot_link, root_link):
        self.root_link = root_link
        self.left_foot_link = left_foot_link
        self.right_foot_link = right_foot_link

        self.model_s = super_robot
        self.model_r = self.model_s.robot_reduced.model
        self.data_r = self.model_r.createData()

        self.rf_id = self.model_r.getFrameId(right_foot_link)
        self.lf_id = self.model_r.getFrameId(left_foot_link)
        #self.r_id = self.model_r.getFrameId(root_link)

        self.q0 = super_robot.robot_reduced.q0
        self.x0 = np.concatenate([self.q0, pinocchio.utils.zero(super_robot.robot_reduced.model.nv)])

        self.state = crocoddyl.StateMultibody(self.model_r)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.mu = 0.7 #friction coefficient

        self.rsurf = np.eye(3)
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.nu)




    def addContact(self,contact_model, name, id):
        contact = crocoddyl.ContactModel6D(
            self.state,
            id,
            pin.SE3.Identity(),
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
            np.array([0, 50]), )
        contact_model.addContact(name + "_contact", contact)

    def frictionConeCost(self, cost_model, name, id):
        cone = crocoddyl.WrenchCone(self.rsurf, self.mu, np.array([0.1, 0.05]))
        wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
            self.state, id, cone, self.nu
        )
        wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub)
        )
        wrench_cone = crocoddyl.CostModelResidual(
            self.state, wrench_activation, wrench_residual
        )
        cost_model.addCost(name + "_wrenchCone", wrench_cone, 1e3)

    def stateCost(self, cost_model, x0 = None, cost = 1e2):

        state_weights = np.array(
            [0] * 3 + [500.0] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv
        )
        if x0 is None:
            state_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        else:
            state_residual = crocoddyl.ResidualModelState(self.state, x0, self.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights**2)
        state_reg = crocoddyl.CostModelResidual(
            self.state, state_activation, state_residual
        )
        cost_model.addCost("state_reg", state_reg, cost)

    def stateTorqueCost(self, cost_model):
        ctrl_residual = crocoddyl.ResidualModelJointEffort(
            self.state, self.actuation, self.nu
        )
        ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        cost_model.addCost("torque_reg", ctrl_reg, 1e-1)

    def comCost(self, cost_model, target_pos, cost = 1e9):
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, target_pos, self.nu)
        com_track = crocoddyl.CostModelResidual(self.state, com_residual)
        cost_model.addCost("com_track", com_track, cost)

    def linkCost(self, cost_model, target, link_id, name):
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, link_id, target, self.nu)
        foot_track = crocoddyl.CostModelResidual(self.state, frame_placement_residual)
        cost_model.addCost(name + "_track", foot_track, 1e8)

    def linkVelCost(self, cost_model, name, link_id):
        frame_vel_res = crocoddyl.ResidualModelFrameVelocity(
            self.state,link_id, pinocchio.Motion.Zero(),
            pinocchio.LOCAL_WORLD_ALIGNED,
            self.nu,
        )
        impulse_cost = crocoddyl.CostModelResidual(self.state, frame_vel_res)
        cost_model.addCost(name + "_impulse", impulse_cost, 1e6)

    def standModel(self, desired_com):
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        self.addContact(contact_model, "left", self.lf_id)
        self.addContact(contact_model, "right", self.rf_id)
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)
        self.frictionConeCost(cost_model, "left", self.lf_id)
        self.frictionConeCost(cost_model, "right", self.rf_id)
        self.stateCost(cost_model)
        self.stateTorqueCost(cost_model)
        self.comCost(cost_model, desired_com)
        model = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_model, cost_model
        )
        return model




    def dualSupportModel(self, ts, dt):
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        self.addContact(contact_model, "left", self.lf_id)
        self.addContact(contact_model, "right", self.rf_id)
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)
        self.frictionConeCost(cost_model, "left", self.lf_id)
        self.frictionConeCost(cost_model, "right", self.rf_id)
        self.stateCost(cost_model)
        self.stateTorqueCost(cost_model)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_model, cost_model
        )
        traj = [crocoddyl.IntegratedActionModelEuler(dmodel, self.control, dt)] * ts
        final = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, 0.0)
        return traj, final

    def dualSupportDModel(self, com_target = None, x0_target = None, cost_factor = 1):
        if x0_target is None:
            state_cost = 1e1
        else:
            state_cost = 1e5 * cost_factor
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        self.addContact(contact_model, "left", self.lf_id)
        self.addContact(contact_model, "right", self.rf_id)
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)
        self.frictionConeCost(cost_model, "left", self.lf_id)
        self.frictionConeCost(cost_model, "right", self.rf_id)
        self.stateCost(cost_model, x0 = x0_target, cost = state_cost)
        self.stateTorqueCost(cost_model)
        if com_target is not None:
            self.comCost(cost_model, com_target, cost = 1e6 * cost_factor)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_model, cost_model
        )
        return dmodel

    def singleSupportDModel(self, foot_gnd_id, foot_swing_id, com_target, foot_target, x0_target = None, cost_factor = 1):
        if x0_target is None:
            state_cost = 1e1
        else:
            state_cost = 1e5 * cost_factor
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        self.addContact(contact_model, "ground_foot", foot_gnd_id)
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)
        self.frictionConeCost(cost_model, "ground_foot", foot_gnd_id)
        self.stateCost(cost_model, x0 = x0_target, cost = state_cost)
        self.stateTorqueCost(cost_model)
        self.comCost(cost_model, com_target, cost = 1e5)
        self.linkCost(cost_model, pinocchio.SE3(np.eye(3), foot_target), foot_swing_id, "swing_foot")
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_model,
                                                                     cost_model)
        return dmodel

    def singleFinalDModel(self, foot_gnd_id, foot_swing_id, x0_target = None, cost_factor = 1):
        if x0_target is None:
            state_cost = 1e1
        else:
            state_cost = 1e5 * cost_factor
        contact_end = crocoddyl.ContactModelMultiple(self.state, self.nu)
        self.addContact(contact_end, "ground_foot", foot_gnd_id)
        cost_end = crocoddyl.CostModelSum(self.state, self.nu)
        self.frictionConeCost(cost_end, "ground_foot", foot_gnd_id)
        self.stateCost(cost_end, x0 = x0_target, cost = state_cost)
        self.stateTorqueCost(cost_end)
        self.linkVelCost(cost_end, "swing_foot", foot_swing_id)
        dmodel_end = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_end,
                                                                         cost_end)
        return dmodel_end

    def makeD2M(self, dmodel, dt):
        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, dt)
        return model


    # q0 is the robot pos sstate
    def getPos(self, q0):
        pinocchio.forwardKinematics(self.model_r, self.data_r, q0)
        pinocchio.updateFramePlacements(self.model_r, self.data_r)

        com_pos = np.array(pinocchio.centerOfMass(self.model_r, self.data_r, q0))

        rf_pos = np.array(self.data_r.oMf[self.rf_id].translation)
        lf_pos = np.array(self.data_r.oMf[self.lf_id].translation)
        return lf_pos, rf_pos, com_pos


class InterpolatePose:
    def __init__(self, ts_xs, get_pos, nq):
        horizon = 5
        self.timestamps = np.array(ts_xs[0])[: horizon]
        xvals = ts_xs[1].copy()[:horizon, :]
        pos_arr = np.zeros([len(self.timestamps), 9])
        for c in range(len(self.timestamps)):

            lf, rf, com = get_pos(xvals[c, :nq])
            pos_arr[c, :] = np.concatenate([lf, rf, com], axis = 0)

        cs_list = np.concatenate([xvals, pos_arr], axis = 1)
        self.cs_list = cs_list
        self.nq = nq
        self.x_size = xvals.shape[1]


    def getInterpolation(self, ct):
        #y = self.cs(time)
        st = time.time()
        y = np.zeros([self.cs_list.shape[1]])
        for c in range(self.cs_list.shape[1]):
            y[c] = np.interp(ct, self.timestamps, self.cs_list[:, c])
        xs_mix = y[:self.x_size]
        xs_mix[3:7] = xs_mix[3:7] / (np.sum(xs_mix[3:7] ** 2) ** 0.5)
        pos_list = y[self.x_size:]
        lf = pos_list[0:3]
        rf = pos_list[3:6]
        com = pos_list[6:9]
        return xs_mix, lf, rf, com


class SquatSM:
    def __init__(self, robot, com_pos, left_foot, right_foot):
        self.poser = BipedalPoser(robot, left_foot, right_foot, None)
        self.fast_dt = 0.01
        self.slow_dt = 0.05
        self.com_pos = com_pos
        #self.com_pos = np.array([0., 0., np.sin(time.time()) * 0.1 + 0.5])
        self.ts_xs = None
        self.xs = None

    def setState(self, x0):
        self.poser.x0 = x0
        self.poser.model_s.robot_reduced.q0 = x0[:self.poser.state.nq]
        self.state_update_time = time.time()
        self.com_pos = np.array([0., 0., np.sin(time.time()) * 0.1 + 0.5])

    def makeSquatProblem(self, timesteps, dt, ts_xs = None):
        timestamps = (np.arange(timesteps) + 1) * dt + time.time()
        if ts_xs is None:
            dmodel = self.poser.dualSupportDModel(com_target = self.com_pos)
            model = self.poser.makeD2M(dmodel, dt)
            models = [model] * timesteps
            final = self.poser.makeD2M(dmodel , 0.)
            return timestamps, models, final
        interp = InterpolatePose(ts_xs, self.poser.getPos, self.poser.state.nq)
        models = []
        for c in range(timesteps):
            x0, lf, rf, com = interp.getInterpolation(timestamps[c])
            dmodel = self.poser.dualSupportDModel(com_target = com, x0_target = x0)
            models += [self.poser.makeD2M(dmodel, dt)]
        final = self.poser.makeD2M(dmodel , 0.)
        return timestamps, models, final

    def longHorizonSolver(self):
        st = time.time()
        timestamps, traj, final = self.makeSquatProblem(9, 0.01)
        x0 = self.poser.x0.copy()
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverFDDP(problem)
        fddp.th_stop = 1e5
        init_xs = [x0] * (problem.T + 1)
        init_us = []
        maxiter = 15
        regInit = 0.1
        fddp.solve(init_xs, init_us, maxiter, False, regInit)
        xs = np.array(fddp.xs)
        us = np.array(fddp.us)
        self.ts_xs = (timestamps, xs[1:, :], us)
        return self.ts_xs


class fullbody_dual_ddf(Node):
    def __init__(self):
        super().__init__('fullbody_dual_ddf')

        self.joint_pos = None
        self.names = None
        self.x0 = None

        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)


        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        model = pinocchio.buildModelFromUrdf(urdf_config_path,
                pinocchio.JointModelFreeFlyer())

        #urdf_config_path = '/home/aurum/PycharmProjects/cropSandbox/urdf/g1.urdf'
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.joint_traj_pub2 = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)



        robot_s = SuperRobot(urdf_config_path, JOINT_LIST, LEG_JOINTS, np.array([0. , 0. , 0.7]), model, logger = self.get_logger().info)
        self.get_logger().info("abcdefg")

        self.squat_sm = SquatSM(robot_s, np.array([0.00, 0., 0.7]), 'left_ankle_roll_link', 'right_ankle_roll_link')
        self.subscription_pos = self.create_subscription(
            StateVector,
            '/state_vector',
            self.state_vec_callback,
            10
        )
        self.get_logger().info("Start sub")
        self.timer = self.create_timer(0.01, self.timer_callback)
        #while rclpy.ok():
        #    if self.joint_pos is not None:
        #        self.get_logger().info("Run ddf")
        #        self.squat_sm.poser.model_s.setJoints(self.joint_pos, self.names)
        #        self.squat_sm.setState(self.x0)
        #        ts_xs = self.squat_sm.longHorizonSolver()
        #        self.tsxs2JointTraj(ts_xs)


    def timer_callback(self):
        if self.joint_pos is not None:
            self.squat_sm.poser.model_s.setJoints(self.joint_pos, self.names)
            self.squat_sm.setState(self.x0)
            ts_xs = self.squat_sm.longHorizonSolver()
            self.tsxs2JointTraj(ts_xs)
            self.joint_effort_publish(ts_xs)


    def joint_effort_publish(self, ts_xs):
        x = ts_xs[0]
        y = ts_xs[2]
        cs = scipy.interpolate.CubicSpline(x, y, axis = 0)
        current_effort = cs(time.time())
        jtp = JointTrajectoryPoint()
        efforts = np.zeros([len(JOINT_LIST)])
        for c in range(len(LEG_JOINTS)):
            name = LEG_JOINTS[c]
            efforts[JOINT_LIST.index(name)] = current_effort[c]

        joint_traj = JointTrajectory()
        joint_traj.joint_names = JOINT_LIST

        jtp.effort = efforts
        duration = Duration()
        duration.sec = 9999
        duration.nanosec = 0
        jtp.time_from_start = duration

        joint_traj.points = [jtp]

        self.joint_traj_pub2.publish(joint_traj)

    def state_vec_callback(self, msg):
        names = msg.joint_name
        joint_pos = np.array(msg.joint_pos)
        self.joint_pos = joint_pos
        self.names = names

        pos_vec = np.array(msg.pos_vec)
        orien_quat = np.array(msg.orien_quat)

        self.q_global = np.concatenate([pos_vec, orien_quat], axis = 0)

        q_joint = np.zeros([len(LEG_JOINTS)])
        for leg_joint_name in LEG_JOINTS:
            index = self.squat_sm.poser.model_s.robot_reduced.model.getJointId(leg_joint_name)
            q_joint[index - 2] = joint_pos[names.index(leg_joint_name)]
        self.q0 = np.concatenate([self.q_global, q_joint], axis = 0)
        l, r, com = self.squat_sm.poser.getPos(self.q0)
        #ave_feet_pos = (l + r) * 0.5
        #new_pos_vec = np.array([0., 0., ave_feet_pos[2] * -1])
        #self.q_global = np.concatenate([new_pos_vec, orien_quat], axis=0)
        self.q0 = np.concatenate([self.q_global, q_joint], axis=0)

        v0 = pinocchio.utils.zero(self.squat_sm.poser.model_s.robot_reduced.model.nv)
        self.x0 = np.concatenate([self.q0, v0], axis = 0)

    def tsxs2JointTraj(self, ts_xs):
        joint_traj_desired = JointTrajectoryST()
        joint_traj_desired.jointstates = []
        joint_traj_desired.timestamps = ts_xs[0]
        for c in range(len(ts_xs[0])):
            joint_state = JointState()
            joint_pos = ts_xs[1][c, 7:7 + len(LEG_JOINTS)]
            joint_vel = ts_xs[1][c, 13 + len(LEG_JOINTS):]
            joint_state.name = JOINT_LIST
            joint_state.position = [0] * len(JOINT_LIST)
            joint_state.velocity = [0] * len(JOINT_LIST)
            for c in range(len(JOINT_LIST)):
                name = JOINT_LIST[c]
                if name in LEG_JOINTS:
                    index = self.squat_sm.poser.model_s.robot_reduced.model.getJointId(name)
                    joint_state.position[c] = joint_pos[index - 2]
                    joint_state.velocity[c] = joint_vel[index -2]
            joint_traj_desired.jointstates += [joint_state]
        self.joint_traj_pub.publish(joint_traj_desired)


def main():
    rclpy.init(args=None)

    fb = fullbody_dual_ddf()

    rclpy.spin(fb)

    fb.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()