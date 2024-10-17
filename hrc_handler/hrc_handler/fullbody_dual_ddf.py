
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
from geometry_msgs.msg import Wrench
from pinocchio.shortcuts import buildModelsFromUrdf, createDatas
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
import scipy
import sys
from ament_index_python.packages import get_package_share_directory

helper_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "helpers")

sys.path.append(helper_path)
import helpers

JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
class BipedalPoser():
    def __init__(self, urdf_path, joint_list, leg_joints, left_foot_link, right_foot_link):

        self.joint_list = joint_list
        self.leg_joints = leg_joints

        self.model = pin.buildModelFromUrdf(urdf_path,
                pin.JointModelFreeFlyer())
        self.get_lock_joint()
        self.reduceRobot(None)

        self.add_freeflyer_limits(self.model)
        self.add_freeflyer_limits(self.model_r)

        self.left_foot_link = left_foot_link
        self.right_foot_link = right_foot_link

        self.data_r = self.model_r.createData()

        self.rf_id = self.model_r.getFrameId(right_foot_link)
        self.lf_id = self.model_r.getFrameId(left_foot_link)

        self.q0 = np.zeros([len(self.leg_joints) + 7])
        self.q0[6] = 1
        self.x0 = np.concatenate([self.q0, pin.utils.zero(self.model_r.nv)])

        self.x = self.x0.copy()

        self.state = crocoddyl.StateMultibody(self.model_r)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.mu = 0.2 #friction coefficient

        self.rsurf = np.eye(3)
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.nu)

    def get_lock_joint(self):
        self.lock_joints = []
        for joint in self.joint_list:
            if not(joint in self.leg_joints):
                num = self.model.getJointId(joint)
                if num not in self.lock_joints:
                    self.lock_joints += [num]

    def config2Vec(self, config_dict, reduced = False):
        if reduced:
            num_joints = len(self.leg_joints)
            vec = np.zeros([num_joints])
            if config_dict is None:
                return vec
            for key in config_dict.keys():
                if key in self.leg_joints:
                    index = self.model_r.getJointId(key) - 2
                    vec[index] = config_dict[key]
            return vec
        else:
            num_joints = len(self.joint_list)
            vec = np.zeros([num_joints + 1])
            if config_dict is None:
                return vec
            for key in config_dict.keys():
                index = self.model.getJointId(key) - 2
                vec[index + 1] = config_dict[key]
            return vec

    def setState(self, pos, config_dict, orien = None, vel = None, ang_vel = None, config_vel = None):
        q_joints = self.config2Vec(config_dict, reduced = True)
        if orien is None:
            rot = self.x[3:7]
        else:
            rot = np.array(orien)
        #print("current joints vs desired", q_joints[0:4], self.x[7:11])
        q = np.concatenate([pos, rot, q_joints], axis = 0)
        if config_vel is None:
            jvel = np.zeros([self.model_r.nv - 6])
        else:
            jvel = self.config2Vec(config_vel, reduced = True)
        if vel is None:
            frame_vel = self.x[len(self.leg_joints) + 7:len(self.leg_joints) + 13]
        else:
            if ang_vel is None:
                frame_vel = self.x[len(self.leg_joints) + 7:len(self.leg_joints) + 13]
                frame_vel[0:3] = np.array(vel)
            else:
                frame_vel = np.concatenate([np.array(vel), np.array(ang_vel)], axis = 0)
        vels = np.concatenate([frame_vel, jvel], axis = 0)
        self.x = np.concatenate([q, vels])

    def reduceRobot(self, config_dict):
        vec = self.config2Vec(config_dict)
        self.model_r = pin.buildReducedModel(self.model,
                                                    list_of_joints_to_lock=self.lock_joints,
                                                    reference_configuration=vec)
        self.add_freeflyer_limits(self.model_r)


    def add_freeflyer_limits(self, model):
        ub = model.upperPositionLimit
        ub[:7] = 10
        model.upperPositionLimit = ub
        lb = model.lowerPositionLimit
        lb[:7] = -10
        model.lowerPositionLimit = lb


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
        cost_model.addCost(name + "_wrenchCone", wrench_cone, 1e1)

    def stateCost(self, cost_model, x0 = None, cost = 1e-3):

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
        cost_model.addCost("torque_reg", ctrl_reg, 1e-4)

    def comCost(self, cost_model, target_pos, cost = 1e2):
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, target_pos, self.nu)
        com_track = crocoddyl.CostModelResidual(self.state, com_residual)
        cost_model.addCost("com_track", com_track, cost)

    def linkCost(self, cost_model, target, link_id, name):
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, link_id, target, self.nu)
        foot_track = crocoddyl.CostModelResidual(self.state, frame_placement_residual)
        cost_model.addCost(name + "_track", foot_track, 1e8)

    def linkVelCost(self, cost_model, name, link_id):
        frame_vel_res = crocoddyl.ResidualModelFrameVelocity(
            self.state,link_id, pin.Motion.Zero(),
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
        )
        impulse_cost = crocoddyl.CostModelResidual(self.state, frame_vel_res)
        cost_model.addCost(name + "_impulse", impulse_cost, 1e6)

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
            self.comCost(cost_model, com_target, cost = 1e2 * cost_factor)
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
        self.linkCost(cost_model, pin.SE3(np.eye(3), foot_target), foot_swing_id, "swing_foot")
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
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)

        com_pos = np.array(pin.centerOfMass(self.model_r, self.data_r, q0))

        rf_pos = np.array(self.data_r.oMf[self.rf_id].translation)
        lf_pos = np.array(self.data_r.oMf[self.lf_id].translation)
        return lf_pos, rf_pos, com_pos


    def getJointConfig(self, x):
        names = self.model_r.names.tolist()
        joint_dict = {}
        joint_vels = {}
        for c in range(len(names) - 2):
            joint_dict[names[c+2]] = x[c + 7]
            joint_vels[names[c+2]] = x[c + 13 + len(self.leg_joints)]
        pos = x[0:3]
        return pos, joint_dict, joint_vels



class SquatSM:
    def __init__(self, poser, com_pos):
        self.poser = poser
        self.fast_dt = 0.01
        self.slow_dt = 0.05
        self.com_pos = com_pos
        #self.com_pos = np.array([0., 0., np.sin(time.time()) * 0.1 + 0.5])
        self.ts_xs = None
        self.xs = None
        self.y = None


    def makeSquatProblem(self, timesteps, dt):
        dmodel = self.poser.dualSupportDModel(com_target=self.com_pos)
        model = self.poser.makeD2M(dmodel, dt)
        models = [model] * timesteps
        final = self.poser.makeD2M(dmodel , 0.)
        return models, final


    def simpleNextMPC(self, init_xs):
        traj, final = self.makeSquatProblem(9, 0.02)
        x0 = self.poser.x.copy()
        q0 = x0[0:7+len(LEG_JOINTS)]
        l, r, com = self.poser.getPos(q0)
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverFDDP(problem)
        fddp.th_stop = 1e5
        if init_xs is None:
            init_xs = [x0] * (problem.T + 1)
        init_us = []
        maxiter = 20
        regInit = 0.1
        solved = fddp.solve(init_xs, init_us, maxiter, False, regInit)
        #print(solved)
        xs = np.array(fddp.xs)
        return xs


class fullbody_dual_ddf_gz(Node):
    def __init__(self):
        super().__init__('fullbody_dual_ddf')


        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)


        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        #urdf_config_path = '/home/aurum/PycharmProjects/cropSandbox/urdf/g1.urdf'
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST, LEG_JOINTS, "left_ankle_roll_link", "right_ankle_roll_link")

        self.ji = helpers.JointInterpolation(len(LEG_JOINTS), 0.1, 0.1)

        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.55]))

        self.get_logger().info("Start sub")
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.state_time = None

        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_callback,
            10)

    def state_callback(self, msg):
        names = msg.joint_name
        j_pos = msg.joint_pos
        self.state_time = msg.time
        j_pos_config = dict(zip(names, j_pos))
        pos = msg.pos
        orien = msg.orien_quat
        ang_vel = msg.ang_vel
        self.poser.x[7 + len(LEG_JOINTS):] = 0
        #self.poser.setState(pos, j_pos_config, orien = orien, vel = msg.vel ,config_vel = dict(zip(names, msg.joint_vel)), ang_vel = ang_vel)
        self.poser.setState(pos, j_pos_config)
        self.squat_sm.com_pos = np.array([0., 0., 0.55 + 0.05 * np.cos(self.state_time)])

    def timer_callback(self):
        if self.state_time is None:
            self.get_logger().info("No sim time")
            return
        timeseries = np.arange(10)* 0.02 + self.state_time
        if self.ji.hasHistory():
            x = self.ji.getSeedX(timeseries)
            x[0] = self.poser.x.copy()
        else:
            x = None

        y = self.squat_sm.simpleNextMPC(x)
        self.ji.updateX(timeseries, y)

        self.joint_trajst_publish(timeseries, y)

    def r2whole(self, joint_dict):
        pos_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in joint_dict.keys():
                pos_list[c] = joint_dict[name]
        return pos_list

    def joint_trajst_publish(self, timestamps, y):
        js_list = []
        joint_traj_desired = JointTrajectoryST()
        joint_traj_desired.timestamps = timestamps
        for x0 in y:
            pos, joint_dict, joint_vels = self.poser.getJointConfig(x0)
            js = JointState()
            js.name = JOINT_LIST
            pos_list = self.r2whole(joint_dict)
            vel_list = self.r2whole(joint_vels)

            js0 = JointState()
            js0.name = JOINT_LIST
            js0.position = pos_list
            js0.velocity = vel_list
            js_list += [js0]
        joint_traj_desired.jointstates = js_list
        self.joint_traj_pub.publish(joint_traj_desired)

def ddf_gz():
    rclpy.init(args=None)

    fb = fullbody_dual_ddf_gz()

    rclpy.spin(fb)

    fb.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    ddf_gz()