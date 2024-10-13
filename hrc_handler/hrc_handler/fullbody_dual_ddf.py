
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

    def stateCost(self, cost_model, x0 = None, cost = 1e1):

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

    def comCost(self, cost_model, target_pos, cost = 1e5):
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

    def rneaTest(self, q0):
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)
        v0 = np.zeros([len(self.leg_joints) + 6])
        a0 = v0.copy()

        zero_force = pin.Force(np.array([0., 0., 0.]), np.zeros([3]))
        grav_force = pin.Force(np.array([0., 0., -500]), np.zeros([3]))

        contact_list = [zero_force] * (len(self.leg_joints))
        for c in range(len(self.leg_joints)):
            index = self.model_r.getJointId(self.leg_joints[c]) - 2
            if self.leg_joints[c] == "right_ankle_roll_joint" or self.leg_joints[c] == "left_ankle_roll_joint":
                contact_list[index] = pin.Force(np.array([0., 0., 250]), np.zeros([3]))
        std_vec = pin.StdVec_Force()
        std_vec.append(zero_force)
        std_vec.append(grav_force)
        for c in range(len(self.leg_joints)):
            std_vec.append(contact_list[c])
        pin.rnea(self.model_r, self.data_r, q0, v0, a0, std_vec)
        #pin.computeStaticTorque()
        #print(self.data_r.tau[6:], "sfbf")
        return self.data_r.tau[6:]

        #print(self.data_r.tau, "RNEA")



    # q0 is the robot pos sstate
    def getPos(self, q0):
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)

        com_pos = np.array(pin.centerOfMass(self.model_r, self.data_r, q0))

        rf_pos = np.array(self.data_r.oMf[self.rf_id].translation)
        lf_pos = np.array(self.data_r.oMf[self.lf_id].translation)
        return lf_pos, rf_pos, com_pos

    def getGravTorques(self, q0, efforts = None):
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)
        #pin.computeGeneralizedGravity(self.model_r, self.data_r, q0)
        v0 = np.zeros([len(self.leg_joints) + 6])


        gravs = self.rneaTest(q0)
        pin.computeAllTerms(self.model_r, self.data_r, q0, v0)
        bad_gravs = self.data_r.g[6:]
        print(gravs[0:6], bad_gravs[0:6], "GRAVS")
        if efforts is not None:
            #gravs = -0 * efforts + gravs
            gravs = gravs #-1 * efforts
        names = self.model_r.names.tolist()
        joint_grav = {}
        for c in range(len(names) - 2):
            joint_grav[names[c+2]] = gravs[c]
        return joint_grav

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

        self.prev_xs = None
        self.prev_us = None

    def makeSquatProblem(self, timesteps, dt):
        dmodel = self.poser.dualSupportDModel(com_target=self.com_pos)
        model = self.poser.makeD2M(dmodel, dt)
        models = [model] * timesteps
        final = self.poser.makeD2M(dmodel , 0.)
        return models, final


    def simpleNextMPC(self):
        traj, final = self.makeSquatProblem(9, 0.02)
        x0 = self.poser.x.copy()
        q0 = x0[0:7+len(LEG_JOINTS)]
        l, r, com = self.poser.getPos(q0)

        pin.computeGeneralizedGravity(self.poser.model_r, self.poser.data_r, q0)
        #print(self.poser.data_r.g, len(self.poser.data_r.g))

        self.com_pos[0:2] = ((l + r) * 0.5)[0:2]
        #print(l,r,com)
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverFDDP(problem)
        fddp.th_stop = 1e5
        if self.prev_xs is None:
            init_xs = [x0] * (problem.T + 1)
            init_us = []
        else:
            init_xs = [x0]
            for c in range(self.prev_xs.shape[0]):
                init_xs += [self.prev_xs[c, :]]
            init_us = []
        maxiter = 10
        regInit = 0.1
        solved = fddp.solve(init_xs, init_us, maxiter, False, regInit)
        #print(solved)
        xs = np.array(fddp.xs)
        us = np.array(fddp.us)
        f_l, f_r, f_com = self.poser.getPos(xs[1, 0:7 + len(LEG_JOINTS)])
        #print(xs[1, 7:8], xs[2, 7:8], xs[0, 7:8], xs[0, 0:3], com, f_com)
        #print(us[2, 0:4], us.shape)
        usy = us[0,:]
        #print(usy)
        #print(us[0,:])
        if solved:

            self.prev_us = us
            self.prev_xs = np.concatenate([xs[2:,:], xs[-1:, :]], axis = 0)
            self.y = xs[1,:]
            return self.y, usy, solved
        else:
            self.prev_us = None
            self.prev_xs = None
            return xs[1,:], usy, False


class fullbody_dual_ddf_rviz(Node):
    def __init__(self):
        super().__init__('fullbody_dual_ddf')

        self.joint_pos = None
        self.names = None
        self.x0 = None

        qos_profile = QoSProfile(depth=10)
        #self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        #urdf_config_path = '/home/aurum/PycharmProjects/cropSandbox/urdf/g1.urdf'
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST, LEG_JOINTS, "left_ankle_roll_link", "right_ankle_roll_link")
        self.x = self.poser.x.copy()
        self.x[0:3] = np.array([0., 0., 0.75])

        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))
        self.poser.x = self.x

        self.get_logger().info("Start sub")
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.c = 0

    def timer_callback(self):
        y, _ = self.squat_sm.simpleNextMPC()
        self.joint_state_publish(y)

    def joint_state_publish(self, y):
        self.c +=1
        pos, joint_dict, _ = self.poser.getJointConfig(y)
        if self.c == 100:
            self.c = 0
            y[7:7 + len(LEG_JOINTS)] = y[7:7 + len(LEG_JOINTS)]

        self.poser.x = y.copy()
        js = JointState()
        js.name = JOINT_LIST
        pos_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in joint_dict.keys():
                pos_list[c] = joint_dict[name]
        now = self.get_clock().now()
        js.header.stamp = now.to_msg()
        js.position = pos_list
        self.joint_state_pub.publish(js)

class fullbody_dual_ddf_gz(Node):
    def __init__(self):
        super().__init__('fullbody_dual_ddf')

        self.joint_pos = None
        self.names = None
        self.x0 = None
        self.efforts = np.zeros([len(LEG_JOINTS)])

        qos_profile = QoSProfile(depth=10)
        self.joint_traj_pub = self.create_publisher(JointTrajectoryST, 'joint_trajectory_desired', qos_profile)

        self.joint_grav_pub = self.create_publisher(JointState, 'joint_grav', qos_profile)
        #self.joint_state_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")

        #urdf_config_path = '/home/aurum/PycharmProjects/cropSandbox/urdf/g1.urdf'
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

        self.poser = BipedalPoser(urdf_config_path, JOINT_LIST, LEG_JOINTS, "left_ankle_roll_link", "right_ankle_roll_link")
        self.x = self.poser.x.copy()
        self.x[0:3] = np.array([0., 0., 0.75])

        self.squat_sm = SquatSM(self.poser, np.array([0.00, 0., 0.65]))
        self.poser.x = self.x

        self.get_logger().info("Start sub")
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.subscription = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_callback,
            10)

    def state_callback(self, msg):
        names = msg.joint_name
        j_pos = msg.joint_pos
        j_vel = msg.joint_vel
        j_pos_config = dict(zip(names, j_pos))
        j_vel_config = dict(zip(names, j_vel))
        pos = msg.pos
        orien = msg.orien_quat
        #pos = self.poser.x[0:3]
        self.poser.x[7 + len(LEG_JOINTS):] = 0
        self.poser.setState(pos, j_pos_config, orien = orien)

        q0 = self.poser.x[0:7 + len(LEG_JOINTS)]
        grav_dict = self.poser.getGravTorques(q0, efforts = self.efforts)
        joint_grav = JointState()
        joint_grav.name = JOINT_LIST
        grav_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in grav_dict.keys():
                grav_list[c] = grav_dict[name]
        joint_grav.effort = grav_list

        #self.poser.rneaTest(q0)

        self.joint_grav_pub.publish(joint_grav)

        self.squat_sm.com_pos = np.array([0., 0., 0.6 + 0.05 * np.sin(0.8 * time.time())])

        #self.poser.setState(pos, j_pos_config, orien = orien, config_vel = j_vel_config)

    def timer_callback(self):

        y, us, solved = self.squat_sm.simpleNextMPC()
        self.efforts = us
        if solved:
            self.x = y.copy()
        self.joint_trajst_publish(y)

    def joint_trajst_publish(self, y):
        pos, joint_dict, joint_vels = self.poser.getJointConfig(y)
        self.poser.x = y.copy()
        js = JointState()
        js.name = JOINT_LIST
        pos_list = [0.] * len(JOINT_LIST)
        vel_list = [0.] * len(JOINT_LIST)
        for c in range(len(JOINT_LIST)):
            name = JOINT_LIST[c]
            if name in joint_dict.keys():
                pos_list[c] = joint_dict[name]
                vel_list[c] = joint_vels[name]

        joint_traj_desired = JointTrajectoryST()
        joint_traj_desired.timestamps = [time.time() + 0.02]
        js = JointState()
        js.name = JOINT_LIST
        js.position = pos_list
        js.velocity = vel_list
        joint_traj_desired.jointstates = [js]

        self.joint_traj_pub.publish(joint_traj_desired)

def ddf_gz():
    rclpy.init(args=None)

    fb = fullbody_dual_ddf_gz()

    rclpy.spin(fb)

    fb.destroy_node()
    rclpy.shutdown()

def ddf_rviz():
    rclpy.init(args=None)

    fb = fullbody_dual_ddf_rviz()

    rclpy.spin(fb)

    fb.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ddf_rviz()