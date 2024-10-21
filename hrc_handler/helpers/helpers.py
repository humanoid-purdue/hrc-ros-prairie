
import numpy as np
import scipy
import crocoddyl
import pinocchio as pin
import os

JOINT_LIST = ['pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'head_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'logo_joint', 'imu_joint', 'left_palm_joint', 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint', 'right_palm_joint', 'right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']

LEG_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']


class JointInterpolation:
    def __init__(self, joint_num, position_error, velocity_error):
        self.pos_err = position_error
        self.vel_err = velocity_error
        self.joint_num = joint_num

        self.pos_arr = None
        self.vel_arr = None
        self.timelist = None
        self.cs_pos = None
        self.cs_vel = None
        self.cs_tau = None
        self.cs_centroid = None


        self.consecutive_fails = 0
        self.fail_thresh = 5

    def updateJointState(self, timelist, pos, vel, centroid_vec = None):
        #Of shape (seq len, joint num)
        #make existing sequency with cs and interpolate and filter
        if self.cs_pos is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            new_pos = pos * 0.5 + match_pos * 0.5
            new_vel = vel * 0.5 + match_vel * 0.5
        else:
            new_pos = pos
            new_vel = vel
        cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis = 0)
        cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis = 0)
        if centroid_vec is not None:
            self.cs_centroid = scipy.interpolate.CubicSpline(timelist, centroid_vec, axis = 0)

        if self.pos_arr is None:
            self.pos_arr = pos
            self.vel_arr = vel
            self.timelist = timelist
            self.cs_pos = cs_pos
            self.cs_vel = cs_vel
            return True, 0, 0
        check_pos = cs_pos(timelist[1])
        check_vel = cs_vel(timelist[1])
        inrange = self.checkDelta(timelist[1], check_pos, check_vel)
        if inrange:
            self.pos_arr = pos
            self.vel_arr = vel
            self.timelist = timelist
            self.cs_pos = cs_pos
            self.cs_vel = cs_vel
            return True, np.mean(np.abs(pos - check_pos)), np.mean(np.abs(vel - check_vel))
        else:
            self.consecutive_fails += 1
            if self.consecutive_fails > self.fail_thresh:
                self.consecutive_fails = 0
                self.pos_arr = pos
                self.vel_arr = vel
                self.timelist = timelist
                self.cs_pos = cs_pos
                self.cs_vel = cs_vel
            return False, np.mean(np.abs(pos - check_pos)), np.mean(np.abs(vel - check_vel))

    def forceUpdateState(self, timelist, pos, vel, efforts):
        if self.cs_pos is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            match_tau = self.cs_tau(timelist)

            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * (( np.arange(sz) - 1 ) / sz )
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = vel * weight_vec + match_vel * (1 - weight_vec)
            new_tau = efforts * weight_vec + match_tau * (1 - weight_vec)
        else:
            new_pos = pos
            new_vel = vel
            new_tau = efforts
        self.cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis=0)
        self.cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis=0)
        self.cs_tau = scipy.interpolate.CubicSpline(timelist, new_tau, axis=0)

    def updateX(self, timelist, x):
        centroid_pos = x[:, 0:7]
        pos = x[:, 7:7 + self.joint_num]
        centroid_vel = x[:, 7 + self.joint_num: 13 + self.joint_num]
        vel = x[:, 13 + self.joint_num:]
        centroid = np.concatenate([centroid_pos, centroid_vel], axis = 1)
        return self.updateJointState(timelist, pos, vel, centroid_vec = centroid)


    def checkDelta(self, check_time, pos, vel):
        check_pos = self.cs_pos(check_time)
        check_vel = self.cs_vel(check_time)
        pc = np.sum(np.abs(pos - check_pos) < self.pos_err)
        vc = np.sum(np.abs(vel - check_vel) < self.vel_err)
        return pc == 0 and vc == 0

    def getInterpolation(self, timestamp, pos_delta = 0):
        pos = self.cs_pos(timestamp + pos_delta)
        vel = self.cs_vel(timestamp)
        if self.cs_tau is None:
            return pos, vel
        else:
            tau = self.cs_tau(timestamp)
            return pos, vel, tau

    def getX(self, t):
        pos, vel = self.getInterpolation(t)
        centroid = self.cs_centroid(t)
        centroid[3:7] = centroid[3:7] / (np.sum(centroid[3:7] ** 2) ** 0.5)
        x0 = np.concatenate([centroid[:7], pos, centroid[7:], vel], axis=0)
        return x0

    def getSeedX(self, timestamps):
        x = []
        for t in timestamps:
            x0 = self.getX(t)
            x += [x0]
        return x

    def hasHistory(self):
        return not(self.cs_pos is None)

class SignalFilter:
    def __init__(self, params, freq, cutoff):
        #self.b, self.a = scipy.signal.butter(4, cutoff, btype='low', analog=False, fs = freq)
        nyquist = 0.5 * freq
        normal_cutoff = cutoff / nyquist
        self.sos = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False, output='sos')
        self.zi = []
        self.y = np.zeros(params)
        for c in range(params):
            self.zi += [scipy.signal.sosfilt_zi(self.sos)]

    def update(self, vec):
        for c in range(vec.shape[0]):
            filtered_point, self.zi[c] = scipy.signal.sosfilt(self.sos, vec[c:c+1], zi=self.zi[c], axis = 0)
            self.y[c] = filtered_point[0]
    def get(self):
        return self.y

class CSVDump:
    def __init__(self, params, name_list):
        self.abs_path = "/home/aurum/RosProjects/prairie/datadump"
        self.name_list = name_list
        self.max_num = 5000
        self.arr = np.zeros([self.max_num, params , len(name_list)])

    def update(self, vecs):
        self.arr = np.roll(self.arr, -1, axis = 0)
        for c in range(len(vecs)):
            self.arr[-1, :, c] = vecs[c]

    def save(self):
        if os.path.exists(self.abs_path):
            for c in range(len(self.name_list)):
                name = self.name_list[c]
                path = self.abs_path + "/{}.csv".format(name)
                np.savetxt(path, self.arr[:, :, c], delimiter = ',')

class discreteIntegral:
    def __init__(self, params):
        self.integral = np.zeros([params])
        self.prev_time = -1
    def update(self, timestamp, x):
        if self.prev_time == -1:
            self.prev_time = timestamp
        else:
            self.integral += x * (timestamp - self.prev_time)
        return self.integral

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
        self.pelvis_id = self.model_r.getFrameId("pelvis")

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
        cost_model.addCost(name + "_wrenchCone", wrench_cone, 1e2)

    def stateCost(self, cost_model, x0 = None, cost = 4e-3):

        state_weights = np.array(
            [0] * 3 + [5000.0] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv
        )
        if x0 is None:
            self.x0[3:7] = np.array([0., 0., 0., 1])
            state_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        else:
            x0[3:7] = np.array([0., 0., 0., 1])
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

    def linkCost(self, cost_model, target, link_id, name, cost = 1e8):
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, link_id, target, self.nu)
        foot_track = crocoddyl.CostModelResidual(self.state, frame_placement_residual)
        cost_model.addCost(name + "_track", foot_track, cost)

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
            self.linkCost(cost_model, pin.SE3(np.eye(3), com_target + np.array([0., 0., 0.1])), self.pelvis_id, "pelvis", cost = 1e3)
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


    def getJointConfig(self, x, efforts = None):
        names = self.model_r.names.tolist()
        joint_dict = {}
        joint_vels = {}
        joint_efforts = {}
        for c in range(len(names) - 2):
            joint_dict[names[c+2]] = x[c + 7]
            joint_vels[names[c+2]] = x[c + 13 + len(self.leg_joints)]
            if efforts is not None:
                joint_efforts[names[c+2]] = efforts[c]
        pos = x[0:3]
        return pos, joint_dict, joint_vels, joint_efforts



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
        self.us = None


    def makeSquatProblem(self, timesteps, dt):
        dmodel = self.poser.dualSupportDModel(com_target=self.com_pos)
        model = self.poser.makeD2M(dmodel, dt)
        models = [model] * timesteps
        final = self.poser.makeD2M(dmodel , 0.)
        return models, final


    def simpleNextMPC(self, init_xs):
        traj, final = self.makeSquatProblem(9, 0.03)
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
        self.us = np.array(fddp.us)
        xs = np.array(fddp.xs)
        return xs

if __name__ == "__main__":
    v1 = JointInterpolation(5, 0.1, 0.1)
    data = np.ones([10, 5])
    v1.forceUpdateState(np.arange(10), data, data, data)
    print(v1.cs_pos(np.arange(10)).shape)