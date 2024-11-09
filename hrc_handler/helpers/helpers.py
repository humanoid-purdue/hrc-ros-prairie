try:
    import numpy as np
    import scipy
    import crocoddyl
    import pinocchio as pin
except:
    print("Unable to load PY dependencies")

try:
    from geometry_msgs.msg import Point, Pose, Quaternion
    from hrc_msgs.msg import InverseCommand, BipedalCommand
except:
    print("Unable to load ROS dependencies")
import os
import yaml


def makeJointList():
    try:
        from ament_index_python.packages import get_package_share_directory
        joint_path = os.path.join(
                    get_package_share_directory('hrc_handler'),
                    "config/joints_list.yaml")
    except:
        joint_path = os.getcwd()[:-7] + "config/joints_list.yaml"
    with open(joint_path, 'r') as infp:
        pid_txt = infp.read()
        joints_dict = yaml.load(pid_txt, Loader = yaml.Loader)
    JOINT_LIST_COMPLETE = []
    JOINT_LIST_MOVABLE = []
    JOINT_LIST_LEG = []
    for c in range(len(joints_dict.keys())):
        JOINT_LIST_COMPLETE += [joints_dict[c]['name']]
        if joints_dict[c]['movable']:
            JOINT_LIST_MOVABLE += [joints_dict[c]['name']]
        if joints_dict[c]['leg']:
            JOINT_LIST_LEG += [joints_dict[c]['name']]
    return JOINT_LIST_COMPLETE, JOINT_LIST_MOVABLE, JOINT_LIST_LEG



def quaternion_rotation_matrix(Q):
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

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

    def forceUpdateState(self, timelist, pos, vel, tau):
        if self.cs_pos is not None and self.cs_pos is not None:
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
            new_tau = tau * weight_vec + match_tau * (1 - weight_vec)
        else:
            new_pos = pos
            new_vel = vel
            new_tau = tau
        self.cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis=0)
        self.cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis=0)
        self.cs_tau = scipy.interpolate.CubicSpline(timelist, new_tau, axis=0)

    def updateMixState(self, current_time, timelist, pos, vel, tau):
        self.cs_tau = scipy.interpolate.CubicSpline(timelist, tau, axis=0)
        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis = 0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0] : ]

            new_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)(new_timelist)
            new_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)(new_timelist)

            match_pos = self.cs_pos(new_timelist)
            match_vel = self.cs_vel(new_timelist)
            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * ((np.arange(sz) - 1) / sz)
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = new_pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = new_vel * weight_vec + match_vel * (1 - weight_vec)

            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)

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
        return not(self.cs_pos is None or self.cs_vel is None)


class JointSpaceFilter:
    def __init__(self, joint_num, position_error, velocity_error):

        self.timelist = None
        self.cs_pos = None
        self.cs_vel = None
        self.cs_tau = None

        self.joint_num = joint_num

        self.max_points = 100

        self.state_samples = np.zeros([0, joint_num * 2])
        self.tau_samples = np.zeros([0, joint_num])

        self.A = np.zeros([joint_num, joint_num * 2])
        self.B = np.zeros([joint_num])

    def getInterpolation(self, pos, vel, timestamp):
        state_r = np.concatenate([pos, vel], axis = 0)
        error = np.linalg.norm(self.state_samples[:, :self.joint_num] - state_r[:self.joint_num], axis = 1)
        index = np.argmin(error)
        ref = self.tau_samples[index, :]
        weight = 0.5 + 0.0 * np.arange(timestamp.shape[0]) / timestamp.shape[0]
        weight[weight > 1] = 1

        pos = self.cs_pos(timestamp)
        vel = self.cs_vel(timestamp)
        tau = self.cs_tau(timestamp) # * weight[:, None] + np.tile(ref[None, :], [timestamp.shape[0], 1]) * (1 - weight)[:, None]


        return pos, vel, tau

    def hasHistory(self):
        return not(self.cs_pos is None or self.cs_vel is None or self.cs_tau is None)



    def updateMixState(self, current_time, timelist, pos, vel, tau):
        state = np.concatenate([np.array(pos[1:, :]), np.array(vel[1:, :])], 1)
        self.state_samples = np.concatenate([state, self.state_samples], axis = 0)
        self.tau_samples = np.concatenate([tau[1:, :], self.tau_samples], axis = 0)
        if self.state_samples.shape[0] > self.max_points:
            self.state_samples = self.state_samples[:self.max_points, :]
            self.tau_samples = self.tau_samples[:self.max_points, :]
        self.cs_tau = scipy.interpolate.CubicSpline(timelist[:], tau[:, :], axis = 0)

        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis = 0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0] : ]

            new_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)(new_timelist)
            new_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)(new_timelist)

            match_pos = self.cs_pos(new_timelist)
            match_vel = self.cs_vel(new_timelist)
            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * ((np.arange(sz) - 1) / sz)
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = new_pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = new_vel * weight_vec + match_vel * (1 - weight_vec)

            self.cs_pos = scipy.interpolate.CubicSpline(new_timelist, new_pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(new_timelist, new_vel, axis=0)

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

class ForwardPoser:
    def __init__(self, urdf_path, joint_dict):
        self.joint_dict = joint_dict
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q = None

    def config2Vec(self, config_dict):
        num_joints = len(self.joint_dict)
        vec = np.zeros([num_joints])
        for key in config_dict.keys():
            index = self.model.getJointId(key) - 2
            vec[index] = config_dict[key]
        return vec

    def updateData(self, centroid_pos, centroid_orien, joint_pos_dict):
        #pos: xyz, orien: xyzw
        q_joints = self.config2Vec(joint_pos_dict)
        self.q = np.concatenate([centroid_pos, centroid_orien, q_joints], axis = 0)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def getLinkPose(self, link_name):
        id = self.model.getFrameId(link_name)
        pos = np.array(self.data.oMf[id].translation)
        return pos

    def getCOMPos(self):
        if self.q is not None:
            com_pos = np.array(pin.centerOfMass(self.model, self.data, self.q))
            return com_pos
        return None

    def jacobianCOMCorrection(self, desired_cpos, desired_corien, desired_jpos, contacts):
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        current_com = np.array(pin.centerOfMass(self.model, self.data, self.q))

        jac_com = pin.jacobianCenterOfMass(self.model, self.data, self.q)[:, 6:]

        jac_contacts = np.zeros([jac_com.shape[0], jac_com.shape[1]])
        for contact in contacts:
            jac_contact = pin.computeFrameJacobian(self.model, self.data, self.q,
                                            self.model.getFrameId(contact))
            jac_contacts += jac_contact[:3, 6:] / len(contacts)

        jac = jac_com - jac_contacts
        inv_jac = np.linalg.pinv(jac)

        q_joints = self.config2Vec(desired_jpos)
        q_desired = np.concatenate([desired_cpos, desired_corien, q_joints], axis=0)
        pin.forwardKinematics(self.model, self.data, q_desired)
        pin.updateFramePlacements(self.model, self.data)
        desired_com = np.array(pin.centerOfMass(self.model, self.data, q_desired))
        delta_com = desired_com - current_com
        delta_r = np.sum(inv_jac * np.tile(delta_com[None, :], [inv_jac.shape[0], 1]), axis = 1)

        names = self.model.names.tolist()
        joint_dict = {}
        for c in range(len(names) - 2):
            joint_dict[names[c + 2]] = delta_r[c]

        return joint_dict

    def jacobianTorqueForce(self, torques, contacts):
        torque_vec = self.config2Vec(torques)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        contact_forces = []
        for contact in contacts:
            jac_contacts = pin.computeFrameJacobian(self.model, self.data, self.q,
                                                    self.model.getFrameId(contact))[:3, 6:]
            contact_force = np.sum(jac_contacts * torque_vec[None, :], axis = 1)
            contact_forces += [contact_force]
        return contact_forces


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

        self.pelvis_id = self.model_r.getFrameId("pelvis")
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

        self.pose_dict = None

    def updateReducedModel(self, inverse_joints, config_dict):
        self.leg_joints = inverse_joints
        self.get_lock_joint()
        self.reduceRobot(config_dict)


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
        q0 = self.x[:7 + len(self.leg_joints)]
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)

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

    def stateLimitCost(self, cost_model , cost = 1e0):
        x_lb = np.concatenate([self.state.lb[1: self.state.nv + 1], self.state.lb[-self.state.nv:]])
        x_ub = np.concatenate([self.state.ub[1: self.state.nv + 1], self.state.ub[-self.state.nv:]])

        x_lb[12 + len(self.leg_joints):] = -np.inf
        x_ub[12 + len(self.leg_joints):] = np.inf

        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )
        x_bounds = crocoddyl.CostModelResidual(
            self.state,
            activation_xbounds,
            crocoddyl.ResidualModelState(self.state, 0 * self.x0, self.actuation.nu),
        )
        cost_model.addCost("xBounds", x_bounds, cost)


    def centroidalVelCost(self, cost_model, max_vel, max_angvel, cost):
        x_lb = np.full([12 + 2 * len(self.leg_joints)], -1 * np.inf)
        x_ub = np.full([12 + 2 * len(self.leg_joints)], np.inf)
        x_lb[6 + len(self.leg_joints) : 9 + len(self.leg_joints)] = -1 * max_vel
        x_ub[6 + len(self.leg_joints): 9 + len(self.leg_joints)] = 1 * max_vel

        x_lb[9 + len(self.leg_joints): 12 + len(self.leg_joints)] = -1 * max_angvel
        x_ub[9 + len(self.leg_joints): 12 + len(self.leg_joints)] = 1 * max_angvel

        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )
        x_bounds = crocoddyl.CostModelResidual(
            self.state,
            activation_xbounds,
            crocoddyl.ResidualModelState(self.state, 0 * self.x0, self.actuation.nu),
        )
        cost_model.addCost("centroidBounds", x_bounds, cost)


    def frictionConeCost(self, cost_model, name, id, cost = 1e2):

        cone = crocoddyl.WrenchCone(self.rsurf, self.mu, np.array([0.07, 0.03]))
        wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
            self.state, id, cone, self.nu
        )
        wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub)
        )

        wrench_cone = crocoddyl.CostModelResidual(
            self.state, wrench_activation, wrench_residual
        )
        cost_model.addCost(name + "_wrenchCone", wrench_cone, cost)

    def stateCost(self, cost_model, x0 = None, cost = 4e-3):

        state_weights = np.array(
            [0] * 3 + [500.0] * 3 + [0.01] * (self.state.nv - 6) + [0.1, 0.1, 0.1, 10, 10, 10] + [10] * (self.state.nv - 6)
        )
        ideal_state = self.x0.copy()
        ideal_state[7 + len(self.leg_joints) : 13 + len(self.leg_joints)] = ideal_state[7 + len(self.leg_joints) : 13 + len(self.leg_joints)] * 0.9
        if x0 is None:
            state_residual = crocoddyl.ResidualModelState(self.state, ideal_state, self.nu)
        else:
            state_residual = crocoddyl.ResidualModelState(self.state, x0, self.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights**2)
        state_reg = crocoddyl.CostModelResidual(
            self.state, state_activation, state_residual
        )
        cost_model.addCost("state_reg", state_reg, cost)

    def stateTorqueCost(self, cost_model, cost = 1e-4):
        ctrl_residual = crocoddyl.ResidualModelJointEffort(
            self.state, self.actuation, self.nu
        )
        ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        cost_model.addCost("torque_reg", ctrl_reg, cost)

    def jointAccCost(self, cost_model, cost):
        acc_residual = crocoddyl.ResidualModelJointAcceleration(self.state,
                                                                np.zeros([6 + len(self.leg_joints)]), self.nu)
        x_lb = np.full([6 + len(self.leg_joints)], -30)
        x_ub = np.full([6 + len(self.leg_joints)], 30)

        x_lb[:2] = -30
        x_ub[:2] = 30
        x_lb[2] = -10
        x_ub[2] = 10

        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )

        acc_reg = crocoddyl.CostModelResidual(self.state, activation_xbounds, acc_residual)
        cost_model.addCost("joint_acc", acc_reg, cost)


    def comCost(self, cost_model, target_pos, cost = 1e2):
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, target_pos, self.nu)
        state_weights = np.array( [1, 1, 1] )
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights ** 2)
        com_track = crocoddyl.CostModelResidual(self.state, state_activation, com_residual)
        cost_model.addCost("com_track", com_track, cost)

    def linkCost(self, cost_model, target, link_id, name, cost = 1e8):
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, link_id, target, self.nu)
        foot_track = crocoddyl.CostModelResidual(self.state, frame_placement_residual)
        cost_model.addCost(name + "_track", foot_track, cost)

    def linkClipCost(self, cost_model, link_name, z_height, cost):
        link_id = self.model_r.getFrameId(link_name)
        pose = pin.SE3(np.eye(3), np.zeros([3]))
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, link_id, pose, self.nu)

        x_lb = np.full([6], -np.inf)
        x_ub = np.full([6], np.inf)

        x_lb[2] = z_height + 0.005

        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )

        link_target = crocoddyl.CostModelResidual(self.state, activation_xbounds, frame_placement_residual)
        cost_model.addCost(link_name + "_clip", link_target, cost)
        return


    def linkWeightedCost(self, cost_model, pos, orien, link_name, cost, ang_weight):
        rot_mat = quaternion_rotation_matrix(orien)
        id = self.model_r.getFrameId(link_name)
        pose = pin.SE3(rot_mat, pos)
        if ang_weight > 10:
            weights = np.array([0] * 3 + [1] * 3)
        else:
            weights = np.array([1, 1, 10] + [ang_weight] * 3)
        activation_link = crocoddyl.ActivationModelWeightedQuad(weights ** 2)
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(self.state, id, pose, self.nu)
        link_target = crocoddyl.CostModelResidual(
            self.state,
            activation_link,
            frame_placement_residual
        )
        cost_model.addCost(link_name + "_link", link_target, cost)

    def linkVelCost(self, cost_model, link_name, cost):
        link_id = self.model_r.getFrameId(link_name)
        frame_vel_res = crocoddyl.ResidualModelFrameVelocity(
            self.state,link_id, pin.Motion.Zero(),
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
        )
        x_lb = np.full([6], -1 * np.inf)
        x_ub = np.full([6], np.inf)
        x_lb[:3] = -0.8
        x_ub[:3] = 0.8
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )
        impulse_cost = crocoddyl.CostModelResidual(self.state, activation_xbounds, frame_vel_res)
        cost_model.addCost(link_name + "_impulse", impulse_cost, cost)

    def contactCoPCost(self, cost_model, link_name, cost):

        id = self.model_r.getFrameId(link_name)

        cref = crocoddyl.CoPSupport(np.eye(3), np.array([0.20, 0.08]))
        cop_residual = crocoddyl.ResidualModelContactCoPPosition(
            self.state, id, cref, self.nu, True
        )
        cop_cost = crocoddyl.CostModelResidual(self.state, cop_residual)
        cost_model.addCost(link_name + "_cop", cop_cost, cost)

    def contactForceCost(self, cost_model, link_name, cost):
        link_id = self.model_r.getFrameId(link_name)
        fref = pin.Force(np.zeros(6))
        x_lb = np.full([6], -np.inf)
        x_ub = np.full([6], np.inf)

        x_lb[2] = 100

        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(x_lb, x_ub)
        )

        res_force = crocoddyl.ResidualModelContactForce(self.state, link_id, fref, 6, self.nu, True)
        f_cost = crocoddyl.CostModelResidual(self.state, activation_xbounds, res_force)
        cost_model.addCost(link_name + "_zforce", f_cost, cost)

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
        self.stateLimitCost(cost_model)
        self.contactForceCost(cost_model, "left_ankle_roll_link", 1e7)
        if com_target is not None:
            self.comCost(cost_model, com_target, cost = 1e6 * cost_factor)
            self.linkCost(cost_model, pin.SE3(np.eye(3), com_target + np.array([0., 0., 0.1])), self.pelvis_id, "pelvis", cost = 1e3)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_model, cost_model
        )
        return dmodel

    def makeD2M(self, dmodel, dt):
        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, dt)
        #model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.three, dt)
        return model


    # q0 is the robot pos sstate
    def getPos(self, q0):
        if q0 is None:
            q0 = self.x[:7 + len(self.leg_joints)]
        pin.forwardKinematics(self.model_r, self.data_r, q0)
        pin.updateFramePlacements(self.model_r, self.data_r)

        com_pos = np.array(pin.centerOfMass(self.model_r, self.data_r, q0))

        rf_pos = np.array(self.data_r.oMf[self.rf_id].translation)
        lf_pos = np.array(self.data_r.oMf[self.lf_id].translation)
        return lf_pos, rf_pos, com_pos

    def getLinkPose(self, link_list):

        pose_dict = {}
        for link in link_list:
            id = self.model_r.getFrameId(link)
            pos = np.array(self.data_r.oMf[id].translation)
            rot_mat = np.array(self.data_r.oMf[id].rotation)
            pose_dict[link] = {"pos": pos, "rot_mat": rot_mat}
        return pose_dict

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
        quat = x[3:7]
        return pos, quat, joint_dict, joint_vels, joint_efforts

    def makeInverseCmdDmodel(self, inverse_command):
        _, _, com_pos = self.getPos(None)

        self.pose_dict = self.getLinkPose(inverse_command.link_contacts)
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)

        ground_heights = []

        for contact_name, friction_cost, cop_cost, force_cost in zip(inverse_command.link_contacts,
                                                                  inverse_command.friction_contact_costs,
                                                                  inverse_command.cop_costs,
                                                                  inverse_command.force_limit_costs):
            id = self.model_r.getFrameId(contact_name)
            self.addContact(contact_model, contact_name + "_contact", id)
            self.frictionConeCost(cost_model, contact_name + '_friction', id, cost = friction_cost)
            if cop_cost > 0:
                self.contactCoPCost(cost_model, contact_name, cop_cost)
            if force_cost > 0:
                self.contactForceCost(cost_model, contact_name, force_cost)

            ground_heights += [self.data_r.oMf[id].translation[2]]

        ground_height = np.mean(np.array(ground_heights))

        if inverse_command.state_cost > 0:
            self.stateCost(cost_model, cost = inverse_command.state_cost)
        if inverse_command.torque_cost > 0:
            self.stateTorqueCost(cost_model, cost = inverse_command.torque_cost)
        for poses, link_name, link_costs, link_orien_weight, link_vel_cost, link_clip_cost in zip(inverse_command.link_poses,
                                                                   inverse_command.link_pose_names,
                                                                   inverse_command.link_costs,
                                                                   inverse_command.link_orien_weight,
                                                                   inverse_command.link_vel_costs,
                                                                   inverse_command.link_clip_costs):
            pos = np.array([poses.position.x, poses.position.y, poses.position.z])
            orien = np.array([poses.orientation.x, poses.orientation.y, poses.orientation.z, poses.orientation.w])
            self.linkWeightedCost(cost_model, pos, orien, link_name, link_costs, link_orien_weight)
            if link_vel_cost > 0:
                self.linkVelCost(cost_model, link_name, link_vel_cost)
            if link_clip_cost > 0:
                self.linkClipCost(cost_model, link_name, ground_height, link_clip_cost)

        com_pos = np.array([inverse_command.com_pos.x, inverse_command.com_pos.y, inverse_command.com_pos.z])
        if inverse_command.com_cost > 0:
            self.comCost(cost_model, com_pos, cost = inverse_command.com_cost)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_model,
                                                                         cost_model)
        if inverse_command.state_limit_cost > 0:
            self.stateLimitCost(cost_model, inverse_command.state_limit_cost)
        if inverse_command.centroid_vel_cost > 0:
            self.centroidalVelCost(cost_model, inverse_command.max_linear_vel, inverse_command.max_ang_vel,
                                   inverse_command.centroid_vel_cost)
        if inverse_command.joint_acceleration_cost > 0:
            self.jointAccCost(cost_model, inverse_command.joint_acceleration_cost)
        return dmodel

class SimpleFwdInvSM:
    def __init__(self, poser):
        self.poser = poser
        self.us = None

    def makeFwdInvProblem(self, timestamps, inverse_commands):
        self.poser.pose_dict = None
        ts_prev = 0
        models = []
        for timestamp, inverse_command in zip(timestamps, inverse_commands):
            dmodel = self.poser.makeInverseCmdDmodel(inverse_command)
            model = self.poser.makeD2M(dmodel, timestamp - ts_prev)
            ts_prev = timestamp
            models += [model]
        final_dmodel = self.poser.makeInverseCmdDmodel(inverse_commands[-1])
        final_model = self.poser.makeD2M(final_dmodel, 0)
        return models, final_model

    def nextMPC(self, timestamps, inverse_commands, xs):
        traj, final = self.makeFwdInvProblem(timestamps, inverse_commands)
        x0 = self.poser.x.copy()
        q0 = x0[0:7 + len(self.poser.leg_joints)]
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverBoxFDDP(problem)
        fddp.th_stop = 1e5
        if xs is None:
            init_xs = [x0] * (problem.T + 1)
        else:
            init_xs = []
            for c in range(xs.shape[0]):
                init_xs += [xs[c, :]]

        init_us = []
        if len(inverse_commands) > 50:
            maxiter = 200
        else:
            maxiter = 4
        regInit = 1e-1
        solved = fddp.solve(init_xs, init_us, maxiter, False, regInit)
        # print(solved)
        xs = np.array(fddp.xs)
        self.us = np.array(fddp.us)
        return xs, self.us

def makeFwdTraj(current_state, target):
    delta = target - current_state
    delta_range = (np.arange(100) + 1) * 0.001
    if delta > 0:
        delta_list = delta - delta_range
        delta_list[delta_list < 0] = 0
    else:
        delta_list = delta + delta_range
        delta_list[delta_list > 0] = 0
    new_pos = delta_list + current_state
    new_vel = ( new_pos[1:] - new_pos[:-1] ) / 0.001
    new_vel = np.concatenate([new_vel, np.array([0])], axis = 0)
    return new_pos, new_vel

def constrained_regression(X, Y, fixed_point_X, fixed_point_y):
    n, m = X.shape

    D = np.hstack([X, np.ones((n, 1))])  # n x (m+1)

    M = D.T @ D  # (m+1) x (m+1)
    DTy = D.T @ Y  # (m+1) x 1

    C = np.append(fixed_point_X, 1)  # (m+1) vector

    AUG_MAT = np.zeros((m+2, m+2))
    AUG_MAT[:m+1, :m+1] = M
    AUG_MAT[:m+1, m+1] = C
    AUG_MAT[m+1, :m+1] = C
    AUG_MAT[m+1, m+1] = 0

    AUG_VEC = np.zeros(m+2)
    AUG_VEC[:m+1] = DTy
    AUG_VEC[m+1] = fixed_point_y

    SOL = np.linalg.solve(AUG_MAT, AUG_VEC)
    beta = SOL[:m+1]

    A = beta[:m]
    B = beta[m]

    return A, B

def makePose(pos, rot):
    pose = Pose()
    point = Point()
    point.x = float(pos[0])
    point.y = float(pos[1])
    point.z = float(pos[2])
    orien = Quaternion()
    orien.x = float(rot[0])
    orien.y = float(rot[1])
    orien.z = float(rot[2])
    orien.w = float(rot[3])
    pose.position = point
    pose.orientation = orien
    return pose

class BipedalGait:

    def __init__(self, step_length, step_height):
        self.step_height = step_height
        self.step_length = step_length
        self.jlc, self.jlm, self.leg_joints = makeJointList()

    def dualSupport(self, com, com_vel, support_contact):
        ic = InverseCommand()

        ic.state_cost = float(1e1)
        ic.torque_cost = float(1e0)
        pelvis_pose = makePose([0, 0, 0], [0, 0, 0, 1])
        ic.link_poses = [pelvis_pose]
        ic.link_pose_names = ["pelvis"]
        ic.link_costs = [1e8]
        ic.link_orien_weight = [10000]
        ic.link_vel_costs = [0.]
        ic.link_clip_costs = [0.]
        ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
        ic.friction_contact_costs = [float(2e1), float(2e1)]

        if support_contact == "left_ankle_roll_link":
            ic.cop_costs = [float(1e2), float(0)]
            ic.force_limit_costs = [float(0.), float(0.)]
        else:
            ic.cop_costs = [float(0), float(1e2)]
            ic.force_limit_costs = [float(0.), float(0.)]
        com_pos = Point()
        com_pos.x = float(com[0])
        com_pos.y = float(com[1])
        com_pos.z = float(com[2])
        ic.com_pos = com_pos
        ic.com_cost = float(1e9)
        ic.max_linear_vel = 0.8
        ic.max_ang_vel = 0.8
        ic.state_limit_cost = 1e4
        ic.centroid_vel_cost = 0.
        ic.joint_acceleration_cost = 0.
        return ic

    def singleSupport(self, contact_link, move_link, move_pos, move_orien, com):
        ic = InverseCommand()

        ic.state_cost = float(1e1)
        ic.torque_cost = float(1e0)
        move_pose = makePose(move_pos, move_orien)
        zero_pos = makePose(np.array([0, 0, 0]), np.array([0,0,0,1]))
        ic.link_poses = [move_pose, zero_pos]
        ic.link_pose_names = [move_link, "pelvis"]
        ic.link_costs = [float(1e7), float(1e3)]
        ic.link_orien_weight = [float(1), float(10000)]
        ic.link_vel_costs = [float(1e8), 0.]
        ic.link_clip_costs = [1e5, 0.]
        ic.link_contacts = [contact_link]
        ic.friction_contact_costs = [float(1e3)]
        ic.force_limit_costs = [float(1e1)]
        ic.cop_costs = [float(1e1)]
        com_pos = Point()
        com_pos.x = float(com[0])
        com_pos.y = float(com[1])
        com_pos.z = float(com[2])
        ic.com_pos = com_pos
        ic.com_cost = float(1e2)
        ic.max_linear_vel = 0.8
        ic.max_ang_vel = 0.8
        ic.state_limit_cost = 1e7
        ic.centroid_vel_cost = 0.
        ic.joint_acceleration_cost = 0.
        return ic

    def swingTrajectory(self, initial_pos, final_pos, prop):
        #initial pos and final pos are  3d with 0 in index 2
        ref_xy = initial_pos * (1 - prop) + final_pos * prop
        z = ( prop * (1 - prop) ) * 4 * self.step_height
        ref_xy[2] = ref_xy[2] + z
        return ref_xy

    def reflessWalk(self):
        left_pos = np.array([-0.003, 0.12, 0.01])
        right_pos = np.array([-0.003, -0.12, 0.01])
        com_pos = np.array([0., 0., 0.65])

        bpc = BipedalCommand()

        timestamps = 0.02 + np.arange(160) * 0.02
        #Dual support settle for 0.4 secs or 40 ts
        ics = []
        for c in range(40):
            prop = c / 39
            #settle_com = com_pos * (1 - prop) + prop * np.array([0., 0.08, 0.55])
            ic = self.dualSupport(np.array([0., 0.08, 0.55]), None, "left_ankle_roll_link")
            ics += [ic]
        com_pos = np.array([0., 0.12, 0.55])
        #single support left foot
        right_pos2 = right_pos.copy() + np.array([self.step_length/2, 0., 0.0])
        #right_pos2 = np.array([0.1, -0.12, 0.01])
        for c in range(30):
            prop = c / 29
            rlink = self.swingTrajectory(right_pos, right_pos2, prop)
            shift_com = np.array([self.step_length/2, -0.09, 0.55])
            ic = self.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", rlink, np.array([0,0,0,1]), shift_com)
            ics += [ic]
        right_pos = right_pos2.copy()
        com_pos = np.array([self.step_length/2, -0.09, 0.55])
        #dual support 0.1 s
        for c in range(5):
            ic = self.dualSupport(com_pos, None, "right_ankle_roll_link")
            ics += [ic]

        left_pos2 = left_pos + np.array([self.step_length, 0., 0.0])
        for c in range(40):
            prop = c / 39
            llink = self.swingTrajectory(left_pos, left_pos2, prop)
            shift_com = np.array([self.step_length, 0.09, 0.55])
            ic = self.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", llink, np.array([0,0,0,1]), shift_com)
            ics += [ic]

        com_pos = np.array([self.step_length, 0.07, 0.55])

        for c in range(5):
            ic = self.dualSupport(com_pos, None, "left_ankle_roll_link")
            ics += [ic]

        right_pos2 = right_pos.copy() + np.array([self.step_length, 0., 0.0])
        for c in range(40):
            prop = c / 39
            rlink = self.swingTrajectory(right_pos, right_pos2, prop)
            shift_com = np.array([self.step_length*3/2, -0.09, 0.55])
            ic = self.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", rlink, np.array([0,0,0,1]), shift_com)
            ics += [ic]

        com_pos = np.array([self.step_length*3/2, -0.07, 0.55])
        for c in range(5):
            ic = self.dualSupport(com_pos, None, "right_ankle_roll_link")
            ics += [ic]

        bpc.inverse_timestamps = timestamps
        bpc.inverse_commands = ics
        bpc.inverse_joints = self.leg_joints
        return bpc

def idleInvCmd():
    bpc = BipedalCommand()
    bpc.inverse_timestamps = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    ics = []

    for c in range(9):
        ic = InverseCommand()

        ic.state_cost = float(2e2)
        ic.torque_cost = float(1e-2)
        pelvis_pose = makePose([0, 0, 0.7], [0, 0, 0, 1])
        ic.link_poses = [pelvis_pose]
        ic.link_pose_names = ["pelvis"]
        ic.link_costs = [float(1e6)]
        ic.link_orien_weight = [float(100000), float(1), float(1)]
        ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
        ic.friction_contact_costs = [float(1e3), float(1e3)]
        ic.contact_lock_costs = [float(0), float(0)]
        com_pos = Point()
        com_pos.x = 0.03
        com_pos.y = 0.0
        com_pos.z = 0.63
        ic.com_pos = com_pos
        ic.com_cost = float(0)

        ics += [ic]

    bpc.inverse_commands = ics
    _, _, leg_joints = makeJointList()
    bpc.inverse_joints = leg_joints
    return bpc

class WalkingSM:
    def __init__(self):
        #4 states DS_SR, DS_CR, SR, DS_SL, DS_CL, SL
        #DS_SR: Dual support state end cond, prep for right swing
        #DS_CR: Dual support countdown end cond, prep for right swing
        #SR: right swing foot
        #DS_SL: Dual support state end cond, prep for left swing
        #DS_CL: Dual support countdown end cond, prep for left swing
        #SL: left swing foot
        JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = makeJointList()
        self.current_state = "DS_SR"
        self.countdown_start = 0
        self.countdown_duration = 0.05
        from ament_index_python.packages import get_package_share_directory
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.fwd_poser = ForwardPoser(urdf_config_path, JOINT_LIST)

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
            if xy < 0.10 and z < 0.02:
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
            if xy < 0.10:
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

if __name__ == "__main__":
    import time
    joint_list, joint_list_movable, leg_joints = makeJointList()
    urdf_config_path = os.getcwd()[:-7] + "urdf/g1_meshless.urdf"
    poser = BipedalPoser(urdf_config_path, joint_list, leg_joints, "left_ankle_roll_link",
                                      "right_ankle_roll_link")
    dmodel = poser.dualSupportDModel()
    q = poser.x[0 : 7 + len(leg_joints)]
    mat = pin.jacobianCenterOfMass(poser.model_r, poser.data_r, q)
    print(mat.shape)


    mat2 = pin.computeFrameJacobian(poser.model_r, poser.data_r, q, poser.model_r.getFrameId("left_ankle_roll_link"))

    print(mat2.shape)

    print(mat2[:, 6])
    pin.forwardKinematics(poser.model_r, poser.data_r, q)
    pin.updateFramePlacements(poser.model_r, poser.data_r)
    pos_i = poser.data_r.oMf[poser.model_r.getFrameId("left_ankle_roll_link")].translation.copy()
    com_i = pin.centerOfMass(poser.model_r, poser.data_r)
    q[7] = 0.01
    pin.forwardKinematics(poser.model_r, poser.data_r, q)
    pin.updateFramePlacements(poser.model_r, poser.data_r)
    com_f = pin.centerOfMass(poser.model_r, poser.data_r)
    pos_f = poser.data_r.oMf[poser.model_r.getFrameId("left_ankle_roll_link")].translation.copy()
    emp_j = (com_f - com_i) / 0.01
    emp_p = (pos_f - pos_i) / 0.01
    print(emp_j)
    print(emp_p)