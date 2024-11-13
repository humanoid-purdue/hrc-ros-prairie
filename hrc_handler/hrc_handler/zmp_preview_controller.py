import rclpy
from rclpy.node import Node
import numpy as np
from hrc_msgs.msg import StateVector, CentroidalTrajectory, InverseCommand
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
from control.matlab import dare

#From : https://github.com/chauby/ZMP_preview_control

def calculatePreviewControlParams(A, B, C, Q, R, N):
    [P, _, _] = dare(A, B, C.T*Q*C, R)
    K = (R + B.T*P*B).I*(B.T*P*A)

    f = np.zeros((1, N))
    for i in range(N):
        f[0,i] = (R+B.T*P*B).I*B.T*(((A-B*K).T)**i)*C.T*Q

    return K, f

def calculatePreviewControlParams2(A, B, C, Q, R, N):
    C_dot_A = C*A
    C_dot_B = C*B

    A_tilde = np.matrix([[1, C_dot_A[0,0], C_dot_A[0,1], C_dot_A[0,2]],
                            [0, A[0,0], A[0,1], A[0,2]],
                            [0, A[1,0], A[1,1], A[1,2]],
                            [0, A[2,0], A[2,1], A[2,2]]])
    B_tilde = np.matrix([[C_dot_B[0,0]],
                            [B[0,0]],
                            [B[1,0]],
                            [B[2,0]]])
    C_tilde = np.matrix([[1, 0, 0, 0]])

    [P_tilde, _, _] = dare(A_tilde, B_tilde, C_tilde.T*Q*C_tilde, R)
    K_tilde = (R + B_tilde.T*P_tilde*B_tilde).I*(B_tilde.T*P_tilde*A_tilde)

    Ks = K_tilde[0, 0]
    Kx = K_tilde[0, 1:]

    Ac_tilde = A_tilde - B_tilde*K_tilde

    G = np.zeros((1, N))

    G[0] = -Ks
    I_tilde = np.matrix([[1],[0],[0],[0]])
    X_tilde = -Ac_tilde.T*P_tilde*I_tilde

    for i in range(N):
        G[0,i] = (R + B_tilde.T*P_tilde*B_tilde).I*(B_tilde.T)*X_tilde
        X_tilde = Ac_tilde.T*X_tilde

    return Ks, Kx, G

#-----------

class zmp_preview_controller(Node):
    def __init__(self):
        super().__init__('zmp_preview_controller')
        self.start_time = time.time()
        self.publisher2 = self.create_publisher(CentroidalTrajectory, 'centroidal_trajectory', 10)
        timer_period = 0.00  # seconds

        self.subscription_1 = self.create_subscription(
            StateVector,
            'state_vector',
            self.state_vector_callback,
            10, callback_group=None)

        self.state_time = 0
        self.state_dict = None

        self.dt = 0.01
        self.g = 9.81
        self.t_preview = 1.0
        self.simple_plan = helpers.SimpleFootstepPlan()

        self.A = np.mat(([1, self.dt, self.dt ** 2 / 2],
                    [0, 1, self.dt],
                    [0, 0, 1]))
        self.B = np.mat((self.dt ** 3 / 6, self.dt ** 2 / 2, self.dt)).T
        self.C = np.mat((1, 0, - self.simple_plan.z_height / self.g))

        Q = 1
        R = 1e-6

        self.horizon_steps = 4 #complete current steps then add 3 steps to the zmp reference

        self.n_preview = int(self.t_preview / self.dt)

        self.K, self.f = calculatePreviewControlParams(self.A, self.B, self.C, Q, R, self.n_preview)

        self.Ks, self.Kx, self.G = calculatePreviewControlParams2(self.A, self.B, self.C, Q, R, self.n_preview)

        self.walking_sm = helpers.WalkingSM()

        self.timer = self.create_timer(0.001, self.timer_callback)

        self.prev_state = "0000"

    def timer_callback(self):
        if len(self.simple_plan.plan) == 0:
            return
        if self.state_dict is not None:
            initial_pos = self.simple_plan.plan[0][2]
            swing_target = self.simple_plan.plan[0][1]
            support_loc = self.simple_plan.plan[0][3]
            self.walking_sm.updateState(self.state_dict, self.state_time, swing_target)
            current_state = self.walking_sm.current_state
            #determine time remaining in current states to construct zmp plan
            pos_l = self.walking_sm.fwd_poser.getLinkPose("left_ankle_roll_link")
            pos_r = self.walking_sm.fwd_poser.getLinkPose("right_ankle_roll_link")
            com = self.walking_sm.fwd_poser.getCOMPos()


            if current_state == "SR":
                pos_c = pos_r
            else:
                pos_c = pos_l

            if current_state[0] == "S":
                time_remaining = np.linalg.norm(swing_target[:2] - pos_c[:2]) / self.simple_plan.step_speed
                time_remaining = min(self.simple_plan.swing_time, max(time_remaining, 0))

            if current_state[0:4] == "DS_C":
                time_remaining = self.walking_sm.countdown_duration - (self.state_time - self.walking_sm.countdown_start)
                time_remaining = time_remaining + self.simple_plan.swing_time

            if current_state[0:4] == "DS_S":
                if current_state[-1] == "L":
                    com_pos = pos_r
                else:
                    com_pos = pos_l
                time_remaining = np.linalg.norm(com_pos[:2] - com[:2]) / self.simple_plan.com_speed
                time_remaining = time_remaining + self.walking_sm.countdown_duration + self.simple_plan.swing_time

            iters = int(np.ceil(time_remaining / self.dt))
            zmp_x_ref = [support_loc[0]] * iters
            zmp_y_ref = [support_loc[1]] * iters

            for c in range(self.horizon_steps):
                support_loc = self.simple_plan.plan[c + 1][3]
                iters = int(np.ceil((self.simple_plan.swing_time + self.simple_plan.support_time) / self.dt))
                zmp_x_ref += [support_loc[0]] * iters
                zmp_y_ref += [support_loc[1]] * iters

            zmp_x_ref = np.asmatrix(np.array(zmp_x_ref)[:, None])
            zmp_y_ref = np.asmatrix(np.array(zmp_y_ref)[:, None])

            com_x = np.asmatrix(np.zeros([3, zmp_x_ref.shape[0]]))
            com_y = np.asmatrix(np.zeros([3, zmp_x_ref.shape[0]]))

            com_x[0, :] = com[0]
            com_y[0, :] = com[1]
            com_x[1, :] = self.state_dict["vel"][0]
            com_y[1, :] = self.state_dict["vel"][1]

            zmp_x_record = []
            zmp_y_record = []

            com_x_record = []
            com_y_record = []

            ux_1 = np.asmatrix(np.zeros((zmp_x_ref.shape[0], 1)))
            uy_1 = np.asmatrix(np.zeros((zmp_x_ref.shape[0], 1)))

            point_list = []

            for k in range(zmp_x_ref.shape[0] - self.n_preview):

                zmp_x_preview = zmp_x_ref[k:k + self.n_preview, :]
                zmp_y_preview = zmp_y_ref[k:k + self.n_preview, :]

                zmp_x = self.C * com_x[:, k]
                zmp_y = self.C * com_y[:, k]
                zmp_x_record += [zmp_x[0, 0]]
                zmp_y_record += [zmp_y[0, 0]]

                # update u
                ux_1[k] = -self.K * com_x[:, k] + self.f * zmp_x_preview
                uy_1[k] = -self.K * com_y[:, k] + self.f * zmp_y_preview
                # update COM state
                com_x[:, k + 1] = self.A * com_x[:, k] + self.B * ux_1[k]
                com_y[:, k + 1] = self.A * com_y[:, k] + self.B * uy_1[k]
                com_x_record += [com_x[0, k]]
                com_y_record += [com_y[0, k]]

                point = Point()
                point.x = com_x[0, k]
                point.y = com_y[0, k]
                point.z = self.simple_plan.z_height

                point_list += [point]

            timestamps = np.arange(len(point_list)) * self.dt + self.state_time
            centroid_traj = CentroidalTrajectory()
            centroid_traj.timestamps = timestamps
            centroid_traj.com_pos = point_list
            self.publisher2.publish(centroid_traj)

            #debug_save = np.zeros([zmp_x_ref.shape[0] - self.n_preview, 4])
            #debug_save[:, 0] = np.array(zmp_x_record)
            #debug_save[:, 1] = np.array(zmp_y_record)
            #debug_save[:, 2] = np.array(com_x_record)
            #debug_save[:, 3] = np.array(com_y_record)
            #np.savetxt("/home/aurum/RosProjects/prairie/datadump/preview_data", debug_save, delimiter = ',')



            if self.prev_state[0] == "S" and current_state[0:2] == "DS":
                self.simple_plan.plan.pop(0)
            self.prev_state = current_state



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

def main(args=None):
    rclpy.init(args=args)

    zmp = zmp_preview_controller()

    rclpy.spin(zmp)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    zmp.destroy_node()
    rclpy.shutdown()