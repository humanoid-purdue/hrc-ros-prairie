float64 state_cost
float64 torque_cost
geometry_msgs/Pose[] link_poses
string[] link_pose_names
float64[] link_costs
float64[] link_orien_weight
float64[] link_vel_costs

string[] link_contacts
float64[] link_clip_costs
float64[] cop_costs
float64[] friction_contact_costs
float64[] force_limit_costs
float64[] contact_vel_costs
float64[] contact_force_cap

geometry_msgs/Point com_pos
geometry_msgs/Point com_vel
geometry_msgs/Point com_acc
float64[] com_cost
float64 max_linear_vel
float64 max_ang_vel
float64 state_limit_cost
float64 centroid_vel_cost
float64 joint_acceleration_cost
bool com_height_only