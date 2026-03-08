# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from sweet_ocp.models.overhead_crane_models import export_free_time_crane_model, export_free_time_crane_model_with_sep_hyperplane
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs
from matplotlib.patches import Rectangle, Circle

@dataclass
class CraneProblemParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 25
    nx : int = 7
    nu : int = 5
    Tf : float = 1.0
    # Initial states
    l0 : float = 0.9
    x0 : float = 0.0
    theta0 : float = 0.0
    l_dot0 : float = 0.0
    x_dot0 : float = 0.0
    theta_dot0 : float = 0.0
    initial_state = np.array([l0, x0, theta0, l_dot0, x_dot0, theta_dot0])

    # Terminal states
    l_f : float = 0.9
    x_f : float = 0.5
    theta_f : float = 0.0
    x_dot_f : float = 0.0
    l_dot_f : float = 0.0
    theta_dot_f : float = 0.0
    terminal_state = np.array([l_f, x_f, theta_f, x_dot_f, l_dot_f, theta_dot_f])

    # bounds for path constraints
    crane_angle_limit = 0.75  # in radians
    cart_pos_min = -0.1  # in metres
    cart_pos_max = 0.6  # in metres
    cart_max_vel = 0.4  # in m/s
    l_min = 1e-2
    l_max = 2
    l_dot_max = 0.25
    lower_path_bounds = np.array([l_min, cart_pos_min, -crane_angle_limit, -l_dot_max, -cart_max_vel])
    upper_path_bounds = np.array([l_max, cart_pos_max, crane_angle_limit, l_dot_max, cart_max_vel])

    l_ddot_max = 5
    x_ddot_max = 5
    upper_control_bounds = np.array([l_ddot_max, x_ddot_max])
    lower_control_bounds = np.array([-l_ddot_max, -x_ddot_max])

def create_problem(opts : AcadosOcpOptions,
                   WITH_RECTANGULAR_OBSTACLE : bool = True):

    params = CraneProblemParameters()
    # nx = 7
    # nu = 2
    # if WITH_RECTANGULAR_OBSTACLE:
    #     nu = 5

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_time_crane_model()
    if WITH_RECTANGULAR_OBSTACLE:
        model = export_free_time_crane_model_with_sep_hyperplane()
    ocp.model = model


    ###########################################################################
    # Define cost
    ###########################################################################
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.append([0.0], params.initial_state)
    ocp.constraints.ubx_0 = np.append([ACADOS_INFTY], params.initial_state)
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5, 6])
    ocp.constraints.idxbxe_0 = np.array([1, 2, 3, 4, 5, 6])

    # Path constraints on states
    ocp.constraints.lbx = params.lower_path_bounds
    ocp.constraints.ubx = params.upper_path_bounds
    ocp.constraints.idxbx = np.array([1, 2, 3, 4, 5])

    # Path constraints on control
    if WITH_RECTANGULAR_OBSTACLE:
        ocp.constraints.lbu = np.append(params.lower_control_bounds, [-1.0 , -1.0, -1.0])
        ocp.constraints.ubu = np.append(params.upper_control_bounds, [1.0 , 1.0, 1.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
    else:
        ocp.constraints.lbu = params.lower_control_bounds
        ocp.constraints.ubu = params.upper_control_bounds
        ocp.constraints.idxbu = np.array([0, 1])

    # Terminal constraints
    ocp.constraints.lbx_e = np.append([0.0], params.terminal_state)
    ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state)
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5, 6])

    payload_position = cs.vertcat(model.x[2] + model.x[1]*cs.sin(model.x[3]), -model.x[1]*cs.cos(model.x[3]))
    # with circular obstacle
    if not WITH_RECTANGULAR_OBSTACLE:
        obs_x = 0.25
        obs_y = 0.8
        obs_rad = 0.15
        obstacle_position = cs.vertcat(obs_x,-obs_y)
        ocp.model.con_h_expr = (payload_position[0] - obstacle_position[0])**2 + (payload_position[1] - obstacle_position[1])**2
        ocp.constraints.lh = np.array([obs_rad**2])
        ocp.constraints.uh = np.array([ACADOS_INFTY])
        # copy for terminal
        ocp.constraints.uh_e = ocp.constraints.uh
        ocp.constraints.lh_e = ocp.constraints.lh
        ocp.model.con_h_expr_e = ocp.model.con_h_expr
    # with rectangular obstacle
    else:
        rad = 0.08
        x_obstacle = [0.1, 0.2]
        y_obstacle = [-2.0, -0.7]

        # Obstacle Avoiding Constraint
        # adding separating hyperplane constraint for crane
        hyperplane_expression = cs.vertcat(
            model.u[2]*payload_position[0] + model.u[3]*payload_position[1] + model.u[4],
            model.u[2]*x_obstacle[0] + model.u[3]*y_obstacle[0] + model.u[4],
            model.u[2]*x_obstacle[0] + model.u[3]*y_obstacle[1] + model.u[4],
            model.u[2]*x_obstacle[1] + model.u[3]*y_obstacle[0] + model.u[4],
            model.u[2]*x_obstacle[1] + model.u[3]*y_obstacle[1] + model.u[4]
        )
        hyperplane_ub = np.array([-rad, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY])
        hyperplane_lb = np.array([-ACADOS_INFTY, 0.0, 0.0, 0.0, 0.0])

        ocp.model.con_h_expr = hyperplane_expression
        ocp.constraints.lh = hyperplane_lb
        ocp.constraints.uh = hyperplane_ub


    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess(WITH_RECTANGULAR_OBSTACLE=True, variant : int = 0):
    params = CraneProblemParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    # Initial guess
    T0 = 2.0
    tau = np.linspace(0, 1, params.N+1)
    x_guess = np.linspace(params.x0, params.x_f, params.N+1)
    xc_dot = (params.x_f-params.x0)/T0
    l_guess = params.l0*(1-tau)+tau*params.l_f-0.5*np.sin(np.pi*tau)


    if WITH_RECTANGULAR_OBSTACLE:
        if variant == 0:
            for i in range(params.N):
                init_X[i,:] = np.array([T0, 0.6, x_guess[i], 0, 0, xc_dot, 0])
                init_U[i,:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            init_X[params.N,:] = np.array([T0, params.l_f, x_guess[params.N], 0, 0, xc_dot, 0])
        else:
            for i in range(params.N):
                init_X[i,:] = np.array([T0, params.l0, params.x0, params.theta0, params.l_dot0, params.x_dot0, params.theta_dot0])
                init_U[i,:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            init_X[params.N,:] = np.array([T0, params.l0, params.x0, params.theta0, params.l_dot0, params.x_dot0, params.theta_dot0])
    else:
        if variant == 0:
            for i in range(params.N):
                init_X[i,:] = np.array([T0, l_guess[i], x_guess[i], 0, 0, xc_dot, 0])
                init_U[i,:] = np.array([0.0, 0.0])
            init_X[i,:] = np.array([T0, l_guess[i], x_guess[params.N], 0, 0, xc_dot, 0])
        else:
            for i in range(params.N):
                init_X[i,:] = np.array([T0, params.l0, params.x0, params.theta0, params.l_dot0, params.x_dot0, params.theta_dot0])
                init_U[i,:] = np.array([0.0, 0.0])
            init_X[params.N,:] = np.array([T0, params.l0, params.x0, params.theta0, params.l_dot0, params.x_dot0, params.theta_dot0])

    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U :np.array, WITH_RECTANGULAR_OBSTACLE=True):
    """
    Plot the optimal trajectory of the crane load.
    """
    obs_x = 0.25
    obs_y = 0.8
    obs_rad = 0.15

    _, ax = plt.subplots()
    ax.set_aspect('equal')
    payload_x1_position = sol_X[:, 2] + sol_X[:, 1]*np.sin(sol_X[:, 3])
    payload_x2_position = -sol_X[:, 1] * np.cos(sol_X[:, 3])

    if WITH_RECTANGULAR_OBSTACLE:
        rect = Rectangle((0.1, -2.0), 0.1, 1.3, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        rect = Circle((obs_x, -obs_y), obs_rad, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.plot(payload_x1_position, payload_x2_position, label='payload trajectory')
    ax.plot(np.linspace(0, 0.5, 26), -0.6*np.ones(26), label='initial guess')
    ax.set_ylim((-1, 0.0))
    ax.set_xlim((-0.05, 0.55))
    plt.legend(loc='upper right')
    plt.xlabel('$x_1$-axis')
    plt.ylabel('$x_2$-axis')
    plt.title('Overhead Crane, P2P Motion')
    plt.show()

if __name__ == '__main__':
    pass
