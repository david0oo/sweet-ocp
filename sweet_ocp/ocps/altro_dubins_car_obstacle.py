# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from acados_test_problems.models.unicycle_models import export_fixed_time_unicycle_model, export_free_time_unicycle_model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import casadi as cs
import scipy.linalg

@dataclass
class DubinsCarObstacleParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 101
    nx : int = 3
    nu : int = 2
    Tf : float = 5.0
    # Initial states
    Q = np.diag([1.0, 1.0, 1.0])
    R = np.diag([0.5, 0.5])
    Qf = 10.0*np.diag([1.0, 1.0, 1.0])

    initial_state  = np.array([0., 0., 0.])
    terminal_state  = np.array([3., 3., 0.])

    u_min = np.array([0., -3.])
    u_max = np.array([3., 3.])

    u_init = np.array([0.01, 0.01])

    r_car : float = 0.175
    r_circle_obs : float = 0.25

    circle_x = np.array([0.75, 1.5, 2.25])
    circle_y = np.array([0.75, 1.5, 2.25])
    obs1_pos = np.array([circle_x[0], circle_y[0]])
    obs2_pos = np.array([circle_x[1], circle_y[1]])
    obs3_pos = np.array([circle_x[2], circle_y[2]])

def obstacle_constraint_expr(pos_obs, pos_car):
    return (pos_car[0] - pos_obs[0])**2 + (pos_car[1] - pos_obs[1])**2

def create_problem(opts : AcadosOcpOptions, mode : str ='FIXED_TIME'):

    params = DubinsCarObstacleParameters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    if mode == 'FIXED_TIME':
        model = export_fixed_time_unicycle_model()
    else:
        model = export_free_time_unicycle_model()

    ocp.model = model

    ###########################################################################
    # Define cost: time optimal
    ###########################################################################
    if mode == 'FIXED_TIME':
        ny = params.nx + params.nu

        Vx = np.zeros((ny, params.nx))
        Vx[:params.nx, :] = np.eye(params.nx)
        Vu = np.zeros((ny, params.nu))
        Vu[params.nx:ny, :] = np.eye(params.nu)
        yref = np.zeros(ny,)
        yref_e = np.zeros(params.nx,)

        # Add costs of starting and terminal costs
        ocp.cost.cost_type_0 = 'LINEAR_LS'
        ocp.cost.W_0 = scipy.linalg.block_diag(params.Q, params.R)
        ocp.cost.Vx_0 = Vx
        ocp.cost.Vu_0 = Vu
        ocp.cost.yref_0 = yref

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(params.Q, params.R)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.yref  = yref

        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = params.Q
        ocp.cost.Vx_e = np.eye(params.nx)
        ocp.cost.yref_e  = yref_e
    else:
        # time-optimal
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = model.x[0]

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    if mode == 'FIXED_TIME':
        ocp.constraints.lbx_0 = params.initial_state
        ocp.constraints.ubx_0 = params.initial_state
        ocp.constraints.idxbx_0 = np.array([0, 1, 2])
        ocp.constraints.idxbxe_0 = np.array([0, 1, 2])
    else:
        ocp.constraints.lbx_0 = np.append([0.0], params.initial_state)
        ocp.constraints.ubx_0 = np.append([ACADOS_INFTY], params.initial_state)
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

    # Path constraints on controls
    ocp.constraints.lbu = params.u_min
    ocp.constraints.ubu = params.u_max
    ocp.constraints.idxbu = np.array([0, 1])

    # Terminal constraints
    if mode == 'FIXED_TIME':
        ocp.constraints.lbx_e = params.terminal_state
        ocp.constraints.ubx_e = params.terminal_state
        ocp.constraints.idxbx_e = np.array([0, 1, 2])
    else:
        ocp.constraints.lbx_e = np.append([0.0], params.terminal_state)
        ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state)
        ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

    if mode == 'FIXED_TIME':
        car_pos = ocp.model.x[:2]
    else:
        car_pos = ocp.model.x[1:3]

    # Obstacle constraints
    ocp.model.con_h_expr = cs.vertcat(obstacle_constraint_expr(params.obs1_pos, car_pos),
                                      obstacle_constraint_expr(params.obs2_pos, car_pos),
                                      obstacle_constraint_expr(params.obs3_pos, car_pos))
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.lh = (params.r_car+params.r_circle_obs)**2*np.ones(3)

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    if mode == 'FIXED_TIME':
        ocp.solver_options.tf = params.Tf
    else:
        ocp.solver_options.tf = 1.0
    return ocp

def create_initial_guess():
    params = DubinsCarObstacleParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    for i in range(params.N):
        init_X[i,:] = params.initial_state
        init_U[i,:] = np.zeros(params.nu)
    init_X[params.N,:] = params.initial_state
    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    params = DubinsCarObstacleParameters()
    sol_x_y = sol_X[:,:2]
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    rect1 = Circle((params.obs1_pos[0], params.obs1_pos[1]), params.r_circle_obs, linewidth=1, edgecolor='tab:gray', facecolor='tab:gray')
    ax.add_patch(rect1)
    rect2 = Circle((params.obs2_pos[0], params.obs2_pos[1]), params.r_circle_obs, linewidth=1, edgecolor='tab:gray', facecolor='tab:gray')
    ax.add_patch(rect2)
    rect3 = Circle((params.obs3_pos[0], params.obs3_pos[1]), params.r_circle_obs, linewidth=1, edgecolor='tab:gray', facecolor='tab:gray')
    ax.add_patch(rect3)

    ax.plot(sol_x_y[:,0], sol_x_y[:,1], label='car trajectory')
    # ax.set_ylim((-1, 0.0))
    # ax.set_xlim((-0.05, 0.55))
    plt.legend(loc='upper right')
    plt.xlabel('$x$-axis')
    plt.ylabel('$y$-axis')
    # plt.title('Reedsheps car obstacle')
    plt.show()


if __name__ == '__main__':
    pass
