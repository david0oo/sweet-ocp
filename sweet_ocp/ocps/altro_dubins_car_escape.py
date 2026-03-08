# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from sweet_ocp.models.unicycle_models import export_fixed_time_unicycle_model, export_free_time_unicycle_model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import casadi as cs
import scipy.linalg
from scipy.interpolate import interp1d


@dataclass
class DubinsCarEscapeParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 22
    nx : int = 3
    nu : int = 2
    Tf : float = 3.0

    initial_state  = np.array([2.5, 2.5, 0.])
    terminal_state  = np.array([7.5, 2.5, 0.])

    u_min = np.array([-5., -5.])
    u_max = np.array([5., 5.])

    u_init = np.array([0.01, 0.01])

    r_car : float = 0.25
    r_circle_obs : float = 0.25

    # Initial states
    Q = 1e-2*np.diag([1.0, 1.0, 1.0])
    R = 1e-2*np.diag([1.0, 1.0])
    Qf = 100.0*np.diag([1.0, 1.0, 1.0])


    circle_x = np.array([0.75, 1.5, 2.25])
    circle_y = np.array([0.75, 1.5, 2.25])
    obs1_pos = np.array([circle_x[0], circle_y[0]])
    obs2_pos = np.array([circle_x[1], circle_y[1]])
    obs3_pos = np.array([circle_x[2], circle_y[2]])

def obstacle_constraint_expr(pos_obs, pos_car):
    return (pos_car[0] - pos_obs[0])**2 + (pos_car[1] - pos_obs[1])**2

def get_obstacle_positions():
    #define obstacle positions
    # lower line
    x_pos_lower = np.linspace(0,10, 50)
    obs_pos_list = [np.array([x, 0]) for x in x_pos_lower]

    # 3 vertical lines
    y_pos_lower = np.linspace(0, 5, 30)[1:]
    obs_pos_list1 = [np.array([0, y]) for y in y_pos_lower]
    obs_pos_list2 = [np.array([5, y]) for y in y_pos_lower]
    obs_pos_list3 = [np.array([10, y]) for y in y_pos_lower]

    obs_pos_list += obs_pos_list1
    obs_pos_list += obs_pos_list2
    obs_pos_list += obs_pos_list3

    # two upper lines
    x_pos_upper1 = np.linspace(0,3, 15)[1:]
    obs_pos_list4 = [np.array([x, 5]) for x in x_pos_upper1]
    x_pos_upper2 = np.linspace(5,8, 15)[1:]
    obs_pos_list5 = [np.array([x, 5]) for x in x_pos_upper2]

    obs_pos_list += obs_pos_list4
    obs_pos_list += obs_pos_list5
    return obs_pos_list

def get_initial_guess(N):
    # Given points
    X_guess = np.array([
        [2.5, 2.5, 0.],
        [4., 5., 0.785],
        [5., 6.25, 0.],
        [7.5, 6.25, -0.261],
        [9., 5., -1.57],
        [7.5, 2.5, 0.]
    ])

    # Create a parameter (t) for interpolation
    t = np.linspace(0, 1, len(X_guess))  # Normalize parameter values from 0 to 1

    # Define interpolation functions for each coordinate
    interp_funcs = [interp1d(t, X_guess[:, i], kind='cubic') for i in range(3)]

    # Generate 101 evenly spaced points along the parameter
    t_interp = np.linspace(0, 1, N)

    # Compute interpolated values
    X_interp = np.column_stack([f(t_interp) for f in interp_funcs])
    return X_interp



def create_problem(opts : AcadosOcpOptions, mode : str ='FIXED_TIME'):

    params = DubinsCarEscapeParameters()
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
        ocp.constraints.idxbxe_0 = np.array([1, 2, 3])

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
    obs_pos = get_obstacle_positions()
    n_obs = len(obs_pos)
    obs_expr = []

    for obs in obs_pos:
        cas_expr = obstacle_constraint_expr(obs, car_pos)
        obs_expr = cs.vertcat(obs_expr, cas_expr)

    ocp.model.con_h_expr = obs_expr
    ocp.constraints.uh = ACADOS_INFTY*np.ones(n_obs)
    ocp.constraints.lh = (params.r_car+params.r_circle_obs)**2*np.ones(n_obs)

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
    params = DubinsCarEscapeParameters()
    # Given points
    X_guess = np.array([
        [2.5, 2.5, 0.],
        [4., 5., 0.785],
        [5., 6.25, 0.],
        [7.5, 6.25, -0.261],
        [9., 5., -1.57],
        [7.5, 2.5, 0.]
    ])

    # Create a parameter (t) for interpolation
    t = np.linspace(0, 1, len(X_guess))  # Normalize parameter values from 0 to 1

    # Define interpolation functions for each coordinate
    interp_funcs = [interp1d(t, X_guess[:, i], kind='cubic') for i in range(3)]

    # Generate 101 evenly spaced points along the parameter
    t_interp = np.linspace(0, 1, params.N+1)

    # Compute interpolated values
    init_X = np.column_stack([f(t_interp) for f in interp_funcs])
    # return X_interp

    # init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    for i in range(params.N):
        # init_X[i,:] = params.initial_state
        init_U[i,:] = np.zeros(params.nu)
    # init_X[params.N,:] = params.initial_state
    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):
    sol_x_y = sol_X[:,:2]
    params = DubinsCarEscapeParameters()
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    obs_pos = get_obstacle_positions()
    for obs in obs_pos:
        rect1 = Circle((obs[0], obs[1]), params.r_circle_obs, linewidth=1, edgecolor='tab:gray', facecolor='tab:gray')
        ax.add_patch(rect1)

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
