# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from sweet_ocp.models.simple_pendulum_models import export_fixed_time_simple_pendulum_model, export_free_time_simple_pendulum_model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import casadi as cs
import scipy.linalg

@dataclass
class SimplePendulumParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 50
    nx : int = 2
    nu : int = 1
    Tf : float = 3.0
    # Initial states
    Q = 1e-3*np.diag([1.0, 1.0])
    R = 1e-3*np.diag([1.0])
    Qf = np.diag([1.0, 1.0])

    initial_state  = np.array([0., 0.])
    terminal_state  = np.array([np.pi, 0.])

    u_max = np.array([3.])
    u_min = -u_max
    u_init = np.array([0.0])

def create_problem(opts: AcadosOcpOptions, mode: str = 'FIXED_TIME'):
    """
    Create the optimal control problem (OCP) for the simple pendulum problem.

    Args:
        opts (AcadosOcpOptions): Solver options for the OCP.
        mode (str): Mode of the problem, either 'FIXED_TIME' or 'FREE_TIME'.
        create_json (bool): Whether to create a JSON file for the solver.

    Returns:
        AcadosOcpSolver: Solver for the formulated OCP.
    """
    params = SimplePendulumParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    if mode == 'FIXED_TIME':
        model = export_fixed_time_simple_pendulum_model()
    else:
        model = export_free_time_simple_pendulum_model()

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
        ocp.cost.yref = yref

        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = params.Q
        ocp.cost.Vx_e = np.eye(params.nx)
        ocp.cost.yref_e = yref_e
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
        ocp.constraints.idxbx_0 = np.array([0, 1])
        ocp.constraints.idxbxe_0 = np.array([0, 1])
    else:
        ocp.constraints.lbx_0 = np.append([0.0], params.initial_state)
        ocp.constraints.ubx_0 = np.append([ACADOS_INFTY], params.initial_state)
        ocp.constraints.idxbx_0 = np.array([0, 1, 2])
        ocp.constraints.idxbxe_0 = np.array([1, 2])

    # Path constraints on controls
    ocp.constraints.lbu = params.u_min
    ocp.constraints.ubu = params.u_max
    ocp.constraints.idxbu = np.array([0])

    # Terminal constraints
    if mode == 'FIXED_TIME':
        ocp.constraints.lbx_e = params.terminal_state
        ocp.constraints.ubx_e = params.terminal_state
        ocp.constraints.idxbx_e = np.array([0, 1])
    else:
        ocp.constraints.lbx_e = np.append([0.0], params.terminal_state)
        ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state)
        ocp.constraints.idxbx_e = np.array([0, 1, 2])

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
    """
    Create an initial guess for the states and controls for the simple pendulum problem.

    Returns:
        tuple: A tuple containing the initial guess for the states (init_X) and controls (init_U).
    """
    params = SimplePendulumParameters()
    init_X = np.zeros((params.N + 1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    for i in range(params.N):
        init_X[i, :] = params.initial_state
        init_U[i, :] = params.u_init
    init_X[params.N, :] = params.initial_state
    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    params = SimplePendulumParameters()
    tf = params.Tf
    time_scale = np.linspace(0, tf, params.N+1)

    theta = sol_X[:, 0]
    theta_dot = sol_X[:, 1]

    plt.figure(figsize=(8, 8))
    plt.title("Simple Pendulum Solution")
    plt.subplot(3, 1, 1)
    plt.plot(time_scale, theta, label="theta")
    # plt.ylim((900, 1000))
    # plt.xlim((0, 120))
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_scale, theta_dot, label="theta_dot")
    # plt.ylim((9, 14))
    # plt.xlim((0, 120))
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_scale[:-1], sol_U, label="torque")
    # plt.ylim((0.6, 1.4))
    # plt.xlim((0, 120))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    pass
