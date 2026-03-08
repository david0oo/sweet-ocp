# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from sweet_ocp.models.free_flying_robot_model import export_free_flying_robot_model
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

#
# This problem is from the Bett's book on optimal control: Free flying robot
#

@dataclass
class FreeFlyingRobotProblemParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 100
    Tf : float = 12.0
    nx : int = 6
    nu : int = 4
    initial_state = np.array([-10.0, -10.0, np.pi/2, 0.0, 0.0, 0.0])
    terminal_state = np.zeros(nx)

def create_problem(opts: AcadosOcpOptions, mode: str = 'FIXED_TIME'):
    """
    Create the optimal control problem (OCP) for the free-flying robot problem.

    Args:
        opts (AcadosOcpOptions): Solver options for the OCP.
        mode (str): Mode of the problem, either 'FIXED_TIME' or 'FREE_TIME'.
        create_json (bool): Whether to create a JSON file for the solver.

    Returns:
        AcadosOcpSolver: Solver for the formulated OCP.
    """
    params = FreeFlyingRobotProblemParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_flying_robot_model()
    ocp.model = model

    ###########################################################################
    # Define cost
    ###########################################################################
    ocp.cost.cost_type_0 = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_0 = model.u[0] + model.u[1] + model.u[2] + model.u[3]
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.u[0] + model.u[1] + model.u[2] + model.u[3]

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = params.initial_state
    ocp.constraints.ubx_0 = params.initial_state
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.idxbxe_0 = np.array([0, 1, 2, 3, 4, 5])

    # Path constraints on controls
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Linear constraints on controls
    ocp.constraints.D = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    ocp.constraints.C = np.zeros((2, params.nx))
    ocp.constraints.lg = np.array([-ACADOS_INFTY, -ACADOS_INFTY])
    ocp.constraints.ug = np.array([1.0, 1.0])

    # Terminal constraints
    ocp.constraints.lbx_e = params.terminal_state
    ocp.constraints.ubx_e = params.terminal_state
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = FreeFlyingRobotProblemParameters()

    lin_distr = np.linspace(params.initial_state, params.terminal_state, params.N+1)

    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    for i in range(params.N):
        init_X[i,:] = lin_distr[i]
        init_U[i,:] = np.zeros(params.nu)
    init_X[params.N,:] = lin_distr[params.N]
    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    params = FreeFlyingRobotProblemParameters()
    tf = params.Tf
    time_scale = np.linspace(0, tf, params.N+1)

    x1 = sol_X[:, 0]
    x2 = sol_X[:, 1]
    theta = sol_X[:, 2]
    v1 = sol_X[:, 3]
    v2 = sol_X[:, 4]
    omega = sol_X[:, 5]

    u1 = sol_U[:, 0]
    u2 = sol_U[:, 1]
    u3 = sol_U[:, 2]
    u4 = sol_U[:, 3]

    plt.figure(figsize=(8, 8))
    plt.suptitle('States')
    plt.subplot(3, 2, 1)
    plt.plot(time_scale, x1, label="$x_1$")
    plt.ylim((-11, 1))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_scale, x2, label="$x_2$")
    plt.ylim((-11, 1))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time_scale, v1, label="$v_1$")
    plt.ylim((-0.2, 2.0))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time_scale, v2, label="$v_2$")
    plt.ylim((-0.2, 2.0))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time_scale, theta, label="$\\theta$")
    plt.ylim((-0.2, 1.7))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_scale, omega, label="$\\omega$")
    plt.ylim((-0.38, 0.17))
    plt.xlim((-1, 12))
    plt.legend()

    plt.figure(figsize=(8, 8))
    plt.suptitle('Controls')
    plt.subplot(3, 2, 1)
    plt.plot(time_scale[:-1], u1, label="$u_1$")
    plt.ylim((-0.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_scale[:-1], u3, label="$u_3$")
    plt.ylim((-0.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time_scale[:-1], u2, label="$u_2$")
    plt.ylim((-0.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time_scale[:-1], u4, label="$u_4$")
    plt.ylim((-0.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time_scale[:-1], u1-u2, label="$T_1 = u_1 - u_2$")
    plt.ylim((-1.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_scale[:-1], u3-u4, label="$T_2 = u_3 - u_4$")
    plt.ylim((-1.2, 1.2))
    plt.xlim((-1, 12))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    pass
