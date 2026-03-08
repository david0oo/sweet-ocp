# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from acados_test_problems.models.overhead_crane_models import export_fixed_time_crane_model, get_payload_position
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs
import scipy.linalg

#############################################
# This problem is taken from the GRAMPC paper
#############################################


@dataclass
class CraneProblemParameters:
    """
    Parameters used for setting up the crane problem.
    """
    Tf : float = 7.0
    N : int = 100
    nx : int = 6
    nu : int = 2
    # Initial states
    l0 : float = 2.0 # [m]
    x0 : float = -2.0 # [m]
    theta0 : float = 0.0
    l_dot0 : float = 0.0
    x_dot0 : float = 0.0
    theta_dot0 : float = 0.0
    initial_state = np.array([l0, x0, theta0, l_dot0, x_dot0, theta_dot0])

    # Terminal states
    l_f : float = 2.0
    x_f : float = 2.0
    theta_f : float = 0.0
    x_dot_f : float = 0.0
    l_dot_f : float = 0.0
    theta_dot_f : float = 0.0
    desired_state = np.array([l_f, x_f, theta_f, l_dot_f, x_dot_f, theta_dot_f])

    u_max : float = 2.0
    theta_dot_max : float = 0.3
    Q = 2*np.diag([2, 1, 1, 1, 2, 4])
    R = 2*np.diag([0.05, 0.05])

def create_problem(opts : AcadosOcpOptions):

    params = CraneProblemParameters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_fixed_time_crane_model()
    ocp.model = model


    ###########################################################################
    # Define cost
    ###########################################################################
    ny = params.nx + params.nu

    Vx = np.zeros((ny, params.nx))
    Vx[:params.nx, :] = np.eye(params.nx)
    Vu = np.zeros((ny, params.nu))
    Vu[params.nx:ny, :] = np.eye(params.nu)
    yref = np.zeros(ny,)
    yref[:params.nx] = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    # Add costs of starting and terminal costs
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.W_0 = scipy.linalg.block_diag(params.Q, params.R)
    ocp.cost.Vx_0 = np.vstack((np.eye(params.nx), np.zeros((params.nu, params.nx))))
    ocp.cost.Vu_0 = np.vstack((np.zeros((params.nx, params.nu)), np.eye(params.nu)))
    ocp.cost.yref_0 = np.append(params.desired_state, [0,0])

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(params.Q, params.R)
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.yref  = yref

    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W_e = params.Q
    ocp.cost.Vx_e = np.eye(params.nx)
    ocp.cost.yref_e  = params.desired_state

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = params.initial_state
    ocp.constraints.ubx_0 = params.initial_state
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.idxbxe_0 = np.array([0, 1, 2, 3, 4, 5])

    # # Path constraints on states
    ocp.constraints.lbx = np.array([-params.theta_dot_max])
    ocp.constraints.ubx = np.array([params.theta_dot_max])
    ocp.constraints.idxbx = np.array([5])

    ocp.constraints.lbu = np.array([-params.u_max, -params.u_max])
    ocp.constraints.ubu = np.array([params.u_max, params.u_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # payload_position = cs.vertcat(model.x[1] + model.x[0]*cs.sin(model.x[2]), -model.x[0]*cs.cos(model.x[2]))
    # constraint is y_pos >= -0.2*x_pos**2 + 1.25
    ocp.model.con_h_expr = cs.cos(model.x[2])*model.x[0] -0.2*(model.x[1] + model.x[0]*cs.sin(model.x[2]))**2 - 1.25
    ocp.constraints.lh = np.array([-ACADOS_INFTY])
    ocp.constraints.uh = np.array([0.0])

    # copy for terminal
    ocp.constraints.uh_e = ocp.constraints.uh
    ocp.constraints.lh_e = ocp.constraints.lh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess(variant=0):
    params = CraneProblemParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    # Initial guess
    x = np.linspace(-2.0, 2.0, params.N+1)

    if variant == 0:
        init_X[0,:] = params.initial_state
        for i in range(1, params.N):
            init_X[i,:] = params.initial_state
        init_X[params.N,:] = params.desired_state
        # solver_wfqp.set(i, "x", np.array([1.0, x[i], 0.0, 0.0, 0.0, 0.0]))
    else:
        init_X[0,:] = params.initial_state
        for i in range(1, params.N):
            init_X[i,:] = np.array([1.0, x[i], 0.0, 0.0, 0.0, 0.0])
        init_X[params.N,:] = params.desired_state
    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):
    """
    Plot the optimal trajectory of the crane load.
    """
    payload_x1_position, payload_x2_position = get_payload_position(sol_X)

    x_points = np.linspace(-2.3, 2.3, 100)

    plt.figure()
    plt.plot(x_points, 0*x_points, linestyle='--', color='k')
    plt.plot(x_points, -0.2*x_points**2 - 1.25)
    plt.plot(payload_x1_position, payload_x2_position, marker='o', label='payload trajectory')
    plt.legend()
    plt.xlim((-2.3, 2.3))
    plt.ylim((-2.3, 0.3))
    plt.xlabel('$x_1$-axis')
    plt.ylabel('$x_2$-axis')
    plt.show()

if __name__ == '__main__':
    pass
