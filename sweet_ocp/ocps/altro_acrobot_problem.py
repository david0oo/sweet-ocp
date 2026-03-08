# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from acados_test_problems.models.double_pendulum_models import export_fixed_time_double_pendulum_model, export_free_time_double_pendulum_model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import casadi as cs
import scipy.linalg

@dataclass
class AcrobotParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 100
    nx : int = 4
    nu : int = 1
    Tf : float = 5.0
    # Initial states
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([0.01])
    Qf = 100.0*Q

    initial_state  = np.array([-np.pi/2, 0., 0., 0.])
    terminal_state  = np.array([np.pi/2, 0., 0., 0.])

    u_min = np.array([-15.])
    u_max = np.array([15.])
    u_init = np.array([0.0])

def create_problem(opts : AcadosOcpOptions, mode : str ='FIXED_TIME'):

    params = AcrobotParameters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    if mode == 'FIXED_TIME':
        model = export_fixed_time_double_pendulum_model()
    else:
        model = export_free_time_double_pendulum_model()

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
        ocp.cost.W_e = params.Qf
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
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
        ocp.constraints.idxbxe_0 = np.array([0, 1, 2, 3])
    else:
        ocp.constraints.lbx_0 = np.append([0.0], params.initial_state)
        ocp.constraints.ubx_0 = np.append([ACADOS_INFTY], params.initial_state)
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])
        ocp.constraints.idxbxe_0 = np.array([1, 2, 3, 4])

    # Path constraints on controls
    ocp.constraints.lbu = params.u_min
    ocp.constraints.ubu = params.u_max
    ocp.constraints.idxbu = np.array([0])

    # Terminal constraints
    if mode == 'FIXED_TIME':
        ocp.constraints.lbx_e = params.terminal_state
        ocp.constraints.ubx_e = params.terminal_state
        ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])
    else:
        ocp.constraints.lbx_e = np.append([0.0], params.terminal_state)
        ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state)
        ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4])

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
    params = AcrobotParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    for i in range(params.N):
        init_X[i,:] = params.initial_state
        init_U[i,:] = np.zeros(params.nu)
    init_X[params.N,:] = params.initial_state
    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    pass

if __name__ == '__main__':
    pass
