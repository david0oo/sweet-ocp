# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from acados_test_problems.models.moonlander_models import export_free_time_moonlander_model, transf
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

@dataclass
class MoonlanderProblemParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 50
    nx : int = 7
    nu : int = 2
    Tf : float = 1.0
    # Initial states
    p0  = np.array([0., 0.])
    dp0  = np.array([0., 0.])
    theta0  = 0.0
    dtheta0  = 0.0
    initial_state = np.append(np.append(p0, dp0), [theta0, dtheta0])

    # Terminal states
    target = np.array([5., 5.])
    p_f = target
    dp_f = np.array([0., 0.])
    terminal_state = np.append(p_f, dp_f)

    g : float = 9.81
    max_thrust : float = 2*g


def create_problem(opts : AcadosOcpOptions):

    params = MoonlanderProblemParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_time_moonlander_model()
    ocp.model = model

    ###########################################################################
    # Define cost: time optimal
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

    # Path constraints on controls
    ocp.constraints.lbu = np.array([0.0, 0.0])
    ocp.constraints.ubu = np.array([params.max_thrust, params.max_thrust])
    ocp.constraints.idxbu = np.array([0, 1])

    # Terminal constraints
    ocp.constraints.lbx_e = np.append([0.0], params.terminal_state)
    ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state)
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

"""
Taken from Lander's repository
"""
def visualize(F_r_sol, Ftot_sol, F1_sol, F2_sol):
    plt.quiver(F_r_sol[:,0,2], F_r_sol[:,1,2], Ftot_sol[:,0], Ftot_sol[:,1])
    plt.show()

    # visualize F1_sol and F2_sol using matplotlib
    plt.plot(F1_sol)
    plt.plot(F2_sol)
    plt.show()

def create_initial_guess():
    params = MoonlanderProblemParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    # Initial guess
    T0 = 5.0

    for i in range(params.N):
        init_X[i, :] = np.array([T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        init_U[i, :] = np.array([5.0, 5.0])
    init_X[params.N,:] = np.array([T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    # visualize
    params = MoonlanderProblemParameters()
    F1_sol = sol_U[:, 0]
    F2_sol = sol_U[:, 1]
    p_sol = sol_X[:, 2:4]
    theta_sol = sol_X[:, 5]

    F_r_sol = np.zeros((params.N, 3, 3))
    F_tot_sol = np.zeros((params.N, 2))

    for i in range(params.N):
        F_r_sol[i, :, :] = transf(theta_sol[i], p_sol[i, :])
        F_tot_sol[i, :] = np.array((F_r_sol[i,:,:] @ cs.vertcat(0, F1_sol[i] + F2_sol[i], 1))[:2]).squeeze()
    visualize(F_r_sol, F_tot_sol, F1_sol, F2_sol)


if __name__ == '__main__':
    pass
