# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, latexify_plot, AcadosOcpOptions
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

@dataclass
class HanggliderProblemParameters:
    """
    Parameters used for setting up the crane problem.
    """
    N : int = 100
    nx : int = 5
    nu : int = 1
    initial_state = np.array([0, 1000, 13.2275675, -1.287500500])
    terminal_state = np.array([0, 900,  13.2275675, -1.287500500])
    c_max : float = 1.4
    Tf : float = 1.0
    # Initial guess
    T0 : float = 1.0


@dataclass
class HanggliderModelParameters:
    uM : float = 2.5
    R : float = 100 # radius of the upward draft [m]
    C0 : float = 0.034 # drag coefficient
    m : float = 100 # mass [kg]
    g : float = 9.80665 # [m/s^2]
    S : float = 14 # wingsize [m^2]
    rho : float = 1.13 # [kg/m^3]
    k : float = 0.069662


def export_free_time_hangglider_model() -> AcadosModel:

    model_name = 'free_time_hangglider_model'

    params = HanggliderModelParameters()

    ## system dimensions
    nx = 4 + 1 # +1 for the time

    # Define the states
    T = cs.SX.sym('T')
    p_x = cs.SX.sym('p_x', 1)
    p_y = cs.SX.sym('p_y', 1)
    v_x = cs.SX.sym('v_x', 1)
    v_y = cs.SX.sym('v_y', 1)
    states = cs.vertcat(T, p_x, p_y, v_x, v_y)

    # define controls
    C_L = cs.SX.sym('C_L', 1)
    controls = cs.vertcat(C_L)

    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    C_D = params.C0 + params.k*C_L**2

    # vertical distance to the of the upward draft
    X = (p_x/params.R - 2.5)**2
    # upward draft
    ua_px = cs.Function('ua', [p_x], [params.uM*(1 - X)*cs.exp(-X)])

    # Drag
    V_y = v_y - ua_px(p_x)
    v_r = cs.sqrt(v_x**2 + V_y**2)
    D = 0.5*params.rho*params.S*C_D*v_r**2

    # Lift
    L = 0.5*params.rho*params.S*C_L*v_r**2

    sin_eta = V_y / v_r
    cos_eta = v_x / v_r

    ## dynamics
    f_expl = cs.vertcat(0,
                        T*cs.vertcat(v_x,
                                     v_y,
                                     1 / params.m * (-L * sin_eta - D * cos_eta),
                                     1 / params.m * (L * cos_eta - D * sin_eta - params.m * params.g)))
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model


def create_problem(opts : AcadosOcpOptions):

    params = HanggliderProblemParameters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_time_hangglider_model()
    ocp.model = model

    ###########################################################################
    # Define cost: maximize the x-position
    ###########################################################################
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = -model.x[1]

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.append([0.0], params.initial_state)
    ocp.constraints.ubx_0 = np.append([ACADOS_INFTY], params.initial_state)
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])
    ocp.constraints.idxbxe_0 = np.array([1, 2, 3, 4])

    # Path constraints controls
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([params.c_max])
    ocp.constraints.idxbu = np.array([0])


    # Terminal constraints
    ocp.constraints.lbx_e = np.append([0.0], params.terminal_state[1:4])
    ocp.constraints.ubx_e = np.append([ACADOS_INFTY], params.terminal_state[1:4])
    ocp.constraints.idxbx_e = np.array([0, 2, 3, 4])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = HanggliderProblemParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    p_x_init = params.initial_state[0] + params.initial_state[2]*np.linspace(0, params.T0, params.N+1)
    print(p_x_init.shape)
    p_y_init = params.initial_state[1] + params.initial_state[3]*np.linspace(0, params.T0, params.N+1)

    for i in range(params.N):
        init_X[i,:] = np.array([params.T0, p_x_init[i], p_y_init[i], params.initial_state[2], params.initial_state[3]])
        init_U[i,:] = np.array([0.5*params.c_max])
    init_X[params.N,:] = np.array([params.T0, p_x_init[params.N], p_y_init[params.N], params.initial_state[2], params.initial_state[3]])

    return init_X, init_U

def plot_trajectory(sol_X, sol_U):
    params = HanggliderProblemParameters()
    tf = sol_X[0, 0]
    time_scale = np.linspace(0, tf, params.N+1)

    altitude = sol_X[:, 2]
    v_x = sol_X[:, 3]
    v_y = sol_X[:, 4]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(time_scale, altitude, label="altitude")
    plt.ylim((900, 1000))
    plt.xlim((0, 120))
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_scale[:-1], sol_U, label="$c_L$")
    plt.ylim((0.6, 1.4))
    plt.xlim((0, 120))
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time_scale, v_x, label="v_x")
    plt.ylim((9, 14))
    plt.xlim((0, 120))
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(time_scale, v_y, label="v_y")
    plt.ylim((-2.5, 2))
    plt.xlim((0, 120))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    pass
