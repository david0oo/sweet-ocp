# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
from dataclasses import dataclass

@dataclass
class GoddardRocketModelParameters:
    nx : int = 4
    nu : int = 1
    N : int = 200
    M : int = 4
    T0 : float = 1.0
    Tf : float = 1.0 # It is a time optimal problem with time scaling

    # Parameters
    g0 : float = 1.0
    m0 : float = 1.0
    h0 : float = 1.0

    v_c : float = 620
    h_c : float = 500
    m_c : float = 0.6
    T_c : float = 3.5

    m_f : float = m_c*m0
    D_c : float = 0.5*v_c*(m0/g0)
    c : float = 0.5*ca.sqrt(g0*m0)
    m_f : float = m_c*m0
    thrust_max : float = T_c*g0*m0

def free_time_goddard_rocket_model() -> AcadosModel:

    model_name = 'free_time_goddard_rocket_model'

    params = GoddardRocketModelParameters()

    # set up states & controls
    T = ca.SX.sym('T')
    h = ca.SX.sym('h')
    v = ca.SX.sym('v')
    m = ca.SX.sym('m')

    states = ca.vertcat(T, h, v, m)

    thrust = ca.SX.sym('thrust')

    controls = ca.vertcat(thrust)

    # xdot
    T_dot = ca.SX.sym('T_dot')
    h_dot = ca.SX.sym('h_dot')
    v_dot = ca.SX.sym('v_dot')
    m_dot = ca.SX.sym('m_dot')

    states_dot = ca.vertcat(T_dot, h_dot, v_dot, m_dot)

    Dhv = params.D_c * v**2 * ca.exp(-params.h_c*((h-params.h0)/params.h0))
    gh = params.g0*(params.h0/h)**2

    # dynamics
    f_expl = ca.vertcat(0,
                        T*(v),
                        T*(((thrust - Dhv) - m*gh)/m),
                        T*((-thrust/params.c))
                        )

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

    params = GoddardRocketModelParameters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = free_time_goddard_rocket_model()
    ocp.model = model

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = -model.x[1]
    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, 1.0, 0.0, 1.0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, 1.0, 0.0, 1.0])
    ocp.constraints.idxbx_0 = np.array([0,1,2,3])
    ocp.constraints.idxbxe_0 = np.array([1,2,3])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, params.m_f])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, params.m_f])
    ocp.constraints.idxbx_e = np.array([0,3])

    # Path constraints on states
    ocp.constraints.lbx = np.array([0.0, params.h0, 0.0, params.m_f])
    ocp.constraints.ubx = np.array([ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, params.m0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    # Path constraints on control
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([params.thrust_max])
    ocp.constraints.idxbu = np.array([0])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = GoddardRocketModelParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    ts = np.linspace(0, params.T0, params.N+1)
    vs = (ts/params.T0) * (1-(ts/params.T0))
    ms = (params.m_f - params.m0) * (ts/params.T0) + params.m0

    for i in range(params.N):
        init_X[i,:] = np.array([params.T0, 1.0, vs[i], ms[i]])
        init_U[i,:] = np.array([params.thrust_max/2])
    init_X[params.N,:] = np.array([params.T0, 1.0, vs[params.N-1], ms[params.N-1]])
    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):
    """
    This function plots the height and the control.
    """
    params = GoddardRocketModelParameters()
    T_sol = sol_X[0,0]
    T_grid = np.linspace(0, T_sol, params.N+1)
    h_sol = sol_X[:,1].squeeze()
    v_sol = sol_X[:,2].squeeze()
    m_sol = sol_X[:,3].squeeze()
    thrust_sol = sol_U.squeeze()

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('Altitude $h$')
    plt.plot(T_grid, h_sol)
    plt.ylim((1, 1.015))
    plt.xlim((0, 0.2))
    plt.xlabel("time")
    plt.ylabel("$h$")

    plt.subplot(2, 2, 2)
    plt.title('Mass $m$')
    plt.plot(T_grid, m_sol)
    plt.xlim((0, 0.2))
    plt.ylim((0.5, 1))
    plt.xlabel("time")
    plt.ylabel("$m$")

    plt.subplot(2, 2, 3)
    plt.title('Velocity $v$')
    plt.plot(T_grid, v_sol)
    plt.xlim((0, 0.2))
    plt.ylim((0, 0.14))
    plt.xlabel("time")
    plt.ylabel("$m$")

    plt.subplot(2, 2, 4)
    plt.title('Thrust $thrust$')
    plt.plot(T_grid[:-1], thrust_sol)
    plt.xlim((0, 0.2))
    plt.ylim((-0.1, 3.6))
    plt.xlabel("time")
    plt.ylabel("$m$")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
