# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import casadi as ca

@dataclass
class ParticleSteeringProblemParameters:
    N : int = 100
    M : int = 4
    Tf : float = 1.0
    nx : int = 5
    nu : int = 1
    T0 : float = 1.0 # initial guess for the time

def free_time_particle_steering() -> AcadosModel:

    model_name = 'free_time_particle_steering'

    # parameters
    a = 100

    # set up states & controls
    T = ca.SX.sym('T')
    y1 = ca.SX.sym('y1')
    y2 = ca.SX.sym('y2')
    y1_dot = ca.SX.sym('y1_dot')
    y2_dot = ca.SX.sym('y2_dot')

    states = ca.vertcat(T, y1, y2, y1_dot, y2_dot)

    u = ca.SX.sym('u')

    controls = ca.vertcat(u)

    # xdot
    T_dot = ca.SX.sym('T_dot')
    y1_ddot = ca.SX.sym('y1_dot')
    y2_ddot = ca.SX.sym('y2_dot')
    y1_dddot = ca.SX.sym('y1_ddot')
    y2_dddot = ca.SX.sym('y2_ddot')

    states_dot = ca.vertcat(T_dot, y1_ddot, y2_ddot, y1_dddot, y2_dddot)

    # dynamics
    f_expl = ca.vertcat(0,
                     T*y1_dot,
                     T*y2_dot,
                     T*a*ca.cos(u),
                     T*a*ca.sin(u)
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

    params = ParticleSteeringProblemParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = free_time_particle_steering()
    ocp.model = model

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]
    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0,1,2,3,4])
    ocp.constraints.idxbxe_0 = np.array([1,2,3,4])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, 5.0, 45.0, 0.0])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, 5.0, 45.0, 0.0])
    ocp.constraints.idxbx_e = np.array([0,2,3,4])

    # Path constraints on control
    ocp.constraints.lbu = np.array([-np.pi/2])
    ocp.constraints.ubu = np.array([np.pi/2])
    ocp.constraints.idxbu = np.array([0])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = ParticleSteeringProblemParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    ts = np.linspace(0, 1, params.N+1)
    y1s = 5*(ts/params.T0)
    y1_dots = 45*(ts/params.T0)

    for i in range(params.N):
        init_X[i,:] = np.array([params.T0, y1s[i], 0.0, y1_dots[i], 0.0])
    init_X[params.N,:] = np.array([params.T0, y1s[params.N-1], 0.0, y1_dots[params.N-1], 0.0])

    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):
    """
    This function plots the height and the control.
    """
    y1_sol = sol_X[:,1].squeeze()
    y2_sol = sol_X[:,2].squeeze()
    u_sol = sol_U.squeeze()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Solution of $y_2$')
    plt.plot(y1_sol, y2_sol)
    plt.xlim((0, 14))
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")

    plt.subplot(1, 2, 2)
    plt.title('Solution of $u$')
    plt.plot(y1_sol[:-1], u_sol)
    plt.xlim((0, 14))
    plt.xlabel("$y_1$")
    plt.ylabel("$u$")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
