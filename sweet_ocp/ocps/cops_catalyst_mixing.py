# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
from dataclasses import dataclass


@dataclass
class CatalystMixingProblemParameters:
    Tf : float = 1.0
    nx : int = 2
    nu : int = 1
    N : int = 200
    M : int = 4

def catalyst_mixing_model() -> AcadosModel:

    model_name = 'catalyst_mixing_model'

    # set up states & controls
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')

    states = ca.vertcat(x1, x2)

    u = ca.SX.sym('u')

    controls = ca.vertcat(u)

    # xdot
    x1_dot = ca.SX.sym('x1_dot')
    x2_dot = ca.SX.sym('x2_dot')

    states_dot = ca.vertcat(x1_dot, x2_dot)

    # dynamics
    f_expl = ca.vertcat(u*(10*x2-x1),
                        u*(x1-10*x2) - (1-u)*x2)

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

    params = CatalystMixingProblemParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = catalyst_mixing_model()
    ocp.model = model

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = -1 + model.x[0] + model.x[1]
    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([1.0, 0.0])
    ocp.constraints.ubx_0 = np.array([1.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0,1])
    ocp.constraints.idxbxe_0 = np.array([0,1])

    # Path constraints on control
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = CatalystMixingProblemParameters()

    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    for i in range(params.N):
        init_X[i,:] = np.array([1.0, 0.0])
        init_U[i,:] = np.array([0.0])
    init_X[params.N,:] = np.array([1.0, 0.0])

    return init_X, init_U


def plot_trajectory(sol_X, sol_U):
    """
    This function plots the height and the control.
    """
    params = CatalystMixingProblemParameters()
    time_sol = np.linspace(0, 1, params.N+1)
    plt.figure(figsize=(10, 4))
    plt.title('Solution of $u$')
    plt.plot(time_sol[:-1], sol_U)
    plt.xlim((0, 1))
    plt.ylim((-0.1, 1.1))
    plt.xlabel("time")
    plt.ylabel("$u$")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
