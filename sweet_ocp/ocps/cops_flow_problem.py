# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
from dataclasses import dataclass
from typing import Union

@dataclass
class FlowModelParameters:
    L : float = 5.0
    N : int = 20
    # N : int = 200
    M : int = 4 # Needed for integrator
    Tf : float = 1.0 # initial guess for the time
    nx : int = 4
    nu : int = 0

def flow_in_channel_model() -> AcadosModel:

    model_name = 'flow_in_channel'

    # set up states & controls
    u = ca.SX.sym('u')
    du = ca.SX.sym('du')
    ddu = ca.SX.sym('ddu')
    dddu = ca.SX.sym('dddu')

    states = ca.vertcat(u, du, ddu, dddu)

    controls = []

    # xdot
    ddddu = ca.SX.sym('ddddu')

    states_dot = ca.vertcat(du, ddu, dddu, ddddu)

    R = ca.SX.sym('R')

    # dynamics
    f_expl = ca.vertcat(du,
                        ddu,
                        dddu,
                        R*(du*ddu - u*dddu))

    f_impl = states_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.p = R
    model.name = model_name

    return model

def create_problem(opts : AcadosOcpOptions):

    params = FlowModelParameters()

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = flow_in_channel_model()
    ocp.model = model

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = 0.0
    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, 0.0])
    ocp.constraints.ubx_0 = np.array([0.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0,1])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([1.0, 0.0])
    ocp.constraints.ubx_e = np.array([1.0, 0.0])
    ocp.constraints.idxbx_e = np.array([0,1])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf

    ocp.parameter_values = np.array([10])

    return ocp

def create_initial_guess(variant : int = 1):
    """
    This function creates an initial guess for the flow problem.
    Choose the reynolds_number from [0, 10, 100, 1000]
    """
    params = FlowModelParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    init_P = np.zeros((params.N+1, 1))

    reynolds_number = 0.0
    if variant == 0:   
        reynolds_number = 0.0 
    elif variant == 1:   
        reynolds_number = 10.0 
    elif variant == 2:   
        reynolds_number = 100.0 
    elif variant == 3:   
        reynolds_number = 1000.0
    # else use default value 0.0        

    # Initial guess
    ts = np.linspace(0, params.Tf, params.N+1)
    u_init = ts**2*(3-2*ts)
    for i in range(params.N):
        init_X[i,:] = np.array([u_init[i], 0.0, 0.0, 0.0])
        init_P[i] = np.array([reynolds_number])
    init_X[params.N,:] = np.array([u_init[i], 0.0, 0.0, 0.0])
    init_P[params.N] = np.array([reynolds_number])
    return init_X, init_U, init_P


def plot_trajectory(sol_X: np.array, sol_U: np.array):
    """
    This function plots the height and the control.
    """
    params = FlowModelParameters()
    T0 = 1.0
    ts = np.linspace(0, T0, params.N+1)
    plt.figure(figsize=(4, 4))
    plt.title('$u^{\\prime}$')
    plt.plot(ts, sol_X[:, 1])
    plt.ylim((0.0, 1.6))
    plt.xlim((0.0, 1.0))
    plt.xlabel("time")
    plt.tight_layout()
    plt.show()

def plot_different_reynolds_numbers(du_sols: list, labels=None):
    """
    This function plots the height and the control.
    """
    params = FlowModelParameters()
    T0 = 1.0
    ts = np.linspace(0, T0, params.N+1)
    plt.figure(figsize=(4, 4))
    plt.title('$u^{\\prime}$')
    for i in range(len(du_sols)):
        plt.plot(ts, du_sols[i], label=labels[i])
    plt.ylim((0.0, 1.6))
    plt.xlim((0.0, 1.0))
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
