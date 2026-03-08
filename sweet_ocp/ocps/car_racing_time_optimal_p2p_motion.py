# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from dataclasses import dataclass
from sweet_ocp.models.bicycle_models import export_free_time_bicycle
import numpy as np
from matplotlib import pyplot as plt

def centerLine(x):
    return np.arctan(4*(x-5))

@dataclass
class RacingParameters:
    #Initial states
    x0 : float = 0.0
    y0 : float= centerLine(0.0)
    theta0 : float = 0.0
    v0 : float = 1.0

    xf : float = 20.0
    vf : float = 0.0

    #Constraints on velocity and acceleration
    v_min : float = 0.0
    v_max : float= 1.0
    a_max : float = 1.0
    a_min : float = -2.0
    delta_max : float = np.pi/4
    theta_max : float = np.pi/2
    x_min : float = 0.0
    x_max : float = 20.0

    N : int = 50
    M : int = 10 # Needed for integrator
    T0 : float = 10 # initial guess for the time
    Tf : float = 1.0
    nx : int = 5
    nu : int = 2

def create_problem(opts : AcadosOcpOptions):

    # create ocp object to formulate the OCP
    params = RacingParameters()
    ocp = AcadosOcp()

    # set model
    model = export_free_time_bicycle()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]
    # ocp.cost.cost_type = 'EXTERNAL'
    # ocp.model.cost_expr_ext_cost = 1e-5*model.u[0]**2
    ###########################################################################
    # Define constraints
    ###########################################################################
    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, params.x0, params.y0, params.theta0, params.v0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, params.x0, params.y0, params.theta0, params.v0])
    ocp.constraints.idxbx_0 = np.array([0,1,2,3,4])
    ocp.constraints.idxbxe_0 = np.array([1,2,3,4])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, params.xf, -params.theta_max, params.vf])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, params.xf, params.theta_max, params.vf])
    ocp.constraints.idxbx_e = np.array([0,1,3,4])

    # Path constraints on control
    ocp.constraints.lbu = np.array([-params.delta_max, params.a_min])
    ocp.constraints.ubu = np.array([+params.delta_max, params.a_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # Path constraints on states
    ocp.constraints.lbx = np.array([params.x_min, -params.theta_max, params.v_min])
    ocp.constraints.ubx = np.array([params.x_max, params.theta_max, params.v_max])
    ocp.constraints.idxbx = np.array([1, 3, 4])

    # Nonlinear Constraints
    ocp.model.con_h_expr = centerLine(model.x[1]-0.3) -model.x[2]
    ocp.constraints.lh = np.array([-0.5])
    ocp.constraints.uh = np.array([0.5])

    ocp.model.con_h_expr_e = centerLine(model.x[1]-0.3) -model.x[2]
    ocp.constraints.lh_e = np.array([-0.5])
    ocp.constraints.uh_e = np.array([0.5])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = RacingParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    for i in range(params.N):
        init_X[i,:] = np.array([10.0, params.x0, params.y0, params.theta0, params.v0])
        init_U[i,:] = np.array([0, 0])
    init_X[params.N,:] = np.array([10.0, params.x0, params.y0, params.theta0, params.v0])

    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):

    plt.figure()
    ts = np.linspace(0, 20, 1000)
    fs = centerLine(ts)
    plt.plot(ts.squeeze(), fs.squeeze(), ':', color=[0.5,0.5,0.5])
    plt.plot(ts, centerLine(ts+0.3)+0.5, color=[0.5,0.5,0.5])
    plt.plot(ts, centerLine(ts-0.3)-0.5, color=[0.5,0.5,0.5])
    plt.plot(sol_X[:,1], sol_X[:,2], marker='o', label="Solution Trajectory")
    plt.show()


if __name__ == '__main__':
    pass
