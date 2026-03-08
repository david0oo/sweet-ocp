# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, AcadosOcpOptions
from acados_test_problems.models.bicycle_models import export_free_time_simple_bicycle
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

@dataclass
class HourGlassParameters:
    x0 : float = 0.0
    y0 : float = 0.0
    theta0 : float = np.pi/2
    xf : float = 0.0
    yf : float = 11.0
    thetaf : float = np.pi/2
    v_max : float = 1.0
    delta_max : float = np.pi/6
    Tf: float = 1.0
    N : int = 20
    M : int = 2
    nx : int = 4
    nu : int = 2

def create_problem(opts : AcadosOcpOptions):

    params = HourGlassParameters()
    # The flag denotes, if the problem should be transformed into a feasibility
    # problem, or if the unconstrained OCP should be solved.

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_time_simple_bicycle()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    ###########################################################################
    # Define cost
    ###########################################################################
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]
    # works much better with regularization
    # ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = 1e-5*model.u[0]**2

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, params.x0, params.y0, params.theta0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, params.x0, params.y0, params.theta0])
    ocp.constraints.idxbx_0 = np.array([0,1,2,3])
    ocp.constraints.idxbxe_0 = np.array([1,2,3])

    # Path constraints on control
    ocp.constraints.lbu = np.array([-params.delta_max, 0.0])
    ocp.constraints.ubu = np.array([+params.delta_max, +params.v_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([ACADOS_INFTY])
    ocp.constraints.idxbx = np.array([0])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, params.xf, params.yf, params.thetaf])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, params.xf, params.yf, params.thetaf])
    ocp.constraints.idxbx_e = np.array([0,1,2,3])

    # Convex over Nonlinear Constraints
    # r = ca.SX.sym('r', 1, 1)
    # ocp.model.con_phi_expr = r**2
    # ocp.model.con_r_in_phi = r
    # ocp.model.con_r_expr = (5*(model.x[1]+3)) / (ca.sqrt(1+15*(model.x[2]-5)**2))

    # ocp.constraints.lphi = np.array([0])
    # ocp.constraints.uphi = np.array([1])

    # Otherwise, exact Hessian is not possible
    ocp.model.con_h_expr = ((5*(model.x[1]+3)) / (ca.sqrt(1+15*(model.x[2]-5)**2)))**2
    ocp.constraints.lh = np.array([0])
    ocp.constraints.uh = np.array([1])

    # set prediction horizon
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess(variant = 0):
    params = HourGlassParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    if variant == 0:
        arr = np.array([18.0, 0.0, 0.0, 0.0]).reshape(1, params.nx)
        init_X = np.vstack([arr] * (params.N+1))
        init_U = np.zeros((params.N, params.nu))
    elif variant == 1:
        # Load andx set the initial guess
        with open(current+'/hour_glass_initial_guess.npy', 'rb') as f:
            init_X = np.load(f).T
            init_U = np.load(f).T
    else:
        raise ValueError("variant must be 0 or 1")

    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U :np.array):
    params = HourGlassParameters()

    # Number of control intervals
    N = params.N

    # Plot the x and y solutions + tunnel shape
    plt.figure()
    plt.title('TUNNEL N=' +str(N)+' Start: (' + str("{:.2f}".format(params.x0)) + ',' + str("{:.2f}".format(params.y0)) + ') End: (' + str("{:.2f}".format(params.xf)) + ',' + str("{:.2f}".format(params.yf)) + ')')
    plt.axis([-8, 5, -1 , 11])

    # Plot the tunnel
    y_co = np.linspace(params.y0, params.yf, 1000) # Points along the path
    r = np.sqrt(1+15*(y_co-5)**2)/5    # Distance between the path and the tunnel walls
    plt.plot(-3+r,y_co, 'r-')
    plt.plot(-3-r,y_co, 'r-')
    for i in range(len(y_co)):
        plt.hlines(y= y_co[i], xmin=-3-r[i], xmax=-3+r[i], color='r')
    angle = np.linspace(0,np.pi/4,500, endpoint=True)
    rf1 = np.sqrt(1+15*(params.yf-5)**2)/5
    rf2 = np.sqrt(1+15*(params.y0-5)**2)/5
    for i in range(500):
        plt.hlines(y= params.yf+rf1*np.sin(angle[i]), xmin=-3-rf1*np.cos(angle[i]), xmax=-3+rf1*np.cos(angle[i]), color='r')
        plt.hlines(y= params.y0-rf2*np.sin(angle[i]), xmin=-3-rf2*np.cos(angle[i]), xmax=-3+rf2*np.cos(angle[i]), color='r')

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    plt.plot(-3+rf1*cos_angle,params.yf+rf1*sin_angle, 'r-')
    plt.plot(-3-rf1*cos_angle,params.yf+rf1*sin_angle, 'r-')
    plt.plot(-3+rf2*cos_angle,params.y0-rf2*sin_angle, 'r-')
    plt.plot(-3-rf2*cos_angle,params.y0-rf2*sin_angle, 'r-')

    # Plot x and y coordinates
    plt.plot(sol_X[:,1], sol_X[:,2], label = 'Solution', marker='o', markersize=4)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass
