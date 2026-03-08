# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
from dataclasses import dataclass

@dataclass
class RobotModelParameters:
    L : float = 5.0
    N : int = 200
    M : int = 4 # Needed for integrator
    T0 : float = 10.0 # initial guess for the time
    nx : int = 7
    nu : int = 3
    Tf : float = 1.0

def free_time_robot_model() -> AcadosModel:

    model_name = 'free_time_robot_model'

    # parameters
    params = RobotModelParameters()

    # set up states & controls
    T = ca.SX.sym('T')
    rho = ca.SX.sym('rho')
    theta = ca.SX.sym('theta')
    phi = ca.SX.sym('phi')
    rho_dot = ca.SX.sym('rho_dot')
    theta_dot = ca.SX.sym('theta_dot')
    phi_dot = ca.SX.sym('phi_dot')

    states = ca.vertcat(T, rho, theta, phi, rho_dot, theta_dot, phi_dot)

    u_rho = ca.SX.sym('u_rho')
    u_theta = ca.SX.sym('u_theta')
    u_phi = ca.SX.sym('u_phi')

    controls = ca.vertcat(u_rho, u_theta, u_phi)

    # xdot
    T_dot = ca.SX.sym('T_dot')
    rho_ddot = ca.SX.sym('rho_ddot')
    theta_ddot = ca.SX.sym('theta_ddot')
    phi_ddot = ca.SX.sym('phi_ddot')

    I_theta = ((params.L - rho)**3 + rho**3)/3 * ca.sin(phi)**2
    I_phi = ((params.L - rho)**3 + rho**3)/3

    states_dot = ca.vertcat(T_dot, rho_dot, theta_dot, phi_dot, rho_ddot, theta_ddot, phi_ddot)

    # dynamics
    f_expl = ca.vertcat(0,
                        T*ca.vertcat(
                            rho_dot,
                            theta_dot,
                            phi_dot,
                            (1/params.L)*u_rho,
                            (1/I_theta)*u_theta,
                            (1/I_phi)*u_phi)
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

    params = RobotModelParameters()
    # The flag denotes, if the problem should be transformed into a feasibility
    # problem, or if the unconstrained OCP should be solved.

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = free_time_robot_model()
    ocp.model = model

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]
    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, 4.5, 0.0, np.pi/4, 0.0, 0.0, 0.0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, 4.5, 0.0, np.pi/4, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0,1,2,3,4,5,6])
    ocp.constraints.idxbxe_0 = np.array([1,2,3,4,5,6])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, 4.5, (2*np.pi)/3, np.pi/4, 0.0, 0.0, 0.0])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, 4.5, (2*np.pi)/3, np.pi/4, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_e = np.array([0,1,2,3,4,5,6])

    # Path constraints on states
    ocp.constraints.lbx = np.array([0.0, 0.0, -np.pi, 0.0])
    ocp.constraints.ubx = np.array([ACADOS_INFTY, params.L, np.pi, np.pi])
    ocp.constraints.idxbx = np.array([0,1,2,3])

    # Path constraints on control
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = RobotModelParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))

    # Initial guess
    T0 = 1.0

    ts = np.linspace(0, T0, params.N+1)
    theta_init = (2*np.pi)/3 * (ts/T0)**2

    for i in range(params.N):
        init_X[i,:] = np.array([T0, 4.5, theta_init[i], np.pi/4, 0.0, 0.0, 0.0])
    init_X[params.N,:] = np.array([T0, 4.5, theta_init[params.N], np.pi/4, 0.0, 0.0, 0.0])
    return init_X, init_U

def plot_trajectory(sol_X: np.array, sol_U: np.array):
    """
    This function plots the height and the control.
    """
    params = RobotModelParameters()
    T_sol = sol_X[0,0]
    T_grid = np.linspace(0, T_sol, params.N+1)
    rho_sol = sol_X[:,1].squeeze()
    theta_sol = sol_X[:,2].squeeze()
    phi_sol = sol_X[:,3].squeeze()
    u_rho_sol = sol_U[:,0].squeeze()
    u_theta_sol = sol_U[:,1].squeeze()
    u_phi_sol = sol_U[:,2].squeeze()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.title('$\\rho$')
    plt.plot(T_grid, rho_sol)
    plt.ylim((3.4, 4.8))
    plt.xlim((0.0, 10.0))
    plt.xlabel("time")

    plt.subplot(2, 3, 2)
    plt.title('$\\theta$')
    plt.plot(T_grid, theta_sol)
    plt.ylim((0.0, 2.5))
    plt.xlim((0.0, 10.0))
    plt.xlabel("time")

    plt.subplot(2, 3, 3)
    plt.title('$\\phi$')
    plt.plot(T_grid, phi_sol)
    plt.xlim((0.0, 10.0))
    plt.ylim((0.5, 0.85))
    plt.xlabel("time")

    plt.subplot(2, 3, 4)
    plt.title('$u_{\\rho}$')
    plt.plot(T_grid[:-1], u_rho_sol)
    plt.ylim((-1.1, 1.1))
    plt.xlim((0.0, 10.0))
    plt.xlabel("time")

    plt.subplot(2, 3, 5)
    plt.title('$u_{\\theta}$')
    plt.plot(T_grid[:-1], u_theta_sol)
    plt.xlim((0.0, 10.0))
    plt.ylim((-1.1, 1.1))
    plt.xlabel("time")

    plt.subplot(2, 3, 6)
    plt.title('$u_{\\phi}$')
    plt.plot(T_grid[:-1], u_phi_sol)
    plt.xlim((0.0, 10.0))
    plt.ylim((-1.1, 1.1))
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
