# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY, latexify_plot, AcadosOcpOptions
from sweet_ocp.models.cart_pendulum_models import export_free_time_pendulum_ode_model
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from sweet_ocp.visualization.cart_pendulum_visualization import CartPendulumVisualization, VisualizationFunctor

@dataclass
class PendulumParmeters:
    N : int = 100
    nx : int = 5
    nu : int = 1
    Tf : float = 1.0

    # Parameters
    max_f : float = 5.
    max_x1 : float = 1.0
    max_v : float = 2.0

    x1_0 : float = 0.0
    theta_0 : float = np.pi
    dx1_0 : float = 0.0
    dtheta_0 : float = 0.0

    theta_f : float = 0.0
    dx1_f : float = 0.0
    dtheta_f : float = 0.0

def create_problem(opts : AcadosOcpOptions):
    params = PendulumParmeters()
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_free_time_pendulum_ode_model()
    ocp.model = model

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, params.x1_0, params.theta_0, params.dx1_0, params.dtheta_0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, params.x1_0, params.theta_0, params.dx1_0, params.dtheta_0])
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])
    ocp.constraints.idxbxe_0 = np.array([1, 2, 3, 4])

    # Actuator constraints
    ocp.constraints.lbu = np.array([-params.max_f])
    ocp.constraints.ubu = np.array([+params.max_f])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([0.0, -params.max_x1, -params.max_v])
    ocp.constraints.ubx = np.array([ACADOS_INFTY, params.max_x1, params.max_v])
    ocp.constraints.idxbx = np.array([0, 1, 3])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, params.theta_f, params.dx1_f, params.dtheta_f])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, params.theta_f, params.dx1_f, params.dtheta_f])
    ocp.constraints.idxbx_e = np.array([0, 2, 3, 4])

    ###########################################################################
    # Define objective function
    ###########################################################################
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    return ocp

def create_initial_guess():
    params = PendulumParmeters()
    init_X = np.zeros((params.N+1, params.nx))
    init_U = np.zeros((params.N, params.nu))
    T0 = 1.0
    for i in range(params.N):
        init_X[i,:] = np.array([T0, 0.0, np.pi, 0.0, 0.0])
        init_U[i,:] = np.array([0.0])
    init_X[params.N,:] = np.array([T0, 0.0, np.pi, 0.0, 0.0])
    return init_X, init_U

def plot_trajectory(sol_X : np.array, sol_U: np.array):
    """"
    Inspired by 
    
    """"
    params = PendulumParmeters()
    T_final = sol_X[0, 0]
    fps = int(params.N/T_final)
    res_x = sol_X[:, 1].squeeze()
    res_theta = sol_X[:, 2].squeeze()
    vis = CartPendulumVisualization()
    ani = FuncAnimation(
        vis.fig, VisualizationFunctor(vis, res_x, res_theta),
        frames=range(params.N),
        interval=34)
    ani.save('cart_pendulum_swing_up.gif', writer=PillowWriter(fps=20))
    plt.show(block=True)

if __name__ == '__main__':
    pass
