# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY, AcadosOcpOptions
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
from dataclasses import dataclass

@dataclass
class FlowModelParameters:
    L : float = 5.0
    N : int = 3#20
    # N : int = 200
    M : int = 4 # Needed for integrator
    T0 : float = 10.0 # initial guess for the time
    nx : int = 4
    nu : int = 0

def free_time_flow_in_channel_model() -> AcadosModel:

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
    model = free_time_flow_in_channel_model()
    ocp.model = model

    Tf = 1.0

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
    ocp.solver_options.tf = Tf

    ocp.parameter_values = np.array([10])

    return ocp

def create_initial_guess(reynolds_numer : float = 0.0):
    """
    This function creates an initial guess for the flow problem.
    Choose the reynolds_number from [0, 10, 100, 1000]
    """
    params = FlowModelParameters()
    init_X = np.zeros((params.N+1, params.nx))
    init_P = np.zeros((params.N+1, 1))

    # Initial guess
    T0 = 1.0

    ts = np.linspace(0, T0, params.N+1)
    u_init = ts**2*(3-2*ts)
    for i in range(params.N):
        init_X[i,:] = np.array([u_init[i], 0.0, 0.0, 0.0])
        init_P[i] = np.array([reynolds_numer])
    init_X[params.N,:] = np.array([u_init[i], 0.0, 0.0, 0.0])
    init_P[params.N] = np.array([reynolds_numer])
    return init_X, init_P

def plot_trajectory(du_sols, labels):
    """
    This function plots the height and the control.
    """
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
    params = FlowModelParameters()

    opts = AcadosOcpOptions()
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # opts.qp_solver = 'FULL_CONDENSING_HPIPM'
    opts.qp_solver_mu0 = 1e3 # bad option!!!
    opts.hessian_approx = 'EXACT'
    opts.regularize_method = 'MIRROR'
    # opts.hessian_approx = 'GAUSS_NEWTON'
    opts.integrator_type = 'ERK'
    opts.print_level = 4
    opts.nlp_solver_max_iter = 10
    opts.qp_solver_iter_max = 1000

    # Search direction
    opts.nlp_solver_type = 'SQP_WITH_FEASIBLE_QP'#'SQP'#
    # opts.search_direction_mode = 'BYRD_OMOJOKUN'
    # opts.allow_direction_mode_switch_to_nominal = False

    # Globalization
    opts.globalization = 'FUNNEL_L1PEN_LINESEARCH'#'MERIT_BACKTRACKING'#
    opts.globalization_full_step_dual = True
    opts.globalization_funnel_use_merit_fun_only = False
    # opts.globalization_funnel_fraction_switching_condition = 1e-3
    # opts.globalization_funnel_init_increase_factor = 1.25
    # opts.globalization_funnel_init_upper_bound = 100.0
    # opts.globalization_eps_sufficient_descent = 1e-4
    # opts.globalization_funnel_kappa = 0.5
    # opts.globalization_funnel_sufficient_decrease_factor = 0.9

    # Scaling
    opts.qpscaling_type = 'OBJECTIVE_GERSHGORIN'
    opts.qpscaling_scale_qp_objective = False
    opts.qpscaling_scale_qp_dynamics = False
    opts.qpscaling_scale_qp_constraints = False
    # opts.qpscaling_lb_norm_inf_grad_obj = 1.
    # opts.qpscaling_ub_norm_inf_grad_obj = 100.0
    # opts.qpscaling_ub_max_abs_eig = 1e5

    ocp = create_problem(opts)
    solver = AcadosOcpSolver(ocp, verbose = True)

    sol_X = np.zeros((params.N+1, params.nx))

    # Initial guess
    T0 = 1.0

    ts = np.linspace(0, T0, params.N+1)
    u_init = ts**2*(3-2*ts)

    # R_range = [0, 10, 100, 1000]
    R_range = [10]#, 10, 100, 1000]
    du_sols = []
    for R in R_range:
        for i in range(params.N):
            solver.set(i, "x", np.array([u_init[i], 0.0, 0.0, 0.0]))
            solver.set(i, "p", np.array([R]))
        solver.set(params.N, "x", np.array([u_init[params.N], 0.0, 0.0, 0.0]))
        solver.set(params.N, "p", np.array([R]))

        # Solve the problem
        status = solver.solve()

        for i in range(params.N):
            sol_X[i,:] = solver.get(i, "x")
        sol_X[params.N,:] = solver.get(params.N, "x")

        du_sols.append(np.copy(sol_X[:,1].squeeze()))

    plot_trajectory(du_sols, ["R = 0", "R = 10", "R = 100", "R = 1000"])
    print("R = 10000 and bigger does not work yet!!")
