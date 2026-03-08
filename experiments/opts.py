# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcpOptions

def create_acados_options(globalization='funnel'):
    opts = AcadosOcpOptions()
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # opts.qp_solver = 'PARTIAL_CONDENSING_CLARABEL'
    # opts.qp_solver = 'FULL_CONDENSING_HPIPM'
    # opts.qp_solver = 'FULL_CONDENSING_DAQP'

    # with scaling we can also solve for higher accuracy
    # opts.hpipm_mode = 'ROBUST'
    # opts.qp_solver_tol_comp = 1e-7
    # opts.qp_solver_tol_stat = 1e-7
    # opts.qp_solver_tol_eq = 1e-7
    # opts.qp_solver_tol_ineq = 1e-7
    opts.qp_solver_mu0 = 1e3 # bad option!!!
    # opts.qp_solver_t0_init = 0
    opts.hessian_approx = 'EXACT'
    opts.regularize_method = 'MIRROR'
    opts.reg_epsilon = 1e-6
    # opts.byrd_omojokon_slack_relaxation_factor = 1.0+1e-2
    # opts.qp_solver_ric_alg = 0

    opts.nlp_solver_max_iter = 600
    opts.qp_solver_iter_max = 300
    opts.store_iterates = True

    # opts.hessian_approx = 'GAUSS_NEWTON'
    # opts.levenberg_marquardt = 1e-6
    # opts.integrator_type = 'ERK'
    opts.print_level = 1
    # opts.nlp_solver_max_iter = 15
    # Search direction
    # opts.nlp_solver_type = 'SQP'#
    opts.nlp_solver_type = 'SQP_WITH_FEASIBLE_QP'#'SQP'#
    # opts.search_direction_mode = 'BYRD_OMOJOKUN'
    # opts.allow_direction_mode_switch_to_nominal = False

    # Globalization
    if globalization == 'funnel':
        opts.globalization = 'FUNNEL_L1PEN_LINESEARCH'
    elif globalization == 'merit_backtracking':
        opts.globalization = 'MERIT_BACKTRACKING'
    elif globalization == 'fixed_step':
        opts.globalization = 'FIXED_STEP'
    opts.globalization_full_step_dual = True
    opts.globalization_funnel_use_merit_fun_only = False
    opts.globalization_alpha_reduction = 0.7
    opts.globalization_eps_sufficient_descent = 1e-4

    # opts.globalization_funnel_fraction_switching_condition = 1e-3
    # opts.globalization_funnel_init_increase_factor = 1.25
    # opts.globalization_funnel_init_upper_bound = 100.0
    # opts.globalization_eps_sufficient_descent = 1e-4
    # opts.globalization_funnel_kappa = 0.5
    # opts.globalization_funnel_sufficient_decrease_factor = 0.9

    # Scaling
    opts.qpscaling_scale_objective = 'OBJECTIVE_GERSHGORIN'
    # opts.qpscaling_scale_constraints = 'INF_NORM'
    opts.qpscaling_lb_norm_inf_grad_obj = 1e-4
    opts.qpscaling_ub_max_abs_eig = 1e5
    # opts.nlp_qp_tol_strategy = 'ADAPTIVE_QPSCALING'

    return opts
