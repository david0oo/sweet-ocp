# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import sweet_ocp.ocps.time_optimal_swing_up as tocp_cart_pendulum
from sweet_ocp.ocps.ocp_utils import set_initial_guess
from opts import create_acados_options, extract_solution
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosCasadiOcpSolver
import numpy as np

def solve_ocp(use_acados_sol, casadi_solver_str: str = 'ipopt'):
    opts = create_acados_options()
    ocp : AcadosOcp = tocp_cart_pendulum.create_problem(opts=opts)
    acados_solver = AcadosOcpSolver(ocp, verbose = False, generate = False)
    casadi_solver = AcadosCasadiOcpSolver(ocp, solver=casadi_solver_str)
    init_X, init_U = tocp_cart_pendulum.create_initial_guess()

    # Initialize at initial guess
    N = ocp.solver_options.N_horizon
    nx = ocp.model.x.shape[0]
    nu = ocp.model.u.shape[0]
    
    set_initial_guess(N, init_X, init_U, acados_solver)
    set_initial_guess(N, init_X, init_U, casadi_solver)

    # Solve the problem with acados
    acados_status = acados_solver.solve()
    casadi_status = casadi_solver.solve()

    if use_acados_sol:
        sol_X, sol_U = extract_solution(N, nx, nu, acados_solver)
    else:
        sol_X, sol_U = extract_solution(N, nx, nu, casadi_solver)

    # Plot the solution
    tocp_cart_pendulum.plot_trajectory(sol_X, sol_U)

if __name__ == "__main__":
    solve_ocp(use_acados_sol=False, casadi_solver_str='ipopt')