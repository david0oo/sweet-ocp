# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import sweet_ocp.ocps.time_optimal_swing_up as tocp_cart_pendulum
from opts import create_acados_options
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

def set_initial_guess(N, init_X, init_U, solver):
    for i in range(N):
        solver.set(i, "x", init_X[i, :])
        solver.set(i, "u", init_U[i, :])
    solver.set(N, "x", init_X[N, :])

def extract_solution(N, nx, nu, solver):
    # Extract Solution
    acados_sol_X = np.zeros((N+1, nx))
    acados_sol_U = np.zeros((N, nu))
    # get solution
    for i in range(N):
        acados_sol_X[i,:] = solver.get(i, "x")
        acados_sol_U[i,:] = solver.get(i, "u")
    acados_sol_X[N,:] = solver.get(N, "x")

    return acados_sol_X, acados_sol_U

if __name__ == "__main__":
    solve_ocp(use_acados_sol=False, casadi_solver_str='ipopt')