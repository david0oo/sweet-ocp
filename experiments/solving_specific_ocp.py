# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import acados_test_problems.ocps.time_optimal_overhead_crane_problem as tocp_crane
from opts import create_acados_options
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosCasadiOcpSolver
import numpy as np

def solve_ocp():
    opts = create_acados_options()
    ocp : AcadosOcp = tocp_crane.create_problem(opts=opts)
    acados_solver = AcadosOcpSolver(ocp, verbose = False)
    # casadi_solver = AcadosCasadiOcpSolver(ocp, solver='ipopt', casadi_solver_opts=opts)
    init_X, init_U = tocp_crane.create_initial_guess()

    # Initialize at initial guess
    N = ocp.solver_options.N_horizon
    nx = ocp.model.x.shape[0]
    nu = ocp.model.u.shape[0]
    for i in range(N):
        acados_solver.set(i, "x", init_X[i, :])
        # casadi_solver.set(i, "x", init_X[i, :])
        acados_solver.set(i, "u", init_U[i, :])
        # casadi_solver.set(i, "u", init_U[i, :])
    acados_solver.set(N, "x", init_X[N, :])
    # casadi_solver.set(N, "x", init_X[N, :])

    # Solve the problem with acados
    acados_status = acados_solver.solve()
    # Solve the problem with acados
    # ipopt_status = casadi_solver.solve()

    # Extract Solution
    acados_sol_X = np.zeros((N+1, nx))
    acados_sol_U = np.zeros((N, nu))
    # get solution
    for i in range(N):
        acados_sol_X[i,:] = acados_solver.get(i, "x")
        acados_sol_U[i,:] = acados_solver.get(i, "u")
    acados_sol_X[N,:] = acados_solver.get(N, "x")

    # Plot the solution
    tocp_crane.plot_trajectory(acados_sol_X, acados_sol_U)

if __name__ == "__main__":
    solve_ocp()