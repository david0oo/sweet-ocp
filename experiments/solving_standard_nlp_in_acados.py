# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

# import sweet_ocp.standard_nlps.inconsistent_qp_linearization as polynomial
# import sweet_ocp.standard_nlps.globalized_convex_problem as polynomial
# import sweet_ocp.standard_nlps.fourth_order_polynomial as polynomial
import sweet_ocp.standard_nlps.maratos_test_problem as polynomial
from opts import create_acados_options
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosCasadiOcpSolver

def solve_nlp():
    opts = create_acados_options()
    ocp : AcadosOcp = polynomial.create_problem(opts=opts)
    acados_solver = AcadosOcpSolver(ocp, verbose = False)
    init_X, _ = polynomial.create_initial_guess()

    # Initialize at initial guess
    N = ocp.solver_options.N_horizon
    acados_solver.set(0, "x", init_X)

    # Solve the problem with acados
    acados_status = acados_solver.solve()

    # Extract Solution
    acados_sol = acados_solver.get(0, "x")
    print(acados_sol)

if __name__ == "__main__":
    solve_nlp()