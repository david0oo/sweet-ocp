# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import sweet_ocp.standard_nlps.fourth_order_polynomial as polynomial

from acados_template import AcadosCasadiOcpSolver, AcadosOcpSolver

try:
    from .opts import create_acados_options
except ImportError:  # pragma: no cover - allows running the example as a script
    from opts import create_acados_options


def solve_nlp():
    ocp = polynomial.create_problem(opts=create_acados_options())
    # solver = AcadosCasadiOcpSolver(ocp, solver="ipopt")
    solver = AcadosOcpSolver(ocp)
    initial_x, _ = polynomial.create_initial_guess()
    solver.set(0, "x", initial_x)

    solver.solve()
    print(solver.get(0, "x"))

if __name__ == "__main__":
    solve_nlp()
