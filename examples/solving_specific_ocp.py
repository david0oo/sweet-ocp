# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import tempfile

import sweet_ocp.ocps.altro_dubins_car_obstacle as tocp_cart_pendulum
from sweet_ocp.ocps.ocp_utils import set_initial_guess, extract_solution
from acados_template import AcadosOcpSolver, AcadosCasadiOcpSolver

try:
    from .opts import create_acados_options
except ImportError:  # pragma: no cover - allows running the example as a script
    from opts import create_acados_options


def solve_ocp(use_acados: bool = True, casadi_solver_str: str = "ipopt"):
    opts = create_acados_options()
    ocp = tocp_cart_pendulum.create_problem(opts=opts)
    init_x, init_u = tocp_cart_pendulum.create_initial_guess()

    horizon = ocp.solver_options.N_horizon
    state_dim = ocp.model.x.shape[0]
    control_dim = ocp.model.u.shape[0]

    with tempfile.TemporaryDirectory(prefix="sweet_ocp_") as codegen_dir:
        ocp.code_gen_options.code_export_directory = codegen_dir
        if use_acados:
            solver = AcadosOcpSolver(ocp, verbose=False, generate=False)
        else:
            solver_opts = {"uno": {"preset": "ipopt"}} if casadi_solver_str == "uno" else {}
            solver = AcadosCasadiOcpSolver(
                ocp,
                solver=casadi_solver_str,
                casadi_solver_opts=solver_opts,
            )

        set_initial_guess(horizon, init_x, init_u, solver)
        solver.solve()
        sol_x, sol_u = extract_solution(horizon, state_dim, control_dim, solver)
        del solver

    tocp_cart_pendulum.plot_trajectory(sol_x, sol_u)


if __name__ == "__main__":
    solve_ocp(use_acados=True)