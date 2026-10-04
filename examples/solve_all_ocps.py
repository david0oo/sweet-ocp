# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import os
import importlib
import tempfile
from opts import create_acados_options
from sweet_ocp.ocps.ocp_utils import get_ocp_names, set_initial_guess, extract_solution

def solve_problem(file_name, opts):
    prob = importlib.import_module(f'sweet_ocp.ocps.{file_name}', package='sweet_ocp')
    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="sweet_ocp_") as tmp_dir:
        os.chdir(tmp_dir)
        try:
            ocp: AcadosOcp = prob.create_problem(opts=opts)
            init_X, init_U = prob.create_initial_guess()
            ocp_solver = AcadosOcpSolver(ocp, verbose=False, generate=False)

            horizon = ocp.solver_options.N_horizon
            set_initial_guess(horizon, init_X, init_U, ocp_solver)
            status = ocp_solver.solve()
            extract_solution(
                horizon,
                ocp.model.x.shape[0],
                ocp.model.u.shape[0],
                ocp_solver,
            )

            return (
                status,
                ocp_solver.get_stats('nlp_iter'),
                ocp_solver.get_stats('time_tot'),
                ocp_solver.get_cost(),
            )
        finally:
            os.chdir(previous_cwd)

def solve_all_problems():
    ocp_names = get_ocp_names()
    print("Number of test problems: ", len(ocp_names))

    status = False

    n_iters = []
    times = []
    statuses = []
    solved_problem = []
    f_opts = []
    failures = []
    for name in ocp_names:
        print('Solving '+ name + '.....')
        opts = create_acados_options()
        status, n_iter, time_tot, f_opt = solve_problem(name, opts)
        statuses.append(status)
        n_iters.append(n_iter)
        times.append(time_tot)
        solved_problem.append(name)
        f_opts.append(f_opt)
        if status != 0:
            failures.append(name)

    name_width = max(
        len("Problem Name"),
        max((len(name) for name in solved_problem), default=0),
    )
    print(
        f"{'Problem Name':<{name_width}} {'Status':>8} {'n_iter':>8} "
        f"{'Wall time':>10} {'f_opt':>12}"
    )
    print("-" * (name_width + 42))
    for i in range(len(statuses)):
        print(
            f"{solved_problem[i]:<{name_width}} {statuses[i]:>8} {n_iters[i]:>8} "
            f"{times[i]:>10.2e} {f_opts[i]:>12.2e}"
        )

    print(f"Failed instances: {failures}")
    print(f"Total number of solved problems: {len(solved_problem)}")

if __name__ == '__main__':
    solve_all_problems()
