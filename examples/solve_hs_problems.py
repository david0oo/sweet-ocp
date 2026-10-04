# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import importlib
import os
import tempfile

import numpy as np
from acados_template import AcadosOcpSolver
from sweet_ocp.utils.standard_nlp_util import create_standard_nlp_from_casadi_expression

import casadi as cs
try:
        cs.GlobalOptions.setNumpyMode(1)
except AttributeError:
        pass

if __package__:
    from .opts import create_acados_options
else:  # pragma: no cover - allows running the example as a script
    from opts import create_acados_options

# file names
def create_file_names():
    skipped = {"hs082", "hs087", "hs094", "hs115"}
    file_names = []

    for i in range(12, 120):
        file_name = f"hs{i:03d}"
        if file_name in skipped:
            print(f"{file_name}.....not available or excluded")
            continue
        file_names.append(file_name)

    return file_names

def solve_problem(file_name, opts):
    """Solve a single Hock-Schittkowski NLP without writing into the repo."""
    prob = importlib.import_module(
        f'sweet_ocp.standard_nlps.hock_schittkowski.{file_name}', package='sweet_ocp'
    )
    hock_schittkowsky_func = getattr(prob, file_name)
    (x_opt, f_opt, x, f, g, lbg, ubg, lbx, ubx, x0) = hock_schittkowsky_func()

    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="sweet_ocp_") as tmp_dir:
        os.chdir(tmp_dir)
        try:
            N = 0
            ocp = create_standard_nlp_from_casadi_expression(
                file_name, x, f, g, lbg, ubg, lbx, ubx, opts, N
            )
            ocp_solver = AcadosOcpSolver(ocp, verbose=False, generate=False)

            xinit = np.asarray(x0).squeeze()
            for i in range(N + 1):
                ocp_solver.set(i, "x", xinit)

            status = ocp_solver.solve()
            solution = ocp_solver.get(0, "x")
            lam_sol = ocp_solver.get(0, 'lam')
            print("Found solution: ", solution)
            print("Found lam_solution: ", lam_sol)

            cost_value = ocp_solver.get_cost()
            sol_err = float(np.linalg.norm(solution - x_opt, ord=np.inf))
            f_error = float(abs(f_opt - cost_value))
            print("Sol Error: ", sol_err)
            print("Obj error: ", f_error)
            return status, ocp_solver.get_stats('nlp_iter'), ocp_solver.get_stats('time_tot'), sol_err, f_error
        finally:
            os.chdir(previous_cwd)

def solve_all_problems():
    file_names = create_file_names()
    print(f"Number of test problems: {len(file_names)}")
    opts = create_acados_options()

    results = []
    for name in file_names:
        print(f"Solving {name}.....")
        result = solve_problem(name, opts)
        results.append((name, *result))

    print(f"{'Problem Name':<13} {'Status':>8} {'n_iter':>8} {'Wall time':>10} {'Sol_Error':>12} {'f_Error':>12}")
    print("-" * 75)
    for name, status, n_iter, time_tot, sol_error, f_error in results:
        print(
            f"{name:<13} {status:>8} {n_iter:>8} {time_tot:>10.2e} {sol_error:>12.2e} {f_error:>12.2e}"
        )

    failures = [name for name, status, *_ in results if status != 0]
    print("Failed instances:", failures)
    print("Total number of solved problems: ", len(results))

if __name__ == '__main__':
    solve_all_problems()
