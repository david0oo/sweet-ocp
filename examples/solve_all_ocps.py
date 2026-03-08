# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import os
import importlib
from opts import create_acados_options
from sweet_ocp.ocps.ocp_utils import get_ocp_names, set_initial_guess, extract_solution

def solve_problem(file_name, opts):

    # print("Solving ", file_name)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.system(f"rm -f {dir_path}/acados_ocp.json")
    os.system(f"rm -rf {dir_path}/c_generated_code")

    prob = importlib.import_module(f'sweet_ocp.ocps.{file_name}', package='sweet_ocp')
    ocp : AcadosOcp = prob.create_problem(opts=opts)
    init_X, init_U = prob.create_initial_guess()
    ocp_solver = AcadosOcpSolver(ocp, verbose=False, generate=False)
    
    # Initialize at initial guess
    N = ocp.solver_options.N_horizon
    nx = ocp.model.x.shape[0]
    nu = ocp.model.u.shape[0]

    set_initial_guess(N, init_X, init_U, ocp_solver)

    # solve
    status = ocp_solver.solve()

    # get solution
    sol_X, sol_U = extract_solution(N, nx, nu, ocp_solver)

    cost_value = ocp_solver.get_cost()

    return status, ocp_solver.get_stats('nlp_iter'), ocp_solver.get_stats('time_tot'), cost_value

def solve_all_problems():
    ocp_names = get_ocp_names()
    print("Number of test problems: ", len(ocp_names))

    solved_problems = 0
    unsolved_problems = []
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

    print(f"{"Problem Name":<25} {"Status":>8}\t{"n_iter":>8}\t{"Wall time":>8}\t{"f_opt":>8}")
    print("-" * 80)
    for i in range(len(statuses)):
        print(f"{solved_problem[i]:<25} {statuses[i]:>8}\t{n_iters[i]:>8}\t{times[i]:>8.2e}\t{f_opts[i]:>8.2e}")

    print("Failed instances:", failures)
    print("Total number of solved problems: ", len(solved_problem))

if __name__ == '__main__':
    solve_all_problems()

    # opts = create_acados_options()
    # solve_problem('hs109', opts)
