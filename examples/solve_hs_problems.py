# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcpSolver
import numpy as np
import os
import importlib
from opts import create_acados_options
from sweet_ocp.utils.standard_nlp_util import create_standard_nlp_from_casadi_expression

# file names
def create_file_names():
    file_names = []
    for i in range(12, 120):
        if i < 10:
            file_name = "hs00" + str(i)
        elif i < 100:
            file_name = "hs0" + str(i)
        else:
            file_name = "hs" + str(i)

        if file_name == 'hs087':
            print(file_name +
                '.....nonsmooth objective function...not in our scope')
            continue
        if file_name == 'hs082' or file_name == 'hs094' or file_name == 'hs115'\
                or file_name == 'hs087':
            print(file_name + '.....not existant...')
            continue

        file_names.append(file_name)
    return file_names

def solve_problem(file_name, opts):

    # print("Solving ", file_name)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.system(f"rm -f {dir_path}/acados_ocp.json")
    os.system(f"rm -rf {dir_path}/c_generated_code")

    prob = importlib.import_module(f'sweet_ocp.standard_nlps.hock_schittkowski.{file_name}', package='sweet_ocp')
    hock_schittkowsky_func = getattr(prob, file_name)
    (x_opt, f_opt, x, f, g, lbg, ubg, lbx, ubx, x0) = hock_schittkowsky_func()

    N = 0
    # create ocp object to formulate the OCP
    ocp = create_standard_nlp_from_casadi_expression(file_name, x, f, g, lbg, ubg, lbx, ubx, opts, N)
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    # initialize solver
    xinit = np.array(x0).squeeze()
    for i in range(N+1):
        ocp_solver.set(i, "x", xinit)

    # solve
    status = ocp_solver.solve()

    # get solution
    solution = ocp_solver.get(0, "x")
    lam_sol = ocp_solver.get(0, 'lam')
    print("Found solution: ", solution)
    print("Found lam_solution: ", lam_sol)

    cost_value = ocp_solver.get_cost()
    # print("Found cost: ", cost_value)

    # compare to analytical solution
    sol_err = np.linalg.norm(solution - x_opt, ord=np.inf)
    print("Sol Error: ", sol_err)
    f_error = abs(f_opt - cost_value)
    print("Obj error: ", f_error)
    return status, ocp_solver.get_stats('nlp_iter'), ocp_solver.get_stats('time_tot'), sol_err, f_error

def solve_all_problems():
    file_names = create_file_names()
    print("Number of test problems: ", len(file_names))
    # file_names = ['hs001', 'hs002']
    # file_names = ['hs016']
    file_names = file_names[87:]
    solved_problems = 0
    unsolved_problems = []
    status = False
    opts = create_acados_options()
    n_iters = []
    times = []
    statuses = []
    solved_problem = []
    sol_errors = []
    f_errors = []
    failures = []
    for name in file_names:
        print('Solving '+ name + '.....')
        status, n_iter, time_tot, sol_error, f_error = solve_problem(name, opts)
        statuses.append(status)
        n_iters.append(n_iter)
        times.append(time_tot)
        solved_problem.append(name)
        sol_errors.append(sol_error)
        f_errors.append(f_error)
        if status != 0:
            failures.append(name)

    print(f"{"Problem Name":<13} {"Status":>8}\t{"n_iter":>8}\t{"Wall time":>8}\t{"Sol_Error":>8}\t{"f_Error":>8}")
    print("-" * 75)
    for i in range(len(statuses)):
        print(f"{solved_problem[i]:<13} {statuses[i]:>8}\t{n_iters[i]:>8}\t{times[i]:>8.2e}\t{sol_errors[i]:>8.2e}\t{f_errors[i]:>8.2e}")

    print("Failed instances:", failures)
    print("Total number of solved problems: ", len(solved_problem))

if __name__ == '__main__':
    solve_all_problems()

    # opts = create_acados_options()
    # solve_problem('hs109', opts)
