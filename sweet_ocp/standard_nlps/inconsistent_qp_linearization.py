# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions, ACADOS_INFTY
import numpy as np
import casadi as cs
from matplotlib import pyplot as plt
# Simplest NLP with Maratos effect
#
# Problem is in one dimension
#
# min -x
#
# s.t.  x <= 1
#       x^2 >= 4
#
# Optimal solution is x* = -2, but started from x > 0, SQP solver converges to
# infeasible point x^ = 1.

def create_problem(opts: AcadosOcpOptions):

    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x = cs.SX.sym('x')

    # dynamics: identity
    model.x = x
    model.name = f'inconsistent_qp_linearization'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = -model.x[0]

    # constraints
    ocp.model.con_h_expr_e = x**2
    ocp.constraints.lh_e = np.array([4.0])
    ocp.constraints.uh_e = np.array([ACADOS_INFTY])

    # add bounds on x
    nx = 1
    ocp.constraints.idxbx_e = np.array(range(nx))
    ocp.constraints.lbx_e = -ACADOS_INFTY * np.ones((nx))
    ocp.constraints.ubx_e = 1 * np.ones((nx))

    ocp.solver_options = opts
    # discretization
    N = 0
    ocp.solver_options.N_horizon = N

    return ocp

def create_initial_guess():
    # initialize solver
    init_X = np.array([0.5])
    init_U = []
    return init_X, init_U

if __name__ == '__main__':
    pass
