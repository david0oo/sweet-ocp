# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions, ACADOS_INFTY
import numpy as np
from casadi import *
# Simple NLP with with degenerate solution
#
# Problem is in one dimension
#
# min -x1
#
# s.t.  -x2 <= 0
#       x2 + x1^5 <= 0
#
# Optimal solution is x* = (0, 0), We start from (5,5)

def create_problem(opts : AcadosOcpOptions):

    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x = SX.sym('x', 2)

    # dynamics: identity
    # model.disc_dyn_expr = x
    model.x = x
    # model.u = SX.sym('u', 0, 0) # [] / None doesnt work
    # model.p = []
    model.name = f'simple_degenerate_problem'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = -model.x[0]

    # constraints
    ocp.model.con_h_expr_e = x[1] + x[0]**5
    ocp.constraints.lh_e = np.array([-ACADOS_INFTY])
    ocp.constraints.uh_e = np.array([0.0])

    # add bounds on x
    ocp.constraints.idxbx_e = np.array([0])
    ocp.constraints.lbx_e = -ACADOS_INFTY * np.ones((1))
    ocp.constraints.ubx_e = 0 * np.ones((1))

    ocp.solver_options = opts
    # discretization
    ocp.solver_options.N_horizon = 0

    return ocp

def create_initial_guess():
    # initialize solver
    init_X = np.array([5, 5])
    init_U = []
    return init_X, init_U

def get_analytical_solution():
    return np.array([0, 0])

if __name__ == '__main__':
    pass
