# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions, ACADOS_INFTY
import numpy as np
from casadi import *
# The problem described here is taken from the paper
# R. H. Byrd, F. E. Curtis, and J. Nocedal, Infeasibility detection and SQP 
# methods for nonlinear optimization, SIAM J. Optim., 20 (2010), pp. 2281–2299.
#
# The problem is called "unique"
# Problem is in two dimension
#
# min x1+x2
#
# s.t.  x2 - x1^2 -1 >= 0
#       0.3(1-exp(x2)) >= 0
#
# The problem has a local minimizer of constraint violation at (0,1) and is 
# started at (3,2)

def create_problem(opts: AcadosOcpOptions):

    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x = SX.sym('x',2)

    # dynamics: identity
    model.x = x
    model.name = f'infeasible_unique'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]+model.x[1]

    # constraints
    ocp.model.con_h_expr_e = vertcat(x[1]-x[0]**2-1,
                                     0.3*(1-exp(x[1])))
    ocp.constraints.lh_e = np.array([0.0, 0.0])
    ocp.constraints.uh_e = np.array([ACADOS_INFTY, ACADOS_INFTY])

    # discretization
    ocp.solver_options = opts
    ocp.solver_options.N_horizon = 0

    return ocp

def create_initial_guess():
    # initialize solver
    init_X = np.array([3, 2]) # Initial point from paper
    init_U = []
    return init_X, init_U

if __name__ == '__main__':
    pass
