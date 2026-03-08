# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions
import numpy as np
from casadi import *
import casadi as cs
# Simplest NLP with Marathos effect
#
# min log(exp(x) + exp(-x))
#

def create_problem(opts: AcadosOcpOptions):

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x = SX.sym('x')

    # dynamics: identity
    # model.disc_dyn_expr = x
    model.x = x
    # model.u = SX.sym('u', 0, 0)
    # model.p = []
    model.name = f'convex_globalization_problem'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = cs.log(cs.exp(x) + cs.exp(-x))

    # set options
    ocp.solver_options = opts
    
    # discretization
    N = 0
    ocp.solver_options.N_horizon = N

    return ocp

def create_initial_guess():
    # initialize solver
    init_X = np.array([1.5])
    init_U = []
    return init_X, init_U

if __name__ == '__main__':
    pass
