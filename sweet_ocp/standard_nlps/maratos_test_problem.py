# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions, AcadosModel
import numpy as np
from casadi import *
from matplotlib import pyplot as plt
# Simplest NLP with Maratos effect
#
# min x_1
#
# s.t. x_1^2 + x_2^2 = 1

def create_problem(opts : AcadosOcpOptions):

    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    # dynamics: identity
    # model.disc_dyn_expr = x
    model.x = x
    # model.u = SX.sym('u', 0, 0) # [] / None doesnt work
    # model.p = []
    model.name = 'maratos_problem'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = x1

    # constarints
    ocp.model.con_h_expr_e = x1 ** 2 + x2 ** 2
    ocp.constraints.lh_e = np.array([1.0])
    ocp.constraints.uh_e = np.array([1.0])

    ocp.solver_options = opts
    # discretization
    N = 0
    ocp.solver_options.N_horizon = N

    return ocp

def create_initial_guess():
    # initialize solver
    rad_init = 0.1 #0.1 #np.pi / 4
    init_X = np.array([np.cos(rad_init), np.sin(rad_init)])
    init_U = []

    return init_X, init_U

def get_analytical_solution():
    return np.array([-1,0])

if __name__ == '__main__':
    pass
