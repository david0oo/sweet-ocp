# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions
import numpy as np
import casadi as cs
from matplotlib import pyplot as plt
# A simple fourth order polynomial
#
# min -0.5*x^2 + x^4
#
# s.t. -10 <= x <= 10

def create_problem(opts: AcadosOcpOptions):

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x = cs.SX.sym('x')

    # dynamics: identity
    # model.disc_dyn_expr = x
    model.x = x
    # model.p = []
    model.name = f'fourth_order_polynomial'
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = 0.5*x**2 - x**4

    # constarints
    ocp.constraints.lbx_e = np.array([-10])
    ocp.constraints.ubx_e = np.array([10])
    ocp.constraints.idxbx_e = np.array([0])

    ocp.solver_options = opts
    # discretization
    N = 0
    ocp.solver_options.N_horizon = N

    return ocp

def create_initial_guess():
    init_X = np.array([0.3])
    init_U = []
    return init_X, init_U

if __name__ == '__main__':
    pass
