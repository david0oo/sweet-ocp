# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosOcp, AcadosOcpOptions, AcadosModel, ACADOS_INFTY
import numpy as np
import casadi as cs
from typing import Union
import re

def create_casadi_expressions_from_nl_file(filepath: str):
    # Create an NLP instance
    nl = cs.NlpBuilder()

    # Parse an NL-file
    path = filepath + ".nl"
    try:
        nl.import_nl(path, {"verbose": True})
    except:
        print("Nl file could not be loaded")
        return -1, -1, -1, -1, -1, -1, -1, -1

    x = cs.vertcat(*nl.x)
    print(x.shape)
    g = cs.vertcat(*nl.g)
    fun_f_g = cs.Function('fun_f_g',[x], [nl.f, g])
    x_sx = cs.SX.sym('x_sx', x.sparsity())
    f_sx, g_sx = fun_f_g(x_sx)

    return x_sx, f_sx, g_sx, nl.g_lb, nl.g_ub, nl.x_lb, nl.x_ub, nl.x_init

def create_standard_nlp_from_casadi_expression(filename: str,
                                               x: cs.MX.sym,
                                               f: cs.MX.sym,
                                               g: cs.MX.sym,
                                               lbg: Union[np.array, cs.DM],
                                               ubg: Union[np.array, cs.DM],
                                               lbx: Union[np.array, cs.DM],
                                               ubx: Union[np.array, cs.DM],
                                               opts: AcadosOcpOptions,
                                               N : int = 0) -> AcadosOcp:

    if N not in [0, 1]:
        raise ValueError("N must be 0 or 1")

    # replace inf with ACADOS_INFTY
    lbx = np.array(lbx).squeeze()
    lbx[lbx == -np.inf] = -ACADOS_INFTY
    ubx = np.array(ubx).squeeze()
    ubx[ubx == np.inf] = ACADOS_INFTY

    lbg = np.array(lbg).squeeze()
    lbg[lbg == -np.inf] = -ACADOS_INFTY
    ubg = np.array(ubg).squeeze()
    ubg[ubg == np.inf] = ACADOS_INFTY

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # dynamics: identity
    model = AcadosModel()
    model.x = x

    new_filename = re.sub(r'-', '_', filename)
    model.name = new_filename
    ocp.model = model

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = f

    if N == 0:
        # constraints
        ocp.model.con_h_expr_e = g
        ocp.constraints.lh_e = np.array(lbg).squeeze()
        ocp.constraints.uh_e = np.array(ubg).squeeze()

        ocp.constraints.lbx_e = lbx
        ocp.constraints.ubx_e = ubx
        ocp.constraints.idxbx_e = np.arange(x.shape[0])
    else:
        ocp.model.u = cs.SX.sym('u', 0, 0) # [] / None doesnt work
        ocp.model.disc_dyn_expr = x

        # _0 is used to get Lagrange Multipliers correct
        # constraints
        ocp.model.con_h_expr_0 = g
        ocp.constraints.lh_0 = np.array(lbg).squeeze()
        ocp.constraints.uh_0 = np.array(ubg).squeeze()

        ocp.constraints.lbx_0 = lbx
        ocp.constraints.ubx_0 = ubx
        ocp.constraints.idxbx_0 = np.arange(x.shape[0])

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options = opts
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.N_horizon = N
    if N == 1:
        ocp.solver_options.tf = 1.0
    return ocp