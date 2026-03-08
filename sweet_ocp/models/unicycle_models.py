# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from dataclasses import dataclass
import casadi as cs

def export_free_time_unicycle_model() -> AcadosModel:

    model_name = 'free_time_unicycle_model'

    ## system dimensions
    nx = 3 + 1 # +1 for the time

    # Define the states
    x = cs.SX.sym('x', 1)
    y = cs.SX.sym('y', 1)
    theta = cs.SX.sym('theta', 1)
    T = cs.SX.sym('T')
    states = cs.vertcat(T, x, y, theta)

    # define controls
    v = cs.SX.sym('v', 1)
    omega = cs.SX.sym('omega', 1)
    controls = cs.vertcat(v, omega)


    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(0,
                     T*v*cs.cos(theta),
                     T*v*cs.sin(theta),
                     T*omega)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_fixed_time_unicycle_model() -> AcadosModel:

    model_name = 'fixed_time_unicycle_model'

    ## system dimensions
    nx = 3

    # Define the states
    x = cs.SX.sym('x', 1)
    y = cs.SX.sym('y', 1)
    theta = cs.SX.sym('theta', 1)
    states = cs.vertcat(x, y, theta)

    # define controls
    v = cs.SX.sym('v', 1)
    omega = cs.SX.sym('omega', 1)
    controls = cs.vertcat(v, omega)


    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(v*cs.cos(theta),
                        v*cs.sin(theta),
                        omega)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model