# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from dataclasses import dataclass
import casadi as cs

def export_fixed_time_simple_pendulum_model() -> AcadosModel:

    model_name = 'fixed_time_simple_pendulum_model'

    nx = 2
    nu = 1

    # Parameters
    mass = 1
    len = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81

    # Define the states
    theta = cs.SX.sym('theta', 1)
    theta_dot = cs.SX.sym('theta_dot', 1)
    states = cs.vertcat(theta, theta_dot)

    # define controls
    u = cs.SX.sym('u', nu)
    controls = cs.vertcat(u)

    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    m = mass*lc*lc
    f_expl = cs.vertcat(theta_dot,
                        u/m - g*cs.sin(theta)/lc -b*theta_dot/m)

    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_free_time_simple_pendulum_model() -> AcadosModel:

    model_name = 'free_time_simple_pendulum_model'

    ## system dimensions
    fixed_time_model = export_fixed_time_simple_pendulum_model()
    nx = fixed_time_model.x.shape[0] + 1

    # Define the states
    T = cs.SX.sym('T', 1)
    states = cs.vertcat(T, fixed_time_model.x)

    # define controls
    controls = fixed_time_model.u


    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(0,
                        T*fixed_time_model.x)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model