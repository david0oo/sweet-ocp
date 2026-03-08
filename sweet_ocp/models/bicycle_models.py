# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, tan

def export_free_time_simple_bicycle() -> AcadosModel:

    model_name = 'free_time_simple_bicycle'

    # constants
    l = 1.0

    # set up states & controls
    T = SX.sym('T')
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')

    states = vertcat(T, x, y, theta)

    delta = SX.sym('delta')
    v = SX.sym('v')

    controls = vertcat(delta, v)

    # xdot
    T_dot = SX.sym('T_dot')
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')

    states_dot = vertcat(T_dot, x_dot, y_dot, theta_dot)

    # dynamics
    f_expl = vertcat(0,
                     T*v*cos(theta),
                     T*v*sin(theta),
                     T*v/l*tan(delta)
                    )

    f_impl = states_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_free_time_bicycle() -> AcadosModel:

    model_name = 'free_time_bicycle'

    # constants
    l = 1.0

    # set up states
    T = SX.sym('T')
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    v = SX.sym('v')

    states = vertcat(T, x, y, theta, v)

    # set up controls
    delta = SX.sym('delta')
    a = SX.sym('a')

    controls = vertcat(delta, a)

    # xdot
    T_dot = SX.sym('T_dot')
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')
    v_dot = SX.sym('v_dot')

    states_dot = vertcat(T_dot, x_dot, y_dot, theta_dot, v_dot)

    # dynamics
    f_expl = vertcat(0,
                     T*v*cos(theta),
                     T*v*sin(theta),
                     T*v/l*tan(delta),
                     T*a)

    f_impl = states_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_space_domain_bicycle() -> AcadosModel:

    model_name = 'free_time_space_domain_bicycle'

    # constants
    l = 1.0

    # set up states
    L = SX.sym('L') #space domain
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    v = SX.sym('v')
    rx = SX.sym('rx')

    states = vertcat(L, x, y, theta, v, rx)

    # set up controls
    delta = SX.sym('delta')
    a = SX.sym('a')

    controls = vertcat(delta, a)

    # parameter
    rp = SX.sym('rp')

    # xdot
    L_dot = SX.sym('L_dot')
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')
    v_dot = SX.sym('v_dot')
    rx_dot = SX.sym('rx_dot')

    states_dot = vertcat(L_dot, x_dot, y_dot, theta_dot, v_dot, rx_dot)

    # dynamics
    f_expl = vertcat(0.0, # L
                     L*cos(theta)/sin(theta), # x
                     L*1, # y
                     L*tan(delta)/(l*sin(theta)), # theta
                     L*a/(v*sin(theta)), # v
                     L*cos(theta)/sin(theta)) # rx

    f_impl = states_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name
    model.p = rp

    return model