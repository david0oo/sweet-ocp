# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from dataclasses import dataclass
import casadi as cs

def export_fixed_time_double_pendulum_model() -> AcadosModel:

    model_name = 'fixed_time_double_pendulum_model'

    ## system dimensions
    nx = 4

    g = 9.81 #m/s^2
    m1 = 1.0
    m2 = 1.0
    J1 = 1.0
    J2 = 1.0
    l1 = 1.0
    l2 = 1.0

    # Define the states
    theta1 = cs.SX.sym('theta1', 1)
    theta1_dot = cs.SX.sym('theta1_dot', 1)
    theta2 = cs.SX.sym('theta2', 1)
    theta2_dot = cs.SX.sym('theta2_dot', 1)
    states = cs.vertcat(theta1, theta2, theta1_dot, theta2_dot)

    s1 = cs.sin(theta1)
    c1 = cs.cos(theta1)
    s2 = cs.sin(theta2)
    c2 = cs.cos(theta2)
    c12 = cs.cos(theta1+theta2)

    # mass matrix
    m11 = m1*l1**2 + J1 + m2*(l1**2 + l2**2 + 2*l1*l2*c2) + J2
    m12 = m2*(l2**2 + l1*l2*c2 + J2)
    m22 = l2**2*m2 + J2
    M = cs.vertcat(cs.horzcat(m11, m12), cs.horzcat(m12, m22))

    # bias term
    tmp = l1*l2*m2*s2
    b1 = -(2 * theta1_dot * theta2_dot + theta2_dot**2)*tmp
    b2 = tmp * theta1_dot**2
    B = cs.vertcat(b1, b2)

    # friction
    c = 1.0
    C = cs.vertcat(c*theta1_dot, c*theta2_dot)

    # gravity term
    g1 = ((m1 + m2)*l2*c1 + m2*l2*c12) * g
    g2 = m2*l2*c12*g
    G = cs.vertcat(g1, g2)

    # define controls
    u = cs.SX.sym('u', 1)
    controls = cs.vertcat(u)

    # equations of motion
    tau = cs.vertcat(0, u)
    theta_ddot = cs.solve(M, tau - B - G - C)

    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(
                     theta1_dot,
                     theta2_dot,
                     theta_ddot[0],
                     theta_ddot[1])
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_free_time_double_pendulum_model() -> AcadosModel:

    model_name = 'free_time_double_pendulum_model'

    ## system dimensions
    nx = 4 + 1
    fixed_time_model = export_fixed_time_double_pendulum_model()

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