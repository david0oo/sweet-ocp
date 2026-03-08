# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
import casadi as cs
from dataclasses import dataclass

@dataclass
class CartPendulumModelParameters:
    # constants
    m_cart : float = 1.0 # mass of the cart [kg]
    m : float = 0.1 # mass of the ball [kg]
    g : float = 9.81 # gravity constant [m/s^2]
    l : float = 0.8 # length of the rod [m]


def export_free_time_cart_pendulum_ode_model() -> AcadosModel:

    model_name = 'free_time_cart_pendulum'

    pars = CartPendulumModelParameters()

    # set up states & controls
    T = cs.SX.sym('T')
    x1      = cs.SX.sym('x1')
    theta   = cs.SX.sym('theta')
    v1      = cs.SX.sym('v1')
    dtheta  = cs.SX.sym('dtheta')

    x = cs.vertcat(T, x1, theta, v1, dtheta)

    F = cs.SX.sym('F')
    u = cs.vertcat(F)

    # xdot
    T_dot       = cs.SX.sym('T_dot')
    x1_dot      = cs.SX.sym('x1_dot')
    theta_dot   = cs.SX.sym('theta_dot')
    v1_dot      = cs.SX.sym('v1_dot')
    dtheta_dot  = cs.SX.sym('dtheta_dot')

    xdot = cs.vertcat(T_dot, x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = pars.m_cart + pars.m - pars.m*cos_theta*cos_theta
    f_expl = cs.vertcat(0,
                        T*v1,
                        T*dtheta,
                        T*((-pars.m*pars.l*sin_theta*dtheta*dtheta + pars.m*pars.g*cos_theta*sin_theta+F)/denominator),
                        T*((-pars.m*pars.l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(pars.m_cart+pars.m)*pars.g*sin_theta)/(pars.l*denominator))
                        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

def export_fixed_time_cart_pendulum_ode_model() -> AcadosModel:

    model_name = 'fixed_time_cart_pendulum'

    pars = CartPendulumModelParameters()

    # set up states & controls
    x1      = cs.SX.sym('x1')
    theta   = cs.SX.sym('theta')
    v1      = cs.SX.sym('v1')
    dtheta  = cs.SX.sym('dtheta')

    x = cs.vertcat(x1, theta, v1, dtheta)

    F = cs.SX.sym('F')
    u = cs.vertcat(F)

    # xdot
    x1_dot      = cs.SX.sym('x1_dot')
    theta_dot   = cs.SX.sym('theta_dot')
    v1_dot      = cs.SX.sym('v1_dot')
    dtheta_dot  = cs.SX.sym('dtheta_dot')

    xdot = cs.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = pars.m_cart + pars.m - pars.m*cos_theta*cos_theta
    f_expl = cs.vertcat(v1,
                        dtheta,
                        ((-pars.m*pars.l*sin_theta*dtheta*dtheta + pars.m*pars.g*cos_theta*sin_theta+F)/denominator),
                        ((-pars.m*pars.l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(pars.m_cart+pars.m)*pars.g*sin_theta)/(pars.l*denominator))
                        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model


