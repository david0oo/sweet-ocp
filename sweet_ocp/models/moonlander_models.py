# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from dataclasses import dataclass
import casadi as cs

@dataclass
class MoonlanderModelParameters:
    m : float = 1.0
    g : float = 9.81
    I : float = 0.1
    D : float = 1.0


def transf(theta, p):
    return cs.vertcat(
        cs.horzcat( cs.cos(theta), -cs.sin(theta), p[0]),
                    cs.horzcat(cs.sin(theta), cs.cos(theta), p[1]),
                    cs.horzcat(0.0, 0.0, 1.))


def export_free_time_moonlander_model() -> AcadosModel:

    model_name = 'free_time_moonlander_model'

    params = MoonlanderModelParameters()

    ## system dimensions
    nx = 6 + 1 # +1 for the time
    g = 9.81 # m/s^2

    # Define the states
    p = cs.SX.sym('p', 2)
    dp = cs.SX.sym('dp', 2)
    theta = cs.SX.sym('theta', 1)
    dtheta = cs.SX.sym('dtheta', 1)
    T = cs.SX.sym('T')
    states = cs.vertcat(T, p, dp, theta, dtheta)

    # define controls
    F1 = cs.SX.sym('x_ddot', 1)
    F2 = cs.SX.sym('l_ddot', 1)
    controls = cs.vertcat(F1, F2)


    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    F_r = transf(theta, p)
    F_tot = (F_r @ cs.vertcat(0, F1 + F2, 1))[:2]

    ## dynamics
    f_expl = cs.vertcat(0,
                     T*dp,
                     T*(1/params.m * F_tot + cs.vertcat(0, -params.g)),
                     T*dtheta,
                     T*(1/params.I * params.D/2 * (F2 - F1)))
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model
