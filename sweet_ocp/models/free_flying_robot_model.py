# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from dataclasses import dataclass
import casadi as cs

@dataclass
class FreeFlyingRobotModelParameters:
    alpha : float = 0.2
    beta : float = 0.2

def export_free_flying_robot_model() -> AcadosModel:

    model_name = 'free_flying_robot_model'

    params = FreeFlyingRobotModelParameters()

    # system dimensions
    nx = 6

    # Define the states
    x1 = cs.SX.sym('x1', 1)
    x2 = cs.SX.sym('x2', 1)
    theta = cs.SX.sym('theta', 1)
    v1 = cs.SX.sym('v1', 1)
    v2 = cs.SX.sym('v2', 1)
    omega = cs.SX.sym('omega', 1)
    states = cs.vertcat(x1, x2, theta, v1, v2, omega)

    # define controls
    u1 = cs.SX.sym('u1', 1)
    u2 = cs.SX.sym('u2', 1)
    u3 = cs.SX.sym('u3', 1)
    u4 = cs.SX.sym('u4', 1)
    controls = cs.vertcat(u1, u2, u3, u4)


    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(v1,
                        v2,
                        omega,
                        (u1-u2+u3-u4)*cs.cos(theta),
                        (u1-u2+u3-u4)*cs.sin(theta),
                        params.alpha*(u1-u2)-params.beta*(u3-u4))
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model
