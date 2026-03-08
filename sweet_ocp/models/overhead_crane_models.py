# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos
import numpy as np

def export_fixed_time_crane_model() -> AcadosModel:

    model_name = 'fixed_time_crane_model'

    ## system dimensions
    nx = 6
    g = 9.81 # m/s^2

    ## named symbolic variables
    # Define states
    length = SX.sym('l', 1)  # length of the pendulum
    x_c = SX.sym('x_c', 1) # position of cart
    theta = SX.sym('theta', 1) # angle
    l_dot = SX.sym('l_dot', 1)  # derivative of length of the pendulum
    x_c_dot = SX.sym('x_c_dot', 1)
    theta_dot = SX.sym('theta_dot', 1)
    states = vertcat(length, x_c, theta, l_dot, x_c_dot, theta_dot)

    # define controls
    x_ddot = SX.sym('x_ddot', 1)
    l_ddot = SX.sym('l_ddot', 1)
    controls = vertcat(l_ddot, x_ddot)

    theta_ddot = (-x_ddot*cos(theta) - 2*l_dot*theta_dot - g*sin(theta))/length

    # define dynamics expression
    states_dot = SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = vertcat(l_dot,
                     x_c_dot,
                     theta_dot,
                     l_ddot,
                     x_ddot,
                     theta_ddot)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model


def export_free_time_crane_model() -> AcadosModel:

    model_name = 'free_time_crane_model'

    ## system dimensions
    nx = 6 + 1 # +1 for the time
    g = 9.81 # m/s^2

    ## named symbolic variables
    # define states
    # Define states
    length = SX.sym('l', 1)  # length of the pendulum
    x_c = SX.sym('x_c', 1) # position of cart
    theta = SX.sym('theta', 1) # angle
    l_dot = SX.sym('l_dot', 1)  # derivative of length of the pendulum
    x_c_dot = SX.sym('x_c_dot', 1)
    theta_dot = SX.sym('theta_dot', 1)
    T = SX.sym('T')
    states = vertcat(T, length, x_c, theta, l_dot, x_c_dot, theta_dot)

    # define controls
    x_ddot = SX.sym('x_ddot', 1)
    l_ddot = SX.sym('l_ddot', 1)
    controls = vertcat(l_ddot, x_ddot)

    theta_ddot = (-x_ddot*cos(theta) - 2*l_dot *
                      theta_dot - g*sin(theta))/length

    # define dynamics expression
    states_dot = SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = vertcat(0,
                     T*l_dot,
                     T*x_c_dot,
                     T*theta_dot,
                     T*l_ddot,
                     T*x_ddot,
                     T*theta_ddot)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def export_free_time_crane_model_with_sep_hyperplane() -> AcadosModel:

    model_name = 'free_time_crane_model_with_separating_hyperplane'

    ## system dimensions
    nx = 6 + 1 # +1 for the time
    g = 9.81 # m/s^2

    ## named symbolic variables
    # define states
    # Define states
    length = SX.sym('l', 1)  # length of the pendulum
    x_c = SX.sym('x_c', 1) # position of cart
    theta = SX.sym('theta', 1) # angle
    l_dot = SX.sym('l_dot', 1)  # derivative of length of the pendulum
    x_c_dot = SX.sym('x_c_dot', 1)
    theta_dot = SX.sym('theta_dot', 1)
    T = SX.sym('T')
    states = vertcat(T, length, x_c, theta, l_dot, x_c_dot, theta_dot)

    # define controls
    x_ddot = SX.sym('x_ddot', 1)
    l_ddot = SX.sym('l_ddot', 1)
    sh = SX.sym('sh', 3)
    controls = vertcat(l_ddot, x_ddot, sh)

    theta_ddot = (-x_ddot*cos(theta) - 2*l_dot *
                      theta_dot - g*sin(theta))/length

    # define dynamics expression
    states_dot = SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = vertcat(0,
                     T*l_dot,
                     T*x_c_dot,
                     T*theta_dot,
                     T*l_ddot,
                     T*x_ddot,
                     T*theta_ddot)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def get_payload_position(sol_X: np.ndarray):
    """
    For non-free-time model!!
    """
    payload_x1_position = sol_X[:, 1] + sol_X[:, 0]*np.sin(sol_X[:, 2])
    payload_x2_position = -sol_X[:, 0] * np.cos(sol_X[:, 2])
    return payload_x1_position, payload_x2_position