# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import numpy as np
from typing import Union
from acados_template import AcadosOcpSolver, AcadosCasadiOcpSolver

# Helper functions
def set_initial_guess(N: int,
                      init_X: np.array,
                      init_U: np.array,
                      solver: Union[AcadosOcpSolver, AcadosCasadiOcpSolver]):
    if N > 0:
        for i in range(N):
            solver.set(i, "x", init_X[i, :])
            solver.set(i, "u", init_U[i, :])
        solver.set(N, "x", init_X[N, :])
    else:
        solver.set(0, "x", init_X[0, :])

def extract_solution(N: int,
                     nx: int,
                     nu: int,
                     solver: Union[AcadosOcpSolver, AcadosCasadiOcpSolver]):
    # Extract Solution
    sol_X = np.zeros((N+1, nx))
    sol_U = np.zeros((N, nu))
    # get solution
    if N > 0:
        for i in range(N):
            sol_X[i,:] = solver.get(i, "x")
            sol_U[i,:] = solver.get(i, "u")
        sol_X[N,:] = solver.get(N, "x")
    else:
        sol_X[0,:] = solver.get(0, "x")
        sol_U = []

    return sol_X, sol_U

def get_ocp_names():
    problems = [
        "altro_acrobot_problem",
        "altro_dubins_car_escape",
        "altro_dubins_car_obstacle",
        "altro_dubins_car_parallel_parking",
        "altro_simple_pendulum_problem",
        "betts_free_flying_robot_problem",
        "car_racing_time_optimal_p2p_motion",
        "cops_catalyst_mixing",
        "cops_flow_problem",
        "cops_goddard_rocket",
        "cops_hangglider_problem",
        "cops_particle_steering",
        "cops_robot_problem",
        "fatrop_moonlander_problem",
        "grampc_overhead_crane_problem",
        "hour_glass_time_optimal_p2p_motion",
        "time_optimal_overhead_crane_problem",
        "time_optimal_swing_up",
    ]
    return problems
