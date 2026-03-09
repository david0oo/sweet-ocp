# sweet-OCP: A test suite of direct optimal control problems🍬

A collection of Optimal Control Problems (OCPs) that are implemented in Python using the `acados` API. The test problems can be exported to Casadi expressions using the `AcadosCasadiSolver`.
The problems are discretized using multiple shooting.

## Installation
Set up and activate a fresh Python virtual environment (required Python >= 3.8)
```
python -m venv env
source env/bin/activate
```
Install acados and its python interface like here.

Clone the repository and navigate to the root directory of the repository
```
git clone git@github.com:david0oo/sweet_ocp.git
cd sweet_ocp
```
Install the python package
```
pip install -e .
```
Test if it was installed correctly by executing the following file:
```
python examples/solving_standard_nlp_in_acados.py
```

## Usage 
Please have a look into the examples folder.

## Contributing
If you have an interesting OCP that you would like to add to the test collection. Please follow the following steps
1. Assume your system is called `abcd`. If your system model is not available yet, please add a file `abcd_models.py` in the folder 
    ```
    sweet_ocp/models
    ```
    and a function that is called `export_abcd_model` for a fixed-time model or `export_free_time_abcd_model` for a free-time model
2. Add your OCP in the folder 
    ```
    sweet_ocp/ocps
    ```
    Please ensure that the `AcadosOcp` is built in a function `create_problem(opts: AcadosOptions)`.
    An initial guess should be available in a function `create_initial_guess()`. A  function for plotting trajectories should be available in `plot_trajectory(sol_X: np.array, sol_U: np.array)`.

## Citing
If you find this project useful, please consider giving it a ⭐ or citing it if your work is scientific:
```
@software{Kiessling2026,
  author = {Kiessling, David},
  license = {BSD-2},
  title = {sweet-OCP: A test suite of direct optimal control problems},
  url = {https://github.com/david0oo/sweet_ocp},
  version = {0.1.},
  year = {2026}
}
```
