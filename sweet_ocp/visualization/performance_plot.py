# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

def latexify():
    params = {
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': True,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

latexify()

class PerformancePlot:

    def __init__(self,
                 list_combinations,
                 linestyles,
                 colors,
                 data) -> None:

        df = {}

        for i in range(len(list_combinations)):
            df[list_combinations[i]] = data[i]

        self.data = df
        self.linestyles = linestyles
        self.colors = colors

    def plot_absolute_performance(self,
                                  save_title: str,
                                  metric: str,
                                  save=True):

        t_sp = np.array([df[metric] for df in self.data.values()]) # add all problems into one np array
        n_p = t_sp.shape[1] # number of problems
        n_s = t_sp.shape[0] # number of solvers
        tm_sp = np.min(t_sp, axis=0) # take minimum of solvers per problem
        tm_sp = np.tile(tm_sp, (n_s, 1)) # duplicate the minima for number of solvers solvers
        r_sp = t_sp # do not divide all entries of the metric by the minimum (stay relative)

        plt.figure(figsize=(4.5, 3.0))
        # plt.save_title('Performance Profile: ' + save_title)
        τ_max = np.max(r_sp[np.isfinite(r_sp)]) # get the maximum tau that is necessary
        for ρ_solver, label in zip(r_sp, self.data.keys()):
            ρ_solver = ρ_solver[np.isfinite(ρ_solver)] #remove all values with inf
            x, y = np.unique(ρ_solver, return_counts=True) #get unique values and count them
            y = (1. / n_p) * np.cumsum(y) # get fraction of solved problems
            x = np.append(x, [τ_max])
            y = np.append(y, [y[-1]])
            plt.step(x, y, '-', where='post', label=label)

        plt.xscale("log")
        plt.xlim(0, τ_max)
        plt.ylim(0, 1)
        plt.xlabel(metric)
        plt.ylabel(r'$\rho_s(t)$')
        plt.legend(loc='lower right')
        if save:
            plt.savefig(save_title+".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.tight_layout()
        plt.show()

    def plot_performance_profile(self,
                                 save_title: str,
                                 metric: str,
                                 save: bool=True,
                                 use_n_problems: bool=False):
        fig, ax = plt.subplots(figsize=(3.5, 3))

        xlim = 0
        print(metric)
        t_sp = np.array([df[metric] for df in self.data.values()]) # add all problems into one np array
        print(t_sp.shape)
        n_p = t_sp.shape[1] # number of problems
        n_s = t_sp.shape[0] # number of solvers
        tm_sp = np.min(t_sp, axis=0) # take minimum between all solvers per problem
        tm_sp = np.tile(tm_sp, (n_s, 1)) # duplicate the minima for number of solvers solvers
        r_sp = t_sp / tm_sp # divide all entries of the metric by the minimum (make it relative)

        τ_max = np.max(r_sp[np.isfinite(r_sp)]) * 2 # get the maximum tau that is necessary
        xlim = max(xlim, τ_max)
        for ρ_solver, label, ls, color in zip(r_sp, self.data.keys(), self.linestyles, self.colors):
            ρ_solver = ρ_solver[np.isfinite(ρ_solver)] #remove all values with inf
            x, y = np.unique(ρ_solver, return_counts=True) #get unique values and count them
            if use_n_problems:
                y = (1. / n_p) * np.cumsum(y) * n_p
            else:
                y = (1. / n_p) * np.cumsum(y) # get fraction of solved problems
            x = np.append(x, [τ_max])
            y = np.append(y, [y[-1]])
            ax.step(x, y, '-', where='post', label=label, color=color)

        ax.set_xscale("log", base=2)
        if use_n_problems:
            ax.set_ylim(0, n_p)
            ax.set_ylabel('number of solved problems')
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('fraction of solved problems')

        ax.set_xlim(1, xlim)
        ax.set_xlabel('ratio of evaluations wrt best')
        ax.legend(loc='lower right')

        plt.tight_layout()

        if save:
            plt.savefig(save_title + "_" + metric + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)

        plt.show()

    def plot_convergence(self,
                         convergence_criterion: str,
                         list_criteria: list,
                         labels: list,
                         colors: list,
                         save_name:str):
        # %% Plot Contraction of convergence criterion
        plt.figure(figsize=(5, 3))
        xlim = 0
        for infeas, label, color in zip(list_criteria, labels, colors):
            iters = np.arange(0, len(list(infeas)))
            print(iters)
            print(infeas)
            xlim = max(xlim, len(infeas))
            plt.semilogy(iters,
                        infeas,
                        label=label,
                        color=color)
        plt.xlabel('iteration number')
        plt.xlim((0, xlim))
        plt.ylim((1e-6, 1e2))
        plt.ylabel(f'{convergence_criterion}')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(save_name + ".pdf",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.01)
        plt.show()
