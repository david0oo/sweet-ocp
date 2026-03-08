# Copyright (c) 2026 David Kiessling
# Licensed under the BSD-2 license. See LICENSE file in the project directory for details.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CartPendulumVisualization:
    """
    Taken and adjusted from fatrop benchmark:
    https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop_benchmarks/-/blob/T1_problems/cart_pendulum/visualization.py?ref_type=heads
    """
    def __init__(self, L=0.8):
        self.L = L
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal')
        # self.ax.grid()
        self.line = patches.Rectangle((-2.0, -0.005), 4., 0.01, fc='gray')
        self.cart = patches.Rectangle((-0.1, -0.1), 0.2, 0.2, fc='r')
        self.pendulum = patches.Rectangle((-0.02, 0), 0.04, L, fc='b')
        self.ax.add_patch(self.line)
        self.ax.add_patch(self.cart)
        self.ax.add_patch(self.pendulum)

    def visualize(self, x, theta):
        self.cart.set_xy([x-0.1, -0.1])
        # rotate the pendulum over an angle theta
        self.pendulum.set_transform(mpl.transforms.Affine2D().rotate_around(
            0.00, 0., (theta)).translate(x, 0) + self.ax.transData)

        # set the position of the pendulum
        self.fig.canvas.draw()

class VisualizationFunctor:
    def __init__(self, vis, x, theta, L=1.):
        self.x = x
        self.theta = theta
        self.vis: CartPendulumVisualization = vis

    def __call__(self, frame):
        self.vis.visualize(self.x[frame], self.theta[frame])
