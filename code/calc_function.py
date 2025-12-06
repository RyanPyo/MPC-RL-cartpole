import math
import numpy as np
from scipy.optimize import least_squares


def calc_AB(M, m, I, l):
    # I = 0.063
    # l = 0.216
    #M = 0.977  # cart mass
    #m = 2.686  # pendulum mass
    b = 0.0  # friction
    g = 9.81  # gravitational acceleration

    px = I * (m + M) + M * m * l ** 2
    A22 = -(I + m * l ** 2) * b / px
    A23 = -(m ** 2 * l ** 2 * g) / px
    A42 = -(m * l * b) / px
    A43 = ((M + m) * m * g * l) / px
    B21 = (I + m * l ** 2) / px
    B41 = (m * l) / px
    Ac = np.matrix([[0, 1, 0, 0],
                    [0, A22, A23, 0],
                    [0, 0, 0, 1],
                    [0, A42, A43, 0]])  # considering theta and theta dot only
    Bc = np.matrix([[0],
                    [B21],
                    [0],
                    [B41]])

    global Ts
    Ts = 0.01
    [nx, nu] = Bc.shape  # number of states and number or inputs

    # Simple forward euler discretization
    Ad = np.eye(nx) + Ac * Ts
    Bd = Bc * Ts
    return Ad, Bd