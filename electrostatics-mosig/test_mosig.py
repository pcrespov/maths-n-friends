from typing import Callable


def definite_integral(primitive: Callable, x1: float, x2: float, y1: float, y2: float):
    return primitive(x1, y1) + primitive(x2, y2) - primitive(x1, y2) - primitive(x2, y1)


import numpy as np

#
# Primitives of Table I
#


def primitive_of_g(u, v, h):
    return (
        u * np.arcsinh(v / np.sqrt(u**2 + h**2))
        + v * np.arcsinh(u / np.sqrt(v**2 + h**2))
        - h * np.arctanh(u * v / (h * np.sqrt(u**2 + v**2 + h**2)))
    )


def primitive_of_ug(u, v, h):
    return 0.5 * (
        v * np.sqrt(u**2 + v**2 + h**2)
        + (u**2 + h**2) * np.arcsinh(v / np.sqrt(u**2 + h**2))
    )


def primitive_of_vg(u, v, h):
    return 0.5 * (
        u * np.sqrt(u**2 + v**2 + h**2)
        + (v**2 + h**2) * np.arcsinh(u / np.sqrt(v**2 + h**2))
    )


def primitive_of_uvg(u, v, h):
    return ((u**2 + v**2 + h**2) ** (3 / 2)) / 3.


def discrete_green_function(xc, yc, zc, a, b):
    u1, v1 = xc + a / 2, yc + b / 2
    u2, v2 = xc - a / 2, yc + b / 2
    result = (
        primitive_of_g(u=u1, v=v1, h=zc)
        + primitive_of_g(u=u2, v=v2, h=zc)
        - primitive_of_g(u=u1, v=v2, h=zc)
        - primitive_of_g(u=u2, v=v1, h=zc)
    )
    S = a * b
    return result / S


def average_discrete_green_function(xc, yc, zc, a, b):
