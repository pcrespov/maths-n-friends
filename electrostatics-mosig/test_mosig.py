from typing import Callable

import numpy as np


def definite_integral(primitive: Callable, x1, x2, y1, y2, h):
    return (
        primitive(x1, y1, h)
        + primitive(x2, y2, h)
        - primitive(x1, y2, h)
        - primitive(x2, y1, h)
    )


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
    return ((u**2 + v**2 + h**2) ** (3 / 2)) / 3.0


def discrete_green_function(xc, yc, zc, a, b):
    u1, v1 = xc + a / 2, yc + b / 2
    u2, v2 = xc - a / 2, yc + b / 2
    result = (
        primitive_of_g(u=u1, v=v1, h=zc)
        + primitive_of_g(u=u2, v=v2, h=zc)
        - primitive_of_g(u=u1, v=v2, h=zc)
        - primitive_of_g(u=u2, v=v1, h=zc)
    )
    assert result == definite_integral(primitive_of_g, u1, u2, v1, v2, zc)
    S = a * b
    return result / S


def average_discrete_green_function(xc, yc, zc, a, b):
    x1, x2, y1, y2 = xc - a, xc, yc - b, yc
    Yuu = (
        x1 * y1 * definite_integral(primitive_of_g, x1, x2, y1, y2, zc)
        - y1 * definite_integral(primitive_of_ug, x1, x2, y1, y2, zc)
        - x1 * definite_integral(primitive_of_vg, x1, x2, y1, y2, zc)
        + definite_integral(primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc - a, xc, yc, yc + b
    Yud = (
        -x1 * y2 * definite_integral(primitive_of_g, x1, x2, y1, y2, zc)
        + y2 * definite_integral(primitive_of_ug, x1, x2, y1, y2, zc)
        + x1 * definite_integral(primitive_of_vg, x1, x2, y1, y2, zc)
        - definite_integral(primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc, xc + a, yc - b, yc
    Ydu = (
        -x2 * y1 * definite_integral(primitive_of_g, x1, x2, y1, y2, zc)
        + y1 * definite_integral(primitive_of_ug, x1, x2, y1, y2, zc)
        + x2 * definite_integral(primitive_of_vg, x1, x2, y1, y2, zc)
        - definite_integral(primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc, xc + a, yc, yc + b
    Ydd = (
        x2 * y2 * definite_integral(primitive_of_g, x1, x2, y1, y2, zc)
        - y2 * definite_integral(primitive_of_ug, x1, x2, y1, y2, zc)
        - x2 * definite_integral(primitive_of_vg, x1, x2, y1, y2, zc)
        + definite_integral(primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    S = a * b
    return (Yuu + Yud + Ydu + Ydd) / S**2
