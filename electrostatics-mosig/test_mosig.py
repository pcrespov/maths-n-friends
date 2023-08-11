from typing import Callable

import numpy as np

#
# Primitives of Table I
#


def _primitive_of_g(u, v, h):
    return (
        u * np.arcsinh(v / np.sqrt(u**2 + h**2))
        + v * np.arcsinh(u / np.sqrt(v**2 + h**2))
        - h * np.arctanh(u * v / (h * np.sqrt(u**2 + v**2 + h**2)))
    )


def _primitive_of_ug(u, v, h):
    return 0.5 * (
        v * np.sqrt(u**2 + v**2 + h**2)
        + (u**2 + h**2) * np.arcsinh(v / np.sqrt(u**2 + h**2))
    )


def _primitive_of_vg(u, v, h):
    return 0.5 * (
        u * np.sqrt(u**2 + v**2 + h**2)
        + (v**2 + h**2) * np.arcsinh(u / np.sqrt(v**2 + h**2))
    )


def _primitive_of_uvg(u, v, h):
    return ((u**2 + v**2 + h**2) ** (3 / 2)) / 3.0


#
# Helper
#


def _definite_integral(primitive: Callable, x1, x2, y1, y2, h):
    return (
        primitive(x1, y1, h)
        + primitive(x2, y2, h)
        - primitive(x1, y2, h)
        - primitive(x2, y1, h)
    )


#
# Integration
#


def discrete_green_function(xc, yc, zc, a, b):
    u1, v1 = xc + a / 2, yc + b / 2
    u2, v2 = xc - a / 2, yc + b / 2
    result = (
        _primitive_of_g(u=u1, v=v1, h=zc)
        + _primitive_of_g(u=u2, v=v2, h=zc)
        - _primitive_of_g(u=u1, v=v2, h=zc)
        - _primitive_of_g(u=u2, v=v1, h=zc)
    )
    assert result == _definite_integral(_primitive_of_g, u1, u2, v1, v2, zc)  # nosec
    S = a * b
    return result / S


def average_discrete_green_function(xc, yc, zc, a, b):
    x1, x2, y1, y2 = xc - a, xc, yc - b, yc
    Yuu = (
        x1 * y1 * _definite_integral(_primitive_of_g, x1, x2, y1, y2, zc)
        - y1 * _definite_integral(_primitive_of_ug, x1, x2, y1, y2, zc)
        - x1 * _definite_integral(_primitive_of_vg, x1, x2, y1, y2, zc)
        + _definite_integral(_primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc - a, xc, yc, yc + b
    Yud = (
        -x1 * y2 * _definite_integral(_primitive_of_g, x1, x2, y1, y2, zc)
        + y2 * _definite_integral(_primitive_of_ug, x1, x2, y1, y2, zc)
        + x1 * _definite_integral(_primitive_of_vg, x1, x2, y1, y2, zc)
        - _definite_integral(_primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc, xc + a, yc - b, yc
    Ydu = (
        -x2 * y1 * _definite_integral(_primitive_of_g, x1, x2, y1, y2, zc)
        + y1 * _definite_integral(_primitive_of_ug, x1, x2, y1, y2, zc)
        + x2 * _definite_integral(_primitive_of_vg, x1, x2, y1, y2, zc)
        - _definite_integral(_primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    x1, x2, y1, y2 = xc, xc + a, yc, yc + b
    Ydd = (
        x2 * y2 * _definite_integral(_primitive_of_g, x1, x2, y1, y2, zc)
        - y2 * _definite_integral(_primitive_of_ug, x1, x2, y1, y2, zc)
        - x2 * _definite_integral(_primitive_of_vg, x1, x2, y1, y2, zc)
        + _definite_integral(_primitive_of_uvg, x1, x2, y1, y2, zc)
    )

    S = a * b
    return (Yuu + Yud + Ydu + Ydd) / S**2


import pytest


@pytest.param(
    "gal,pm,gf,loc",
    [
        (2.9732, 3.5255, np.Inf, (0, 0, 0)),
        (1.1121, 1.0380, 1.0, (1, 0, 0)),
        (0.7490, 0.7247, 0.7071, (1, 1, 0)),
        (0.2513, 0.2506, 0.2500, (4, 0, 0)),
        (2.4674, 2.9533, 10.0, (0, 0, 0.1)),
        (0.8788, 0.9286, 1.0, (0, 0, 1)),
        (0.1987, 0.1933, 0.2, (0, 0, 5)),
    ],
)
def test_integration_values_table(gal, pm, gf, loc):
    assert discrete_green_function(*loc, a=1, b=1) == pytest.approx(pm, abs=0.0001)
    assert average_discrete_green_function(*loc, a=1, b=1) == pytest.approx(
        gal, abs=0.0001
    )
    assert 1 / np.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[3] ** 2) == pytest.approx(
        gf, abs=0.001
    )
