from typing import Callable

import numpy as np
from scipy.constants import epsilon_0

import logging

_logger = logging.getLogger(__name__)

#
# Primitives of Table I
#


def _primitive_of_g(u, v, w):
    return (
        u * np.arcsinh(v / np.sqrt(u**2 + w**2))
        + v * np.arcsinh(u / np.sqrt(v**2 + w**2))
        - w * np.arctan(u * v / (w * np.sqrt(u**2 + v**2 + w**2)))
    )


def _primitive_of_ug(u, v, w):
    return (
        v * np.sqrt(u**2 + v**2 + w**2)
        + (u**2 + w**2) * np.arcsinh(v / np.sqrt(u**2 + w**2))
    ) / 2


def _primitive_of_vg(u, v, w):
    return (
        u * np.sqrt(u**2 + v**2 + w**2)
        + (v**2 + w**2) * np.arcsinh(u / np.sqrt(v**2 + w**2))
    ) / 2


def _primitive_of_uvg(u, v, w):
    return ((u**2 + v**2 + w**2) ** (3 / 2)) / 3.0



#
# Helper
#


def _definite_integral(primitive: Callable, x1, x2, y1, y2, h):
    return (
        primitive(x2, y2, h)
        - primitive(x1, y2, h)
        - primitive(x2, y1, h)
        + primitive(x1, y1, h)
    )


#
# Integration
#

_EPS = np.finfo(float).eps

def green_function(xc, yc, zc):
    # 1/R
    return 1 / np.sqrt(xc ** 2 + yc ** 2 + zc ** 2) 


def discrete_green_function(xc, yc, zc, a, b):
    u1, u2 = xc + a / 2, xc - a / 2
    v1, v2 = yc + b / 2, yc - b / 2
    result = _definite_integral(_primitive_of_g, u1, u2, v1, v2, zc + _EPS)
    S = a * b
    return result / S


def average_discrete_green_function(xc, yc, zc, a, b):
    zc += _EPS
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


def eval_stat_mom(
    x_basis,
    y_basis,
    z_basis,
    x_test,
    y_test,
    z_test,
    *,
    a: float,
    b: float,
    factor_resolution_h: float,
    use_galerkin: bool,
):
    xc = np.abs(x_basis - x_test)
    yc = np.abs(y_basis - y_test)
    zc = np.abs(z_basis - z_test)
    rh = np.sqrt(xc**2 + yc**2)

    use_gf_values = rh > (factor_resolution_h * max(a, b))
    if use_gf_values:
        r = np.sqrt(xc**2 + yc**2 + zc**2)  # distance between cell centers
        return 1 / r

    if use_galerkin:
        return average_discrete_green_function(xc, yc, zc, a, b)

    return discrete_green_function(xc, yc, zc, a, b)


def run_static_plate(
    a_side: float = 1.0, b_side: float = 1.0, m: int = 20, n: int = 20
):
    a = a_side / m
    b = b_side / n

    x = np.linspace(a / 2, a_side - a / 2, m)
    y = np.linspace(b / 2, b_side - b / 2, n)

    x_matrix, y_matrix = np.meshgrid(x, y)
    assert x_matrix.size == m * n  # nosec
    assert y_matrix.size == x_matrix.size  # nosec

    x_vector = np.reshape(x_matrix, newshape=x_matrix.size)
    y_vector = np.reshape(y_matrix, newshape=y_matrix.size)

    # filling the Mom matrix
    # TODO: vectorize
    ndim = x_matrix.size
    tresh = 10
    mom = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            mom[i, j] = eval_stat_mom(
                x_basis=x_vector[j],
                y_basis=y_vector[j],
                z_basis=0,
                x_test=x_vector[i],
                y_test=y_vector[i],
                z_test=0,
                a=a,
                b=b,
                factor_resolution_h=tresh,
                use_galerkin=False,
            )

    # excitation vector
    _logger.debug(f"{mom=}")

    # (1) constant potential
    def _excite_with_constant_potential(v):
        return np.full(shape=x_vector.shape, fill_value=v)

    # (2) a point charge
    def _excite_with_point_charge():
        excv = np.zeros(x_vector.shape)
        xq, yq, zq = 0.5, 0.5, 0.5
        for i in range(excv.size):
            excv[i] = 1.0 / np.sqrt(
                (xq - x_vector[i]) ** 2 + (yq - y_vector[i]) ** 2 + zq**2
            )
        return excv

    excitation_potential = 1
    excv = _excite_with_constant_potential(excitation_potential)
    # excv = _excite_with_point_charge()

    # Solving linear system
    charge = np.linalg.solve(mom, excv)

    # Total charge of the plate
    total_charge = charge.sum()

    # If constant potential, then the normalized capacity is
    normalized_capacity = total_charge / excitation_potential

    capacity = 4.0 * np.pi * epsilon_0  # Farads

    print(f"{total_charge=} Coulombs")
    print(f"{normalized_capacity=}")
    print(f"{capacity=} F")

