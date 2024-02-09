from typing import Callable

import numpy as np
import pytest
from scipy.constants import epsilon_0

from em_statics import _primitive_of_g, _primitive_of_ug, _primitive_of_uvg, _primitive_of_vg
from em_statics import discrete_green_function, average_discrete_green_function, run_static_plate, green_function


def test_center():
    assert 4 * _primitive_of_uvg(0.5, 0.5, 0) == pytest.approx(3.5255, 0.001)


def test_edge_mid_point():
    assert 4 * _primitive_of_uvg(1.0, 0.5, 0) == pytest.approx(2.4061, 0.001)


def test_corner():
    assert 4 * _primitive_of_uvg(1.0, 1.5, 0) == pytest.approx(1.7627, 0.001)


def test_above_center():
    assert 4 * _primitive_of_uvg(0.5, 0.5, 0.5) == pytest.approx(1.5867, 0.001)


# gal,pm,gf,loc
GAL, PM, GF, LOC = 0, 1, 2, 3
_table_1 = [
    (2.9732, 3.5255, np.Inf, (0, 0, 0)),
    (1.1121, 1.0380, 1.0, (1, 0, 0)),
    (0.7490, 0.7247, 0.7071, (1, 1, 0)),
    (0.2513, 0.2506, 0.2500, (4, 0, 0)),
    (2.4674, 2.9533, 10.0, (0, 0, 0.1)),
    (0.8788, 0.9286, 1.0, (0, 0, 1)),
    (0.1987, 0.1993, 0.2, (0, 0, 5)),
]

# E         Obtained: 0.1993379575985088
# E         Expected: 0.1933 ± 1.0e-04


@pytest.mark.parametrize("pm,loc", [(row[PM], row[LOC]) for row in _table_1])
def test_discrete_green_function(pm, loc):
    assert discrete_green_function(*loc, a=1, b=1) == pytest.approx(pm, abs=0.0001)


@pytest.mark.parametrize("gal,loc", [(row[GAL], row[LOC]) for row in _table_1])
def test_average_discrete_green_function(gal, loc):
    assert average_discrete_green_function(*loc, a=1, b=1) == pytest.approx(
        gal, abs=0.0001
    )


@pytest.mark.parametrize("gf,loc", [(row[GF], row[LOC]) for row in _table_1])
def test_green_function(gf, loc):
    assert green_function(*loc) == pytest.approx(
        gf, abs=0.001
    )


def test_static_plate_problem_run():
    run_static_plate()
