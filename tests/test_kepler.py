"""
Test the eurydice.kepler module
"""

import numpy as np
import eurydice.kepler as kepler
import pytest


threshold = 1e-5


def test_kepler_solver():
    """
    Test Kepler Solver by comparing the mean anomaly computed from
    kepler_solver output vs the input mean anomaly
    """
    M_anoms = np.linspace(0, 2.0 * np.pi, 100)
    eccs = np.linspace(0, 0.999999, 100)

    for eccentricity in eccs:
        ecc_array = eccentricity * np.ones_like(M_anoms)
        E_anoms = kepler.kepler_solver(M_anoms, ecc_array)
        calc_mm = E_anoms - eccentricity * np.sin(E_anoms)

        for meas, truth in zip(calc_mm, M_anoms):
            assert (meas - truth) == pytest.approx(0.0, abs=threshold)


def test_transit_to_periastron():
    """
    Test transit_to_periastron using a set of test cases

    Test case 1: 4 day circular orbit
    *technically a circular orbit doesn't have a defined time of periastron
    passage but this is just to check that the function is outputting
    correct numbers*

    Test case 2: time of transit = time of periastron (omega = pi/2 -> f = 0 at transit)

    Test case 3: standard orbit with random standard parameters
    """

    params_list = [
        [58849, 4, 0, 0],
        [58849, 4, 0, np.pi / 2],
        [58849, 2.35, 0.12, np.pi / 4],
    ]

    true_periastrons = [58848, 58849, 58848.7658259028]

    for ii in range(0, 3):
        calc_periastron = kepler.transit_to_periastron(*params_list[ii])
        assert calc_periastron == pytest.approx(true_periastrons[ii], abs=threshold)


def test_calc_mean_anomaly():
    """
    Test test_calc_mean_anomaly function using a standard 4 day orbit
    """
    test_times = np.linspace(55849, 55849 + 4, num=50)
    period = 4
    t_periastron = 55849

    true_mean_anoms = np.linspace(0, 2 * np.pi, num=49, endpoint=False)
    true_mean_anoms = np.append(true_mean_anoms, 0)

    meas_mean_anoms = kepler.calc_mean_anomaly(test_times, period, t_periastron)

    for meas, truth in zip(meas_mean_anoms, true_mean_anoms):
        assert meas == pytest.approx(truth, abs=threshold)


def test_calc_true_anomaly():
    """
    Test transit_to_periastron using a 2 test cases

    Test case 1: ecc = 0, true anomaly = eccentric anomaly

    Test case 2: ecc of 0.3
    """
    test_ecc_anoms = np.linspace(-np.pi, np.pi, num=11)

    ### test case 1:
    meas_true_anoms_zero_ecc = kepler.calc_true_anomaly(test_ecc_anoms, 0)
    for meas, truth in zip(meas_true_anoms_zero_ecc, test_ecc_anoms):
        assert meas == pytest.approx(truth, abs=threshold)

    ### test case 2:
    meas_true_anoms_non_zero_ecc = kepler.calc_true_anomaly(test_ecc_anoms, 0.3)
    truth_true_anoms_non_zero_ecc = np.array(
        [
            -3.141592654,
            -2.673480544,
            -2.161984575,
            -1.560857835,
            -0.8336853887,
            0,
            0.8336853887,
            1.560857835,
            2.161984575,
            2.673480544,
            3.141592654,
        ]
    )
    for meas, truth in zip(meas_true_anoms_non_zero_ecc, truth_true_anoms_non_zero_ecc):
        assert meas == pytest.approx(truth, abs=threshold)


if __name__ == "__main__":
    test_kepler_solver()
    test_transit_to_periastron()
    test_calc_mean_anomaly()
    test_calc_true_anomaly()
    print("Done!")
