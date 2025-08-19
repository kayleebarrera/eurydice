import numpy as np


##### KEPLERIAN SIGNAL HELPER FUNCTIONS #####


def kepler_solver(M_anom, ecc, tolerance=1e-7):
    """
    Solves Kepler's equation using the Newton Raphson method

    Args:
        M_anom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        tolerance (float, optional): absolute tolerance of iterative computatiom ('exit condition'). Defaults to 1e-7.

    Return:
        (np.array): array of eccentric anomalies
    """

    def f(E_anom, ecc, M_enom):
        return E_anom - ecc * np.sin(E_anom) - M_enom

    def df(E_anom, ecc):
        return 1 - ecc * np.cos(E_anom)

    ### intialize guess at E_0 = M ##
    E_anom = np.copy(M_anom)

    ### starting conditions ###
    E_anom -= f(E_anom, ecc, M_anom) / df(E_anom, ecc)
    diff = f(E_anom, ecc, M_anom) / df(E_anom, ecc)
    error = np.abs(diff)
    indx = np.where(error > tolerance)
    n = 0

    ### run algorithm until all values are found within tolerance
    while (indx[0].size > 0) and (n <= 1000000):
        E_anom[indx] -= diff[indx]
        diff[indx] = f(E_anom[indx], ecc[indx], M_anom[indx]) / df(
            E_anom[indx], ecc[indx]
        )
        error[indx] = np.abs(diff[indx])
        indx = np.where(error > tolerance)
        n += 1

    if n > 1000000:
        raise RuntimeError("Could not converge within 1000000 iterations")

    return E_anom


def transit_to_periastron(t_transit, period, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

    Args:
        t_transit (float): time of transit
        period (float): period of orbit [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        (float): time of periastron passage

    """
    f = np.pi / 2 - omega
    E_anom = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(f / 2))
    t_periastron = t_transit - period / (2 * np.pi) * (E_anom - ecc * np.sin(E_anom))

    return t_periastron


def calc_mean_anomaly(times, period, t_periastron):
    """
    Compute the mean anomaly at given times during an orbit

    Args:
        times (np.array): times of measurement
        period (float): period of orbit [days]
        t_periastron (float): time of periastron passage

    Returns:
        (np.array): array of mean anomalies

    """
    return (2 * np.pi / period * (times - t_periastron)) % (2 * np.pi)


def calc_true_anomaly(E_anom, ecc):
    """
    Compute the true anomaly at certain points during an orbit

    Args:
        E_anom (np.array): array of eccentric anomalies
        ecc (float): eccentricity

    Returns:
        (np.array): array of true anomalies

    """
    return 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E_anom / 2))


def calc_keplerian_signal(times, t_transit, period, ecc, omega, K):
    """
    Compute the radial velocity signal of a transiting exoplanet

    Args:
        times (np.array): time measurements
        t_transit (float): time of transit
        period (float): period of orbit [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)
        K (float): Keplarian semi-amplitude

    Returns:
        (np.array): array of radial velocity signals

    """

    if 0 <= ecc < 1:
        pass
    else:
        raise ValueError(
            f"Eccentricity must be between 0 <= ecc < 1. Your input eccentricity was {ecc}"
        )

    t_periastron = transit_to_periastron(t_transit, period, ecc, omega)

    M_anom = calc_mean_anomaly(times, period, t_periastron)

    ecc_array = ecc * np.ones_like(times)

    try:
        E_anom = kepler_solver(M_anom, ecc_array)

    except RuntimeError:
        print(
            f"Sorry, the Kepler solver took more than 1000000 iterations to converge for orbit with parameters [{t_transit}, {period}, {ecc}, {omega}, {K}], so I gave up."
        )
        print("-------------------")
        raise

    else:
        nu = calc_true_anomaly(E_anom, ecc)
        rv = K * (np.cos(omega + nu) + ecc * np.cos(omega))
        return rv
