import numpy as np
import pandas as pd
import eurydice.kepler as kepler
import eurydice.plot as plot
import eurydice.kernel as kernel
import matplotlib.pyplot as plt


def split(data, train_split, random=True):
    """
    Splits data into a training set and test set to perform cross validation

    Args:
        data (dataframe): a dataframe containing timestamps, RV measurements, RV errors, and instrument info
        train_split (float): number from 0 to 1 (inclusive), the fraction of data that will be split into the training set. If split does not divide data into whole numbers, the training split will be rounded down.
        random (bool): whether or not to choose the training set from random. Defaults to True. If set to False, the training set will chosen from the first fraction of data.

    Returns:
        training_data (dataframe), test_data (dataframe): two dataframes split from input data into a training and test set
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("The input data must be organized as a pandas DataFrame.")

    if (train_split < 0) or (train_split > 1):
        raise ValueError(
            f"The train_split value must be between 0 and 1 (inclusive). Your input train_split value was {train_split}"
        )

    n_data = len(data)

    if random:
        training_mask = np.random.choice(
            n_data, size=int(train_split * n_data), replace=False
        )

    else:
        training_mask = np.arange(int(train_split * n_data))

    test_mask = np.setdiff1d(np.arange(n_data), training_mask)

    training_data = data.iloc[training_mask]
    training_data.reset_index(drop=True, inplace=True)

    test_data = data.iloc[test_mask]
    test_data.reset_index(drop=True, inplace=True)

    return training_data, test_data


##### CROSSVALIDATION OBJECT ######
class CrossValidation:
    """
    Args:
        training_data (dataframe): a dataframe containing the times, rv measurements, rv errors, and instruments information of the training set
        test_data (dataframe): a dataframe containing the times, rv measurements, rv errors, and instruments information of the test set
        orbit_params (tuple or list of tuples): (transit time, period, eccentricity, omega, keplerian semi-amplitude) per planet
        inst_params (dict): a dictionary of lists holding the gamma, jitter, and GP amplitudes for each instrument present in the data
        kernel (object): describes the covariance between two points. If none given, uses the defaultKernel
    """

    def __init__(
        self,
        training_data,
        test_data,
        orbit_params,
        inst_params,
        kernel=kernel.defaultKernel(),
    ):
        self.orbit_params = orbit_params
        self.inst_params = inst_params
        self.kernel = kernel

        ## check training_data for proper input type and requirements
        if not isinstance(training_data, pd.DataFrame):
            raise TypeError(
                "The input training data must be organized as a pandas DataFrame."
            )

        for column in ["times", "rv", "err", "inst"]:
            if column not in training_data:
                raise ValueError(
                    f"The input dataframe does not have the required columns needed. The training data is missing a(n) {column} column."
                )

        self.training_times = training_data["times"].to_numpy()
        self.training_rvs = training_data["rv"].to_numpy()
        self.training_errors = training_data["err"].to_numpy()
        self.training_insts = training_data["inst"].to_numpy()

        ## check test_data for proper input type and requirements
        if not isinstance(test_data, pd.DataFrame):
            raise TypeError(
                "The input test data must be organized as a pandas DataFrame."
            )

        for column in ["times", "rv", "err", "inst"]:
            if column not in test_data:
                raise ValueError(
                    f"The input dataframe does not have the required columns needed. The test data is missing a(n) {column} column."
                )

        self.test_times = test_data["times"].to_numpy()
        self.test_rvs = test_data["rv"].to_numpy()
        self.test_errors = test_data["err"].to_numpy()
        self.test_insts = test_data["inst"].to_numpy()

    def calc_total_planetary_signal(self, times):
        """
        Computes the combined radial velocity signals of all transiting exoplanets in the star system

        Args:
            times (np.array): set of times to calculate the planetary signal for

        Returns:
            np.array: total radial velociy signal of the system at given times
        """
        ## check for proper input type
        if (not isinstance(self.orbit_params, list)) and (
            not isinstance(self.orbit_params, tuple)
        ):
            raise TypeError(
                "Your orbital parameters are not given in the correct format. A single planet's parameters must be given in a tuple, and multiple planets should have their parameters given as a list of parameter tuples per planet."
            )

        ## check for multiple planets
        if isinstance(self.orbit_params, list):

            total_rv_signal = np.zeros(len(times))

            for planet_params in self.orbit_params:
                if not isinstance(planet_params, tuple):
                    raise TypeError(
                        f"One of your planets does not have its parameters given as a tuple: {planet_params}"
                    )

                if len(planet_params) != 5:
                    raise ValueError(
                        f"Your orbital parameters should have 5 values per tuple: (transit time, period, eccentricity, omega, keplerian semi-amplitude). One of your planet's tuples has {len(planet_params)} parameters given: {planet_params}"
                    )

                total_rv_signal += kepler.calc_keplerian_signal(times, *planet_params)
            return total_rv_signal

        ## for singular planets
        elif isinstance(self.orbit_params, tuple):
            if len(self.orbit_params) != 5:
                raise ValueError(
                    f"Your orbital parameters tuple should have 5 values: (transit time, period, eccentricity, omega, keplerian semi-amplitude). Your input orbit parameter tuple has {len(self.orbit_params)} parameters given: {self.orbit_params}"
                )
            return kepler.calc_keplerian_signal(times, *self.orbit_params)

    def GP_predict(self, times_predict, include_keplerian=True, inst=None):
        """
        Performs Gaussian Process regression conditioned on the training set of data to predict values at new points

        Args:
            times_predict (np.array): set of times for the GPR to predict values for
            include_keplerian (bool): whether or not to include planetary signal as mean function. Defaults to True.
            inst (str): which instrument parameters the GP is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.

        Returns:
            predictive_means, predictive_variances (np.arrays): mean and variance values for times you want the GP to predict
        """
        if inst is None:
            predict_gamma = self.inst_params[next(iter(self.inst_params))][0]

        else:
            if inst not in self.inst_params:
                raise ValueError(
                    "This instrument does not exist in your inst_params dictionary."
                )
            else:
                predict_gamma = self.inst_params[inst][0]

        gamma_list = []
        jitters_list = []
        for inst in self.training_insts:
            gamma_list.append(self.inst_params[inst][0])
            jitters_list.append(self.inst_params[inst][1])

        training_gamma = np.asarray(gamma_list)
        training_jitter = np.asanyarray(jitters_list)

        if include_keplerian:
            mean_function = (
                self.calc_total_planetary_signal(self.training_times) + training_gamma
            )
            mean_function_predict = (
                self.calc_total_planetary_signal(times_predict) + predict_gamma
            )
        else:
            mean_function = np.zeros_like(self.training_times) + training_gamma
            mean_function_predict = np.zeros_like(times_predict) + predict_gamma

        tot_train_error = np.sqrt((self.training_errors) ** 2 + (training_jitter) ** 2)

        K = self.kernel.calc_covmatrix(
            self.training_times, self.training_times
        ) + tot_train_error**2 * np.identity(len(self.training_times))

        K_star = self.kernel.calc_covmatrix(self.training_times, times_predict)

        K_doublestar = self.kernel.calc_covmatrix(times_predict, times_predict)

        predictive_means = mean_function_predict + np.linalg.multi_dot(
            [
                np.transpose(K_star),
                np.linalg.inv(K),
                (self.training_rvs - mean_function),
            ]
        )

        predictive_covariances = np.subtract(
            K_doublestar,
            np.linalg.multi_dot([np.transpose(K_star), np.linalg.inv(K), K_star]),
        )

        predictive_variances = np.diag(predictive_covariances)

        return predictive_means, predictive_variances

    def run_CV(self, include_keplerian=True, inst=None):
        """
        Args:
            include_keplerian (bool): whether or not to include planetary signal as mean function. Defaults to True.
            inst (str): which instrument parameters the GP is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.

        Returns:
            CV_means, CV_variances (np.arrays): mean and variance values of GP ran on total set of data
        """
        all_times = np.concatenate((self.training_times, self.test_times))
        CV_means, CV_variances = self.GP_predict(all_times, include_keplerian, inst)

        return CV_means, CV_variances

    def plot(self, times_predict, include_keplerian=True, inst=None):
        """
        Wrapper for plot.Plot

        Args:
            times_predict (np.array): set of times for the GPR to predict values for
            include_keplerian (bool): whether or not to include planetary signal as mean function in GP. Defaults to True.
            inst (str): which instrument parameters the GP is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.

        Returns:
            matplotlib.pyplot.Figure
        """
        return plot.plot_cv(self, times_predict, include_keplerian)

    def histogram(self, include_Gaussian=False, include_keplerian=True, inst=None):
        """
        Wrapper for plot.Histogram

        Args:
            include_Gaussian (bool): whether or not to fit and plot Gaussians to residuals. Defaults to False.
            include_keplerian (bool): whether or not to include planetary signal as mean function in CV. Defaults to True.
            inst (str): which instrument parameters the CV is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.

        Returns:
            matplotlib.pyplot.Figure
        """
        return plot.plot_histogram(self, include_Gaussian, include_keplerian, inst=inst)
