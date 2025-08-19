import numpy as np
import pandas as pd
import eurydice.kepler as kepler
import scipy
from scipy.stats import norm
from eurydice.kernel import Kernel

# import eurydice.plot as plot
import plot
import matplotlib.pyplot as plt


class CrossValidation:
    """
    A class for running and analyzing Gaussian Process-based cross validation on radial velocity data.

        Args:
        training_data (pd.DataFrame): Contains 'times', 'rv', 'err', 'inst' columns for conditioning the GP.
        test_data (pd.DataFrame): Contains 'times', 'rv', 'err', 'inst' columns for testing in cross-validation.
        inst_params (dict): Dictionary of {instrument: [gamma, jitter]}.
        kernel (eurydice.kernel object): Kernel object used for computing the GP covariance matrices.
        include_keplerian (bool): Whether or not to use the Keplerian signals of the planet(s) as your GP's mean function. Defaults to False.
        orbit_params (dict of dicts, optional): Dictionary of per-planet parameters.
            If include_keplerian is True, this must be included.
            Each planet's parameters must include keys: 'T0', 'P', 'e', 'omega', 'K'.
    """

    def __init__(
        self,
        training_data,
        test_data,
        inst_params,
        kernel,
        include_keplerian=False,
        orbit_params=None,
    ):
        # check data provided in correct format
        self._validate_dataframe(training_data, "training_data")
        self._validate_dataframe(test_data, "test_data")

        self.training_data = training_data.copy()
        self.test_data = test_data.copy()

        # check that all instrumental parameters included for both sets
        all_instruments = set(self.training_data["inst"]).union(
            set(self.test_data["inst"])
        )
        missing_instruments = all_instruments - set(inst_params.keys())
        if missing_instruments:
            raise ValueError(
                f"inst_params missing parameters for instrument(s): {missing_instruments}"
            )
        self.inst_params = inst_params

        # verify kernel is a eurydice kernel
        if not isinstance(kernel, Kernel) or not kernel.__class__.__module__.startswith(
            "eurydice.kernels"
        ):
            raise TypeError("Kernel must inherit from eurydice.kernels.Kernel.")

        self.kernel = kernel

        # check for Keplerian and orbital parameters
        self.include_keplerian = include_keplerian

        if include_keplerian and orbit_params is None:
            raise ValueError("orbit_params must be specified to include Keplerian.")

        if orbit_params is not None:
            if not isinstance(orbit_params, dict):
                raise TypeError(
                    "orbit_params must be a dictionary with one entry per planet."
                )

            for planet, params in orbit_params.items():
                if not isinstance(params, dict):
                    raise TypeError(
                        f"Orbit parameters for {planet} must be organized in a dictionary."
                    )

                required_keys = {"T0", "P", "e", "omega", "K"}
                if set(params.keys()) != required_keys:
                    raise ValueError(
                        f"Orbit parameters for {planet} must include these keys: {required_keys}"
                    )
        self.orbit_params = orbit_params

        # flags
        self._gp_has_conditioned = False
        self._cv_has_run = False

    def _validate_dataframe(self, df, name):
        """
        Method to validate input DataFrame format.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If required columns are missing.
        """
        required_cols = ["times", "rv", "err", "inst"]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        missing = [col for col in required_cols if col not in df]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    def __repr__(self):
        planets = list(self.orbit_params.keys()) if self.orbit_params else []
        kernel_name = self.kernel.name

        return (
            f"<CrossValidation | "
            f"Training: {len(self.training_data)} pts, "
            f"Test: {len(self.test_data)} pts, "
            f"Instruments: {list(self.inst_params.keys())}, "
            f"Include Keplerian?: {self.include_keplerian}, "
            f"Planets: {len(planets)} ({', '.join(planets) if planets else 'none'}), "
            f"Kernel: {kernel_name}>"
        )

    def calc_total_planetary_signal(self, times):
        """
        Computes the combined radial velocity signal of all exoplanets provided in orbit_params.

        Args:
            times (np.array): Times to calculate the planetary signal at

        Returns:
            (np.array): Total radial velociy signal of the system at given times
        """
        if self.orbit_params is None:
            raise RuntimeError(
                "Orbit parameters must be provided in intialization to compute planetary signal."
            )

        total_signal = np.zeros_like(times)

        # add signal per planet
        for params in self.orbit_params.values():
            total_signal += kepler.calc_keplerian_signal(
                times,
                params["T0"],
                params["P"],
                params["e"],
                params["omega"],
                params["K"],
            )

        return total_signal

    def condition(self):
        """
        Conditions the Gaussian Process using the set of training data provided.
        """
        times = self.training_data["times"].to_numpy()
        rvs = self.training_data["rv"].to_numpy()
        errors = self.training_data["err"].to_numpy()
        insts = self.training_data["inst"].to_numpy()

        # grab instrument specific parameters for each training point
        gammas = np.array([self.inst_params[inst][0] for inst in insts])
        jitters = np.array([self.inst_params[inst][1] for inst in insts])

        mean = (
            self.calc_total_planetary_signal(times) + gammas
            if self.include_keplerian
            else gammas
        )

        total_err = np.sqrt(errors**2 + jitters**2)

        K = self.kernel.calc_covmatrix(
            times, times, errors=total_err, inst1=insts, inst2=insts
        )  # kernel may or may not use instrumental info, depending if the kernel is instrumental aware

        residual = rvs - mean

        # attempt to calculate using cholseky decomp, revert to manual code if not
        try:
            L = np.linalg.cholesky(K)
            self._gp_conditioning = {
                "K": K,
                "L": L,
                "residual": residual,
            }
            self._gp_has_conditioned = True
        except np.linalg.LinAlgError:
            print("Cholesky failed. Falling back to matrix inverse.")
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                raise RuntimeError("Inverse could not be computed.")

            self._gp_conditioning = {
                "K": K,
                "K_inv": K_inv,
                "residual": residual,
            }
            self._gp_has_conditioned = True

    def predict(self, times_predict, inst_predict):
        """
        Performs Gaussian Process regression conditioned on the training set of data to predict values at new points.

        Args:
            times_predict (np.array): Set of times for the GP to predict values at.
            inst_predict (str or np.array): Instrument(s) corresponding to the data you'd like to predict for.
                If a string is provided, uses the same instrument to predict at all times.
                If an array is passed, it must be the same length as times_predict.

        Returns:
            (np.array, np.array): predictive means, predictive variances
        """
        if not self._gp_has_conditioned:
            raise RuntimeError("You must run 'condition' before predicting!")

        if isinstance(inst_predict, str):
            inst_predict = np.array([inst_predict] * len(times_predict))
        else:
            inst_predict = np.array(inst_predict)

        if len(times_predict) != len(inst_predict):
            raise ValueError("times_predict and inst_predict must be the same length.")

        # checks if the GP is trying to predict for an instrument that has no defined instrumental parameters
        unique_insts = np.unique(inst_predict)
        missing = set(unique_insts) - set(self.inst_params.keys())
        if missing:
            raise ValueError(f"Instrument(s) {missing} not found in inst_params.")

        gammas_predict = np.array([self.inst_params[inst][0] for inst in inst_predict])

        if self.include_keplerian:
            mean_predict = (
                self.calc_total_planetary_signal(times_predict) + gammas_predict
            )
        else:
            mean_predict = gammas_predict

        K_star = self.kernel.calc_covmatrix(
            self.training_data["times"].to_numpy(),
            times_predict,
            errors=0.0,
            inst1=self.training_data["inst"].to_numpy(),
            inst2=inst_predict,
        )

        K_doublestar = self.kernel.calc_covmatrix(
            times_predict,
            times_predict,
            errors=0.0,
            inst1=inst_predict,
            inst2=inst_predict,
        )

        residual = self._gp_conditioning["residual"]
        if "L" in self._gp_conditioning:  # calculate using cholesky decomp
            L = self._gp_conditioning["L"]
            y = scipy.linalg.solve_triangular(L, residual, lower=True)
            alpha = scipy.linalg.solve_triangular(L.T, y, lower=False)
            predictive_means = mean_predict + np.transpose(K_star) @ alpha

            v = scipy.linalg.solve_triangular(L, K_star, lower=True)
            predictive_covariances = K_doublestar - np.transpose(v) @ v
        else:  # if cholseky wasnt computed in condition, try manual
            K_inv = self._gp_conditioning["K_inv"]
            predictive_means = mean_predict + np.transpose(K_star) @ (K_inv @ residual)
            predictive_covariances = K_doublestar - np.transpose(K_star) @ (
                K_inv @ K_star
            )

        predictive_variances = np.diag(predictive_covariances)

        return predictive_means, predictive_variances

    def run_CV(self):
        """
        Run cross-validation by determining how well the model conditioned on the training set predicts the values of the test set.
        """
        if not self.gp_has_conditioned:
            self.condition()

        all_times = np.concatenate(
            [
                self.training_data["times"].to_numpy(),
                self.test_data["times"].to_numpy(),
            ]
        )

        all_insts = np.concatenate(
            [
                self.training_data["inst"].to_numpy(),
                self.test_data["inst"].to_numpy(),
            ]
        )

        CV_means, CV_variances = self.predict(
            times_predict=all_times,
            inst_predict=all_insts,
        )

        # calculate normalized residuals per data point

        N_train = len(self.training_data["times"])

        test_resid = (self.test_data["rv"] - CV_means[N_train:]) / np.sqrt(
            CV_variances[N_train:] + self.test_data["err"] ** 2
        )
        train_resid = (self.training_data["rv"] - CV_means[:N_train]) / np.sqrt(
            CV_variances[:N_train] + self.training_data["err"] ** 2
        )

        all_resids = np.concatenate([train_resid, test_resid])

        # fit normal distributions to residuals
        mu_train, std_train = norm.fit(train_resid)
        mu_test, std_test = norm.fit(test_resid)

        # dataframe with detailed per-point info
        results_df = pd.DataFrame(
            {
                "time": all_times,
                "inst": all_insts,
                "pred_mean": CV_means,
                "pred_variance": CV_variances,
                "normalized_residual": all_resids,
                "data_split": ["train"] * N_train
                + ["test"] * (len(all_times) - N_train),
            }
        )

        self._CV_results = {
            "detailed_results": results_df,
            "train_residual_mean": mu_train,
            "train_residual_std": std_train,
            "test_residual_mean": mu_test,
            "test_residual_std": std_test,
        }

        self._cv_has_run = True

    def get_CV_results(self):
        """
        Get cross-validation results.

        Returns:
            tuple: (results_df, residual_stats) where
                results_df (pd.DataFrame): Detailed results DataFrame.
                residual_stats (dict): {
                    "train_residual_mean": float,
                    "train_residual_std": float,
                    "test_residual_mean": float,
                    "test_residual_std": float,
                }

        Raises:
            RuntimeError: If 'run_CV' hasn't been called yet.
        """
        if not self.cv_has_run:
            raise RuntimeError("'run_CV' hasn't been called yet!")

        results_df = self._CV_results.get("detailed_results_df")
        residual_stats = {
            "train_residual_mean": self._CV_results.get("train_residual_mean"),
            "train_residual_std": self._CV_results.get("train_residual_std"),
            "test_residual_mean": self._CV_results.get("test_residual_mean"),
            "test_residual_std": self._CV_results.get("test_residual_std"),
        }

        return results_df, residual_stats

    def plot_CV(
        self,
        include_Gaussian=False,
        save_path=None,
        colors=None,
        labels=None,
        prediction_plot_axis_kwargs=None,
        histogram_axis_kwargs=None,
        legend_kwargs=None,
    ):
        """
        Plot CV results in a multi-panel plot with prediction + residuals on the left and histogram on the right.

        Args:
            include_Gaussian (bool): If True, overlay Gaussian curve fits on histogram. Defaults to False.
            save_path (str or Path, optional): If provided, saves the figure to given path.
            colors (dict, optional): Custom color mapping for 'train' and 'test'.
            labels (dict, optional): Custom label mapping for 'train' and 'test'.
            prediction_plot_axis_kwargs (dict, optional): Axis settings for prediction + residual panel.
            histogram_axis_kwargs (dict, optional): Axis settings for histogram.
            legend_kwargs (dict, optional): Custom legend settings.

        Returns:
            (matplotlib.figure.Figure)
        """

        fig = plt.figure(figsize=(15, 7))
        gs = plt.GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], figure=fig)

        ax_pred = fig.add_subplot(gs[0, 0])
        ax_resid = fig.add_subplot(gs[1, 0], sharex=ax_pred)
        ax_hist = fig.add_subplot(gs[:, 1])

        plot.plot_prediction_and_residuals(
            self,
            colors=colors,
            labels=labels,
            axis_kwargs=prediction_plot_axis_kwargs,
            legend_kwargs=legend_kwargs,
            ax1=ax_pred,
            ax2=ax_resid,
        )

        plot.plot_histogram(
            self,
            include_Gaussian=include_Gaussian,
            colors=colors,
            labels=labels,
            axis_kwargs=histogram_axis_kwargs,
            legend_kwargs=legend_kwargs,
            ax=ax_hist,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig
