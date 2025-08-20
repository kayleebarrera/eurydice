import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_prediction_and_residuals(
    cv_obj,
    colors=None,
    labels=None,
    axis_kwargs=None,
    legend_kwargs=None,
    ax1=None,
    ax2=None,
):
    """
    TODO: add doc string here
    """
    # default dicts
    default_colors = {"train": "#009aa6", "test": "#d82c24"}
    default_labels = {"train": "Training Set", "test": "Test Set"}
    default_axis_kwargs = {
        "xlabel": "Time",
        "ylabel": r"RV [$\mathrm{m\,s^{-1}}$]",
        "residual_ylabel": "RV Residual",
        "title": "GP Prediction",
        "xlabel_kwargs": {"fontsize": 12},
        "ylabel_kwargs": {"fontsize": 12},
        "residual_ylabel_kwargs": {"fontsize": 12},
        "title_kwargs": {"fontsize": 14},
        "xlim": None,
        "ylim": None,
        "ylim_residuals": None,
    }
    default_legend_kwargs = {"fontsize": 10, "loc": "best"}

    # user customization
    colors = {**default_colors, **(colors or {})}
    labels = {**default_labels, **(labels or {})}
    axis_kwargs = {**default_axis_kwargs, **(axis_kwargs or {})}
    legend_kwargs = {**default_legend_kwargs, **(legend_kwargs or {})}

    if axis_kwargs["xlim"] is not None:
        xmin, xmax = axis_kwargs["xlim"]
    else:
        times_all = np.concatenate(
            [cv_obj.training_data["times"], cv_obj.test_data["times"]]
        )
        xmin, xmax = np.min(times_all) - 10, np.max(times_all) + 10

    plot_times = np.linspace(xmin, xmax, int(1e4))

    if not cv_obj._cv_has_run:
        raise RuntimeError("You must run 'run_CV' before plotting!")

    # predictions panel
    pred_instrument = np.unique(cv_obj.test_data["inst"])[0]
    print("Plotting for", pred_instrument)

    plot_means, plot_variance = cv_obj.predict(plot_times, inst_predict=pred_instrument)
    pred_std_dev = np.sqrt(plot_variance)
    predictive_means_zero = plot_means - cv_obj.inst_params[pred_instrument][0]

    train = cv_obj.training_data
    test = cv_obj.test_data

    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(
            2, sharex=True, height_ratios=[2, 1], figsize=(8, 4)
        )
        fig.subplots_adjust(hspace=0)

    ax1.plot(
        plot_times, predictive_means_zero, color="k", label="Mean Prediction", lw=1.5
    )
    ax1.fill_between(
        plot_times,
        predictive_means_zero - pred_std_dev,
        predictive_means_zero + pred_std_dev,
        color="gray",
        alpha=0.3,
    )
    ax1.fill_between(
        plot_times,
        predictive_means_zero - 2 * pred_std_dev,
        predictive_means_zero + 2 * pred_std_dev,
        color="lightgray",
        alpha=0.5,
    )

    train_rv_zero = np.array(train["rv"])
    for inst in np.unique(train["inst"]):
        gamma = cv_obj.inst_params[inst][0]
        mask = train["inst"] == inst
        train_rv_zero[mask] -= gamma

    ax1.errorbar(
        train["times"],
        train_rv_zero,
        train["err"],
        fmt="o",
        color=colors["train"],
        label=labels["train"],
    )
    test_rv_zero = np.array(test["rv"])
    for inst in np.unique(test["inst"]):
        gamma = cv_obj.inst_params[inst][0]
        mask = test["inst"] == inst
        test_rv_zero[mask] -= gamma

    ax1.errorbar(
        test["times"],
        test_rv_zero,
        test["err"],
        fmt="^",
        color=colors["test"],
        label=labels["test"],
    )
    ax1.set_ylabel(axis_kwargs["ylabel"], **axis_kwargs.get("ylabel_kwargs", {}))
    if axis_kwargs.get("title"):
        ax1.set_title(axis_kwargs["title"], **axis_kwargs.get("title_kwargs", {}))

    ax1.set_xlim(xmin, xmax)
    if axis_kwargs.get("ylim") is not None:
        ax1.set_ylim(axis_kwargs["ylim"])

    ax1.legend(**legend_kwargs)

    # residuals panel
    results_df, _ = cv_obj.get_CV_results()

    CV_means = results_df["pred_mean"]

    train_resids = train["rv"].to_numpy() - CV_means[: len(train)].to_numpy()
    test_resids = test["rv"].to_numpy() - CV_means[len(train) :].to_numpy()

    ax2.axhline(0, ls="--", color="k", lw=0.8)
    ax2.errorbar(
        train["times"], train_resids, train["err"], fmt="o", color=colors["train"]
    )
    ax2.errorbar(test["times"], test_resids, test["err"], fmt="^", color=colors["test"])
    ax2.set_xlabel(axis_kwargs["xlabel"], **axis_kwargs.get("xlabel_kwargs", {}))
    ax2.set_ylabel(
        axis_kwargs["residual_ylabel"], **axis_kwargs.get("residual_ylabel_kwargs", {})
    )

    ax2.set_xlim(xmin, xmax)
    if axis_kwargs.get("ylim_residuals") is not None:
        ax2.set_ylim(axis_kwargs["ylim_residuals"])

    return ax1, ax2


def plot_histogram(
    cv_obj,
    include_Gaussian=False,
    colors=None,
    labels=None,
    axis_kwargs=None,
    legend_kwargs=None,
    ax=None,
):
    """
    TODO: add doc string here
    """
    # default styles
    default_colors = {"train": "#009aa6", "test": "#d82c24"}
    default_labels = {"train": "Training Set", "test": "Test Set"}
    default_axis_kwargs = {
        "xlabel": "Standardized Residuals",
        "ylabel": "Density",
        "xlabel_kwargs": {"fontsize": 12},
        "ylabel_kwargs": {"fontsize": 12},
    }
    default_legend_kwargs = {"fontsize": 10, "loc": "best"}

    # user customization
    colors = {**default_colors, **(colors or {})}
    labels = {**default_labels, **(labels or {})}
    axis_kwargs = {**default_axis_kwargs, **(axis_kwargs or {})}
    legend_kwargs = {**default_legend_kwargs, **(legend_kwargs or {})}

    if not cv_obj._cv_has_run:
        raise RuntimeError("You must run 'run_CV' before plotting!")

    # grab cv results
    results_df, residual_stats = cv_obj.get_CV_results()

    N_train = len(cv_obj.training_data)

    test_resids = results_df["normalized_residual"][N_train:]
    train_resids = results_df["normalized_residual"][:N_train]

    mu_train, std_train = (
        residual_stats["train_residual_mean"],
        residual_stats["train_residual_std"],
    )
    mu_test, std_test = (
        residual_stats["test_residual_mean"],
        residual_stats["test_residual_std"],
    )

    labels["train"] += f", $\sigma$ = {std_train:.2f}"
    labels["test"] += f", $\sigma$ = {std_test:.2f}"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.hist(
        train_resids,
        bins=25,
        range=(-5, 5),
        color=colors["train"],
        alpha=0.5,
        label=labels["train"],
        density=True,
    )

    ax.hist(
        test_resids,
        bins=25,
        range=(-5, 5),
        color=colors["test"],
        histtype="step",
        linewidth=1.5,
        label=labels["test"],
        density=True,
    )

    if include_Gaussian:
        x_vals = np.linspace(-5, 5, 500)

        ax.plot(
            x_vals,
            norm.pdf(x_vals, mu_train, std_train),
            color=colors["train"],
            linestyle="-",
            alpha=0.6,
        )
        ax.plot(
            x_vals,
            norm.pdf(x_vals, mu_test, std_test),
            color=colors["test"],
            linestyle="--",
            alpha=0.6,
        )

    ax.set_xlabel(axis_kwargs["xlabel"], **axis_kwargs.get("xlabel_kwargs", {}))
    ax.set_ylabel(axis_kwargs["ylabel"], **axis_kwargs.get("ylabel_kwargs", {}))

    ax.legend(**legend_kwargs)

    return fig
