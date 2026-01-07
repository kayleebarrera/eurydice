import pandas as pd
import numpy as np
import os

required_cols = ["times", "rv", "err", "inst"]

default_aliases = {
    "time": "times",
    "jd": "times",
    "bjd": "times",
    "rv_error": "err",
    "unc": "err",
    "error": "err",
}


def read_data(file_name, inst, column_map=None, delimiter=None):
    """
    Loads a radial velocity data file and returns a standardized DataFrame for eurydice's pipeline.

    Args:å
        file_name (str): Path to .csv or .txt files.
        inst(str): Name of instrument used to take data
        column_map (dict, optional): Optional mapping from custom column names to standard names.
        delimiter (str, optional): Custom delimiter

    Returns:
        pd.DataFrame: Standarized combined dataframe with ['times', 'rv', 'err', 'inst'] columns.

    Raises:
        ValueError: If required columns are missing.

    Note:
        Assumes the input file contains data from a single instrument.
    """

    all_data = []

    ext = os.path.splitext(file_name)[-1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Unsupported file extension for {file_name}")

    df = pd.read_csv(file_name, delimiter=delimiter, comment="#")
    df.columns = df.columns.str.lower()  # make case sensitive
    all_aliases = default_aliases.copy()

    if column_map:
        all_aliases.update({k.lower(): v.lower() for k, v in column_map.items()})

    # rename columns to standardized eurydice labels
    df = df.rename(columns={k: v for k, v in all_aliases.items() if k in df.columns})

    df["inst"] = inst

    # check for missing required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{file_name} is missing required columns: {missing_cols}")

    # check for NaNs in required columns
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"{file_name} contains missing values in column '{col}'")

    all_data.append(df[required_cols])

    return pd.concat(all_data, ignore_index=True)


def split(data, train_split, random=False):
    """
    Splits a DataFrame into training and test sets for cross-validation.

    Args:
        data (pd.DataFrame): a DataFrame organizing radial velocity data
        train_split (float): Fraction of data to use for training (0 to 1, inclusive)
        random (bool): If True, randomly selects training points.
                        If False (default), takes first fraction of points.

    Returns:
        (pd.DataFrame, pd.DataFrame): training_data, test_data
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "The input data must be organized as a pandas DataFrame. Try read_data()."
        )

    if (train_split < 0) or (train_split > 1):
        raise ValueError(
            f"The train_split value must be between 0 and 1 (inclusive). Your input train_split value was {train_split}"
        )

    n_data = len(data)

    # decide which indices are in training
    if random:
        training_mask = np.random.choice(
            n_data, size=int(train_split * n_data), replace=False
        )

    else:
        training_mask = np.arange(int(train_split * n_data))

    # takes remaining as test
    test_mask = np.setdiff1d(np.arange(n_data), training_mask)

    training_data = data.iloc[training_mask]
    training_data.reset_index(drop=True, inplace=True)

    test_data = data.iloc[test_mask]
    test_data.reset_index(drop=True, inplace=True)

    return training_data, test_data
