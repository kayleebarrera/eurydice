import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist


class Kernel(ABC):
    """
    An abstract base class to set up any kernel function to use in CrossValidation object. All kernel objects inherit from this class.

    Note:
        To implement your own custom kernel, create a class that inherits from this class.
    """

    @abstractmethod
    def name(self):
        pass

    def calc_covmatrix(self, x1, x2, errors, **kwargs):
        """
        Calculate the full covariance matrix between x1 and x2

        kwargs can include inst1, inst2 for handling multi-instrument kernels.
        """
        pass


class SqExp(Kernel):
    """
    The squared exponential kernel
    .
    An element, :math:`C_{ij}`, of the squared exoponential kernel matrix is:

    .. math::

        C_{ij} = \\eta_{amp}^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_{length}^2 } )

    Args:
        hyperparams (dict): dictionary of the GP hyperparameters
            of this kernel containing ['length*'] and ['amp*'] terms.
    """

    @property
    def name(self):
        return "SquaredExponential"

    def __init__(self, hyperparams):
        self.covmatrix = None
        self.hparams = {}

        assert (
            len(hyperparams) == 2
        ), "SqExp requires exactly 2 hyperparameters with names 'length*' and 'amp*'."

        for key, val in hyperparams.items():
            if key.startswith("length"):
                self.hparams["length"] = val
            elif key.startswith("amp"):
                self.hparams["amp"] = val

        if "length" not in self.hparams or "amp" not in self.hparams:
            raise ValueError(
                "SqExpKernel requires hyperparameters 'length*' and 'amp*'."
            )

    def __repr__(self):
        length = self.hparams["length"]
        amp = self.hparams["amp"]
        return "SquaredExponential Kernel with length: {}, amp: {}".format(length, amp)

    def calc_covmatrix(self, x1, x2, errors=0.0, **kwargs):
        """
        Compute the full covariance matrix between x1 and x2

        Args:
            x1, x2 (np.arrays): Points to calculate covariance matrix at.
            errors (float or np.array): If the covariance matrix is non-square (x1 and x2 are not the same length),
                this must be set to zero. Otherwise, can add errors to the matrix diagonal.

        Returns:
            (np.array): Covariance matrix of size (len(x1), len(x2))
        """

        amp = self.hparams["amp"]
        length = self.hparams["length"]

        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        dist = cdist(X1, X2, "sqeuclidean")

        K = amp**2 * np.exp(-dist / (length**2))

        if K.shape[0] == K.shape[1]:
            K += (errors**2) * np.eye(K.shape[0])

        self.covmatrix = K
        return K


class QuasiPer(Kernel):
    """
    The quasi periodic kernel
    .
    An element, :math:`C_{ij}`, of the quasi periodic kernel matrix is:

    .. math::

        C_{ij} = \\eta_{amp}^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_{explength}^2 } -
                 \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_{per} } ) }{ 2\\eta_{perlength}^2 } )

    Args:
        hyperparams (dict): dictionary of the GP hyperparameters
            of this kernel containing ['amp*'], ['explength*'], ['per*'], and ['perlength*'] keys.
    """

    @property
    def name(self):
        return "QuasiPeriodic"

    def __init__(self, hyperparams):
        self.covmatrix = None
        self.hparams = {}

        assert (
            len(hyperparams) == 4
        ), "QuasiPer requires exactly 4 hyperparameters with names 'amp*', 'explength*', 'per*', and 'perlength*'."

        for key, val in hyperparams.items():
            if key.startswith("amp"):
                self.hparams["amp"] = val
            elif key.startswith("explength"):
                self.hparams["explength"] = val
            elif key.startswith("per") and not "length" in key:
                self.hparams["per"] = val
            elif key.startswith("perlength"):
                self.hparams["perlength"] = val

        for param in ["amp", "explength", "per", "perlength"]:
            if param not in self.hparams:
                raise ValueError(f"QuasiPer requires hyperparameter {param}.")

    def __repr__(self):
        amp = self.hparams["amp"]
        explength = self.hparams["explength"]
        per = self.hparams["per"]
        perlength = self.hparams["perlength"]
        return "QuasiPeriodic Kernel with amp: {}, explength: {}, per: {}, perlength: {}".format(
            amp, explength, per, perlength
        )

    def calc_covmatrix(self, x1, x2, errors=0.0, **kwargs):
        """
        Compute the full covariance matrix between x1 and x2

        Args:
            x1, x2 (np.arrays): Points to calculate covariance matrix at.
            errors (float or np.array): If the covariance matrix is non-square (x1 and x2 are not the same length),
                this must be set to zero. Otherwise, can add errors to the matrix diagonal.

        Returns:
            (np.array): Covariance matrix of size (len(x1), len(x2))
        """

        amp = self.hparams["amp"]
        explength = self.hparams["explength"]
        per = self.hparams["per"]
        perlength = self.hparams["perlength"]

        X1 = np.array([x1]).T
        X2 = np.array([x2]).T

        dist = cdist(X1, X2, "euclidean")
        sq_dist = cdist(X1, X2, "sqeuclidean")

        K = np.array(
            amp**2
            * np.exp(-sq_dist / (explength**2))
            * np.exp((-np.sin(np.pi * dist / per) ** 2.0) / (2.0 * perlength**2))
        )

        if K.shape[0] == K.shape[1]:
            K += (errors**2) * np.eye(K.shape[0])

        self.covmatrix = K
        return K
