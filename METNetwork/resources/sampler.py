"""
The sampler class for the METNet
"""

import numpy as np

from pathlib import Path
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter


class Sampler:
    """
    A class which calculates weight values for a sample given their location in a
    histogram.

    The weights are calculated as the reciprocal of the histogram
    height with possible linear shifts in their values.

    Once calculated these weights are used in two ways:
        - The smallest weights are used in rejection sampling
        - The largest weights are returned to be used in the loss function

    Weights are renormalised such that no matter the threshold or sampling, the average
    number of weights in the returned batch is always approximately equal to 1
    """

    def __init__(
        self,
        path: Path,
        weight_type: str = "mag",
        weight_from: float = 0.05,
        weight_to: float = 0,
        weight_shift: float = 0,
        weight_ratio: float = 0,
    ):
        """
        args:
            path: Path to the histogram file from which to calculate weights
        kwargs:
            weight_type: Either 'mag' or 'trg', the type of histogram used
            weight_from: The min x-value in the histogram which will receive a weight
            weight_to: The max x-value in the histogram which will receive a weight
                       Essentially flattening up to a this point
            weight_shift: Gradient of a linear shift applied to all weights
            weight_ratio: Allowed ratio between return weights
                          Defines the sampling vs loss threshold
                          0 = loss, 1 = sampl
        """

        print("Creating the sampler class")

        ## Class attributes
        self.weight_type = weight_type
        self.weight_from = weight_from
        self.weight_to = weight_to
        self.weight_shift = weight_shift
        self.weight_ratio = weight_ratio

        ## Make sure weight from and to are in the right order:
        assert weight_from < weight_to

        ## Load the two dimensional histogram of normalised targets
        if weight_type == "trg":

            ## weight_to shouldnt be bigger than three for target space
            if weight_to > 3:
                print("- weight_to is too large ({weight_to}) for target space")
                print("- reducing to 3")
                weight_to = 3

            ## Load the histogram and extract the bin definitions
            source = np.loadtxt(Path(path, "TrgDist.csv"))
            mid_x = source[0]
            mid_y = source[1]
            mid_r = np.linalg.norm(np.dstack(np.meshgrid(mid_x, mid_y)), axis=-1)

            ## Bin width, used to normalise for expectation values
            bin_nrm = (mid_x[1] - mid_x[0]) * (mid_y[1] - mid_y[0])

            ## Get the histogram and applying a gaussian filter (tested 0.1 works well)
            hist = gaussian_filter(source[2:], 0.1)

        ## Load the one dimensional histogram of raw magnitude
        if weight_type == "mag":
            weight_from *= 1e5 ## weight from and to is given in hundred GeV for mag
            weight_to *= 1e5

            ## Load the histogram and extract the bin definitions
            source = np.loadtxt(Path(path, "MagDist.csv"), delimiter=",", skiprows=1)
            mid_x = source[:, 0]
            mid_r = mid_x

            ## Bin width, used to normalise for expectation values
            bin_nrm = mid_x[1] - mid_x[0]

            ## Use the histogram without any smoothing
            hist = source[:, 1]

        ## Calculate the weights using the reciprocal of the histogram
        weights = 1.0 / (hist + 1e-8)

        ## Calculate the max weight allowed by looking at the weight to location
        x_bin = (np.abs(mid_x - weight_to)).argmin()
        if weight_type == "trg":
            y_bin = (np.abs(mid_y)).argmin()  ## Use zero in y
            max_weight = weights[y_bin, x_bin]
        if weight_type == "mag":
            max_weight = weights[x_bin]

        ## Clip the weights according to the max
        weights = np.clip(weights, None, max_weight)

        ## Make all weights for bins with r value smaller than weight from the same
        if weight_type == "trg":
            x_bin = (np.abs(mid_x - weight_from)).argmin()
            y_bin = (np.abs(mid_y)).argmin()
            val = weights[y_bin, x_bin]
        if weight_type == "mag":
            x_bin = (np.abs(mid_x - weight_from)).argmin()
            val = weights[x_bin]
        weights[mid_r < weight_from] = val

        ## Modify the weights using a linear shift
        if weight_shift != 0:

            ## Calculate the line passing through 0.5 at mid and grad bounded by -1, 1
            m = weight_shift / weight_to
            c = (1 - m * weight_to) / 2
            shft = np.clip(m * mid_r + c, 0.0, 1)

            ## Multiply the weights by this shift
            weights = weights * shft

        ## Calculate the weight treshold (v rndm vs loss ^) using the desired ratio
        thresh = np.max(weights) * weight_ratio

        ## The two weight types, clipped from above and below
        rndm_weights = np.clip(weights, None, thresh)
        loss_weights = np.clip(weights, thresh, None)

        ## Normalise the rndm weights as long as the threshold is reachable
        rndm_weights = (
            rndm_weights / np.sum(rndm_weights * hist * bin_nrm) if thresh else 1
        )

        ## Get the normalisation factor for the loss weights
        norm_fac = np.sum(loss_weights * rndm_weights * hist * bin_nrm)

        ## Apply the normalistion factor to the original weights
        weights /= norm_fac
        thresh /= norm_fac

        ## Derive the functions which calculate weights given x (and y) values
        if weight_type == "trg":
            self.func = RectBivariateSpline(mid_x, mid_y, weights)
        if weight_type == "mag":
            self.func = interp1d(
                mid_x,
                weights,
                kind="linear",  ## Cubic is overkill
                bounds_error=False,
                fill_value=(weights[0], weights[-1]),
            )

        ## Save the normalised weight threshold
        self.thresh = thresh

    def apply(self, batch: np.ndarray):
        """Passes a batch through the weight function
        args:
            batch: Truth values for weighting Et, Ex, Ey
        """

        if self.weight_type == "trg":
            trg_x = batch[:, 1]
            trg_y = batch[:, 2]
            weights = self.func.ev(trg_x, trg_y).astype(np.float32)

        elif self.weight_type == "mag":
            true_mag = batch[:, 0]
            weights = self.func(true_mag).astype(np.float32)

        ## Apply the downsampling if required
        if self.thresh > 0:

            ## Get the weight divided by the threshold
            wd = weights / self.thresh

            ## Test if the weights are smaller than the threshold or a random number
            t_mask = wd < 1
            z_mask = wd < np.random.random_sample(len(batch))

            ## If smaller then the threshold, make equal to the threshold
            weights[t_mask] = self.thresh

            ## If smaller than the random numbers, zero out
            weights[z_mask] = 0

        return weights
