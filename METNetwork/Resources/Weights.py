import numpy as np
from scipy.interpolate import interp1d

class SampleWeight:
    """
    A class which calculates weight values for a sample given their location
    in a histogram. The weights are primarily calculated as the reciprocal of the histogram height
    with possible linear shifts in their values.
    There weights are used in two ways:
        - The smallest weights are used for down sampling the data using random number generation
        - The largest weights are returned to be used for the loss function
    The weights are renormalised such that no matter the threshold or sampling, the average number of weights
    in the returned dataset is always approximately equal to 1
    args:
        hist_file: A csv containing the distribution histogram with which to perform the weighting
        w_to:  The maximum point in the histogram to apply the weights (flatten up to a point)
               Must be greater than zero!
        w_rat: Defines the threshold between the downsampling weights and the loss weights
        w_shf: A gradient of a linear shift applied to all weights

    """
    def __init__(self, hist_file, w_to, w_rat, w_shf):

        ## Convert w_to to GeV
        w_to *= 1000

        ## Get the hisogram data from file
        source = np.loadtxt(hist_file, delimiter=",")
        et_vals = source[:,0] ## These are the bin centers of the histogram
        et_hist = source[:,1] ## The height of the histogram
        bin_w = et_vals[1] - et_vals[0] ## Bin width, used for expectation values

        ## Calculate maximum allowed value of the weights calculated at w_to
        max_weight = 1 / et_hist[ (np.abs(et_vals - w_to)).argmin() ]

        ## Calculate the weights for each bin, clipping at the maximum
        weights = np.clip(1.0/et_hist, None, max_weight)

        ## DELETE THIS (MAYBE...)
        # peak = et_vals[np.argmin(weights)]
        # weight_at_peak = np.min(weights)
        # weights = np.where(et_vals < peak, weight_at_peak, weights)

        ## Modify the coarse weights using a linear shift, make sure that m is scaled!
        if w_shf:

            ## Calculate a gradient corresponding to inputs between -1 and 1, preventing any negative weights!
            m = lin_shift / w_to

            ## Multiply the weights by the weights by a linear function passing through 0.5 in the mid
            c = (1 - m*w_to) / 2
            weights *= np.clip(m*et_vals+c, 0.0, 1)

        ## Calculate the weight treshold (v rndm vs loss ^) using the desired weight ratio
        thresh = np.max(weights) * w_rat

        ## The two weight types, clipped from above and below
        rndm_weights = np.clip(weights, None, thresh)
        loss_weights = np.clip(weights, thresh, None)

        ## Normalise the rndm weights as long as the threshold is reachable
        rndm_weights = rndm_weights / np.sum(rndm_weights * et_hist * bin_w) if thresh else 1

        ## Get the normalisation factor for the loss weights
        norm_fac = np.sum(loss_weights * rndm_weights * et_hist * bin_w)

        ## Apply the normalistion factor to the original weights!
        weights /= norm_fac
        thresh  /= norm_fac

        ## Can now derive the function using linear interpolation, we also save the threshold
        self.f = interp1d(et_vals, weights, kind="linear", bounds_error=False, fill_value=tuple(weights[[0,-1]]))
        self.thresh = thresh

    def apply(self, true_et):

        ## Currently only supprts weights based on true magnitude
        true_mag = true_et[0]
        weight = self.f(true_mag)

        ## If the weight is greater then the threshold then we return it for the loss
        if weight > self.thresh:
            return weight

        ## Otherwise we downsample using a random number and return the threshold
        elif weight > self.thresh*random.random():
            return self.thresh

        ## If this too fails then we return 0, indicating that the event should be discarded
        return 0
