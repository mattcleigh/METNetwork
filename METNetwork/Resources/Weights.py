import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

def BiasFunction(x, maxx, bias):
    k = (1-bias)**3
    return 1 - x*k / (x*k-x+maxx)

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
        w_rat: Defines the threshold between the downsampling weights and the loss weights (v rndm vs loss ^)
        w_shf: A gradient of a linear shift applied to all weights

    """
    def __init__(self, folder, w_tp, w_to, w_shf, w_rat):

        ## Get weights from the 2D histogram or targets
        if w_tp == 'trg':
            w_to = min(w_to, 3) ## w_to shouldnt be bigger than three for target space
            source = np.loadtxt(Path(folder, 'TrgDist.csv'))
            mid_x = source[0]
            mid_y = source[1]
            mid_r = np.linalg.norm(np.dstack(np.meshgrid(mid_x, mid_y)), axis=-1)
            bin_nrm = (source[0][1] - source[0][0]) * (source[1][1] - source[1][0]) ## Bin width, used to normalise for expectation values
            hist = gaussian_filter(source[2:], 0.1)

        ## Get weights from the 1D histogram of true met magnitude
        elif w_tp == 'mag':
            w_to *= 1e5 ## w_to is in hundred GeV
            source = np.loadtxt(Path(folder, 'MagDist.csv'), delimiter=',', skiprows=1)
            mid_x = source[:,0]
            mid_r = mid_x
            bin_nrm = mid_x[1] - mid_x[0] ## Bin width, used to normalise for expectation values
            hist = source[:,1]

        ## Calculate maximum allowed value of the weights calculated at w_to
        x_bin = (np.abs(mid_x-w_to)).argmin()

        if w_tp == 'trg':
            y_bin = (np.abs(mid_y)).argmin()
            max_weight = 1.0 / hist[y_bin, x_bin]
        elif w_tp == 'mag':
            max_weight = 1.0 / hist[x_bin]

        ## Calculate the weights for each bin, clipping at the maximum
        weights = np.clip(1.0 / hist, None, max_weight)

        ## For magnitude scaling, dont provide massive weights to the first couple of bins!
        ## This is a hack solution to fix some errors we are getting for not scaling up the histogram correctly
        if w_tp == 'mag':
            weights[:5] = weights[5]

        ## Prevent the magnitude weights from disapearing at zero when using trg
        if w_tp == 'trg':
            r = 0.5
            x_bin = (np.abs(mid_x-r)).argmin()
            y_bin = (np.abs(mid_y)).argmin()
            val = weights[y_bin, x_bin]
            mask = (mid_r < r)
            weights[mask] = val

        ## Modify the weights using a linear shift or a bias function
        if w_shf != 0:
            # weights = BiasFunction(weights, np.max(weights), w_shf) ## Doesnt really work with sliced samples...
            m = w_shf / w_to ## Calculate a gradient corresponding to inputs between -1 and 1, preventing any negative weights!
            c = (1 - m*w_to) / 2 ## Multiply the weights by the weights by a linear function passing through 0.5 in the mid
            shft = np.clip(m * mid_r + c, 0.0, 1)
            weights = weights * shft

        ## Calculate the weight treshold (v rndm vs loss ^) using the desired weight ratio
        thresh = np.max(weights) * w_rat

        ## The two weight types, clipped from above and below
        rndm_weights = np.clip(weights, None, thresh)
        loss_weights = np.clip(weights, thresh, None)

        ## Normalise the rndm weights as long as the threshold is reachable
        rndm_weights = rndm_weights / np.sum(rndm_weights * hist * bin_nrm) if thresh else 1

        ## Get the normalisation factor for the loss weights
        norm_fac = np.sum(loss_weights * rndm_weights * hist * bin_nrm)

        ## Apply the normalistion factor to the original weights!
        weights /= norm_fac
        thresh  /= norm_fac

        ## Can now derive the function using linear interpolation
        if w_tp == 'trg':
            self.f = RectBivariateSpline(mid_x, mid_y, weights)

        elif w_tp == 'mag':
            self.f = interp1d( [0] + mid_x.tolist(),  ## Ensure that the histogram starts at zero
                               [0] + weights.tolist(),
                               kind='cubic', bounds_error=False, fill_value=(0, weights[-1]))

        ## Save the weight threshold, the weighting type and set the do_smpl to true by default
        self.thresh = thresh
        self.w_tp = w_tp

    def apply(self, batch):

        if self.w_tp == 'trg':
            trg_x = batch[:, 1]
            trg_y = batch[:, 2]
            weights = self.f.ev(trg_x, trg_y).astype(np.float32)

        elif self.w_tp == 'mag':
            true_mag = batch[:, 0]
            weights = self.f(true_mag).astype(np.float32)

        ## Apply the downsampling if required
        if self.thresh > 0:

            ## Get the weight divided by the threshold
            wd = weights / self.thresh

            ## Test if the weights are smaller than the threshold or a random number
            t_mask = (wd < 1)
            z_mask = (wd < np.random.random_sample(len(batch)))

            ## If smaller then the threshold, make equal to the threshold
            weights[t_mask] = self.thresh

            ## If smaller than the random numbers, zero out
            weights[z_mask] = 0

        return weights

        ## If the weight is greater then the threshold then we return it for the loss
        # if weight > self.thresh:
            # return weight
        # Otherwise we downsample using a random number and return the threshold
        # elif weight > self.thresh * random.random():
            # return self.thresh
        # If this too fails then we return 0, indicating that the event should be discarded
        # return 0
