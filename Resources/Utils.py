import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Weight_Function(object):

    def __init__( self, type, hist_file, max_weight_diff = 10 ):

        ## Get the hisogram data from file
        source = np.loadtxt( hist_file, delimiter="," )
        et_mu   = source[0, 1]
        et_vals = source[1:,0]
        et_hist = source[1:,1]
        bin_w   = et_vals[1] - et_vals[0]

        ## If no sampling at all return 1
        if type == "":
            self.f = lambda x: x
            self.max_weight = 1

        ## Generate weights using the reciprocal function, return the max possible weight for sampling
        elif type[0] == "i":

            ## Get the parameter from the type
            weight_to = float(type[1:])

            ## Calculate maximum allowed value of the weights the falling edge of the plat
            max_weight = 1 / et_hist[ (np.abs(et_vals - weight_to*1000)).argmin() ]

            ## Calculate the weights for each bin, accounting for the maximum
            coarse_weights = np.clip( 1 / et_hist, None, max_weight )

            ## Normalise the weights using their expectation value
            coarse_weights /= np.sum( coarse_weights * et_hist * bin_w )

            ## Turn the coarse weights into a function and recalculate the new max value used for sampling
            self.f = interp1d( et_vals, coarse_weights, kind="linear", bounds_error=False, fill_value=tuple(coarse_weights[[0,-1]]) )
            self.max_weight = np.max( coarse_weights )
            self.sweight = self.max_weight / max_weight_diff

        # ## Generate weights using a linear shift, return the max possible weight for sampling
        # elif type[0] == "m":
        #     m = float(type[1:]) / et_mu
        #     self.f = lambda x: m * ( x - et_mu ) + 1
        #     self.max_weight = max( self.f(et_bins)  ) ## The max weight is based on the limits of the fit
        #
        # elif type[0] == "b":
        #     ## Do the same for i
        #     weight_to = 100
        #     coarse_weights = 1 / et_hist / et_bins[-1]
        #     inertp = interp1d( et_bins, coarse_weights, kind="cubic", bounds_error=False, fill_value=tuple(coarse_weights[[0,-1]]) )
        #     self.max_weight = inertp( 1000 * weight_to ) ## The max weight is based on where we want to do the fit up to
        #     func = lambda x: np.where( inertp(x) < self.max_weight, inertp(x), self.max_weight )
        #
        #     ## Calculate expectation value of new histogram using sum approximate of integral
        #     et_mu = np.sum( et_bins * func(coarse_weights) ) * ( et_bins[2] - et_bins[1] )
        #
        #     ## Do same for m
        #     m = float(type[1:]) / et_mu
        #     self.f = lambda x: ( m * ( x - et_mu ) + 1 ) * func(x)
        #     self.max_weight = max( self.f(et_bins)  ) ## The max weight is based on the limits of the fit

    def apply(self, true_et):
        return self.f( true_et )
