import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def update_avg( old, new, i ):
    return old + ( new - old ) / i

def print_grad_norm( model ):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print( total_norm )

class BiasFunction(object):
    def __init__(self, bias):
        self.k = (1-bias)**3
        self.ret_id = True if bias == 0 else False
        self.ret_0  = True if bias == 1 else False
    def apply(self, x):
        if self.ret_id: return x
        if self.ret_0: return 0*x
        return (x * self.k) / (x * self.k - x + 1 ) ## Convex shape


def chunk_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))


class Weight_Function(object):

    def __init__( self, hist_file, wto, wratio, lin_shift ):

        ## Get the hisogram data from file
        source = np.loadtxt( hist_file, delimiter="," )
        et_vals = source[:,0]
        et_hist = source[:,1]
        bin_w   = et_vals[1] - et_vals[0]
        coarse_weights = np.ones_like( et_vals )
        self.f = None
        self.thresh = None

        ## Generate weights using the reciprocal function up to a certain value GeV
        if wto > 0:

            ## Calculate maximum allowed value of the weights the falling edge of the plateau
            max_weight = 1 / et_hist[ (np.abs(et_vals - wto*1000)).argmin() ]

            ## Calculate the weights for each bin, accounting for the maximum
            coarse_weights = np.clip( 1 / et_hist, None, max_weight )

            ## DELETE THIS (MAYBE...)
            peak = et_vals[np.argmin(coarse_weights)]
            weight_at_peak = np.min(coarse_weights)
            coarse_weights = np.where( et_vals < peak, weight_at_peak, coarse_weights )

        ## Modify the coarse weights using a linear shift, make sure that m is scaled!
        if lin_shift != 0:

            ## Calculate a gradient corresponding to inputs between -1 and 1
            m = lin_shift / ( et_vals[-1] )

            ## Multiply the weights by the weights by a linear function passing through 0.5 in the mid
            coarse_weights *= np.clip( m * ( et_vals - et_vals[-1]/2 ) + 0.5, 0.0, 1)

        ## Calculate the weight treshold (v rndm vs loss ^) using the desired weight ratio
        thresh = np.max( coarse_weights ) * wratio

        ## The two weight types, clipped from above and below
        rndm_weights = np.clip( coarse_weights, None, thresh )
        loss_weights = np.clip( coarse_weights, thresh, None )

        ## Normalise the rndm weights as long as the threshold is reachable
        if thresh > 0: rndm_weights /= np.sum( rndm_weights * et_hist * bin_w )
        else:          rndm_weights  = 1

        ## Next we get the normalisation factor for the loss weights
        norm_fac = np.sum( loss_weights * rndm_weights * et_hist * bin_w )

        ## But we apply the normalistion factor to the origonal weights!
        coarse_weights /= norm_fac
        thresh /= norm_fac

        ## Can now derive the function using linear interpolation, we also save the threshold
        self.f = interp1d( et_vals,
                           coarse_weights,
                           kind="linear",
                           bounds_error=False,
                           fill_value=tuple(coarse_weights[[0,-1]]) )
        self.thresh = thresh


    def apply(self, true_et):
        return self.f( true_et )
