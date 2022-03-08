"""
A collection of good bin values for each sample and variable
"""

def get_bins(sample, x_var=None, y_var=None):

    ## Return nones as default
    left = None
    right = None
    top = None
    bot = None
    nbins = None

    ## Keep the act mu the same for all samples
    if x_var == "ActMu":
            left = 10
            right = 70
            nbins = 20

    ## Keep the Dlin the same for all samples
    if y_var == "DLin":
            bot = -0.2
            top = 0.6

    ## Per sample bounds
    if "ttbar" in sample:

        if x_var == "ET":
            left = 0
            right = 350
            nbins = 20

        if x_var == "Tight_Final_SumET":
            left = 200
            right = 1000
            nbins = 20

        if y_var == "RMSE":
            top = 75
            bot = 0

        if y_var == "Normalised Entries":
            top = 0.02
            bot = 0

    return left, right, top, bot, nbins
