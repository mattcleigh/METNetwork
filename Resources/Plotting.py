import sys
home_env = '../'
sys.path.append(home_env)

import numpy as np
import matplotlib.pyplot as plt

class loss_plot(object):
    def __init__(self, title = ""):

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)
        self.fig.suptitle(title)

        self.trn_line, = self.ax.plot( [], "-r", label="Train" )
        self.tst_line, = self.ax.plot( [], "-g", label="Test" )

    def update(self, trn_data, tst_data):

        x_data = np.arange(len(trn_data))

        self.trn_line.set_data( x_data, trn_data )
        self.tst_line.set_data( x_data, tst_data )

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
