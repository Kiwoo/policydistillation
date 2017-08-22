"""
Attempting to make this return a similar plot as in the GAIL paper, Figure 1,
and also to return a table with results. You need to supply a results file.
Example:

    python scripts/plot_results.py imitation_runs/classic/checkpoints/results.h5

Note that this will (unless we're using Humanoid) contain more than one task for
us to parse through.

(c) June 2017 by Daniel Seita
"""

import argparse
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from misc_util import header

# Some matplotilb options I like to use
plt.style.use('seaborn-darkgrid')
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 12
mew = 5
error_region_alpha = 0.25

# Meh, a bit of manual stuff.
task_to_name = {'cartpole':    'CartPole-v0',
                'mountaincar': 'MountainCar-v0',
                'hopper':      'Hopper-v1',
                'walker':      'Walker2d-v1',
                'ant':         'Ant-v1',
                'halfcheetah': 'HalfCheetah-v1',
                'humanoid':    'Humanoid-v1',
                'reacher':     'Reacher-v1'}
task_to_random = {'cartpole':     20.08,
                  'mountaincar': -200.0,
                  'hopper':        16.0,
                  'walker':         0.7,
                  'ant':          -39.4,
                  'halfcheetah': -283.4,
                  'humanoid':     127.0,
                  'reacher':      -44.1}
colors = {'red', 'blue', 'yellow', 'black'}

class Plot(object):
    def __init__(self):
        print "Plot function initiated"
    def _plot(result_h5file = None):
        if result_h5file == None:
            print "File name not selected"
            return
        with pd.HDFStore(result_h5file, 'r') as f:
            iter_log = f['iter_log'].values
            epis_log = f['epis_log'].values
            timestep_log = f['timestep_log'].values
            ret_mean_log = f['ret_mean_log'].values
            ret_std_log = f['ret_std_log'].values

            # print ret_mean_log

            iter_log = np.squeeze(iter_log)
            epis_log = np.squeeze(epis_log)
            timestep_log = np.squeeze(timestep_log)
            ret_mean_log = np.squeeze(ret_mean_log)
            ret_std_log = np.squeeze(ret_std_log)
            c = 'red'
            header("Check Dimension")
            header('iter_log : {}'.format(np.shape(iter_log)))
            header('ret_mean_log : {}'.format(np.shape(ret_mean_log)))

            fig = plt.figure(figsize=(10,8))
            plt.plot(timestep_log, ret_mean_log, '-', lw=lw, color=c,
                            markersize=ms, mew=mew, label="iter_mean_reward")
            plt.fill_between(timestep_log, 
                             ret_mean_log-ret_std_log,
                             ret_mean_log+ret_std_log, 
                             alpha=error_region_alpha, 
                             facecolor=c)
            plt.title("Test", fontsize=title_size)
            plt.xlabel("Number of Iteration", fontsize=ysize)
            plt.ylabel("Mean_Reward", fontsize=xsize)
            plt.legend(loc='lower right', ncol=2, prop={'size':legend_size})
            plt.tick_params(axis='x', labelsize=tick_size)
            plt.tick_params(axis='y', labelsize=tick_size)
            plt.tight_layout()
            plt.savefig("Test"+'.png')            
