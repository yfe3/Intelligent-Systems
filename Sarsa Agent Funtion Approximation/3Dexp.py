#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""
# this exp code contains the plot code
# use with the 3Dagent to produce 3D plot

from rl_glue import *  # Required for RL-Glue
from mpl_toolkits.mplot3d import Axes3D # Source:https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
import matplotlib.pyplot as plt
RLGlue("mountaincar", "3Dagent")

import numpy as np

if __name__ == "__main__": #modified for part3
    num_episodes = 1000
    num_runs = 1

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            RL_agent_message('ValueFunction')
    filename = 'value.npy'
    true_value = np.load(filename) # get data

# source; https://matplotlib.org/examples/mplot3d/wire3d_animation_demo.html
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.linspace(-1.2, 0.6, 50)
    ys = np.linspace(-0.07, 0.07, 50)
    X, Y = np.meshgrid(xs, ys)
    #ax.set_zlim(-10, 10)

    print true_value

    ax.plot_wireframe(X, Y,true_value )
    plt.show()