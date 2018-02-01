#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
  modified to actually fitting the requirements,
  modified by Yuan Feng (yfeng3)
"""

from rl_glue import *  # Required for RL-Glue

RLGlue("w1_env", "w1_agent")

import numpy as np
import sys


def save_results(data, data_size, filename):  # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))


def getOptimalAction():  #pass the test working properly
    return int(RL_env_message("get optimal action"))


if __name__ == "__main__":
    num_runs = 2000
    max_steps = 1000

    # array to store the results of each step
    global optimal_action
    optimal_action = np.zeros(max_steps)

    #optimal_action[5]=5
    #print (optimal_action[5]) #test code to determine the type and behaviour of optimal_action

    print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    for k in range(num_runs):
        RL_init()

        RL_start()
        for i in range(max_steps):
            # RL_step returns (reward, state, action, is_terminal); we need only the
            # action in this problem

            action = RL_step()[2]

            if int(action) == int(getOptimalAction()):  #the logic seems pass the test

                global optimal_action
                optimal_action[i] += 1.0

            # update your optimal action statistic here

        #optimal_action = np.sum(optimal_action)
        #print(optimal_action)
        RL_cleanup()
        #print ".", #comment out for faster excution
        sys.stdout.flush()
    #print(optimal_action)
    save_results(optimal_action / num_runs, max_steps, "RL_EXP_OUT.dat")
    print "\nDone"
