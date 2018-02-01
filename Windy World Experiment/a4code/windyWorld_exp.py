#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windyWorld_env", "windyWorld_agent")

import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 300
    max_steps = 8000

    num_runs = 50

    time_steps = np.zeros(max_steps) #index is x axis and value is y axis

    v_over_runs = []


    # dict to hold data for key episodes

    for run in range(num_runs):

      print "run number: ", run
      RL_init()
      print "\n"

      for episode in range(num_episodes):
        RL_episode(max_steps)

      V = RL_agent_message('ValueFunction')# the output
        # V is (n,) np.array
      # for i in range(0, max_steps): #debug
      #   print(V[i])
      v_over_runs.append(V)


      RL_cleanup()
      
    #n = v_over_runs[counter][0].shape[0]
    #n_valueFunc = len(key_episodes)
    average_v_over_runs = np.zeros(max_steps)
    for i in range(0,max_steps):
        temp_avg = 0
        for j in range(0,num_runs):
            temp_avg += v_over_runs[j][i]

        #print(temp_avg)
        average_v_over_runs[i] = float(temp_avg)/num_runs

        # each item in dict is a list (one item per run), and each item is a value function 
        #data = np.matrix(v_over_runs[episode])
        # therefore data is (num_runs x length of value fucntion)
        #average_v_over_runs[i] = np.mean(data, axis=0)
    for i in range(0,max_steps):
        print(average_v_over_runs[i])
    print("plot data on the excel sheet")

    #np.save("ValueFunction", average_v_over_runs)
