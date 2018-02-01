#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""
import random

from rl_glue import *  # Required for RL-Glue
RLGlue("maze_env", "dyna_agent")


import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    num_episodes = 50
    max_steps = 6000

    num_runs = 10

    time_steps = np.zeros(max_steps) #index is x axis and value is y axis

    v_over_runs = []
    output = 1  # need to change 1 or 2, change it at here
    print("Running code for part" + str(output))

    if output == 1: # part1
        average_v_over_runs = np.zeros((3, 51))
        for planningSize in range(0,3):

            RL_agent_message('ChangeStep') #advance the planning step
            v_over_runs = []
            for run in range(num_runs):

                  #random.seed(run*2)  # try out seeds 1, 8

                  np.random.seed(run)

                  print "run number: ", run
                  RL_init()
                  print "\n"

                  for episode in range(num_episodes): # 50 episodes
                    RL_episode(max_steps)

                  V = RL_agent_message('ValueFunction')# the output
                    # V is (n,) np.array
                  # for i in range(0, max_steps): #debug
                  #   print(V[i])
                  v_over_runs.append(V)
                  #print(V)
                  RL_cleanup()

            for i in range(2,50): #get average
                temp_avg = 0
                for j in range(0,num_runs):
                    temp_avg += v_over_runs[j][i]

                average_v_over_runs[planningSize][i] = float(temp_avg)/num_runs
            #print(average_v_over_runs[planningSize])

        x = np.arange(0,51)
        plt.show()

        for i in range(0,3):
            labelStr= 'run '+ str(i+1)
            plt.plot(x, average_v_over_runs[i], label=labelStr)
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.legend()
        plt.show()

    elif output == 2:  # part2
        x = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
        average_v_over_runs = np.zeros(6)

        for alphaSize in range(0,6):
            RL_agent_message('ChangeAlpha')  # change values to part2
            v_over_runs = []
            for run in range(num_runs):
                  # random.seed(run)  # try out seeds 1, 8
                  #
                  # np.random.seed(run)
                  print "run number: ", run
                  RL_init()
                  print "\n"
                  for episode in range(num_episodes): # 50 episodes
                    RL_episode(max_steps)

                  V = RL_agent_message('ValueFunction')# the output
                    # V is (n,) np.array
                  # for i in range(0, max_steps): #debug
                  #   print(V[i])
                  v_over_runs.append(V)
                  #print(V)
                  RL_cleanup()

            temp_avg = 0
            for i in range(0,num_runs):  #get average
                for j in range(2,51):
                    temp_avg += v_over_runs[i][j]
            #print(temp_avg)

            average_v_over_runs[alphaSize] = float(temp_avg)/(num_runs*50)

        plt.show()
        plt.xticks(x)
        plt.axis([0,1,0,60])
        plt.plot(x, average_v_over_runs,'-')
        plt.xlabel('Alpha')
        plt.ylabel('Average Steps per Episode')
        #plt.legend()
        plt.show()