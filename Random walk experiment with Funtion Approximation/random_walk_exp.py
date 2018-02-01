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
RLGlue("random_walk_env", "tabular_agent")


import matplotlib.pyplot as plt
import numpy as np
from rndmwalk_policy_evaluation import compute_value_function
import math
import tiles3

if __name__ == "__main__":
    num_episodes = 5001
    max_steps = 15000

    num_runs = 10

    v_ture = compute_value_function() #calculate true value
    #print(v_ture)

    print("Running code for tabular agent ")

    v_over_runs = np.zeros(5001)  # inite to empty
    value_over_runs = np.zeros(1001)
    average_v_over_runs = np.zeros(5001) # finial output
    for run in range(num_runs):

        random.seed(run*2)  # try out seeds 1, 8

        np.random.seed(run)
    #
        print "run number: ", run
        RL_init()
        print "\n"
    #
        for episode in range(1,num_episodes): # 5000 episodes
            RL_episode(max_steps)
    #
            V = RL_agent_message('ValueFunction')# the output, w
    #                 # V is (n,) np.array
            temp_sum = 0
            for i in range(1,1000):
                s = np.zeros(1001)
                s[i]=1 # feature vector
                #get the value
                value_over_runs[i]=np.dot(V, s)

                temp_sum += np.power((v_ture[i] - np.dot(V,s)),2) #formula
            temp_sum = math.sqrt((float(temp_sum) / 1000))
            v_over_runs[episode] += temp_sum # get Vks for 1 run

        #print(V)
        RL_cleanup()
    #

    v_over_runs= v_over_runs/num_runs
    #print(len(value_over_runs))
    #print(value_over_runs)

    x = np.arange(0, 5001)
    #plt.show()

    plt.plot(x, v_over_runs,'b')

    plt.xlabel('Episode')
    plt.ylabel('RMSE')

    # end of exp with tabular agent


    # v_ture = compute_value_function() #calculate true value
    # print(v_ture)

    RLGlue("random_walk_env", "state_aggregation_agent")
    print("Running code for state aggregation agent ")

    v_over_runs = np.zeros(5001)  # inite to empty
    value_over_runs = np.zeros(1001)
    average_v_over_runs = np.zeros(5001)  # finial output

    for run in range(num_runs):

        random.seed(run * 2)  # try out seeds 1, 8

        np.random.seed(run)
        #
        print "run number: ", run
        RL_init()
        print "\n"
        #
        for episode in range(1, num_episodes):  # 5000 episodes
            RL_episode(max_steps)
            #
            V = RL_agent_message('ValueFunction')  # the output, w
            #                 # V is (n,) np.array
            temp_sum = 0
            for i in range(1, 1000):
                s = np.zeros(11)
                s[i//100] = 1  # feature vector
                # get the value
                value_over_runs[i] = np.dot(V, s)

                temp_sum += np.power((v_ture[i] - np.dot(V,s)),2) #formula
            temp_sum = math.sqrt((float(temp_sum) / 1000))
            v_over_runs[episode] += temp_sum # get Vks for 1 run

        # print(V)
        RL_cleanup()
        #

    v_over_runs= v_over_runs/num_runs
        # print(len(value_over_runs))
        # print(value_over_runs)

    x = np.arange(0, 5001)
        # plt.show()

    plt.plot(x, v_over_runs,'r')

    plt.xlabel('Episode')
    plt.ylabel('RMSE')

        # end of state aggregation agents



        # v_ture = compute_value_function() #calculate true value
        # print(v_ture)

    RLGlue("random_walk_env", "tile_agent")
    print("Running code for tile agent ")

    iht = tiles3.IHT(2000)

    tilings = 50

    v_over_runs = np.zeros(5001)  # inite to empty
    value_over_runs = np.zeros(1001)
    average_v_over_runs = np.zeros(5001) # finial output
    for run in range(num_runs):

        random.seed(run*2)  # try out seeds 1, 8

        np.random.seed(run)
    #
        print "run number: ", run
        RL_init()
        print "\n"
    #
        for episode in range(1,num_episodes): # 5000 episodes
            RL_episode(max_steps)
    #
            V = RL_agent_message('ValueFunction')# the output, w
    #                 # V is (n,) np.array
            temp_sum = 0
            for i in range(1,1000):
                new_feature = np.zeros(1200)
                feature_vect = np.zeros(1)
                feature_vect[0] = float(i)//200 #state
                new_feature_x = tiles3.tiles(iht, tilings, feature_vect)  # 1000/50 = 200, 200 states in 1 tile
                for k in new_feature_x:
                    new_feature[k] = 1

                #get the value
                value_over_runs[i]=np.dot(V, new_feature)

                temp_sum += np.power((v_ture[i] - np.dot(V,new_feature)),2) #formula
            temp_sum = math.sqrt((float(temp_sum) / 1000))
            v_over_runs[episode] += temp_sum # get Vks for 1 run

        #print(V)
        RL_cleanup()
    #

    v_over_runs= v_over_runs/num_runs
    #print(len(value_over_runs))
    #print(value_over_runs)

    x = np.arange(0, 5001)


    plt.plot(x, v_over_runs,'g')

    plt.xlabel('Episode')
    plt.ylabel('RMSE')
    plt.show()
    # end of exp with tabular agent