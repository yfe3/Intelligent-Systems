#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle


Q = np.zeros((100,100))  # 1-99, 99 different states

policy_table = np.zeros(100)  # same as above

return_table = np.zeros((101,101,2)) #s a count value
total_run= np.zeros((100,100))

action = 0

total_step = 0

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way


    global policy_table, return_table,total_run

    total_run = np.zeros((100,100))

    for i in range(1,100):
        policy_table[i] = min(i,100-i) #initialize policy
    print(policy_table)

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global policy_table, action, total_step, return_table, total_run
    return_table = np.zeros((101, 101, 2))

    action = policy_table[state]

    total_step = 1
    total_run[int(state)][int(action)]+=1

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q

    global action, return_table, policy_table, total_step


    return_table[int(state)][int(action)][0] += 1 #count +1

    total_run[int(state)][int(action)] += 1

    total_step+=1

    action = policy_table[state]



    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global return_table, policy_table, total_step, action,total_run

    if(reward == 1.0):
        #print("get reward")
        state = 100 - action -1
        total_step+=1
        return_table[int(state)][int(action)][0] += 1

        total_run[int(state)][int(action)] += 1

        for s in range(1,100):
            for a in range(1,99):
                if return_table[s][a][0] != 0:
                    return_table[s][a][1] = float(1)

                    Q[s][a] = Q[s][a] + (return_table[s][a][1] - Q[s][a])/total_run[s][a]

    else:
        state = action -1
        total_step+=1
        return_table[int(state)][int(action)][0] += 1
        total_run[int(state)][int(action)] += 1
        for s in range(1, 100):
            for a in range(1, 99):
                if return_table[s][a][0] != 0:
                    Q[s][a] = Q[s][a] + (return_table[s][a][1] - Q[s][a])/total_step / total_run[s][a]


    for s in range(1,99):
        temp_action = Q[s][1]
        for a in range(1,min(s,100-s+1)):
            if Q[s][a] >= temp_action:
                temp_action = Q[s][a]
                policy_table[s] = a

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q

    #print(np.max(Q, axis=1)) #debug


    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0) #1
    else:
        return "I don't know what to return!!"

