#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""


import numpy as np
import pickle
import random
import tiles3

alpha = 0.1/8

w = 0

lambd = 0.9 # cannot named as lambda

oldS = 0

newS = 0

error = 0

z = 0

# tile code parameters

iht = tiles3.IHT(4096)


tilings = 8

action = 0

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    # initialize the policy array in a smart way


    global w, oldS, newS, iht,z, tilings


    w = np.zeros(4096)

    z = np.zeros(4096) # trace

    iht = tiles3.IHT(4096)

    for i in range(0,4096): # create weight vector
        w[i] = np.random.uniform(-0.001,0)






def agent_start(current_position):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global w,  oldS, newS, iht, action, z, error

    z = np.zeros(4096)

    # print("agent_stat")
    # print(current_position)

    oldS = current_position
    newS = oldS

    q_list = [] # a empty list
    for A in [0,1,2]: #three actions

        feature_vector = tiles3.tiles(iht, tilings, [8 * oldS[0]/(0.5+1.2), 8*oldS[1]/(0.07+0.07)], [A]) # on book formula

        true_feature_vector = np.zeros(4096)
        for i in feature_vector:
            true_feature_vector[i] = 1
        #print(len(feature_vector))

        q_list.append(np.dot(true_feature_vector, w)) #get q

    A_max = q_list.index(max(q_list)) # get action number

    action = A_max

    return action


def agent_step(reward, position):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global  w, newS, oldS, iht, action,z , error

    error = reward
    newS = position # S'

    feature_vector = tiles3.tiles(iht, tilings, [8 * oldS[0] / (0.5 + 1.2), 8 * oldS[1] / (0.07 + 0.07)],
                                  [action])  # on book formula


    for i in feature_vector:
        error = error - w[i]
        z[i] =  1 # replacing trace


    q_list = [] # a empty list
    for A in [0,1,2]: #three actions

        feature_vector = tiles3.tiles(iht, tilings, [8 * newS[0]/(0.5+1.2), 8*newS[1]/(0.07+0.07)], [A]) # on book formula

        true_feature_vector = np.zeros(4096)
        for i in feature_vector:
            true_feature_vector[i] = 1

        q_list.append(np.dot(true_feature_vector,w)) #get q

    #print(q_list)

    A_max = q_list.index(max(q_list)) # get action number

    action = A_max
    #print(action)
    #print(A_max)


    feature_vector = tiles3.tiles(iht, tilings, [8 * newS[0] / (0.5 + 1.2), 8 * newS[1] / (0.07 + 0.07)],
                                  [A_max])  # on book formula

    for i in feature_vector:
        error = error + w[i]

    w = w + alpha*error*z
    z = 0.9*z
    #print (action)
    oldS = newS
    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global w, newS, oldS, iht,z, error, action

    #print("episode done")
    error = reward

    feature_vector = tiles3.tiles(iht, tilings, [8 * oldS[0] / (0.5 + 1.2), 8 * oldS[1] / (0.07 + 0.07)],
                                  [action])  # on book formula

    for i in feature_vector:
        error = error - w[i]
        z[i] =  1 # replacing trace

    w = w + alpha*error*z


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global w

    # print(np.max(Q, axis=1)) #debug


    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):

        # print(time_steps_result)
        return w  # return result


    else:
        return "I don't know what to return!!"

