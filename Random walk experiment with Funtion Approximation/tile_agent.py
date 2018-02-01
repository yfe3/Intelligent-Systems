#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random
import tiles3

alpha = 0.01/50

w = 0


oldS = 0

newS = 0

# tile code parameters

iht = tiles3.IHT(2000)

tilings = 50

width = 0.2

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    # initialize the policy array in a smart way


    global w, oldS, newS, iht

    iht = tiles3.IHT(2000)

    newS= np.zeros(1) #need to be iterable
    oldS = np.zeros(1)

    w = np.zeros(1200)  # according to TA's suggestion, size = 1200 , to avoid out of index



def agent_start(current_position):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global w, v, oldS

    # print("agent_stat")
    # print(current_position)

    temp_ve = random.randint(0, 1)  # choose left or right, 0 = right, 1 = left

    temp_action = random.randint(1, 100)  # 1 100 random walk

    if temp_ve == 0:  # change sign
        temp_ve = -1

    action = current_position + temp_ve * temp_action  # get index in v

    # print(action)

    oldS[0] = float(current_position)//200  # 1000*0.2 = 200, 200 states in 1 tile

    return action


def agent_step(reward, position):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global v, w, newS, oldS, iht

    # time_steps_counter+=1 # increment time step

    temp_ve = random.randint(0, 1)  # choose left or right, 0 = right, 1 = left

    temp_action = random.randint(1, 100)  # 1 100 random walk

    if temp_ve == 0:  # change sign
        temp_ve = -1

    action = position + temp_ve * temp_action  # get index in v

    newS[0] = float(position) //200  # 1000*0.2 = 200, 200 states in 1 tile

    # do update
    new_feature = np.zeros(1200) # feature vector is in size of tilings , +1 for 0 not use
    old_feature = np.zeros(1200)


    new_feature_x = tiles3.tiles(iht, tilings, newS) # 1000*0.2 = 200, 200 states in 1 tile
    #print(new_feature_x) # debug
    for i in new_feature_x: # get feature vecter
        new_feature[i] = 1

    old_feature_x = tiles3.tiles(iht, tilings, oldS)
    for i in old_feature_x:
        old_feature[i] = 1

    w = w + alpha * (reward + 1 * np.dot(w,new_feature) - np.dot(w,old_feature)) * old_feature  # tile, gamma = 1

    #print(w)

    oldS[0] = newS[0]

    #print(action)

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global v, w, newS, oldS, iht

    # if reward >0: #debug
    #     print(reward)

    # do update
    new_feature = np.zeros(1200)  # feature vector is in size of tilings , +1 for 0 not use
    old_feature = np.zeros(1200)

    # 1000/50= 20

    new_feature_x = tiles3.tiles(iht, tilings, newS)
    # print(new_feature_x) # debug
    for i in new_feature_x:
        new_feature[i] = 1
    old_feature_x = tiles3.tiles(iht, tilings, oldS)
    for i in old_feature_x:
        old_feature[i] = 1

    w = w + alpha * (reward - np.dot(w, old_feature)) * old_feature  # tile, gamma = 1

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

