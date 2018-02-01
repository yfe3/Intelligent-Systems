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

alpha = 0.1

w = 0


oldS = 0

newS = 0


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    # initialize the policy array in a smart way


    global w,  oldS, newS

    w = np.zeros(11)  # 1-10 = 0 ,0 not used



def agent_start(current_position):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global w,  oldS, newS

    # print("agent_stat")
    # print(current_position)
    temp_ve = random.randint(0, 1)  # choose left or right, 0 = right, 1 = left

    temp_action = random.randint(1, 100)  # 1 100 random walk

    if temp_ve == 0:  # change sign
        temp_ve = -1

    action = current_position + temp_ve * temp_action  # get index in v

    #print(action)

    oldS = current_position

    return action


def agent_step(reward, position):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global  w, newS, oldS

    # time_steps_counter+=1 # increment time step

    temp_ve = random.randint(0, 1)  # choose left or right, 0 = right, 1 = left

    temp_action = random.randint(1, 100)  # 1 100 random walk

    if temp_ve == 0:  # change sign
        temp_ve = -1

    action = position + temp_ve * temp_action  # get index in v

    newS = position

    # do update

    temp_w = np.zeros(11)

    temp_w[oldS//100] = 1

    w = w + alpha * (reward + 1 * w[newS//100] - w[oldS//100]) * temp_w  # gamma = 1  for this problem

    oldS = newS

    #print(action)

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global  w, newS, oldS

    # do update

    temp_w = np.zeros(11)

    temp_w[oldS // 100] = 1

    w = w + alpha * (reward - w[oldS // 100]) * temp_w  # tabular


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


