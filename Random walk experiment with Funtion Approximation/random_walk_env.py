#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""
# modify to represent the windy world environment

from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import random

world = None

current_position = None

def env_init():
    global world, current_position

    world = np.zeros(1001) # 1-1000,

    current_position = 500 #initial




def env_start():
    """ returns numpy array """
    global maze, current_position

    current_position = 500

    return current_position

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global world, current_position
    # if action < 1 or action > np.minimum(current_state[0], num_total_states + 1 - current_state[0]):
    #     print "Invalid action taken!!"
    #     print "action : ", action
    #     print "current_state : ", current_state
    #     exit(1)

    is_terminal = False

    # print("env step")
    #print(action)
    reward = 0

    #displacement = random.randrange(1,101) #1-100 random

    # if action == 0: #go left
    #     displacement = - displacement # move left

    #current_position = displacement + current_position #update

    current_position = action #update

    if current_position  >= 1001: # finish on right with 1
        reward = 1
        is_terminal = True
    elif current_position  <= 1:  # finish on left with -1
        reward = -1
        is_terminal = True

    result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}
    #print(reward)

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
