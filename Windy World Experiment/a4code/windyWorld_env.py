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

windy_world = None

current_position = None

def env_init():
    global windy_world

    windy_world = np.zeros((10,7,3)) # create the grid [x][y][x,y,wind]

    for x in range(3,9): #assign wind x 3-8
        for y in range(0,7): #assign wind y 0-7
            windy_world[x][y][2] +=1
            if x == 6 or x == 7:
                windy_world[x][y][2] += 1 # stronger wind




def env_start():
    """ returns numpy array """
    global windy_world, current_position

    start_position = [0,3] #windy_world[0][3][:1] # start [x,y]

    current_position = start_position

    return start_position

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
    global windy_world, current_position
    # if action < 1 or action > np.minimum(current_state[0], num_total_states + 1 - current_state[0]):
    #     print "Invalid action taken!!"
    #     print "action : ", action
    #     print "current_state : ", current_state
    #     exit(1)

    is_terminal = False

    # print("env step")
    #print(action)

    if action[0] > 9 or action[0] < 0: # stay
        reward = -1
        #current_position = action
        #return current_position, reward
        result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

        return result

    if action[1] > 6 or action[1] < 0:
        reward = -1
        #current_position = action
        #return current_position, reward
        result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

        return result

    if windy_world[current_position[0]][current_position[1]][2] != 0: # wind apply
       wind = windy_world[current_position[0]][current_position[1]][2]
    else:
        wind = 0

    current_position = action

    #current_position[0] = current_position[0] + wind #update position
    current_position[1] = current_position[1] + wind

    reward = -1

    if current_position[1] > 6: # if below up out
        current_position[1] = 6

    if current_position[0] == 7 and current_position[1] == 3: #terminal
        reward = -1
        #current_position = [0,3] # start [x,y]
        is_terminal = True



    result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

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
