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

maze = None

current_position = None

def env_init():
    global maze

    maze = np.zeros((9,6,1)) # create the grid [x][y][type]
    # type= 0 noral pass
    # 1 wall
    # 2 start
    # 3 goal

    maze[0][3][0] = 2 # set to start
    maze[8][5][0] = 3 # set to goal

    for y in range(2,5): #assign wall
        maze[2][y][0] = 1

    maze[5][1][0] = 1
    for y in range(3,6):
        maze[7][y][0] = 1 #get all wall set




def env_start():
    """ returns numpy array """
    global maze, current_position

    start_position = [0,3] #maze[0][3] # start [x,y]

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
    global maze, current_position
    # if action < 1 or action > np.minimum(current_state[0], num_total_states + 1 - current_state[0]):
    #     print "Invalid action taken!!"
    #     print "action : ", action
    #     print "current_state : ", current_state
    #     exit(1)

    is_terminal = False

    # print("env step")
    #print(action)
    reward = 0

    if action[0] > 8 or action[0] < 0: # stay

        #current_position = action
        #return current_position, reward
        result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

        return result

    if action[1] > 5 or action[1] < 0:

        #current_position = action
        #return current_position, reward
        result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

        return result

    if maze[action[0]][action[1]][0] == 1: #hit the wall, stay
        result = {"reward": reward, "state": current_position, "isTerminal": is_terminal}

        return result


    current_position = action # if still in the maze

    if current_position[0] == 8 and current_position[1] == 5: #terminal
        reward = 1
        #current_position = [0,3] # start [x,y]
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
