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

max_steps = 15000

episilon = 0.1
alpha = 0.5
num_actions = 8 #differnet setting, manuly changed

Q = None

Qsa = None
action = None

total_result = None

#current_position=0

time_steps_counter = None

time_steps_result = None #total 8000 steps



def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way


    global Q, time_steps_counter, time_steps_result, max_steps, total_result

    Q = np.zeros((10, 7, num_actions)) #grid[x][y][Q]

    total_result = 0

    time_steps_counter=0

    time_steps_result = np.zeros(max_steps)



def agent_start(current_position):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global Q, Qsa, action, time_steps_counter, total_result


    #time_steps_counter+=1

    time_steps_result[time_steps_counter] = total_result

    #print("agent_stat")
    #print(current_position)

    current_position=[0,3]

    if random.randrange(0,100) >= (100 * episilon): #greedy
        action = np.argmax(Q[current_position[0]][current_position[1]]) #get index
    else:
        action = random.randrange(0,num_actions) #random action

    Qsa = current_position

    # print("action")
    # print(action)

    position = []

    # action 0 up 1 down 2 left 3 right 4=02 5=03 6=12 7=13

    if(action == 0):
        position = [current_position[0], current_position[1]+1]

    if(action == 1):
        position = [current_position[0], current_position[1]-1]


    if(action == 2):
        position = [current_position[0]-1, current_position[1]]

    if(action == 3):
        position = [current_position[0]+1, current_position[1]]

    if(action == 4):
        position = [current_position[0]-1, current_position[1]+1]

    if(action == 5):
        position = [current_position[0]+1, current_position[1]+1]

    if(action == 6):
        position = [current_position[0]-1, current_position[1]-1]

    if(action == 7):
        position = [current_position[0]+1, current_position[1]-1]

    if(action == 8):
        position = [current_position[0], current_position[1]]
    #
    # if(position == 0):
    #     print("position error")

    #print(position)

    return position
    # if(action = 9): #stay
    #     position = [current_position[0], current_position[1]]
    #     return position
    #



def agent_step(reward, position): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Q, Qsa, action, alpha, time_steps_counter, total_result

    time_steps_counter+=1
    time_steps_result[time_steps_counter] = total_result

    #print(position)
    #print("after")
    #print(Qsa)
    #print(time_steps_counter)

    if random.randrange(0,100) >= 100 * episilon: #greedy
        action_p = np.argmax(Q[position[0]][position[1]]) #get index
    else:
        action_p = random.randrange(0,num_actions) #random action 1-8

    #update rule
    Q[Qsa[0]][Qsa[1]][action] = Q[Qsa[0]][Qsa[1]][action] + alpha*(reward+Q[position[0]][position[1]][action_p]- Q[Qsa[0]][Qsa[1]][action])



    action = action_p

    Qsa = position

    # action 1 up 2 down 3 left 4 right 5=13 6=14 7=23 8=24

    if(action == 0):
        position = [position[0], position[1]+1]

    if(action == 1):
        position = [position[0], position[1]-1]


    if(action == 2):
        position = [position[0]-1, position[1]]


    if(action == 3):
        position = [position[0]+1, position[1]]

    if(action == 4):
        position = [position[0]-1, position[1]+1]

    if(action == 5):
        position = [position[0]+1, position[1]+1]

    if(action == 6):
        position = [position[0]-1, position[1]-1]

    if(action == 7):
        position = [position[0]+1, position[1]-1]

    if(action == 8):
        position = [position[0], position[1]]

    # if(position == 0):
    #     print("position error")


    return position


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global Q, Qsa, action, alpha, time_steps_counter, time_steps_result, total_result

    #print(time_steps_counter)

    #print("time")
    # #
    # print(time_steps_counter)
    # #
    # print(total_result)

    #time_steps_counter += 1

    total_result+=1

    time_steps_result[time_steps_counter] = total_result

    #print(time_steps_result)
    # print(Qsa[0])
    # print(Qsa[1])


    Q[Qsa[0]][Qsa[1]][action] = Q[Qsa[0]][Qsa[1]][action] + alpha * (reward + 0 - Q[Qsa[0]][Qsa[1]][action] )


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q, time_steps_result

    #print(np.max(Q, axis=1)) #debug


    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):

        #print(time_steps_result)
        return time_steps_result #1
    else:
        return "I don't know what to return!!"

