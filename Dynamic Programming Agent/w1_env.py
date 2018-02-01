#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env *ignores* actions: rewards are all random
  
  modified from thew sample code serve as the environment
  now all reward are random based on their expectations
  modified by Yuan Feng (yfeng3)
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

this_reward_observation = (None, None, None)  # this_reward_observation: (floating point, NumPy array, Boolean)

bandit = None


def find_optimal(bandit): #pass the test working properly
    optimal = np.argmax(bandit)  # find the optimal action
    #print(optimal,"this is optimal") #for debuggging use
    return optimal


def env_init():
    global this_reward_observation
    global bandit
    # intialize the reward of each action here

    bandit = np.zeros(10)  # set the bandit to 10 armed

    for loop_control in range(0, 10, 1):
        bandit[loop_control] = rand_norm(0.0, 1.0)  # set the means of reward value

    local_observation = np.zeros(0)  # An empty NumPy array

    this_reward_observation = (0.0, local_observation, False)


def env_start():  # returns NumPy array
    return this_reward_observation[1]


def env_step(this_action):  # returns (floating point, NumPy array, Boolean), this_action: NumPy array
    global this_reward_observation
    global bandit

    # the_reward = rand_norm(0.0, 1.0)  # rewards drawn from (0, 1) Gaussian
    # now we return reward according to the action

    this_action_int = this_action.tolist()

    the_reward = rand_norm(bandit[this_action_int], 1.0)  # set the reward value


    this_reward_observation = (the_reward, this_reward_observation[1], False)

    return this_reward_observation


def env_cleanup():
    #
    return


def env_message(inMessage):  # returns string, inMessage: string
    global bandit
    if inMessage == "what is your name?":
        return "my name is skeleton_environment!"
    elif inMessage == "get optimal action":
        global bandit
        optimal = find_optimal(bandit)
        return optimal
        # action 3 gives the highest q* value; ie. argmax q*(a) = 3
    # else
    return "I don't know how to respond to your message"
