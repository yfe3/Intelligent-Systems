#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions

  modified from sample agent code to start learning.
  modified by Yuan Feng (yfeng3)

"""

from utils import rand_in_range
import numpy as np
import random

last_action = None # last_action: NumPy array

num_actions = 10 #10 armed bandit problem

epsilon=0  #assign epsilon here to change the learning algorothm
alpha=0.1

action_random= 100*epsilon #control variable to decide when to make a random move

reward_table=None

expectation_table=None

def agent_init():
    global last_action

    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero
 
    global reward_table
    reward_table= np.zeros(10) #size 10 for store all rewards from each action
    
    global action_counter
    action_counter= np.zeros(10) #size 10 for count the frequency of all actions
       
    global expectation_table
    expectation_table= np.zeros(10) #size 10 for store all expectations from each action
    global epsilon
    if(epsilon==0):                         #special case for epsilon = 0, set optimal value
        for values in range(0,10,1):
            expectation_table[values] = 5

             

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action

    last_action[0] = rand_in_range(num_actions)# now should correctly working for random 10 values

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)
    
    
    return local_action[0]


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action #need to select from actions
    #update the reward table and counter first

    global expectation_table

    
    global reward_table
    reward_table[last_action[0]] = reward #update the reward respectively

    
     #now need to calculate the expectations, the core formula for learning

    expectation_table[last_action[0]] = expectation_table[last_action[0]] + alpha*(reward - expectation_table[last_action[0]])
    

    local_action = np.zeros(1)
    if(epsilon==0 or random.randrange(0,100)>action_random ): #work for both greedy and e-greedy
        local_action[0] = np.argmax(expectation_table) #get the index of the max expectation
        #print(local_action[0])
    
    else:
        
        local_action[0] = rand_in_range(num_actions)

    # might do some learning here

    last_action = local_action

    return last_action

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
  
    # else
    return "I don't know how to respond to your message"
