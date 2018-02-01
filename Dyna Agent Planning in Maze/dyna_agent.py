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


episilon = 0.1
alpha = 0.1
num_actions = 4 #differnet setting, manuly changed
planningTable = [0, 5, 50] #all planning stepsize
planning = 0 # planning steps

alpha_table = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] # alphas for part2

Q = None

Qsa = None
action = None

Model = None

observeS = None # record state

observeA = None # record action


#current_position=0

time_steps_counter = None

step_per_episode = None #total 8000 steps

total_runs = 0

a_index = -1
p_index = -1

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way


    global Q, time_steps_counter, step_per_episode, total_runs, Model, observeA, observeS

    Q = np.zeros((9, 6, num_actions)) #grid[x][y][Q]

    Model = np.zeros((9, 6, num_actions,3)) #state[x][y][action][s'x s'y][R]

    #print(Model[3][3][1][2]) #debug check initiallizatio nof array
    total_runs = 0 #a new run

    time_steps_counter=0

    step_per_episode = np.zeros(51)

    observeA = np.zeros((54,4)) # state, action
    observeS = [] # a empty list




def agent_start(current_position):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts

    global Q, Qsa, action, time_steps_counter, total_runs


    time_steps_counter =1 # frist time step, reset when restart

    total_runs += 1

    #time_steps_result[time_steps_counter] = total_result

    #print("agent_stat")
    #print(current_position)

    current_position=[0,3]

    if np.random.randint(0,100) >= (100 * episilon): #random.randrange(0,100) >= (100 * episilon): #greedy
        #action = np.argmax(Q[current_position[0]][current_position[1]]) #get index
        max_list = np.argwhere(Q[current_position[0]][current_position[1]] == np.amax(Q[current_position[0]][current_position[1]]))  # Reference: stack overflow
        max_list.flatten().tolist()  # get a list of max action

        #print(max_list.shape[0])

        n = random.randrange(0, max_list.shape[0])  # get a random max index
        # print(n)
        action = max_list[n]
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



    #print(position)

    return position



def agent_step(reward, position): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Q, Qsa, action, alpha, time_steps_counter, Model, observeA, observeS

    time_steps_counter+=1 # increment time step


    if np.random.randint(0,100) >= (100 * episilon): #Rrandom.randrange(0,100) >= 100 * episilon: #greedy Q s' a
        max_list = np.argwhere(Q[position[0]][position[1]] == np.amax(Q[position[0]][position[1]]))  # Reference: stack overflow
        max_list.flatten().tolist()  # get a list of max action

        #print(max_list.shape[0])

        n = random.randrange(0, max_list.shape[0])  # get a random max index
        # print(n)
        action_p = max_list[n]

        #action_p = np.argmax(Q[position[0]][position[1]]) #get index
    else:
        action_p = random.randrange(0,num_actions) #random action 1-8

    #update rule
    Q[Qsa[0]][Qsa[1]][action] = Q[Qsa[0]][Qsa[1]][action] + alpha*(reward+0.95 * Q[position[0]][position[1]][action_p]- Q[Qsa[0]][Qsa[1]][action])


    # do the model learning
    if Qsa not in observeS: # append the state first time seen
        observeS.append(Qsa) #record state
        index = observeS.index(Qsa) #get index, the same with observeA
        observeA[index][action] = 1  # record the action

    else:
        index = observeS.index(Qsa)  # get index, the same with observeA
        if(observeA[index][action] == 0): #action not in record
            observeA[index][action] = 1

    # Model = np.zeros((9, 6, num_actions,3)) state[x][y][action][Q][s'x s'y][R]
    #print(Model[Qsa[0]][Qsa[1]][action][0])

    Model[int(Qsa[0])][int(Qsa[1])][int(action)][0] = int(position[0]) # x
    Model[int(Qsa[0])][int(Qsa[1])][int(action)][1] = int(position[1]) # y
    Model[int(Qsa[0])][int(Qsa[1])][int(action)][2] = reward  # R

    #print(Qsa)
    #print(Model[Qsa[0]][Qsa[1]][action])

    for k in range(0,planning):
        temp_action_list = [] # choice action
        S = random.choice(observeS) # get random S
        index = observeS.index(S)
        for temp_i in range(0,4): # get a list of observed A in S
            if observeA[index][temp_i] == 1:
                temp_action_list.append(temp_i)
        A = random.choice(temp_action_list)
        # get R S' Model[Qsa[0]][Qsa[1]][A] do update

        #print(Model[S[0]][S[1]][A])
        # Q s' max a
        temp_max_list = np.argwhere(Q[Model[S[0]][S[1]][A][0]][Model[S[0]][S[1]][A][1]] == np.amax(Q[Model[S[0]][S[1]][A][0]][Model[S[0]][S[1]][A][1]]))  # Reference: stack overflow
        temp_max_list.flatten().tolist()  # get a list of max action

        temp_n = random.randrange(0, temp_max_list.shape[0])  # get a random max index

        temp_action_p = temp_max_list[temp_n]

        #R = Model[S[0]][S[1]][A][0]
        #print(alpha*(Model[S[0]][S[1]][A][2]+0.95 * Q[Model[S[0]][S[1]][A][0]][Model[S[0]][S[1]][A][1]][temp_action_p]- Q[S[0]][S[1]][A]))
        Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(Model[S[0]][S[1]][A][2]+0.95 * Q[Model[S[0]][S[1]][A][0]][Model[S[0]][S[1]][A][1]][temp_action_p]- Q[S[0]][S[1]][A])

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




    return position


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global Q, Qsa, action, alpha, time_steps_counter, step_per_episode, total_runs

    step_per_episode[total_runs] = time_steps_counter # record the time steps

    #print(time_steps_counter)

    Q[Qsa[0]][Qsa[1]][action] = Q[Qsa[0]][Qsa[1]][action] + alpha*(reward - Q[Qsa[0]][Qsa[1]][action])



def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q, time_steps_result, planning, planningTable, alpha, alpha_table, episilon, a_index, p_index

    #print(np.max(Q, axis=1)) #debug


    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):

        #print(time_steps_result)
        return step_per_episode #return result
    if (in_message == 'ChangeStep'):

        p_index += 1
        planning = planningTable[p_index] # advance to next index
    if (in_message == 'ChangeAlpha'):
        planning = 5 # 5 steps
        episilon = 0.1
        #print(a_index)
        a_index += 1
        alpha = alpha_table[a_index] # advance to next index
        #print(alpha)

    else:
        return "I don't know what to return!!"

