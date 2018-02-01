import numpy as np


theta = 0.00000000000000000000001 #a non zero positive number
ph = 0.55 #need to change to fit in new seetings

value_table = np.zeros(101)

action_table = np.zeros(101)

def agent_run():
    global  theta, ph, value_table

    value = 0
    delta = 0
    sweep =0
    i=30
    while delta>=0:
        sweep += 1
        delta=0
        for state in range(1,100):
            value = value_table[state]
            for action in range(1,min(state,100-state)+1):

                if action + state >= 100:
                    temp_value = ph*(1 + value_table[state+action]) + (1 - ph)* (0 + value_table[state-action])
                else:
                    temp_value = ph * (0 + value_table[state + action]) + (1 - ph) * (0 + value_table[state - action])
                if temp_value > value:
                    value = temp_value #find max for all action
                    optimal_action = action
                delta = max(delta, abs(value - value_table[state]))
            if value > value_table[state]:
                value_table[state] = value
                action_table[state] = optimal_action

            #print(delta) #debug
        if delta < theta:
            #print(sweep) #debug
            return value_table




output= agent_run()
for i in range(0,100):
    print(action_table[i])
for i in range(0, 100):
    print(value_table[i])
#print(action_table)