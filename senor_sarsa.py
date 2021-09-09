#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import matplotlib
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from gym_minigrid.register import env_list

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-6x6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)
obs = env.reset()
#env.render()
#print(env_list)  #env_list

def step(action):
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info

#Number of Actions selector
#nA = 7
nA = 3

#Initialize Q(s,a) arbitrarily, for all s in S, a in A(s)
Q = {(1,1) : {dirc : {a : 0 for a in range(3)} for dirc in range(4)}}
#Q[(1,2)] = {dirc : {a : 0 for a in range(7)} for dirc in range(4)} - How to add key to a dictionary
#Q: Loc: Dir: A: (Q)

#left = 0
#right = 1
#forward = 2
#pickup = 3
#drop = 4
#toggle = 5
#done = 6

#Initialized policy
Pol = {(1,1) : {dirc : 0 for dirc in range(4)}}

#Repeat (for each episode)
attempts = 0 #Episode counter
end = False
while (not end):
    #E(s,a) = 0, for all s in S, a in A(s)
    E = {(1,1) : {dirc : {a : 0 for a in range(nA)} for dirc in range(4)}}

    #Initialze S,A
    loc = env.agent_start_pos
    dire = env.agent_start_dir
    A = 0

    steps  = 1   #Step counter
    α = 0.35     #Step size
    gamma = 0.9  #Discount Factor
    done = False #Terminal
    lmbd = 0.9   #Lambda

    while((not done) or (steps > 1000)):
        #Action Picker acc to ε-greedy
        A = np.random.randint(0,3)
        ε = 1/steps
        A = Pol[loc][dire] if (np.random.random_sample() > (ε)) else A

        #Observe R, S'
        Obs,R,done,info = step(A)
        nloc = (env.agent_pos[0], env.agent_pos[1])
        ndire = env.agent_dir

        #Add state if it doesn't exist
        if Q.get(nloc) is None:
            Q[nloc] = {dirc : {a : 0 for a in range(nA)} for dirc in range(4)}
        if E.get(nloc) is None:
            E[nloc] = {dirc : {a : 0 for a in range(nA)} for dirc in range(4)}
        if Pol.get(nloc) is None:
            Pol[nloc] = {dirc : 0 for dirc in range(4)}
        
        #Target
        targe = R + (gamma*Q[nloc][ndire][Pol[nloc][ndire]]) - Q[loc][dire][A]

        #Spike Eligibility
        E[loc][dire][A] = E[loc][dire][A] + 1

        #Update Q with backward view, Reduce the Eligibility Traces, Get the new Policy
        print(Q)
        for sloc in Q: #Sweep_loc
            for sdire in Q[sloc]: #Sweep_direction
                mA = Pol[sloc][sdire] #Max_Action
                for sA in Q[sloc][sdire]: #Sweep_Action
                    print(sloc,sdire,sA,Q[sloc][sdire][sA],E[sloc][sdire][sA])
                    Q[sloc][sdire][sA] = Q[sloc][sdire][sA] + (α*targe*E[sloc][sdire][sA])
                    E[sloc][sdire][sA] = gamma*lmbd*E[sloc][sdire][sA]
                    mA = sA if (Q[sloc][sdire][sA] > Q[sloc][sdire][mA]) else mA
                Pol[sloc][sdire] = mA
        #S ← S'; A ← A'
        loc = nloc
        dire = ndire
        #Step Increment
        steps = steps + 1

    attempts = attempts + 1
    end = True if (attempts > 100) else False #Loop Terminator/Number of Episodes dial
    
    env.reset()
    print(env.agent_pos)

print(Q,"\n",Pol)
#render(env)