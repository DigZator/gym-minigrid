#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from gym_minigrid.register import env_list

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-8x8-v0'
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
#print(env_list)  #env_list

def step(action):
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info

end = False
#Q:
#   Loc:
#        Dir:
#            A: Q
#print("\n",np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,8))
#left = 0
#right = 1
#forward = 2
#pickup = 3
#drop = 4
#toggle = 5
#done = 6

Q = {(1,1) : {dirc : {a : 0 for a in range(7)} for dirc in range(4)}}
#Q[(1,2)] = {dirc : {a : 0 for a in range(7)} for dirc in range(4)}
Pol = {(1,1) : {dirc : 0 for dirc in range(4)}}
#print(Q)
attempt = 0
while (not end):
    env.reset()
    E = Q.copy()
    loc = env.agent_start_pos
    dire = env.agent_start_dir
    #print(Q[loc][dire])
    A = 0
    steps  = 1
    α = 0.35
    gamma = 0.9
    done = False
    lmbd = 0.9

    while(not done):
        #Action Picker acc to ε-greedy
        A = np.random.randint(0,7)
        A = Pol[loc][dire] if (np.random.random_sample() > (1/steps)) else A

        Obs,R,done,INFO = step(A)
        nloc = (env.agent_pos[0], env.agent_pos[1])
        ndire = env.agent_dir
        #print(nloc,type(nloc))
        #Add state if it doesn't exist
        if Q.get(nloc) is None:
            Q[nloc] = {dirc : {a : 0 for a in range(7)} for dirc in range(4)}
        if E.get(nloc) is None:
            E[nloc] = {dirc : {a : 0 for a in range(7)} for dirc in range(4)}
        if Pol.get(nloc) is None:
            Pol[nloc] = {dirc : 0 for dirc in range(4)}
        #Target
        targe = R + (gamma*Q[nloc][ndire][Pol[nloc][ndire]]) - Q[loc][dire][A]
        E[loc][dire][A] = E[loc][dire][A] + 1
        for sloc in Q:
            for sdire in Q[sloc]:
                mA = Pol[sloc][sdire]
                for sA in Q[sloc][sdire]:
                    Q[sloc][sdire][sA] = Q[sloc][sdire][sA] + (α*targe*E[sloc][sdire][sA])
                    E[sloc][sdire][sA] = gamma*lmbd*E[sloc][sdire][sA]
                    mA = sA if (Q[sloc][sdire][sA] > Q[sloc][sdire][mA]) else mA
                Pol[sloc][sdire] = mA
        loc = nloc
        dire = ndire
    end = True if (attempt > 10000) else False
    attempt = attempt + 1
    #if (attempt%1000 == 0):
    #    print(attempt)
print(Q,"\n",Pol)
#render(env)