import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

#from HPModel3D_env.envs.hpmodel_env_v3_1 import HPModel3D
import numpy as np
import random
import math

seq = 'HPHPPHHPHPPHPHHPPHPH'

#env = HPModel3D(seq)

#state = env.reset()

'''
env.render_final()

print(state.shape)

while True:

    a = random.randint(0,4)
    state,r,d,_, = env.step(a)
    print(r)
    if d:
        break


env.render_final()
'''
l = 2.3
eps_min = 0.01
eps_max = 1
time_steps = 1000000

t = 400000

eps = eps_min + (eps_max-eps_min)*math.e**(-t*l/time_steps)

print(eps)
