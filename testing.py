import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import utils
from HPModel3D_env.envs.hpmodel_env import HPModel3D


seq = "HPHPPHHPHPPHPHHPPHPH"

env = HPModel3D(seq)

device = 'cuda' if torch.cuda.is_available()  else 'cpu'

test = 50

dqn = utils.DDQN(None,'cpu',None)

env_path="saved_models/PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP_1.0M.pth"

dqn.Q.load_state_dict(torch.load(env_path,map_location=torch.device('cpu')))

max_steps = 1000



for i in range(test):
    s = env.reset()
    j=0
    s = dqn.preprocess_state(s)
    a = int(dqn.Q(s).argmax().cpu().numpy())
    reward=0
    while True:

        s_,r,d,info=env.step(a)
        s_ = dqn.preprocess_state(s_)
        reward+=r
        if d:
            relations = r
            break
        s=s_
        j+=1
        if j==max_steps:break
        a = int(dqn.Q(s).argmax().cpu().numpy())
    print(f"info:{info}")    
    print(f"reward:{reward}")
    print(f"relation:{r}")   
    env.render_final()

