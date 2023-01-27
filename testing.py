import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)



import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import utils
from HPModel3D_env.envs.hpmodel_env_v3_1 import HPModel3D


from sequences import sequences_dict

env_name = input("Enviroment (v1.0,v2.0,v3.0,v3.1): ")

name = input("Enter the name of the sequence: ")

model = input ("Num of model: ")

time_train = float( input("Train Steps (Million): ") )


env = HPModel3D(sequences_dict[name])

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


while True:

    dqn = utils.DDQN(None,'cpu',None,model = int(model))

    best_last = input("Best or last model (B,L): ")

    env_path=f"saved_models/{env_name}_{best_last}{model}_{name}_{time_train}M.pth"


    dqn.Q.load_state_dict(torch.load(env_path,map_location=torch.device('cpu')))

    max_steps = 1000
    step = 0

    s = env.reset()
    s = dqn.preprocess_state(s)
    a = int(dqn.Q(s).argmax().cpu().numpy())
    reward=0
    step = 0




    while True:
        s_,r,d,info=env.step(a)
        print(a)
        if d is None: break
        step += 1
        s_ = dqn.preprocess_state(s_)
        reward+=r
        if (d) | (step==max_steps):
            step = 0
            relations = r
            break
        s=s_
        a = int(dqn.Q(s).argmax().cpu().numpy())

    print(f"info:{info}")    
    print(f"reward:{reward}")
    print(f"relation:{r}")   
    env.render_final()

