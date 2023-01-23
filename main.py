import gymnasium as gym
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gc
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from HPModel3D_env.envs.hpmodel_env import HPModel3D
import arguments
import utils
from sequences import sequences_dict

print(
"""   
██╗░░██╗██████╗░░░░░░░███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░██████╗░██████╗░  ██████╗░██╗░░░░░
██║░░██║██╔══██╗░░░░░░████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░╚════██╗██╔══██╗  ██╔══██╗██║░░░░░
███████║██████╔╝█████╗██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░░█████╔╝██║░░██║  ██████╔╝██║░░░░░
██╔══██║██╔═══╝░╚════╝██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░░╚═══██╗██║░░██║  ██╔══██╗██║░░░░░
██║░░██║██║░░░░░░░░░░░██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗██████╔╝██████╔╝  ██║░░██║███████╗
╚═╝░░╚═╝╚═╝░░░░░░░░░░░╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝╚═════╝░╚═════╝░  ╚═╝░░╚═╝╚══════╝


█░░ █▀▀ █▀█ █▄░█ ▄▀█ █▀█ █▀▄ █▀█   █▀█ █▀▀ █▀█ █▀▀ ▀█
█▄▄ ██▄ █▄█ █░▀█ █▀█ █▀▄ █▄▀ █▄█   █▀▀ ██▄ █▀▄ ██▄ █▄

""")

gc.collect()
torch.cuda.empty_cache()

args=arguments.args

steps_million = args.time_steps/1e+6

best_model = f"saved_models/B{args.sequence}_{steps_million}M.pth"
last_model = f"saved_models/L{args.sequence}_{steps_million}M.pth"

env = HPModel3D(sequences_dict[args.sequence])

action_n = env.action_space.n


device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_ACTION = 5

dqn = utils.DDQN(
    output_shape=N_ACTION,
    device = device,
    sync_steps= args.sync_steps
)


buffer = utils.PrioritizedReplayBuffer(
    size = args.buffer_length,
    alpha = args.alpha,
    beta = args.beta
)


writer = SummaryWriter()
o = env.reset()
o = dqn.preprocess_state(o)
nepisode = 0

eps_max = 1
eps_min = 0.01
reward = 0
game = 0
reward_max = -np.inf
l = 5


max_frames = len(sequences_dict[args.sequence])*2
frame = 0
for t in tqdm(range(1, args.time_steps + 1)):

    eps = eps_min + (eps_max-eps_min)*math.e**(-t*l/args.time_steps)
    a = dqn.take_action(o,eps)
    o_, r, done, info = env.step(a)
    o_ = dqn.preprocess_state(o_)
    done = 1 if done else 0
    buffer.add(o, a, r, o_, done)
    reward += r
    frame += 1
    if t >= args.warm_start and t % args.train_freq == 0:
        *transitions, idxs = buffer.sample(args.batch_size)
        priorities = dqn.train(*transitions,args.gamma)
        priorities = np.clip(np.abs(priorities), 1e-6, None)
        buffer.update_priorities(idxs, priorities)
    if (done) | (frame == max_frames):
        frame = 0
        o = env.reset()
        o = dqn.preprocess_state(o)
        writer.add_scalar("Reward",reward,global_step = game+1)
        game+=1
        if reward>=reward_max:
        
            print("Saving model...")
            print(f"reward :{reward_max}")
            torch.save(dqn.Q.state_dict(),best_model)
            reward_max = reward
        torch.save(dqn.Q.state_dict(),last_model)    
        reward=0
        gc.collect()
        torch.cuda.empty_cache()      
    else:
        o = o_

    buffer.beta += (1 - args.beta) / args.time_steps
 
