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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from HPModel3D_env.envs.hpmodel_env import HPModel3D
import arguments
import utils

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

env_path = f"saved_models/{args.sequence}_{steps_million}M.pth"

env = HPModel3D(args.sequence)

action_n = env.action_space.n


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dqn = utils.DDQN(
    output_shape=6,
    device = device,
    sync_steps= args.sync_steps
)

#dqn.Q.load_state_dict(torch.load(env_path,map_location=torch.device(device)))
#dqn.Q_target.load_state_dict(torch.load(env_path,map_location=torch.device(device)))

buffer = utils.PrioritizedReplayBuffer(
    size = args.buffer_length,
    alpha = args.alpha,
    beta = args.beta
)


writer = SummaryWriter()
o = env.reset()
o = dqn.preprocess_state(o)
nepisode = 0

eps_ = 1
eps = eps_
reward = 0
game = 0
reward_max = -np.inf


for t in tqdm(range(1, args.time_steps + 1)):

    a = dqn.take_action(o,eps)
    o_, r, done, info = env.step(a)
    o_ = dqn.preprocess_state(o_)
    done = 1 if done else 0
    buffer.add(o, a, r, o_, done)
    reward += r
    if t >= args.warm_start and t % args.train_freq == 0:
        *transitions, idxs = buffer.sample(args.batch_size)
        priorities = dqn.train(*transitions,args.gamma)
        priorities = np.clip(np.abs(priorities), 1e-6, None)
        buffer.update_priorities(idxs, priorities)
    if done:
        o = env.reset()
        o = dqn.preprocess_state(o)
        writer.add_scalar("Reward",reward,global_step = game+1)
        game+=1
        if reward>=reward_max:
            torch.save(dqn.Q.state_dict(),env_path)
            #ts_image = t/1e+6
            #path_img = f"saved_images/20merA/{args.sequence}_{ts_image}.fig.pickle" #change this every time that u want
            #env.render_final(show_img = False, save_img = True, path_img = path_img)
            reward_max = reward
        reward=0
        gc.collect()
        torch.cuda.empty_cache()      
    else:
        o = o_

    buffer.beta += (1 - args.beta) / args.time_steps

    eps -= (eps_+0.3)/args.time_steps
 
