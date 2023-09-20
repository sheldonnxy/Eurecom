"""
Reinforcement Learning:
Backtracking Proximal Policy Optimization (BPPO),

Trajectory optimization for radio map estimation

通过优化无人机轨迹，利用无人机在空间不同位置采集具有高信息量的RSS数据，以此来提高估计RM的准确性
Using:
tensorflow 1.14.0
gym 0.8.0
torch 1.6.0

"""


import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import grad
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import datetime

import GAEPPO_conv
import Sampling_env.data_collection_cov


Transition = namedtuple('Transition',['state', 'action', 'act_prob', 'reward', 'next_state'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument("--env_name", default="Sampling-v5")  # environment name
parser.add_argument('--tau',  default=0.01, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)
parser.add_argument('--max_grad_norm', default=0.5, type=int)
parser.add_argument('--num_iter', default=10, type=int)
parser.add_argument('--lr_actor', default=1e-3, type=float)  # leaning rate for actor network
parser.add_argument('--lr_critic', default=5e-3, type=float)   # leaning rate for critic network
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--gamma', default=0.8, type=int) # discount gamma
parser.add_argument('--lambd', default=0.95, type=int) # discount lambd
parser.add_argument('--capacity', default=1024, type=int) # replay buffer size
parser.add_argument('--max_length_of_trajectory', default=500, type=int)   #  num of  step
parser.add_argument('--iteration', default=1000, type=int) #  num of  games
parser.add_argument('--log_interval', default=20, type=int)  #
parser.add_argument('--batch_size', default=128, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--load-dir', default=False)
# optional parameters
parser.add_argument('--load', default=False, type=bool) # load model
args = parser.parse_args()


Grid = np.array([20,20])  # grid size
env = gym.make(args.env_name)
state_dim = env.state_dim_vec
action_dim = env.action_dim  ## action_space



def main():
    agent = GAEPPO_conv.PPO_Agent("agent_%d" % 1, Grid, action_dim, 1, args, None)

    print("====================================")
    print("Collection Experience...")
    print("====================================")

    Dis_label = 1
    run_results = []
    update_num = 0
    location_path = []
    ep_t =[]
    lr1 = args.lr_actor
    lr2 = args.lr_critic
    lr_multi = 1
    ep_nmse = []
    ep_energy = []

    if args.load_dir:
        agent.load()
        for i in range(1):
            location, state = env.reset()
            episode_reward = 0
            t = 0
            done = False
            o2_ep = []
            a_ep = []
            aprob_ep = []
            o_ep = []
            d_ep = []
            while not done and t < args.max_length_of_trajectory:
                t += 1

                action, prob_action = agent.select_action(state)
                next_state, next_location, reward, done, mse, reward_batch_ep, energy \
                    = env.step_next(location, state, t, action, Dis_label, args)

                o_ep.append(state)
                a_ep.append(action)
                aprob_ep.append(prob_action)
                o2_ep.append(next_state)
                d_ep.append(done)

                if t == args.max_length_of_trajectory - 1 or done:
                    episode_reward = sum(reward_batch_ep)
                    ep_nmse.append(mse)
                    ep_energy.append(energy)

                state = next_state
                location = next_location

            print("Ep_i {}, the ep_r is {}".format(i, episode_reward / t))
            run_results.append(episode_reward / t)
            ep_t.append(t)


    else:

        for i in range(args.iteration):

            location, state = env.reset()
            episode_reward = 0
            t = 0
            done = False
            o2_ep = []
            a_ep = []
            aprob_ep = []
            o_ep = []
            d_ep = []

            while not done and t < args.max_length_of_trajectory:
                t += 1

                action, prob_action = agent.select_action(state)
                next_state, next_location, reward, done, mse,reward_batch_ep, energy \
                    = env.step_next(location, state, t, action, Dis_label, args)

                o_ep.append(state)
                a_ep.append(action)
                aprob_ep.append(prob_action)
                o2_ep.append(next_state)
                d_ep.append(done)

                if t == args.max_length_of_trajectory-1 or done:
                    episode_reward = sum(reward_batch_ep)
                    ep_nmse.append(mse)
                    ep_energy.append(energy)
                    for jj in range(t):
                        agent.store(o_ep[jj], a_ep[jj], aprob_ep[jj], np.float32(reward_batch_ep[jj]), o2_ep[jj], d_ep[jj])




                # if done:
                #     agent.update(t)
                #     update_num += 1
                #     print("the num of training is {}".format(update_num))

                if done:
                    if len(agent.buffer) >= args.capacity:
                        agent.update(t,lr1,lr2)
                        update_num += 1
                        print("the num of training is {}".format(update_num))



                if i == args.iteration -1:
                    location_path.append(location)


                state = next_state
                location = next_location

            if (update_num+1) % 1 == 0:
                lr_multi  = 0.99
                lr1 *= lr_multi
                lr2 *= lr_multi

            if (i+1) % args.log_interval == 0:

                agent.save()

            print("Ep_i {}, the ep_r is {}".format(i, episode_reward / t))
            run_results.append(episode_reward / t)
            ep_t.append(t)

        sio.savemat('ep_reward_ATO_BPPOcov_NAE_V9.mat', {'run_results': run_results, 'ep_t':ep_t,'ep_energy':ep_energy
                                                     ,'ep_nmse':ep_nmse})
        sio.savemat('Path_ATO_BPPOcov_NAE_V9.mat', {'location_path': location_path})


if __name__ == '__main__':
    main()






