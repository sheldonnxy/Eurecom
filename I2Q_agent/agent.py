import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# agent act to maximize the reward
# get the optimal trajectory for collecting measurements under the assumption of reward
# use the pre-trained UNet to predict based on the optimal trajectory
# result is slightly better than the original predicted one


import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, num_features, n_actions):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, state_space, n_actions):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class QSS(nn.Module):
    def __init__(self, state_space):
        super(QSS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state1, state2):
        x = torch.cat([state1, state2], dim=1)
        return self.model(x)

class FS(nn.Module):
    def __init__(self, state_space, n_actions):
        super(FS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_space)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        mu = self.model(x)
        return mu + state

class Agent:
    def __init__(self, state_space, n_actions, n_ant, alpha, device='cpu'):
        self.n_actions = n_actions
        self.n_ant = n_ant
        self.state_space = state_space
        self.gamma = 0.99
        self.alpha = alpha
        self.update_type = 1
        self.device = device

        self.actor = [Actor(state_space, n_actions).to(device) for _ in range(n_ant)]
        self.critic = [Critic(state_space, n_actions).to(device) for _ in range(n_ant)]
        self.qss = [QSS(state_space).to(device) for _ in range(n_ant)]
        self.fs = [FS(state_space, n_actions).to(device) for _ in range(n_ant)]

        self.actor_tar = [Actor(state_space, n_actions).to(device) for _ in range(n_ant)]
        self.critic_tar = [Critic(state_space, n_actions).to(device) for _ in range(n_ant)]

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=0.001) for actor in self.actor]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=0.001) for critic in self.critic]
        self.qss_optimizers = [optim.Adam(qss.parameters(), lr=0.001) for qss in self.qss]
        self.fs_optimizers = [optim.Adam(fs.parameters(), lr=0.001) for fs in self.fs]

        self.init_update()

    def train(self, S, A, R, Next_S, D):
        for i in range(self.n_ant):
            self.actor_optimizers[i].zero_grad()
            self.critic_optimizers[i].zero_grad()
            self.qss_optimizers[i].zero_grad()
            self.fs_optimizers[i].zero_grad()

            s = torch.FloatTensor(S[i]).to(self.device)
            a = torch.FloatTensor(A[i]).to(self.device)
            r = torch.FloatTensor(R).to(self.device)
            next_s = torch.FloatTensor(Next_S[i]).to(self.device)
            d = torch.FloatTensor(D).to(self.device)

            # Actor loss
            actor_loss = -self.critic[i](s, self.actor[i](s)).mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            if self.update_type == 0:  # IQL
                Q_target = r + self.gamma * (1 - d) * self.critic_tar[i](next_s, self.actor_tar[i](next_s)).detach()
                critic_loss = nn.MSELoss()(self.critic[i](s, a), Q_target)
                critic_loss.backward()
                self.critic_optimizers[i].step()
            else:  # I2Q
                predicted_Next_S = self.fs[i](s, a)
                q_opt = self.qss[i](s, predicted_Next_S)
                lamb = self.alpha / torch.abs(q_opt).mean().detach()
                fs_loss = -lamb * q_opt.mean() + ((predicted_Next_S - next_s) ** 2).mean(dim=1).mean()
                fs_loss.backward()
                self.fs_optimizers[i].step()

                c = (self.qss[i](s, next_s) < self.qss[i](s, predicted_Next_S)).float()
                predicted_Next_S = predicted_Next_S * c + next_s * (1 - c)
                Q_target1 = r + self.gamma * (1 - d) * self.critic_tar[i](predicted_Next_S, self.actor_tar[i](predicted_Next_S)).detach()
                Q_target2 = r + self.gamma * (1 - d) * self.critic_tar[i](next_s, self.actor_tar[i](next_s)).detach()

                critic_loss = nn.MSELoss()(self.critic[i](s, a), torch.min(Q_target1, Q_target2))
                critic_loss.backward()
                self.critic_optimizers[i].step()

                qss_loss = nn.MSELoss()(self.qss[i](s, next_s), Q_target2)
                qss_loss.backward()
                self.qss_optimizers[i].step()

    def update(self):
        for i in range(self.n_ant):
            for target_param, param in zip(self.actor_tar[i].parameters(), self.actor[i].parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
            for target_param, param in zip(self.critic_tar[i].parameters(), self.critic[i].parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    def init_update(self):
        for i in range(self.n_ant):
            self.actor_tar[i].load_state_dict(self.actor[i].state_dict())
            self.critic_tar[i].load_state_dict(self.critic[i].state_dict())
            

    #  save network parameter
    def save(self):
        os.makedirs(os.path.dirname(self.directory), exist_ok=True)
        torch.save(self.actor.state_dict(), self.directory + 'actor_NAEV1.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic_NAEV1.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    #  load network parameter
    def load(self):
        os.makedirs(os.path.dirname(self.directory), exist_ok=True)
        self.actor.load_state_dict(torch.load(self.directory + 'actor_NAEV1.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic_NAEV1.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


