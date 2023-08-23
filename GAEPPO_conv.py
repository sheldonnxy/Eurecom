"""
Reinforcement Learning:
Proximal Policy Optimization (PPO),

PPO算法


"""


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
Transition = namedtuple('Transition', ['state', 'action', 'act_prob', 'reward', 'next_state', 'done'])




def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

########################### BN #############################

# class Actor_Net(nn.Module):
#     def  __init__(self,map_width,map_length,num_act):
#         super(Actor_Net, self).__init__()
#         self.map_width = map_width
#         self.map_length = map_length
#
#         # common layers
#         self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1,bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#
#
#         # action policy layers
#         self.act_conv1 = nn.Conv2d(64, 4, kernel_size=1,bias=False)
#         self.bn3 = nn.BatchNorm2d(4)
#
#         self.act_fc1 = nn.Linear(4 * self.map_width * self.map_length,
#                                  self.map_width * self.map_length)
#
#
#         self.act_fc2 = nn.Linear(self.map_width * self.map_length,
#                                  5*num_act)
#
#
#         self.act_fc3 = nn.Linear(5 * num_act,
#                                   num_act)
#
#     def forward(self, state_input):
#         # common layers
#         x = F.relu(self.bn1(self.conv1(state_input)))
#         x = F.relu(self.bn2(self.conv2(x)))
#
#
#         # action policy layers
#         x_act = F.relu(self.bn3(self.act_conv1(x)))
#         x_act = x_act.view(-1, 4 * self.map_width * self.map_length)
#         XXXX = self.act_fc1(x_act)
#         x_act = F.relu(self.act_fc1(x_act))
#         x_act = F.relu(self.act_fc2(x_act))
#
#         x_act = F.softmax(self.act_fc3(x_act), dim=1)
#         return x_act
#
# class Critic_Net(nn.Module):
#     def  __init__(self, map_width, map_length):
#         super(Critic_Net, self).__init__()
#         self.map_width = map_width
#         self.map_length = map_length
#
#         # common layers
#         self.conv4 = nn.Conv2d(6, 32, kernel_size=3, padding=1,bias=False)
#         self.bn6 = nn.BatchNorm2d(32)
#
#         self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1,bias=False)
#         self.bn7 = nn.BatchNorm2d(64)
#
#
#         # value layers
#         self.val_conv1 = nn.Conv2d(64, 2, kernel_size=1,bias=False)
#         self.bn8 = nn.BatchNorm2d(2)
#
#         self.val_fc1 = nn.Linear(2 * self.map_width * self.map_length,
#                                  256)
#         self.bn9 = nn.BatchNorm1d(256)
#
#         self.val_fc2 = nn.Linear(256, 32)
#         self.bn10 = nn.BatchNorm1d(32)
#
#         self.val_fc3 = nn.Linear(32, 1)
#
#     def forward(self, state_input):
#         # common layers
#         x = F.relu(self.bn6(self.conv4(state_input)))
#         x = F.relu(self.bn7(self.conv5(x)))
#
#
#         # value layers
#         x_val = F.relu(self.bn8(self.val_conv1(x)))
#
#         x_val = x_val.view(-1, 2 * self.map_width * self.map_length)
#         x_val = F.relu(self.val_fc1(x_val))
#         x_val = F.relu(self.val_fc2(x_val))
#         x_val = F.relu(self.val_fc3(x_val))
#
#         return x_val

######################   Actor_network-cov+MAX_pooling   ##############
class Actor_Net(nn.Module):
    def  __init__(self,map_width,map_length,num_act):
        super(Actor_Net, self).__init__()
        self.map_width = map_width
        self.map_length = map_length

        # common layers
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.MP1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.MP2 = nn.MaxPool2d(kernel_size=2, stride=2)


        # action policy layers
        self.act_conv1 = nn.Conv2d(64, 4, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(4)

        self.act_fc1 = nn.Linear(4 * 25,
                                 25)


        self.act_fc2 = nn.Linear(25,
                                 num_act)


    def forward(self, state_input):
        # common layers
        x = F.relu((self.conv1(state_input)))
        x = self.MP1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.MP2(x)


        # action policy layers
        x_act = F.relu(self.bn3(self.act_conv1(x)))
        x_act = x_act.view(-1, 4 * 25)
        XXXX = self.act_fc1(x_act)
        x_act = F.relu(self.act_fc1(x_act))

        x_act = F.softmax(self.act_fc2(x_act), dim=1)
        return x_act

######################   Critic_network-cov+MAX_pooling   ##############
class Critic_Net(nn.Module):
    def  __init__(self, map_width, map_length):
        super(Critic_Net, self).__init__()
        self.map_width = map_width
        self.map_length = map_length

        # common layers
        self.conv4 = nn.Conv2d(6, 32, kernel_size=3, padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.MP3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.MP4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # value layers
        self.val_conv1 = nn.Conv2d(64, 2, kernel_size=1,bias=False)
        self.bn8 = nn.BatchNorm2d(2)

        self.val_fc1 = nn.Linear(2 * 25,
                                 16)
        self.bn9 = nn.BatchNorm1d(256)

        self.val_fc2 = nn.Linear(16, 1)
        self.bn10 = nn.BatchNorm1d(32)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.bn6(self.conv4(state_input)))
        x = self.MP3(x)
        x = F.relu(self.bn7(self.conv5(x)))
        x = self.MP4(x)


        # value layers
        x_val = F.relu(self.bn8(self.val_conv1(x)))

        x_val = x_val.view(-1, 2 * 25)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.relu(self.val_fc2(x_val))

        return x_val

  ######################   PPO算法主函数   ##############
class PPO_Agent():
    def __init__(self, name, map_grid, num_act, agent_index, args, policy):
        # prepare
        self.name = name
        self.map_width = map_grid[0] + 1 #####  map_grid = [X,Y]
        self.map_length = map_grid[1] + 1
        self.num_act = num_act
        self.agent_index = agent_index
        self.args = args
        self.l2_const = 1e-4

        # build nn
        self.actor = Actor_Net(self.map_width, self.map_length, num_act)
        self.critic = Critic_Net(self.map_width, self.map_length)
        self.num_transition = 0
        # other need
        self.buffer = []
        self.directory = "./tmp/policy/"



        self.optimizer_actor = optim.Adam(self.actor.parameters(), weight_decay=self.l2_const)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), weight_decay=self.l2_const)

    ######################   select action   ##############
    def select_action(self, state):
        state = np.array(state)
        current_state = np.ascontiguousarray(state.reshape(
                -1, 6 , self.map_width, self.map_length))
        with torch.no_grad():
            act_probs = self.actor(Variable(torch.from_numpy(current_state)).float())
            # act_probs = np.exp(log_act_prob.data.numpy().flatten())

        dist = Categorical(act_probs)
        action = dist.sample()

        return action.item(), act_probs[:,action.item()].item()

    # def store_transition(self, transition):
    #     self.buffer.append(transition)

    ######################   store experience trace  ##############
    def store(self, s,  a, a_prob, r, s_, d):

        transition = Transition(s, a, a_prob, r, s_, d)
        # print(index)
        self.buffer.append(transition)

    ######################   update network  ##############
    def update(self, point_index,lr1,lr2):
        # state = Variable(torch.tensor([b.state for b in self.buffer], dtype=torch.float).to(device))
        state = torch.tensor([b.state for b in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([[b.action] for b in self.buffer], dtype=torch.long).to(device)
        old_act_prob = torch.tensor([b.act_prob for b in self.buffer], dtype=torch.float).view(-1, 1).to(device)
        next_state = torch.tensor([b.next_state for b in self.buffer], dtype=torch.float).to(device)
        # next_state = Variable(torch.tensor([b.next_state for b in self.buffer], dtype=torch.float).to(device))
        reward = torch.tensor([b.reward for b in self.buffer], dtype=torch.float).view(-1, 1)

        # compute AC-Gt
        # reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_v = reward + self.args.gamma * self.critic(next_state)

        ######################   compute advantage function  ##############
        adv = (target_v - self.critic(state)).detach()


        # compute GAE-Gt
        # Gt = []
        #         # R = 0
        #         # delta = 0
        #         # for index in range(len(reward[::-1]),0,-1):
        #         #     r = reward[index-1]
        #         #     s = state[index-1]
        #         #     s_ = next_state[index-1]
        #         #     V_s = self.critic(s)
        #         #     V_s_ = self.critic(s_)
        #         #     delta = r+self.args.gamma*V_s_-V_s + self.args.gamma*self.args.lambd*delta
        #         #     R = delta+V_s
        #         #     Gt.insert(0, R)

        # compute MC-Gt
        # Gt1 = []
        # R1 =0
        # for r in reward[::-1]:
        #     R1 = r + self.args.gamma * R1
        #     Gt1.insert(0, R1)
        # sio.savemat('comp_1.mat', {'Gt': Gt, 'Gt1': Gt1})



        # train nn by sample from buffer for iteration times
        for _ in range(self.args.num_iter):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.args.batch_size, False):


                # batch_action_prob1 = self.actor(state[index])
                batch_action_prob = self.actor(state[index]).gather(1, action[index])
                ratio = batch_action_prob / old_act_prob[index]

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.args.epsilon,
                                    1 + self.args.epsilon) * adv[index]

                # update actor-net: only change act_prob
                loss_actor = -torch.min(surr1, surr2).mean()
                self.optimizer_actor.zero_grad()
                set_learning_rate(self.optimizer_actor,lr1)
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.optimizer_actor.step()

                # update critic-net: only change V
                loss_critic = F.mse_loss(self.critic(state[index]), target_v[index])
                self.optimizer_critic.zero_grad()
                set_learning_rate(self.optimizer_critic, lr2)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.optimizer_critic.step()

        # clear buffer
        del self.buffer[:]

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







