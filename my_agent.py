#TODO
#stable update method is needed
import gym
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen = buffer_limit)
        # 记录每个样本被采样的概率
        self.p_sample = None

    def put(self, seq_data):
        self.buffer.append(seq_data)
    
    def sample(self, batch_size, on_policy = False, _all = False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            # 论文里根据TD error和Q值增加了每个样本的重要度，最后需要修改
            mini_batch = random.sample(self.buffer, batch_size)
            # if self.p_sample == None:
            #     mini_batch = random.sample(self.buffer, batch_size)
            # else:
            #     mini_batch = []
            #     for i in range(batch_size):
            #         mini_batch.append(np.random.choice(self.buffer, p = list(self.p_sample)))

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s, a, r, prob, done_mask, is_first = torch.tensor(s_lst, dtype = torch.float), torch.tensor(a_lst), \
                                        r_lst, torch.tensor(prob_lst, dtype = torch.float), done_lst, \
                                        is_first_lst
        return s, a, r, prob, done_mask, is_first
    
    def size(self):
        return len(self.buffer)
        
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2) # 输出维度得改
        self.fc_q = nn.Linear(256, 2)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        pi = F.softmax(x, dim = softmax_dim)
        return pi
    
    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q
      
def train(model, optimizer, memory, c, gamma, clipping, batch_size, on_policy = False, calculate_prio = False):
    s, a, r, prob, done_mask, is_first = memory.sample(batch_size, on_policy)
    if calculate_prio:
        s, a, r, prob, done_mask, is_first = memory.sample(memory.size(), on_policy)
    
    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim = 1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()
    
    rho = pi.detach() / prob
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max = c)

    q_ret = v[-1] * done_mask[-1]
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = (rho_bar[i]+ ((rho_a[i] - c)/rho_a[i]).clamp(min = 0)) * (q_ret - q_a[i]) + v[i] 
        
        if is_first[i] and i!=0:
            q_ret = v[i-1] * done_mask[i-1] # When a new sequence begins, q_ret is initialized  
            
    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype = torch.float).unsqueeze(1)
    
    prob_a = prob.gather(1, a)
    ratio_a = torch.min(pi_a / prob_a, torch.clamp(pi_a / prob_a, 1 - clipping, 1 + clipping))
    grad = - ratio_a * (q_ret - v) # that's all about ppo
    loss = grad.mean() + F.smooth_l1_loss(q_a, q_ret)

    if calculate_prio:
        fi = 0.6
        ci = 0.01
        # 优先级
        p = fi * (abs(loss) + ci) + (1 - fi) * abs(grad)
        # 经验池中每个样本的采样概率
        memory.p_sample = p / sum(p)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()