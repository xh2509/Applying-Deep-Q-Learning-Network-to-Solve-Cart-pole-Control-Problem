import numpy as np
import torch
import torch.optim as optim
import random
from model import Q_network

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda" if use_cuda else "cpu")
from  torch.autograd import Variable
from replay_buffer import ReplayMemory, Transition


class Agent(object):

    def __init__(self, n_states, n_actions, hidden_dim, BATCH_SIZE, lr):
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
         #  ReplayMemory: trajectory is saved here
        self.rememory = ReplayMemory(10000)
        
        self.q_old = Q_network(n_states, n_actions, hidden_dim=hidden_dim).to(device)
        self.q_new = Q_network(n_states, n_actions, hidden_dim=hidden_dim).to(device)
        
        self.loss = torch.nn.MSELoss()
        self.optim = optim.RMSprop(self.q_old.parameters(), lr = lr)
        
        self.n_states = n_states
        self.n_actions = n_actions 

    def get_action(self, state, eps, check_eps=True):
        """Returns an action
        Args:
            state : 2-D tensor of shape (n, input_dim)
            eps (float): eps-greedy for exploration
        Returns: int: action index
        """
        global steps_done
        sample = random.random()

        if check_eps==False or sample > eps:
            with torch.no_grad():
                return self.q_old(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            ran_num = [[random.randrange(self.n_actions)]]
            return torch.tensor(ran_num, device=device) 

    
    def learn(self, experiences, gamma, BATCH_SIZE):
        """Prepare minibatch and train them
        Args:
        experiences (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
        """
        
        if len(self.rememory.memory) < BATCH_SIZE:
            return;
            
        transitions = self.rememory.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
                        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        
        Q_max_action = self.q_old(next_states).detach().max(1)[1].unsqueeze(1)
        Q_next = self.q_new(next_states).gather(1, Q_max_action).reshape(-1)

        Q_targets = rewards + (gamma * Q_next * (1-dones))
        Q_expected = self.q_old(states).gather(1, actions)
                
        self.optim.zero_grad()
       
        self.loss(Q_expected, Q_targets.unsqueeze(1)).backward()
        self.optim.step()
        
        