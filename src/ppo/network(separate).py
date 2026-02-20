#separate actor and critic to prevent the gradients from mixing up
import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), #take 372 inputs, and expand to 512 features
            nn.ReLU(),
            nn.Linear(512, 256), # second hidden layer
            nn.ReLU(),
            nn.Linear(256, 128), #final hidden layer
            nn.ReLU(),
            nn.Linear(128, action_dim), #output 12 raw numbers (for each servo), or whatever action_dim is set to
            nn.Tanh() #squeeze numbers between -1 and 1 for easy scaling
        )

        #separate log_std for the actor
        #create a learnable vector of 12 zeros, representing the uncertainty
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    #to move layers and return distributions
    def forward(self, state):
        mu = self.net(state) #calculate mean, pass through network for ideal action
        std = self.log_std.exp().expand_as(mu) #convert log_std to actual standard deviation
        return Normal(mu, std) #create distribution of probability, allowing it to try slightly different things during training

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # copy of actor function, although it learns by itself
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) #outputs only in 1 number, the value
        )

    def forward(self, state):
        return self.net(state) #return predicted reward