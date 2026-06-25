import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from models.rssm import RSSM
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x




"""
Pass the observation through the CNN

"""

class RepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()

    def forward(self, x):


        pass






class Dreamer:
    def __init__(self, state_size):
        self.reward_hidden_size = 30 # hidden size of the reward network.
        self.reward_net = MLP(state_size, 1, 2,  )



class SequenceDataset(Dataset):
    def __init__(self, observations, actions, rewards):
        self.obs = observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return self.obs.shape[0]  # S

    def __getitem__(self, idx):
        return {
            "observations": self.obs[idx],  # (T, C, H, W)
            "actions": self.actions[idx],  # (T, A)
            "rewards": self.rewards[idx],  # (T, 1)
        }


def train():
    # sequence includes {action, observation, reward} for each timestep t

    # total observations (Sequences, T, C, H, W)
    S = 100
    T = 20
    W = 100
    H = 100
    C = 3
    actions_dim = 30
    low = -1
    high = 1
    observations = torch.rand((S, T, C, H, W))
    actions = low + (high - low) * torch.rand((S, T, actions_dim))
    rewards = torch.rand((S, T, 1)) - 1

    dataset = SequenceDataset(observations, actions, rewards)
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    for batch in loader:
        obs, actions, rewards = batch

        # loss = model(obs, actions, rewards)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()



if __name__ == "__main__":

    dims = [10, 20, 30, 40, 50, 1]

    # [(10, 20), (20, 30), (30, 40), (40, 50), (50, 1)]

    print(list(zip(dims[:-1], dims[1:])))

    train()
