import torch
from torch import nn
from torch.nn import Linear
from torch.nn import Sigmoid, ReLU
from torch.nn.functional import sigmoid, tanh
from torch.distributions import MultivariateNormal
from einops import rearrange
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class RSSMState:
    h: torch.Tensor      # deterministic
    z: torch.Tensor      # stochastic

class CNNDecoder(nn.Module):
    """
    Decodes latent state back into image observation.
    
    Input : (B, h_dim + s_dim)  — combined latent state
    Output: (B, 3, 64, 64)      — reconstructed image
    """

    def __init__(self, latent_dim: int = 230, depth: int = 32):
        super().__init__()

        # project flat latent to spatial tensor before upsampling
        self.input_proj = nn.Linear(latent_dim, 32 * depth)  # 1024

        self.decoder = nn.Sequential(
            # (B, 32*depth, 1, 1) → (B, 4*depth, 5, 5)
            nn.ConvTranspose2d(32 * depth, 4 * depth, kernel_size=5, stride=2),
            nn.ELU(),
            # (B, 4*depth, 5, 5) → (B, 2*depth, 13, 13)
            nn.ConvTranspose2d(4 * depth,  2 * depth, kernel_size=5, stride=2),
            nn.ELU(),
            # (B, 2*depth, 13, 13) → (B, 1*depth, 30, 30)
            nn.ConvTranspose2d(2 * depth,  1 * depth, kernel_size=6, stride=2),
            nn.ELU(),
            # (B, 1*depth, 30, 30) → (B, 3, 64, 64)
            nn.ConvTranspose2d(1 * depth,  3,         kernel_size=6, stride=2),
        )
        # no activation on final layer — output is raw logits or pixel values

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) or (B, T, latent_dim)
        Returns:
            obs: (B, 3, 64, 64) or (B, T, 3, 64, 64)
        """
        # handle both 2D and 3D inputs
        shape = latent.shape[:-1]                        # (B,) or (B, T)
        latent = latent.view(-1, latent.shape[-1])       # flatten to (B*T, latent_dim)

        x = self.input_proj(latent)                      # (B*T, 1024)
        x = x.view(-1, 32 * 32, 1, 1)                   # (B*T, 1024, 1, 1) — seed spatial tensor
        x = self.decoder(x)                              # (B*T, 3, 64, 64)

        return x.view(*shape, *x.shape[1:])              # restore (B, 3, 64, 64) or (B, T, 3, 64, 64)
class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size=256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend(
                [
                    nn.Linear(layer_dim_in, layer_dim_out),
                    nn.GELU() if not is_last else nn.Identity(),
                ]
            )

        self.net = nn.Sequential(*layers, nn.Linear(hidden_size, dim_out))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, input_channels=3, feature_dim=1024):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ELU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ELU()
        )

        self.fc = nn.Linear(256 * 3 * 3, feature_dim)  # assumes 64x64 input

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

"""
Formulas:

reset_gate (r_t) = sigmoid(W_r [h_t-1, x_t] + b_r )
update_gate = sigmoid(W_z [h_t-1, x_t] + b_z)
candidate_h = tanh(W_h [r_t * h_t-1, x_t] + b_h)
h_t = (1 - z_t) * h_t-1 + z_t * candidate_h

"""


"""
1. apply cnn to all the observations

2. 

"""

class GRUCell():
    def __init__(self, h_dim = 20, input_dim = 20,  s_dim = 20):
        super(RSSM, self).__init__()
        self.hidden_dim = h_dim
        self.s_dim = s_dim
    
        self.ih = Linear(in_features=input_dim, out_features=h_dim * 3)
        self.hh = Linear(h_dim, h_dim * 3)
    def forward(self, hprev, sprev):
       
        # now we can compute gru as usual
        xh = self.ih(sprev) 
        hh = self.hh(hprev)

        xh_r, xh_u, xh_c = torch.chunk(xh, 3, 1)
        hh_r, hh_u, hh_c = torch.chunk(hh, 3, 1)

        reset = sigmoid(xh_r + hh_r)
        update = sigmoid(xh_u + hh_u)
        candidate = sigmoid(xh_c + (hh_c * reset))

        ht = (1 - update ) * hprev + update * candidate
        return ht



class RSSM(nn.Module):

    def __init__(self, input_dim = 10, h_dim = 20, s_dim = 20, a_dim = 20, emb_dim = 20):
        super(RSSM, self).__init__()
        self.hidden_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.emb_dim = emb_dim
        self.ih = Linear(in_features=input_dim, out_features=h_dim * 3)
        self.hh = Linear(h_dim, h_dim * 3)
        self.gru_representation = GRUCell(h_dim, input_dim, s_dim)
        self.gru_transition = GRUCell(h_dim, input_dim, s_dim)
        self.cnn = CNN()
        # the output are mu and sigma parameters of the normal distribution
        self.posterior_mlp = MLP(emb_dim, s_dim * 2 ) # to include mu and sigma parameters of normal distribution
        self.transition_mlp = MLP(h_dim, s_dim * 2) 

    def encoder(self, ht, emb):

        """
        somewhere in the course of action we need to also sample the st from the distribution
        to create real st
        """

        # we encode both RNN hidden state and observation passed through CNN 
        x = torch.cat([ht, emb], dim=-1)
        device = x.device
        """
        now we use posterior MLP. What should be the output dim?
        """
        s = self.posterior_mlp(x) # (B, s_dim * 2) # mu and sigma parameters of the normal distribution

        mu, sigma = s.chunk(2, dim = -1) # we predict log var
        
        std = torch.exp(0.5 * sigma)
        
        z = torch.distributions.Normal(mu, std)

        # reparametrize

        eps = torch.randn_like(std).to(device)
        z = mu + eps * sigma
        return z
    
    def observe(self, x):

        """
        Representation model
        p_θ(st |st-1,at-1,ot)
        h_t = gru(h_t-1, s_t-1, a_t-1) (GRU)
        """

        B, L, C, H, W = x.shape #observations

        # initialize h, s and a (hidden, stochastic state, and action) for step 0

        h = torch.zeros((B, self.h_dim)) 
        s = torch.zeros((B, self.s_dim))
        a = torch.zeros((B, self.a_dim))

        # deterministic part h will be stored in hs, stochastic - in ss
        hs = torch.zeros((B, L, self.h_dim ))
        ss = torch.zeros((B, L, self.s_dim))


        for t in range(L):
            """
            should we concat s (stochastic part) and action before passing through GRU?
            """
            s = torch.cat(s, a, dim=-1)
            ht = self.gru_representation(h, s)
            # cnn encoder
            emb = self.cnn(x)
            # now we pass the observation through the stochastic encoder
            emb = torch.cat(ht, emb) # should we cat here? yes, because want the stochastic state depend on the ht latent history
            st = self.encoder(ht, emb)
            ss[t] = st
            hs[t] = ht
        return hs, ss

    def imagine_multiple(self, hs, ss,horizon=10):

        """
        let's say we have (5, 10, 20) input where 5 is batch 10 is the number of steps in trajectory, 20
        is a hidden dimension. How can we generate 50 trajectories in one go?

        if we transform to (5 * 10, 20) we'll 50 states which is 5 trajectories however we want 50 trajectories
        so or each of the 50 states we just call imagine

        Also, I think we should not use imagine used in the transition model because it depends on the 
        representation model states, uses them to predict the next step
        in contrast we need to image trajectory like representation model does but without any observation,
        just starting with some state of the representation model. 
        """
        B, L, H = ss.shape # batch, all states, hidden 
        s_init = rearrange(ss, 'b l s -> (b l) s')   # (B*L, s_dim)
        h_init = rearrange(hs, 'b l h -> (b l) h')   # (B*L, h_dim)
        trajectories = torch.zeros((s_init.shape[0], L, H))

        """
        we have 50 initial states. we need to create a trajectory for each of them
        """
        for t in range(horizon):
            # at the first step we use s_init and h_init (computed by representation model) to kick off the trajectory
            



    
    def imagine_single(self, hs, ss, horizon = 10):
        """
        Predicts next state st of the RSSM without looking at the observations (in contrast to the representation model)
        it should be RSSM

        hs - hidden states from representation model
        ss - stochastic states from representation model

        should we explicitly condition on actions from the action model because
        the paper says this:
        These trajectories branch off of the model states s of τ=t
        sequence batches drawn from the agents dataset of experience and predict 
        forward for the imagination horizon H using actions sampled from the action model
        """

        B, L, H = hs.shape
        # we feed states computed by the representation model and computed next st based on them
        # we don't see any observations
        # how should we use hs and ss from representation model
        transition_states = torch.zeros((B, L, H))
        for t in range(horizon): 
            ht = self.gru_transition(hs[:, t], ss[:, t])
            # predict st
            st = self.transition_mlp(ht) # parameters of the Gaussian distribution
            transition_states[:, t] = st # do we also need to store hs or we compute KL divergence only using st
        return transition_states
    
    def kl_regularizer_loss(posterior, prior):
        """
        the world model loss function. Computed as the KL divergence between 
        """
        posterior_mean = posterior[..., 0]
        posterior_std = posterior[..., 1]
        prior_std = posterior[..., 1]
        prior_mean = posterior[..., 0]
        posterior_dist = MultivariateNormal(posterior_mean, scale_tril=torch.diag_embed(posterior_std))
        prior_dist = MultivariateNormal(prior_mean, scale_tril=torch.diag_embed(prior_std))
        kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        return kl
    

    def forward(self, x):
        B, L, C, H, W = x.shape
        """
        This RSSM is for learning the world model: representation vs transition models
        """

        # first we run representation model

        hs_repr, ss_repr = self.observe(x)

        # run the transition model. it should predict the next states of the representation model
        ss_transition = self.imagine(hs_repr, ss_repr)

        return ss_repr, ss_transition
        


class RewardModel(nn.Module):
    def __init__(self, st_dim, out_dim=1):
        self.mlp = MLP(st_dim, out_dim)
    def forward(self, st):
        return self.mlp(st)

class ObservationModel():
    def __init__(self):
        pass



class Actor(nn.Module):
    """
    q(a_r | s_r)

    the output should be:
    actions are vector valued, each action in the vector is Gaussian output by dense neural net passed through 
    the tanh function. 

    Input: sr_dim - let's say B, L, H
    We want B, L, A, 2, we can use rearrange to achieve this

    Shape: 
    """
    def __init__(self, sr_dim, a_dim):
        super().__init__()  
        self.sr_dim = sr_dim
        self.mlp = MLP(sr_dim, a_dim * 2 )
    def forward(self, sr):
        sr = rearrange(sr, 'B L H -> (BL) H')
        h = self.mlp(sr)
        h = rearrange(h, '(b l) (a p) -> b l a p', p=2)
        B, L, A, P = h.shape
        mean = h[..., 0]
        std     = F.softplus(h[..., 1]) + 0.1
        z = torch.randn((B, L, A))
        ar = F.tanh(mean + std * z )
        return ar




"""
Dreamer uses V_λ, an exponentially-weighted average of the estimates for different k to balance bias and variance
"""
def k_step_return(rewards, values_pred, gamma = 0.99, lam = 0.95, k = 5, horizon = 8):

    discounts = gamma ** torch.arange(k, device=rewards.device)
    reward_part = (rewards[..., : k] * discounts).sum(-1)
    bootstrap = gamma ** k * values_pred
    return reward_part + bootstrap

def lambda_return(rewards, values, lam = 0.95, H = 9):

    discounts = lam ** torch.arange(H)
    returns = torch.tensor([k_step_return(rewards, values, k) for k in range(H)]).to(torch.float32).to(rewards.device)
    return  (1 - lam) * (discounts[: -1] * returns[: -1]).sum(-1) + discounts[-1] * returns[-1]


# optimized O(H) lambda return

def lambda_return_optimized(rewards, values, gamma, lam):
    """
    rewards: [B, H]
    values:  [B, H+1]

    returns:
        lambda_returns: [B, H]
    """
    H = rewards.shape[1]

    returns = torch.empty_like(rewards)

    ret = values[:, -1]

    for t in reversed(range(H)):
        ret = rewards[:, t] + gamma * (
            (1 - lam) * values[:, t + 1]
            + lam * ret
        )
        returns[:, t] = ret

    return returns

"""
Loss: min E[Sum(v_model(s_r) - V(s_r)^2)]
"""

class Critic(nn.Module):
    def __init__(self, s_dim: int = 20,  k_length: int = 4, discount: float = 0.99):
        super().__init__()
        self.mlp = MLP(s_dim, 1)

    def value_loss(self, value_pred, lambda_values):
        loss = F.mse_loss(value_pred, lambda_values)
        return loss
        
    def forward(self, sr):
        lambda_values = lambda_return_optimized(self.k_length, self.discount)
        value_pred = self.mlp(sr)
        total_value = lambda_values + value_pred


"""

Dreamer


"""

class Dreamer(nn.Module):
    def __init__(self):
        self.world_model = RSSM()
        actor = Actor()
        critic = Critic()
        reward_model = RewardModel()
        pass

    def imagine_trajectories(hs, ss, horizon):
        """
        imagine trajectory using transition model but use horizon and do it from each state
        so let's say you have B = 5, L = 10, you'll have 50 trajectories kicking off from each of L = 10
        """

        # flatten B and L into one batch dimension
        s_init = rearrange(ss, 'b l s -> (b l) s')   # (B*L, s_dim)
        h_init = rearrange(hs, 'b l h -> (b l) h')   # (B*L, h_dim)



    def forward(self, x):
        # compute model states using representation model

        

# """

# Recurrent State Space Model

# """
# class RSSM():
#     def __init__(self):
#         pass

#     def forward(self):
#         pass


import torch.nn as nn

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1)

# Print the shapes of the master weight matrices
print(gru.weight_ih_l0.shape)  # Output: torch.Size([60, 10])
print(gru.weight_hh_l0.shape)  # Output: torch.Size([60, 20])



if __name__ == "__main__":
    # gru = GRU(input_size=10, hidden_size=20)

    # x = torch.randn((1,10))

    # h_new = gru(x)
    # print(h_new.shape, "Shape of a new hidden state")

    schedule = discount_schedule(4, 0.99)
    print(schedule, "Discount schedule")


