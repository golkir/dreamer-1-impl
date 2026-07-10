
"""
What RSSM should do?
It splits the state into stochastic and deterministic part
deterministic part h_t is computed as h_t = f(h_t-1, s_t-1, a_t-1)
"""



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
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    


"""
Formulas:

reset_gate (r_t) = sigmoid(W_r [h_t-1, x_t] + b_r )
update_gate = sigmoid(W_z [h_t-1, x_t] + b_z)
candidate_h = tanh(W_h [r_t * h_t-1, x_t] + b_h)
h_t = (1 - z_t) * h_t-1 + z_t * candidate_h

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