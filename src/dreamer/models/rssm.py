import torch
from torch import nn
from dreamer.utils import create_normal_dist_from_output
from dreamer.config import DreamerConfig, RSSMConfig
from dreamer.models.mlp import MLP



class RSSM(nn.Module):

    def __init__(self, action_dim: int, config: DreamerConfig):
        super(RSSM, self).__init__()
        # self.device = config.device
        self.config = config
        self.action_dim = action_dim
        self.gru = nn.GRUCell(
            config.rssm.z_dim + action_dim, config.rssm.h_dim
        )  # should we add activation and linear layers before gru
        self.posterior_mlp = MLP(
            config.rssm.obs_emb_dim + config.rssm.h_dim, config.rssm.z_dim * 2
        )  # * 2 to to include mu and sigma parameters of normal distribution
        self.prior_mlp = MLP(config.rssm.h_dim, config.rssm.z_dim * 2)

    def recurrent(self, z_prev, action_prev, h_prev):
        x = torch.cat([z_prev, action_prev], dim=-1)
        h = self.gru(x, h_prev)  # chec if the order is correct
        return h

    def encoder_posterior(self, h, obs_embed):
        """
        compute posterior distribution given previous deterministic state and observation embedding.
        This posterior becomes the new stochastic state of the representation model
        """


        x = torch.cat([h, obs_embed], dim=-1)
        x = self.posterior_mlp(
            x
        )  # (B, s_dim * 2) # mu and sigma parameters of the normal distribution
        posterior_dist = create_normal_dist_from_output(x)
        posterior = posterior_dist.rsample()
        return posterior, posterior_dist

    def encoder_prior(self, h):
        """
        compute prior stochastic state of the transition model
        q(s_t| s_t-1)
        """
        x = self.prior_mlp(
            h
        )  # (B, s_dim * 2) # mu and sigma parameters of the normal distribution
        prior_dist = create_normal_dist_from_output(x)
        prior = prior_dist.rsample()
        return prior, prior_dist

    def observe(self, ht, obs_embed):
        """
        Representation model
        p_θ(st |st-1,at-1,ot)
        h_t = gru(h_t-1, s_t-1, a_t-1) (GRU)
        """
        # compute new stochastic state using posterior
        # q(z_t | h_t, o_t)
        posterior, posterior_dist = self.encoder_posterior(ht, obs_embed)

        return posterior, posterior_dist

    def transition(self, ht) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Predicts next state st of the RSSM without looking at the observations (in contrast to the representation model)
        it should be RSSM
        conditioned only the deterministic hidden state computed by the representation model. Why?
        q(z_t | h_t)
        """

        prior, prior_dist = self.encoder_prior(ht)
        return prior, prior_dist
    
    def state_init(self, B):
        prior = torch.zeros(B, self.config.rssm.z_dim)
        h = torch.zeros(B, self.config.rssm.h_dim)
        return prior, h
