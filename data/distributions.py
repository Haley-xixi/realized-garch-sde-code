import torch
from torch.distributions import Normal, Gamma, Beta, Uniform, StudentT, LogNormal, MultivariateNormal

__all__ = ["Normal","Gamma","Beta","Uniform","StudentT","LogNormal","MultivariateNormal","StructDist"]

class StructDist:
    def __init__(self, dists_dict):
        self.dists = dists_dict

    def sample(self, sample_shape=torch.Size()):
        return {k: dist.sample(sample_shape) for k, dist in self.dists.items()}

    def rsample(self, sample_shape=torch.Size()):
        out = {}
        for k, dist in self.dists.items():
            out[k] = dist.rsample(sample_shape) if hasattr(dist, "rsample") else dist.sample(sample_shape)
        return out

    def log_prob(self, values):
        lp = 0.0
        for k, dist in self.dists.items():
            lp = lp + dist.log_prob(values[k]).sum()
        return lp
