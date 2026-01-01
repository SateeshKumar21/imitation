import torch

class FlowTimeSampler:
    def __init__(self, flow_sampling, flow_sig_min, flow_alpha, flow_beta):
        self.flow_sampling = flow_sampling
        self.flow_sig_min = flow_sig_min
        
        self.flow_t_max = 1 - self.flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        else:
            raise NotImplementedError(f"Unknown flow sampling: {self.flow_sampling}")
        return t