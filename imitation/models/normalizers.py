import torch
import torch.nn as nn

class DictNormalizer(nn.Module):

    def __init__(self, normalization_stats, key_to_norm_type, device=None) -> None:
        super(DictNormalizer, self).__init__()

        self.normalizers = nn.ModuleDict()
        for key, value in normalization_stats.items():
            if key not in key_to_norm_type or key_to_norm_type[key] == None:
                continue
            if key_to_norm_type[key] == 'gaussian':
                normalizer_type = GaussianNormalizer
            elif key_to_norm_type[key] == 'bound':
                normalizer_type = BoundNormalizer
            else:
                raise ValueError(f"Normalizer type {type} not recognized")
            self.normalizers[key] = normalizer_type(key, value)


    def normalize(self, x):
        for key in x.keys():
            if key in self.normalizers:
                x[key] = self.normalize_by_key(x[key], key)
        return x
    
    def normalize_by_key(self, x, key):
        if key in self.normalizers:
            return self.normalizers[key].normalize(x)
        return x

    def unnormalize(self, x):
        for key in x.keys():
            if key in self.normalizers:
                x[key] = self.unnormalize_by_key(x[key], key)
        return x
    
    def unnormalize_by_key(self, x, key):
        if key in self.normalizers:
            return self.normalizers[key].unnormalize(x)
        return x

class GaussianNormalizer(nn.Module):

    def __init__(self, key, normalization_stat) -> None:
        super(GaussianNormalizer, self).__init__()

        self.key = key
        self.register_buffer('mean', torch.from_numpy(normalization_stat['mean']))
        self.register_buffer('std', torch.from_numpy(normalization_stat['std']))

    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def unnormalize(self, x):
        return x * self.std + self.mean
    
    def __repr__(self):
        return f"GaussianNormalizer({self.key})\nmean: {self.mean}\nstd: {self.std}\n"
    
class BoundNormalizer(nn.Module):
    
    def __init__(self, key, normalization_stat) -> None:
        super(BoundNormalizer, self).__init__()

        self.key = key
        self.register_buffer('min', torch.from_numpy(normalization_stat['min']))
        self.register_buffer('max', torch.from_numpy(normalization_stat['max']))
    
    def normalize(self, x):
        return (2 * (x - self.min) / (self.max - self.min)) - 1

    def unnormalize(self, x):
        return ((x + 1) / 2) * (self.max - self.min) + self.min

    def __repr__(self):
        return f"BoundNormalizer({self.key})\nmin: {self.min}\nmax: {self.max}\n"