import torch
import math

class ChannelModel:
    def __init__(self, delta):
        self.delta = delta  # AWGN std

    def apply(self, X, return_h=False):
        raise NotImplementedError


class AWGNChannel(ChannelModel):
    def apply(self, X, return_h=False):
        noise = self.delta * torch.randn_like(X)
        y = X + noise
        if return_h:
            h = torch.ones_like(X)  # no fading
            return y, h
        return y


class RayleighChannel(ChannelModel):
    def apply(self, X, return_h=False):
        h = torch.randn_like(X)  # real Gaussian fading
        noise = self.delta * torch.randn_like(X)
        y = h * X + noise
        if return_h:
            return y, h
        return y


class RicianChannel(ChannelModel):
    def __init__(self, delta, K=3):
        super().__init__(delta)
        self.K = K

    def apply(self, X, return_h=False):
        los = torch.ones_like(X)
        nlos = torch.randn_like(X)
        h = math.sqrt(self.K / (self.K + 1)) * los + math.sqrt(1 / (self.K + 1)) * nlos
        noise = self.delta * torch.randn_like(X)
        y = h * X + noise
        if return_h:
            return y, h
        return y


class NakagamiChannel(ChannelModel):
    def __init__(self, delta, m=2):
        super().__init__(delta)
        self.m = m

    def apply(self, X, return_h=False):
        gamma = torch.distributions.Gamma(self.m, 1.0 / self.m)
        h_abs = torch.sqrt(gamma.sample(X.shape).to(X.device))  # envelope

        # Normalize power: E[h^2] = 1
        power = torch.mean(h_abs ** 2, dim=1, keepdim=True)
        h = h_abs / torch.sqrt(power + 1e-8)

        noise = self.delta * torch.randn_like(X)
        y = h * X + noise
        if return_h:
            return y, h
        return y
