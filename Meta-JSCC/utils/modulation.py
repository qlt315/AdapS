import torch 
import numpy as np
import math
from . import channel

def channel_est(h_true: torch.Tensor, est_var: float) -> torch.Tensor:
    """
    Additive Gaussian noise model for imperfect channel estimation:
        ĥ = h + ε, where ε ~ N(0, est_var)
    """
    noise = torch.randn_like(h_true) * math.sqrt(est_var)
    return h_true + noise


class PSK:
    def __init__(self, M, PSNR, channel_type='awgn', channel_args=None):
        if channel_args is None:
            channel_args = {}
        self.M = M
        self.p = 1
        self.delta = self.compute_noise(PSNR)
        self.constellation = self.build()
        self.channel = self.build_channel(channel_type,channel_args)
    
    def compute_noise(self, PSNR):
        delta_2 = self.p/torch.pow(torch.tensor(10), PSNR/10).float()  
        return torch.sqrt(delta_2/2)

    def apply_channel(self, X, est_var: float = 0.0):
        """
        Pass X through channel with optional equalization.
        If `equalize` is True, estimate the channel and apply ZF equalization:
            y_eq = y / ĥ
        """
        eps = 1e-6
        y, h = self.channel.apply(X, return_h=True)  # real channel
        h_hat = channel_est(h, est_var)  # channel estimation
        y = y / (h_hat + eps) # ZF equalization
        return y

    def build_channel(self, channel_type, channel_args):
        if channel_type == 'awgn':
            return channel.AWGNChannel(self.delta)
        elif channel_type == 'rayleigh':
            return channel.RayleighChannel(self.delta)
        elif channel_type == 'rician':
            return channel.RicianChannel(self.delta, **channel_args)
        elif channel_type == 'nakagami':
            return channel.NakagamiChannel(self.delta, **channel_args)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

    def build(self):
        constellation = torch.ones(self.M, 2)
        for i in range(self.M):
            constellation[i,0] = np.cos(2*np.pi*i/self.M)
            constellation[i,1] = np.sin(2*np.pi*i/self.M)
        return constellation
    
    def modulate(self, z:torch.Tensor):
        m = z.shape[0]
        X = torch.ones(int(m), 2)
        for i in range(m):
            X[i] = self.constellation[int(z[i])]
        return X
    
    def demodulate(self, X):
        inner = np.matmul(X, self.constellation.T)
        return np.argmax(inner, axis=1)


    
def ser(p, d):
    N= 2*d**2
    return 1.5*math.erfc(math.sqrt(p/(10*N)))



class QAM:
    def __init__(self, M, PSNR, channel_type='awgn', channel_args=None):
        self.M = M
        self.max = int(np.sqrt(self.M))-1
        self.constellation, self.map = self.build()
        self.p = (M-1)/6
        self.delta = self.compute_noise(PSNR)
        self.channel = self.build_channel(channel_type, channel_args)
        
    def compute_noise(self, PSNR):
        delta_2 = self.p/torch.pow(torch.tensor(10), PSNR/10).float()  
        return torch.sqrt(delta_2/2)
        
    def build(self):
        l = []
        d = {}
        m = int(np.sqrt(self.M))
        for i in range(m):
            for j in range(m):
                l.append((i,j))
                
        for i in range(self.M):
            d[l[i]] = i
        return l, d
    
    def modulate(self, z:torch.Tensor):
        m = z.shape[0]
        # print(m)
        X = torch.ones(int(m), 2)
        for i in range(m):
            x, y = self.constellation[int(z[i])]
            X[i,0] = x
            X[i,1] = y 
        return X
    
    def apply_channel(self, X, est_var: float = 0.0):
        """
        Pass X through channel with optional equalization.
        If `equalize` is True, estimate the channel and apply ZF equalization:
            y_eq = y / ĥ
        """
        eps = 1e-6
        y, h = self.channel.apply(X, return_h=True)  # real channel
        h_hat = channel_est(h, est_var)  # channel estimation
        y = y / (h_hat + eps) # ZF equalization
        return y

    def build_channel(self, channel_type, channel_args):
        if channel_type == 'awgn':
            return channel.AWGNChannel(self.delta)
        elif channel_type == 'rayleigh':
            return channel.RayleighChannel(self.delta)
        elif channel_type == 'rician':
            return channel.RicianChannel(self.delta, **channel_args)
        elif channel_type == 'nakagami':
            return channel.NakagamiChannel(self.delta, **channel_args)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

    def demodulate(self, X):
        m = X.shape[0]
        Z = torch.ones(m).long()
        for i in range(m):
            x = self.assign(X[i,0])
            y = self.assign(X[i,1])
            Z[i] = self.map[(x,y)]
        return Z
    
    def assign(self, ele):
        num = int(torch.round(ele))
        if num > self.max:
            num = self.max
        if num < 0:
            num = 0
        return num


