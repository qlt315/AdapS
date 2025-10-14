import torch.nn as nn
from .modules import Resblock, MaskAttentionSampler    
    
class DTJSCC_CIFAR10(nn.Module):
    def __init__(self, in_channels, latent_channels, out_classes, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels
        self.prep = nn.Sequential(
                    nn.Conv2d(in_channels, latent_channels//8,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//8),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(latent_channels//8,latent_channels//4, kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//4),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(latent_channels//4,latent_channels//2,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(latent_channels//2,latent_channels,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(),
                    # nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = False)
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )

        self.encoder = nn.Sequential(
            self.prep,                    # 64x32x32
            self.layer1,                  # 128x16x16
            Resblock(latent_channels//4), # 128x16x16
            self.layer2,                  # 256x8x8
            self.layer3,                  # 512x4x4
            # Resblock(latent_channels),    # 512x4x4
            Resblock(latent_channels)     # 512x4x4
        )
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),      # 512x1x1
            nn.Flatten(),                 # 512
            nn.Linear(latent_channels, out_classes)
        )

    def encode(self, X):
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape

    def decode(self, features, former_shape):
        b, c , h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0,3,1,2).contiguous()
        tilde_X = self.decoder(features)
        return tilde_X

    def forward(self, X, mod=None):
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        tilde_X = self.decode(out, former_shape)

        return tilde_X, dist



class DTJSCC_CINIC10(nn.Module):
    def __init__(self, in_channels, latent_channels, out_classes, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels
        self.prep = nn.Sequential(
                    nn.Conv2d(in_channels, latent_channels//8,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//8),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(latent_channels//8,latent_channels//4, kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//4),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(latent_channels//4,latent_channels//2,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(latent_channels//2,latent_channels,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(),
                    # nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = False)
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )

        self.encoder = nn.Sequential(
            self.prep,                    # 64x32x32
            self.layer1,                  # 128x16x16
            Resblock(latent_channels//4), # 128x16x16
            self.layer2,                  # 256x8x8
            self.layer3,                  # 512x4x4
            # Resblock(latent_channels),    # 512x4x4
            Resblock(latent_channels)     # 512x4x4
        )
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),      # 512x1x1
            nn.Flatten(),                 # 512
            nn.Linear(latent_channels, out_classes)
        )

    def encode(self, X):
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape

    def decode(self, features, former_shape):
        b, c , h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0,3,1,2).contiguous()
        tilde_X = self.decoder(features)
        return tilde_X

    def forward(self, X, mod=None):
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        tilde_X = self.decode(out, former_shape)

        return tilde_X, dist

class DTJSCC_CIFAR100(nn.Module):
    def __init__(self, in_channels=3, latent_channels=512, out_classes=100, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels

        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels // 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 8),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(latent_channels // 8, latent_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(latent_channels // 4, latent_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(latent_channels // 2, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.encoder = nn.Sequential(
            self.prep,                      #  L/8 x 32 x 32
            self.layer1,                    #  L/4 x 16 x 16
            Resblock(latent_channels // 4), #  L/4 x 16 x 16
            self.layer2,                    #  L/2 x  8 x  8
            self.layer3,                    #   L  x  4 x  4
            Resblock(latent_channels),      #   L  x  4 x  4
            Resblock(latent_channels)       #   L  x  4 x  4
        )

        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)

        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),        #  L x 1 x 1
            nn.Flatten(),                   #  L
            nn.Linear(latent_channels, out_classes)  # default 100 classes
        )

    def encode(self, X):
        en_X = self.encoder(X)                                  # (B, L, 4, 4)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)  # (B*16, L)
        return en_X, former_shape

    def decode(self, features, former_shape):
        b, c, h, w = former_shape                                # c = L
        features = features.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        logits = self.decoder(features)                          # (B, 100)
        return logits

    def forward(self, X, mod=None):
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        logits = self.decode(out, former_shape)
        return logits, dist



    
