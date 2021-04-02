import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, planes=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=2, bias=False, padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=2, bias=False, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        print("Out: ", out.shape)
        print("Residual: ", residual.shape)
        out = out + residual
        return out


class SRGAN_g(torch.nn.Module):
    def __init__(self):
        super(SRGAN_g, self).__init__()
        nf = 64

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 2, bias=True, padding_mode='circular')
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.conv_down = nn.Conv2d(nf, nf, 3, 1, 2, bias=True, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(nf, affine=True)

        basic_block = functools.partial(ResidualBlock, planes=nf)
        self.recon_trunk = make_layer(basic_block, 16)

        self.conv_out = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf, affine=True)

        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = DepthToSpace(2)

        self.conv_last = nn.Conv2d(nf, 3, 1, 1, 0, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()


    def forward(self, x):

        
        fea = self.relu(self.conv_first(x))

        fea = self.avg_pool(fea)

        fea = self.relu(self.bn1(self.conv_down(fea)))

        fea0 = fea

        out = self.recon_trunk(fea)
        out = self.bn2(self.conv_out(out))

        out = out + fea0
        out = self.relu(self.pixel_shuffle(self.upconv1(out)))
        out = self.tanh(self.conv_last(out))

        return out