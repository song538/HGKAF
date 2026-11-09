import torch
import torch.nn as nn
import torch.nn.functional as F

class FuzzyLayer3(nn.Module):
    def __init__(self, fuzzynum, channel, dropout=0.9):
        super(FuzzyLayer3, self).__init__()
        self.n = fuzzynum
        self.channel = channel
        self.mu = nn.Parameter(torch.randn((self.channel, self.n)))  # 模糊隶属函数的均值
        self.sigma = nn.Parameter(torch.randn((self.channel, self.n)))  # 模糊隶属函数的标准差
        self.bn = nn.BatchNorm1d(self.channel, affine=True)  # 批归一化，用于稳定训练

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)  # Shape: [B, C, WH, 1]
        mu_expanded = self.mu.view(1, self.channel, 1, self.n)  # Shape: [1, C, 1, N]
        sigma_expanded = self.sigma.view(1, self.channel, 1, self.n)  # Shape: [1, C, 1, N]

        # 计算高斯隶属函数
        tmp = -((x_expanded - mu_expanded) / sigma_expanded) ** 2
        tmp = torch.logsumexp(tmp, dim=-1)

        fNeural = self.bn(torch.exp(tmp))
        return fNeural + x
