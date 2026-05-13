import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torchdiffeq import odeint
from torch_scatter import scatter_mean, scatter_sum
import torch.distributions as dist

import numpy as np


class PIFBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def continuous_var_interpolation_update(self, t, x, s0, s1, prior=None):
        # """
        # x: [N, D]
        # """

        if prior is None:
            prior = [torch.zeros_like(x).to(x.device), s0 * torch.ones_like(x).to(x.device)]  # std

        gamma = t  # [0,1]
        
        s1_modify = (1 - gamma) * s1  # s1_modify is changed from s1 to 0


        e_coeff = torch.sqrt(s0**2*s1_modify**2/((1-gamma)*s1_modify**2 + gamma*s0**2))
        # u_coeff = gamma * (e_coeff/s1)**2
        u_coeff = gamma * s0**2/((1-gamma)*s1_modify**2 + gamma*s0**2)

        x_flow = [u_coeff * x, e_coeff]

        
        mu = x_flow[0] + x_flow[1] * torch.randn_like(x).to(x.device)


        # laplace_dist = torch.distributions.Laplace(x_flow[0], x_flow[1])  # laplace prior
        # mu = laplace_dist.sample().to(self.device)

        return mu


    def discrete_var_interpolation_update(self, t, x, K, s1, prior=None):
        # """
        # x: [N, K]
        # """
        if prior is None:
            # prior = torch.ones_like(x).to(x.device) / K  
            prior = torch.ones_like(x).to(x.device)  # better than 1/K

        gamma = t  # [0,1]

        s1_modify = torch.clamp((1 - gamma) * s1, min=1e-3)
        s1_modify = K * s1_modify / (1 - s1_modify + K * s1_modify)  # to make (1-s1_modify) / (s1_modify/K) = [1-(1 - gamma) * s1]/[(1 - gamma) * s1], see eq. below. Not necessary.

        soft_x = (1 - s1_modify) * x + s1_modify / K
        x_flow = gamma * soft_x + (1 - gamma) * prior


        # s1_modify = torch.clamp((1 - gamma) * s1, min=1e-3)  # Dirichlet Flow setup
        # x_multi = x / s1_modify + prior
        # x_flow = gamma * x_multi  + (1 - gamma) * prior


        dirichlet_dist = torch.distributions.Dirichlet(x_flow)
        theta = dirichlet_dist.sample().to(x.device)

        return theta


    def dtime4continuous_interpolation_loss(self, t, N, x_pred, x, s0, s1, segment_ids=None):
        gamma = t

        s0_expand = s0 * torch.ones([x_pred.shape[0],1]).to(x_pred.device)

        s1_modify = torch.clamp((1 - gamma) * s1, min=1e-3)
        s1_expand = s1_modify * torch.ones([x_pred.shape[0],1]).to(x_pred.device)

        e_coeff = torch.sqrt(s0_expand**2*s1_expand**2/((1-gamma)*s1_expand**2 + gamma*s0_expand**2))
        u_coeff = gamma * s0**2/((1-gamma)*s1_modify**2 + gamma*s0**2)

        theta1 = (u_coeff * x_pred, e_coeff)
        theta2 = (u_coeff * x, e_coeff)

        if segment_ids is not None:
            loss = scatter_mean(
                self.gauss_kl_batch(theta1, theta2), segment_ids, dim=0
            )

        return loss


    def gauss_kl_batch(self, theta1, theta2) -> torch.Tensor:
        u1, s1 = theta1
        u2, s2 = theta2

        term1 = torch.log(s1/s2)

        term2 = ((s1**2 + (u1 - u2)**2) / (2 * s2**2) - 1/2).sum(dim=1)

        return term1 + term2


    def fisher_info_isotropic_gaussian_batch(self, a):
        """
        Batch FIM for 3D isotropic Gaussian, input a: (N,4)
        """
        N = a.shape[0]
        eta1 = a[:, :3]          # (N,3)
        tau = a[:, 3]            # (N,)
        s = (eta1**2).sum(dim=1) # (N,)
        
        # I_11
        I_11 = -0.5 / tau.view(N,1,1) * torch.eye(3, device=a.device).unsqueeze(0)  # (N,3,3)

        # I_12
        I_12 = (0.5 / tau**2).view(N,1,1) * eta1.unsqueeze(2)  # (N,3,1)

        # I_22
        I_22 = -s / (2 * tau**3) + 3 / (2 * tau**2)  # (N,)

        # Assemble FIM
        F = torch.zeros(N, 4, 4, device=a.device)
        F[:, :3, :3] = I_11
        F[:, :3, 3:4] = I_12
        F[:, 3:4, :3] = I_12.transpose(1,2)
        F[:, 3, 3] = I_22

        return F


    def dtime4discrete_interpolation_loss_prob(
        self, t, N, p_0, one_hot_x, K, s1, segment_ids=None
    ):

        gamma = t
        
        s1_modify = torch.clamp((1 - gamma) * s1, min=1e-3)
        s1_modify = K * s1_modify / (1-s1_modify+K * s1_modify)


        prior = torch.ones_like(p_0).to(p_0.device)
        soft_p_0 = (1 - s1_modify) * p_0 + s1_modify / K
        soft_one_hot_x = (1 - s1_modify) * one_hot_x + s1_modify / K

        alpha1 = gamma * soft_p_0  + (1 - gamma) * prior
        alpha2 = gamma * soft_one_hot_x  + (1 - gamma) * prior



        # s1_modify = torch.clamp((1 - gamma) * s1, min=1e-3)  # Dirichlet Flow setup
        # prior = torch.ones_like(p_0).to(p_0.device)
        # p_0_multi = p_0 / s1_modify + prior
        # one_hot_x_multi = one_hot_x / s1_modify + prior

        # alpha1 = gamma * p_0_multi  + (1 - gamma) * prior
        # alpha2 = gamma * one_hot_x_multi  + (1 - gamma) * prior

        

        if segment_ids is not None:
            loss = scatter_mean(
                self.dirichlet_kl_batch(alpha1, alpha2), segment_ids, dim=0
            )

        return loss


    def dirichlet_kl_batch(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        """
        计算批量 Dirichlet 分布 KL 散度。
        alpha1, alpha2: 张量形状 [B, K]，表示 B 个样本，每个样本有 K 个浓度参数。
        返回: 形状 [B] 的 KL 散度。
        """
        # 1) 计算每个批次的浓度参数和，形状 [B]
        sum1 = alpha1.sum(dim=1)       # 
        sum2 = alpha2.sum(dim=1)       # 

        # 2) ln Γ(∑α1) - ln Γ(∑α2)，形状 [B]
        term1 = torch.lgamma(sum1) - torch.lgamma(sum2)

        # 3) ∑ [ln Γ(α2_i) - ln Γ(α1_i)]，形状 [B]
        term2 = torch.lgamma(alpha2).sum(dim=1) - torch.lgamma(alpha1).sum(dim=1)

        # 4) ∑ (α1_i - α2_i) [ψ(α1_i) - ψ(∑α1)]，形状 [B]
        term3 = ((alpha1 - alpha2) *
                (torch.digamma(alpha1) - torch.digamma(sum1).unsqueeze(1))
        ).sum(dim=1)

        return term1 + term2 + term3

    def fisher_info_dirichlet_batch(self, eta):
        """
        Compute Fisher Information Matrices for a batch of Dirichlet distributions
        using natural parameters eta = alpha - 1.

        Parameters
        ----------
        eta : torch.Tensor, shape (N, K)
            Natural parameters of Dirichlet (> -1)

        Returns
        -------
        F : torch.Tensor, shape (N, K, K)
            Batch of Fisher Information Matrices
        """
        N, K = eta.shape
        device = eta.device

        # 转回 concentration 参数
        alpha = eta + 1  # alpha_i > 0

        # trigamma(alpha_i) 对角项
        psi1_alpha = torch.polygamma(1, alpha)                # (N, K)

        # trigamma(sum(alpha)) 全部元素相同
        psi1_sum = torch.polygamma(1, alpha.sum(dim=-1, keepdim=True))  # (N,1)

        # 构造对角矩阵 diag(trigamma(alpha))
        F = torch.zeros(N, K, K, device=device)
        idx = torch.arange(K)
        F[:, idx, idx] = psi1_alpha

        # 减去 trigamma(sum(alpha)) * 1_{KxK}
        F = F - psi1_sum.unsqueeze(-1) * torch.ones(N, K, K, device=device)

        return F

    def categorize_kl_batch(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # 计算 p * (log p - log q)
        term = p * torch.log(p/q)
        # sum over last dim
        kl = term.sum(dim=-1)  # shape: (batch,) 或 scalar
        return kl
