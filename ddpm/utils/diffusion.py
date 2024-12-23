import torch
from torch import Tensor


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(
            start=beta_start, end=beta_end, steps=num_timesteps
        ).reshape(num_timesteps, 1, 1, 1)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1 - self.alphas_cum_prod)

    def forward(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Forward Process.

        Args:
            x0 (Tensor): original images of shape (B, C, H, W)
            noise (Tensor): Gaussian noises of shape (B, C, H, W)
            t (Tensor): timesteps of shape (B,)

        Returns:
            Tensor: noisy images after diffusion process
        """
        assert x0.shape == noise.shape, "x0.shape != noise.shape"
        sqrt_alphas_cum_prod = self.sqrt_alphas_cum_prod.to(t.device)[t]
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.to(
            t.device
        )[t]
        return sqrt_alphas_cum_prod * x0 + sqrt_one_minus_alphas_cum_prod * noise

    def reverse(self, xt: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Reverse Process.

        Args:
            xt (Tensor): noisy images of shape (B, C, H, W)
            noise (Tensor): predicted noises by U-Net of shape (B, C, H, W)
            t (Tensor): timesteps of shape (B,)

        Returns:
            Tensor: more clear images after reverse process
        """
        assert xt.shape == noise.shape, "xt.shape != noise.shape"
        z = torch.randn_like(xt).to(t.device)
        z[t == 0] = 0
        alpha = self.alphas.to(t.device)[t]
        beta = self.betas.to(t.device)[t]
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.to(
            t.device
        )[t]
        return (
            xt - (1 - alpha) / sqrt_one_minus_alphas_cum_prod * noise
        ) * alpha**-0.5 + torch.sqrt(beta) * z
