"""U-Net building blocks."""

import torch
from torch import nn, Tensor


class ConvBlock(nn.Module):
    """GroupNorm + SiLU + [Dropout] + Conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        drop_prob: float = 0.0,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.swish = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(x)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.conv(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        drop_prob: float = 0.0,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_size, num_groups)
        self.conv_block2 = ConvBlock(
            out_channels, out_channels, kernel_size, num_groups, drop_prob
        )
        self.proj = nn.Sequential(
            nn.SiLU(inplace=True), nn.Linear(t_emb_dim, out_channels)
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Forward pass of the residual block.

        Args:
            x (Tensor): input tensor of shape (batch_size, in_channels, H, W)
            t_emb (Tensor): time embedding tensor of shape (batch_size, t_emb_dim)

        Returns:
            Tensor: output tensor of shape (batch_size, out_channels, H, W)
        """
        out = self.conv_block1(x)
        out += self.proj(t_emb)[:, :, None, None]
        out = self.conv_block2(out)
        out += self.shortcut(x)
        return out


class AttentionBlock(nn.Module):
    """Self-attention block with group normalization and residual connection."""

    def __init__(self, num_channels: int, num_groups: int = 32):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.proj_q = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.proj_k = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.proj_o = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        x = self.norm(x)
        # Project input tensor to query, key, and value tensors
        q = torch.reshape(self.proj_q(x), (B, C, -1)).permute(0, 2, 1)  # (B, HxW, C)
        k = torch.reshape(self.proj_k(x), (B, C, -1))  # (B, C, HxW)
        v = torch.reshape(self.proj_v(x), (B, C, -1)).permute(0, 2, 1)  # (B, HxW, C)
        # Compute attention scores and attention weights
        score = torch.bmm(q, k) * C**-0.5
        attn = self.softmax(score)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        out = self.proj_o(out)
        return out + x


class DownBlock(nn.Module):
    """Down Block in downsampling path of U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_blocks: int = 2,
        drop_prob: float = 0.0,
        attention: bool = False,
        downsample: bool = True,
        resample_with_conv: bool = True,
        num_groups: int = 32,
    ):
        super(DownBlock, self).__init__()
        self.num_blocks = num_blocks
        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    t_emb_dim,
                    drop_prob=drop_prob,
                    num_groups=num_groups,
                )
                for i in range(num_blocks)
            ]
        )
        # Attention blocks
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(out_channels, num_groups) if attention else nn.Identity()
                for _ in range(num_blocks)
            ]
        )
        self.downsample = (
            Downsample(out_channels, resample_with_conv)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> tuple[Tensor, list[Tensor]]:
        outs: list[Tensor] = []
        out = x
        for i in range(self.num_blocks):
            out = self.res_blocks[i](out, t_emb)
            out = self.attn_blocks[i](out)
            outs.append(out)
        out = self.downsample(out)
        return out, outs


class UpBlock(nn.Module):
    """Up Block in upsampling path of U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_blocks: int = 2,
        drop_prob: float = 0.0,
        attention: bool = False,
        upsample: bool = True,
        resample_with_conv: bool = True,
        num_groups: int = 32,
    ):
        super(UpBlock, self).__init__()
        self.num_blocks = num_blocks
        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels,
                    out_channels,
                    t_emb_dim,
                    drop_prob=drop_prob,
                    num_groups=num_groups,
                )
                for _ in range(num_blocks)
            ]
        )
        # Attention blocks
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(out_channels, num_groups) if attention else nn.Identity()
                for _ in range(num_blocks)
            ]
        )
        self.upsample = (
            Upsample(out_channels, resample_with_conv) if upsample else nn.Identity()
        )

    def forward(self, dec: Tensor, enc: list[Tensor], t_emb: Tensor) -> Tensor:
        out = dec
        for i in range(self.num_blocks):
            out = torch.cat([out, enc.pop()], dim=1)
            out = self.res_blocks[i](out, t_emb)
            out = self.attn_blocks[i](out)
        out = self.upsample(out)
        return out


class MiddleBlock(nn.Module):
    """Middle Block in U-Net for refining."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_blocks: int = 2,
        drop_prob: float = 0.0,
        attention: bool = True,
    ):
        super(MiddleBlock, self).__init__()
        self.num_blocks = num_blocks
        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    t_emb_dim,
                    drop_prob=drop_prob,
                )
                for i in range(num_blocks)
            ]
        )
        # Attention blocks
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(out_channels) if attention else nn.Identity()
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        out = x
        for i in range(self.num_blocks):
            out = self.res_blocks[i](out, t_emb)
            out = self.attn_blocks[i](out)
        return out


class Upsample(nn.Module):
    """Upsampling layer with biliear interpolation or transpose convolution."""

    def __init__(self, num_channels: int, with_conv: bool = True) -> None:
        super(Upsample, self).__init__()
        out_channels = num_channels // 2
        self.upsample = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(num_channels, out_channels, kernel_size=1, bias=False),
            )
            if not with_conv
            else nn.Sequential(
                nn.ConvTranspose2d(
                    num_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)


class Downsample(nn.Module):
    """Downsampling layer with strided convolution or average pooling."""

    def __init__(self, num_channels: int, with_conv: bool = True) -> None:
        super(Downsample, self).__init__()
        self.downsample = (
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
            if with_conv
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)
