"""U-Net backbone for DDPM."""

from torch import nn, Tensor
from .blocks import DownBlock, MiddleBlock, UpBlock
from ..utils.time_embedding import get_time_embedding


class UNet(nn.Module):
    """U-Net backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        basic_width: int = 64,
        ch_mult: list[int] = [1, 2, 4, 8],
        num_blocks: int = 2,
        num_mid_blocks: int = 2,
        num_groups: int = 32,
        drop_prob: float = 0.1,
        attention: list[bool] = [False, True, False, False],
        resample_with_conv: bool = True,
    ):
        super(UNet, self).__init__()
        assert len(ch_mult) == len(attention), "len(ch_mult) != len(attention)"
        num_resolutions = len(ch_mult)
        self.basic_width = basic_width
        self.t_emb_dim = basic_width * 4
        # Time embedding projection.
        self.te_proj = nn.Sequential(
            nn.Linear(basic_width, self.t_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )
        # Initial convolution
        self.in_conv = nn.Conv2d(in_channels, basic_width, kernel_size=3, padding=1)
        # Downsampling blocks
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_channels=basic_width if i == 0 else basic_width * ch_mult[i - 1],
                    out_channels=basic_width * ch_mult[i],
                    t_emb_dim=self.t_emb_dim,
                    num_blocks=num_blocks,
                    drop_prob=drop_prob,
                    attention=attention[i],
                    downsample=i < num_resolutions - 1,
                    resample_with_conv=resample_with_conv,
                    num_groups=num_groups,
                )
                for i in range(num_resolutions)
            ]
        )
        # Middle block
        self.middle_block = MiddleBlock(
            in_channels=basic_width * ch_mult[-1],
            out_channels=basic_width * ch_mult[-1],
            t_emb_dim=self.t_emb_dim,
            num_blocks=num_mid_blocks,
            drop_prob=drop_prob,
            attention=True,  # always use attention in the middle block
        )
        # Upsampling blocks
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    in_channels=2 * basic_width * ch_mult[i],
                    out_channels=basic_width * ch_mult[i],
                    t_emb_dim=self.t_emb_dim,
                    num_blocks=num_blocks,
                    drop_prob=drop_prob,
                    attention=attention[i],
                    upsample=i > 0,
                    resample_with_conv=resample_with_conv,
                    num_groups=num_groups,
                )
                for i in reversed(range(num_resolutions))
            ]
        )
        # Final Convolution
        self.out_conv = nn.Conv2d(basic_width, out_channels, kernel_size=1)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        t_emb = self.te_proj(get_time_embedding(timesteps, self.basic_width))
        out = self.in_conv(x)
        enc_outs: list[list[Tensor]] = []
        # Downsampling path
        for down_block in self.down_blocks:
            out, enc_out = down_block(out, t_emb)
            enc_outs.append(enc_out)
        # Refining path
        out = self.middle_block(out, t_emb)
        # Upsampling path
        for up_block in self.up_blocks:
            out = up_block(out, enc_outs.pop(), t_emb)
        out = self.out_conv(out)
        return out
