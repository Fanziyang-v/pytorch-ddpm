import os
import torch
from torch import Tensor
from ddpm.models.unet import UNet
from ddpm.utils.diffusion import GaussianDiffusion
from torchvision.utils import save_image, make_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _denormomalize(x: Tensor) -> Tensor:
    return torch.clamp(x, min=0, max=1)


def _build_model(cfg: dict) -> UNet:
    """Get U-Net backbone."""
    return UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        basic_width=cfg["basic_width"],
        ch_mult=cfg["ch_mult"],
        num_blocks=cfg["num_blocks"],
        num_mid_blocks=cfg["num_mid_blocks"],
        num_groups=cfg["num_groups"],
        drop_prob=cfg["drop_prob"],
        attention=cfg["attention"],
        resample_with_conv=cfg["resample_with_conv"],
    )


def sample(cfg: dict, model: UNet, diffusion: GaussianDiffusion) -> None:
    num_samples = cfg["num_samples"]
    num_timesteps = cfg["num_timesteps"]
    model.eval()
    x = torch.randn(num_samples, 3, 32, 32).to(device)
    for i in reversed(range(num_timesteps)):
        timesteps = torch.zeros(num_samples, dtype=torch.int32).to(device)
        timesteps[:] = i
        output = model(x, timesteps)
        x = diffusion.reverse(x, output, timesteps)
        torch.cuda.empty_cache()
    img_grid = make_grid(_denormomalize(x), nrow=int(num_samples**0.5), padding=2)
    save_image(img_grid, os.path.join(cfg["sample_dir"], "samples.png"))


def main(cfg: dict) -> None:
    model = _build_model(cfg["unet"]).to(device)
    # load noise predicter checkpoints
    model.load_state_dict(
        torch.load(
            os.path.join(
                cfg["ckpt_dir"],
                cfg["dataset"],
                "noise_predicter_" + str(cfg["num_epochs"] + ".pth"),
            )
        )
    )
    diffusion = GaussianDiffusion(
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )
    sample(model, diffusion)


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    if not os.path.exists(cfg["sample_dir"]):
        os.makedirs(cfg["sample_dir"])
    main(cfg)
