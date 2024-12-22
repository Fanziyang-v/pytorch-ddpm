import os
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from ddpm.models.unet import UNet
from ddpm.utils.diffusion import GaussianDiffusion

def ddp_setup(rank: int, world_size: int, gpu_id: int) -> None:
    """Setup for DDP"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(gpu_id)
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


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


def _build_dataset(name: str) -> Dataset:
    """Get dataset."""
    # input is dataset name. e.g. 'cifar10'
    if name == "cifar10":
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            ),
        )
    else:
        raise (f"Unkown dataset: {name}")
    return dataset


def _build_data_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Get data loader."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def train(
    rank: int,
    cfg: dict,
    data_loader: DataLoader,
    model: DDP,
    optimizer: Adam,
    criterion: nn.MSELoss,
    diffusion: GaussianDiffusion,
) -> None:
    """Model Training"""
    num_epochs = cfg["num_epochs"]
    gpu_id = cfg["gpu_ids"][rank]
    for i in range(num_epochs):
        total_loss = 0
        for images, _ in tqdm(data_loader):
            batch_size = len(images)
            images: Tensor = images.to(gpu_id)
            # Sample timesteps from a uniform distribution.
            timesteps = torch.randint(0, cfg["num_timesteps"], (batch_size,)).to(gpu_id)
            noise = torch.randn_like(images).to(gpu_id)
            # Add noise
            noisy_images = diffusion.forward(images, noise, timesteps)
            # Forward pass(estimate noise)
            output = model(noisy_images, timesteps)
            loss: Tensor = criterion(noise, output)
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"GPU[{gpu_id}], Epoch: [{i + 1}]/[{num_epochs}], loss: {avg_loss}")
        if (i + 1) % cfg["interval"] == 0 and gpu_id == cfg["gpu_ids"][rank]:
            torch.save(model.module.state_dict(), os.path.join(cfg["ckpt_dir"], f"{cfg["dataset"]}_{i + 1}.pth"))


def main(rank: int, cfg: dict) -> None:
    gpu_id = cfg["gpu_ids"][rank]
    world_size = len(cfg["gpu_ids"])
    ddp_setup(rank, world_size, gpu_id)
    data_loader = _build_data_loader(_build_dataset(cfg["dataset"]), cfg["batch_size"])
    # UNet backbone
    model = _build_model(cfg).to(gpu_id)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters())
    model = DDP(model, device_ids=[gpu_id])
    diffusion = GaussianDiffusion(
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )
    train(rank, cfg, data_loader=data_loader, model=model, optimizer=optimizer, criterion=criterion, diffusion=diffusion)
    destroy_process_group()


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    mp.spawn(main, args=(cfg), nprocs=len(cfg["gpu_ids"]))
