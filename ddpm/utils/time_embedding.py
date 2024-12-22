import torch
from torch import Tensor


def get_time_embedding(timesteps: Tensor, t_emb_dim: int) -> Tensor:
    """Generate time embeddings for the input timesteps.

    Args:
        timesteps (Tensor): timesteps tensor of shape (batch_size,)
        t_emb_dim (int): time embedding dimension

    Raises:
        ValueError: if t_emb_dim is not divisible by 2

    Returns:
        Tensor: time embeddings of shape (batch_size, t_emb_dim)
    """
    if t_emb_dim % 2 != 0:
        raise ValueError("t_emb_dim must be divisible by 2")
    batch_size = timesteps.size()[0]
    _2i = torch.arange(
        start=0, end=t_emb_dim, step=2, dtype=torch.float32, device=timesteps.device
    )
    t_emb = torch.zeros(batch_size, t_emb_dim, device=timesteps.device)
    t_emb[:, 0::2] = torch.sin(timesteps.unsqueeze(1) / 10000 ** (_2i / t_emb_dim))
    t_emb[:, 1::2] = torch.cos(timesteps.unsqueeze(1) / 10000 ** (_2i / t_emb_dim))
    return t_emb
