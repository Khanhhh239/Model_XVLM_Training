"""XBM (Cross-Batch Memory) Queue for ITC Loss.

Maintains a FIFO queue of past embeddings to provide more negatives for contrastive learning,
compensating for small batch sizes on limited hardware (Kaggle T4).

References:
  - XBM: Cross-Batch Memory for Embedding Learning (Wang et al., CVPR 2020, arXiv:1912.06798)
  - Adapted for text-image contrastive learning (ITC)

Usage:
    queue = XBMQueue(size=8192, dim=256)
    
    # In training loop:
    loss_itc = itc_loss(img_feat, txt_feat, queue_img=queue.img_queue, queue_txt=queue.txt_queue)
    queue.enqueue(img_feat.detach(), txt_feat.detach())  # Update after backward

Note: Features are stored detached (no grad) and L2-normalized for cosine similarity.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class XBMQueue(nn.Module):
    """FIFO queue storing past image and text embeddings as extra negatives."""
    
    def __init__(self, size: int = 8192, dim: int = 256):
        """
        Args:
            size: Queue capacity (number of embeddings to store).
            dim: Embedding dimension.
        """
        super().__init__()
        self.size = size
        self.dim = dim
        
        # Register as buffers (moved with model.to(device), not trained)
        self.register_buffer("img_queue", torch.zeros(size, dim))
        self.register_buffer("txt_queue", torch.zeros(size, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def enqueue(self, img_feat: Tensor, txt_feat: Tensor):
        """Add new embeddings to the queue (FIFO: dequeue old, enqueue new).
        
        Args:
            img_feat: [B, D] image features (will be detached & L2-normalized)
            txt_feat: [B, D] text features (will be detached & L2-normalized)
        """
        batch_size = img_feat.size(0)
        assert img_feat.size(1) == self.dim and txt_feat.size(1) == self.dim
        
        # Normalize for cosine similarity
        img_feat = torch.nn.functional.normalize(img_feat.detach(), dim=-1)
        txt_feat = torch.nn.functional.normalize(txt_feat.detach(), dim=-1)
        
        ptr = int(self.ptr)
        
        # Wrap around if queue full
        if ptr + batch_size <= self.size:
            self.img_queue[ptr:ptr + batch_size] = img_feat
            self.txt_queue[ptr:ptr + batch_size] = txt_feat
            ptr = (ptr + batch_size) % self.size
        else:
            # Split: fill remaining space, then wrap to beginning
            remaining = self.size - ptr
            self.img_queue[ptr:] = img_feat[:remaining]
            self.txt_queue[ptr:] = txt_feat[:remaining]
            overflow = batch_size - remaining
            self.img_queue[:overflow] = img_feat[remaining:]
            self.txt_queue[:overflow] = txt_feat[remaining:]
            ptr = overflow
        
        self.ptr[0] = ptr
        
    def get_queue(self) -> tuple[Tensor, Tensor]:
        """Return current queue contents (for computing extra negatives in ITC).
        
        Returns:
            (img_queue, txt_queue): [size, dim] tensors (L2-normalized, detached)
        """
        return self.img_queue.clone(), self.txt_queue.clone()
    
    @property
    def is_full(self) -> bool:
        """Check if queue has been filled at least once (ptr wrapped around)."""
        return int(self.ptr) < self.img_queue.size(0) // 2  # Heuristic: if ptr is small, likely wrapped


# Helper function to integrate queue into ITC loss
def compute_sim_with_queue(
    img_feat: Tensor,
    txt_feat: Tensor,
    queue_img: Tensor | None = None,
    queue_txt: Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[Tensor, Tensor]:
    """Compute similarity matrices including queue negatives.
    
    Args:
        img_feat: [B, D] current batch image features (L2-normalized)
        txt_feat: [B, D] current batch text features (L2-normalized)
        queue_img: [Q, D] queued image features (or None)
        queue_txt: [Q, D] queued text features (or None)
        temperature: Scaling factor
        
    Returns:
        sim_i2t: [B, B+Q] image-to-text similarity
        sim_t2i: [B, B+Q] text-to-image similarity
    """
    # In-batch similarity
    sim_i2t_batch = (img_feat @ txt_feat.t()) / temperature  # [B, B]
    sim_t2i_batch = (txt_feat @ img_feat.t()) / temperature  # [B, B]
    
    if queue_img is None or queue_txt is None:
        return sim_i2t_batch, sim_t2i_batch
    
    # Add queue negatives
    sim_i2t_queue = (img_feat @ queue_txt.t()) / temperature  # [B, Q]
    sim_t2i_queue = (txt_feat @ queue_img.t()) / temperature  # [B, Q]
    
    # Concatenate: [B, B+Q]
    sim_i2t = torch.cat([sim_i2t_batch, sim_i2t_queue], dim=1)
    sim_t2i = torch.cat([sim_t2i_batch, sim_t2i_queue], dim=1)
    
    return sim_i2t, sim_t2i
