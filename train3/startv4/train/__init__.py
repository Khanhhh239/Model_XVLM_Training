"""Training: SigLIP retrieval trainer + X-VLM-v4 cross-encoder trainer + hard-neg mining."""
from .mining import mine_hard_negatives
from .trainer import SiglipTrainer
from .trainer_xvlm import XVLMTrainer

__all__ = ["SiglipTrainer", "XVLMTrainer", "mine_hard_negatives"]
