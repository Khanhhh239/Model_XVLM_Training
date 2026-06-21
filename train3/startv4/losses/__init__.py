"""Loss functions for STAR-v4 (all pure-torch, no model dependency)."""
from .box_loss import box_giou, box_loss
from .filip import filip_loss, filip_sim
from .infonce import info_nce
from .sigmoid_loss import siglip_sigmoid_loss
from .smoothap import smooth_ap_loss

__all__ = [
    "siglip_sigmoid_loss",
    "info_nce",
    "filip_sim",
    "filip_loss",
    "box_giou",
    "box_loss",
    "smooth_ap_loss",
]
