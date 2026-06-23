"""Sim2real augmentation mapped to the wrong-case taxonomy (moderate p so it can't break the 80-mAP base).

#9 blur / sim2real : motion-blur + gaussian-blur + JPEG re-compress + downscale->upscale
#8 occlusion       : RandomErasing (CMP already uses p~0.6; kept)
robustness         : mild color-jitter
#6 OCR             : NOT erasing text hard (keep legible) -> erasing kept mild here

Geometric ops (resize/flip) must be applied IDENTICALLY to the image and its pose map. We therefore do
photometric degradation only in this transform (pose maps should NOT get blur/JPEG — they are clean
skeletons); flip is handled by the dataset with a shared random state.
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class JPEGCompress:
    def __init__(self, p=0.3, q=(35, 75)):
        self.p, self.q = p, q
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(*self.q)); buf.seek(0)
        return Image.open(buf).convert("RGB")


class Downscale:
    def __init__(self, p=0.3, factor=(1.5, 3.0)):
        self.p, self.factor = p, factor
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h = img.size; f = random.uniform(*self.factor)
        small = img.resize((max(1, int(w / f)), max(1, int(h / f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p=0.3, radius=(0.5, 2.0)):
        self.p, self.radius = p, radius
    def __call__(self, img):
        if random.random() > self.p: return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


class MotionBlur:
    def __init__(self, p=0.2, ksize=(5, 13)):
        self.p, self.ksize = p, ksize
    def __call__(self, img):
        if random.random() > self.p: return img
        k = random.randrange(self.ksize[0], self.ksize[1], 2)
        arr = np.asarray(img).astype(np.float32)
        kernel = np.zeros((k, k), np.float32)
        if random.random() < 0.5: kernel[k // 2, :] = 1.0          # horizontal
        else: kernel[:, k // 2] = 1.0                              # vertical
        kernel /= k
        from scipy.ndimage import convolve  # scipy is in CMP deps
        for c in range(3): arr[..., c] = convolve(arr[..., c], kernel, mode="reflect")
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_ft_image_transform(h, w, erasing_p=0.5, sim2real_p=1.0):
    """Photometric sim2real degradation + resize + normalize (CLIP stats, matches CMP)."""
    degr = []
    if sim2real_p > 0:
        degr = [transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.3),
                RandomBlur(0.3), MotionBlur(0.2), JPEGCompress(0.3), Downscale(0.3)]
    return transforms.Compose([
        *degr,
        transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
        # flip is done JOINTLY with pose in dataset_ft (kept out here for image/pose consistency)
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        transforms.RandomErasing(probability=erasing_p, mean=[0.0, 0.0, 0.0]),
    ])


def build_pose_transform(h, w):
    """Pose maps: clean resize+normalize ONLY (no blur/JPEG). Flip handled jointly in the dataset."""
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
