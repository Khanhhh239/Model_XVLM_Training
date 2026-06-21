import torch
from PIL import Image

from startv4.data.augment import (
    DownscaleUpscale,
    GaussianNoiseTensor,
    JpegCompression,
    MotionBlurTensor,
    build_eval_transform,
    build_train_transform,
)


def test_train_transform_shape_and_finite():
    img = Image.new("RGB", (64, 64), (120, 130, 140))
    out = build_train_transform(32)(img)
    assert out.shape == (3, 32, 32)
    assert torch.isfinite(out).all()


def test_eval_transform_deterministic():
    img = Image.new("RGB", (64, 64), (120, 130, 140))
    t = build_eval_transform(32)
    assert torch.allclose(t(img), t(img))


def test_motion_blur_preserves_shape():
    x = torch.rand(3, 16, 16)
    out = MotionBlurTensor(p=1.0)(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_noise_keeps_range():
    x = torch.rand(3, 8, 8)
    out = GaussianNoiseTensor(p=1.0, std=0.1)(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_jpeg_and_downscale_return_same_size():
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    assert JpegCompression(p=1.0)(img).size == (32, 32)
    assert DownscaleUpscale(p=1.0)(img).size == (32, 32)
