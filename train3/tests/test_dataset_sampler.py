from _helpers import make_dataset

from startv4.data import (
    BalancedBucketSampler,
    PABDatasetV4,
    PairBatchSampler,
    collate_fn,
)


def test_dataset_items_and_collate(tmp_path):
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=True, max_token=16)
    assert len(ds) == 8
    item = ds[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["keypoints"].shape == (17, 3)
    assert item["bbox"].shape == (4,)
    assert bool(item["has_bbox"])
    batch = collate_fn([ds[i] for i in range(4)])
    assert batch["pixel_values"].shape == (4, 3, 32, 32)
    assert batch["input_ids"].shape[0] == 4
    assert len(batch["image_id"]) == 4


def test_dataset_pairs_and_buckets(tmp_path):
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=False)
    assert len(ds.pairs()) == 8
    assert set(ds.buckets()) <= {0, 1}


def test_pair_batch_sampler():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    s = PairBatchSampler(pairs, batch_size=4, seed=1)
    batches = list(iter(s))
    assert len(batches) == 2
    assert all(len(b) == 4 for b in batches)
    assert sorted(sum(batches, [])) == list(range(8))


def test_balanced_bucket_sampler():
    s = BalancedBucketSampler([0, 0, 0, 0, 1, 1, 1, 1], batch_size=4, seed=1)
    batches = list(iter(s))
    assert len(batches) == 2
    assert all(len(b) == 4 for b in batches)
