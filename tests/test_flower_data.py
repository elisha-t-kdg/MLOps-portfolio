import pathlib

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from flower_classifier.flower_data import augment_and_save_dataset, split_images

SPLIT_CFG = OmegaConf.create({"train": 0.7, "val": 0.15, "test": 0.15})


def make_paths(n):
    return [pathlib.Path(f"img{i}.jpg") for i in range(n)]


def test_split_sizes_follow_ratios():
    train, val, test = split_images(make_paths(100), SPLIT_CFG, seed=123)

    assert (len(train), len(val), len(test)) == (70, 15, 15)


def test_split_has_no_overlap_and_loses_nothing():
    paths = make_paths(37)
    train, val, test = split_images(paths, SPLIT_CFG, seed=123)

    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert set(train) | set(val) | set(test) == set(paths)


def test_split_is_reproducible_regardless_of_input_order():
    paths = make_paths(50)
    first = split_images(paths, SPLIT_CFG, seed=123)
    second = split_images(list(reversed(paths)), SPLIT_CFG, seed=123)

    assert first == second


def test_split_changes_with_different_seed():
    paths = make_paths(50)

    assert split_images(paths, SPLIT_CFG, seed=1) != split_images(paths, SPLIT_CFG, seed=2)


@pytest.fixture
def data_cfg(tmp_path):
    """Create a tiny fake raw dataset (2 classes x 10 images) and a matching data config."""
    raw_dir = tmp_path / "raw"

    for class_name in ["daisy", "roses"]:
        (raw_dir / class_name).mkdir(parents=True)

        for i in range(10):
            image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(raw_dir / class_name / f"img{i}.jpg"), image)

    return OmegaConf.create({
        "raw_dir": str(raw_dir),
        "processed_dir": str(tmp_path / "processed"),
        "image_height": 180,
        "image_width": 180,
        "load_padding": 20,
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "augmentation": {"copies_per_image": 2, "rotation": 0.2},
    })


def test_only_train_set_is_augmented(data_cfg):
    output_dir = augment_and_save_dataset(data_cfg, seed=123)

    for class_name in ["daisy", "roses"]:
        # 10 images per class -> 7 train, 1 val, 2 test
        assert len(list((output_dir / "train" / class_name).glob("*"))) == 7 * (1 + 2)
        assert len(list((output_dir / "val" / class_name).glob("*"))) == 1
        assert len(list((output_dir / "test" / class_name).glob("*"))) == 2
        assert not list((output_dir / "val" / class_name).glob("*_aug*"))
        assert not list((output_dir / "test" / class_name).glob("*_aug*"))


def test_no_image_leaks_between_splits(data_cfg):
    output_dir = augment_and_save_dataset(data_cfg, seed=123)

    def originals(split):
        """Return 'class/original_name' for every file in a split (augmented copies included)."""
        return {f"{p.parent.name}/{p.stem.split('_aug')[0]}"
                for p in (output_dir / split).rglob("*.jpg")}

    assert not originals("train") & originals("val")
    assert not originals("train") & originals("test")
    assert not originals("val") & originals("test")
