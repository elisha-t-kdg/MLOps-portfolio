import pathlib
import cv2
import numpy as np
import tensorflow as tf
import hydra
import random
import shutil

from omegaconf import DictConfig
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation(data_cfg):
    """Data augmentation used to generate extra training images: random rotation + random crop."""
    return keras.Sequential([
        layers.RandomRotation(data_cfg.augmentation.rotation),
        layers.RandomCrop(data_cfg.image_height, data_cfg.image_width),
    ], name="data_augmentation")

def split_images(image_paths, split_cfg, seed):
    """Shuffle the images of one class and split them into train, validation and test lists."""
    paths = sorted(image_paths)
    random.Random(seed).shuffle(paths)

    n_train = int(len(paths) * split_cfg.train)
    n_val = int(len(paths) * split_cfg.val)

    return paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:]

def augment_and_save_dataset(data_cfg, seed):
    """Split the original images into train/val/test, ONLY THEN, expand the train set.

    The split happens BEFORE augmenting, so augmented copies of an image always end up
    in the same split as the original (no data leakage). Validation and test images
    are saved as (center-cropped) originals only.

    Result: <processed_dir>/train|val|test/<class>/... Returns the new directory.
    """
    flowers_data = pathlib.Path(data_cfg.raw_dir)
    output_dir = pathlib.Path(data_cfg.processed_dir)
    height, width = data_cfg.image_height, data_cfg.image_width
    # Images are loaded a bit larger than the target size so RandomCrop has room to crop from.
    load_height, load_width = height + data_cfg.load_padding, width + data_cfg.load_padding

    # Start from an empty folder so no files from a previous run are left behind.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    augmentation = build_augmentation(data_cfg)
    center_crop = layers.CenterCrop(height, width)

    class_dirs = [d for d in flowers_data.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        train_paths, val_paths, test_paths = split_images(list(class_dir.glob('*')), data_cfg.split, seed)
        splits = {"train": train_paths, "val": val_paths, "test": test_paths}

        for split_name, image_paths in splits.items():
            out_class_dir = output_dir / split_name / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image_resized = cv2.resize(image, (load_width, load_height))
                batch = np.expand_dims(image_resized, axis=0)

                original_cropped = center_crop(batch).numpy()[0]
                original_cropped = np.clip(original_cropped, 0, 255).astype(np.uint8)
                cv2.imwrite(str(out_class_dir / image_path.name), original_cropped)

                if split_name != "train":
                    continue  # validation and test images are NOT augmented

                for i in range(data_cfg.augmentation.copies_per_image):
                    augmented = augmentation(batch, training=True).numpy()[0]
                    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
                    augmented_path = out_class_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                    cv2.imwrite(str(augmented_path), augmented)

    print(f"Expanded dataset saved to {output_dir}")
    return output_dir


def load_datasets(data_cfg, seed):
    """Load the train, validation and test datasets from the split processed directory."""
    data_dir = pathlib.Path(data_cfg.processed_dir)
    image_size = (data_cfg.image_height, data_cfg.image_width)

    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir / "train",
        seed=seed,
        image_size=image_size,
        batch_size=data_cfg.batch_size)

    validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir / "val",
        shuffle=False,
        image_size=image_size,
        batch_size=data_cfg.batch_size)

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir / "test",
        shuffle=False,
        image_size=image_size,
        batch_size=data_cfg.batch_size)

    image_cat = train_set.class_names
    print(image_cat)

    return train_set, validation_set, test_set, image_cat


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Standalone entry point: build the augmented dataset on disk from the DVC-tracked raw data."""
    augment_and_save_dataset(cfg.data, cfg.seed)


if __name__ == "__main__":
    main()
