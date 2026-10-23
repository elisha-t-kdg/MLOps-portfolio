import pathlib
import cv2
import numpy as np
import tensorflow as tf
import hydra

from omegaconf import DictConfig
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation(data_cfg):
    """Data augmentation used to generate extra training images: random rotation + random crop."""
    return keras.Sequential([
        layers.RandomRotation(data_cfg.augmentation.rotation),
        layers.RandomCrop(data_cfg.image_height, data_cfg.image_width),
    ], name="data_augmentation")


def augment_and_save_dataset(data_cfg):
    """Expand the flower photos dataset on disk.

    For every original image, save a (center-cropped) copy of the original plus
    `copies_per_image` randomly rotated/cropped copies into `data_cfg.processed_dir`,
    mirroring the original class-subfolder structure. Returns the new directory.
    """
    flowers_data = pathlib.Path(data_cfg.raw_dir)
    output_dir = pathlib.Path(data_cfg.processed_dir)
    height, width = data_cfg.image_height, data_cfg.image_width

    # Images are loaded a bit larger than the target size so RandomCrop has room to crop from.
    load_height, load_width = height + data_cfg.load_padding, width + data_cfg.load_padding

    augmentation = build_augmentation(data_cfg)
    center_crop = layers.CenterCrop(height, width)

    class_dirs = [d for d in flowers_data.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob('*'):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_resized = cv2.resize(image, (load_width, load_height))
            batch = np.expand_dims(image_resized, axis=0)

            original_cropped = center_crop(batch).numpy()[0]
            original_cropped = np.clip(original_cropped, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_class_dir / image_path.name), original_cropped)

            for i in range(data_cfg.augmentation.copies_per_image):
                augmented = augmentation(batch, training=True).numpy()[0]
                augmented = np.clip(augmented, 0, 255).astype(np.uint8)
                augmented_path = out_class_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                cv2.imwrite(str(augmented_path), augmented)

    print(f"Expanded dataset saved to {output_dir}")
    return output_dir


def load_datasets(data_cfg, seed):
    """Build the training and validation datasets from the processed flower photos directory."""
    train_set, validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        data_cfg.processed_dir,
        validation_split=0.2,
        subset="both",
        seed=seed,
        image_size=(data_cfg.image_height, data_cfg.image_width),
        batch_size=data_cfg.batch_size)
    image_cat = train_set.class_names
    print(image_cat)

    return train_set, validation_set, image_cat


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Standalone entry point: build the augmented dataset on disk from the DVC-tracked raw data."""
    augment_and_save_dataset(cfg.data)


if __name__ == "__main__":
    main()
