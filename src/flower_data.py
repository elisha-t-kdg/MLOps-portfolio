import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


HEIGHT, WIDTH = 180, 180
# Images are loaded a bit larger than the target size so RandomCrop has room to crop from.
LOAD_HEIGHT, LOAD_WIDTH = HEIGHT + 20, WIDTH + 20
TRAINING_BATCH_SIZE = 32
DATASET_DIRNAME = "flower_photos"
AUGMENTED_DIRNAME = "flower_photos_augmented"
COPIES_PER_IMAGE = 5


def download_flowers_data():
    """Download the flower photos dataset and return its local path."""
    flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    extracted_dir = tf.keras.utils.get_file(DATASET_DIRNAME, origin=flowers_url, untar=True)
    # The archive's own top-level folder is also called "flower_photos", so the actual
    # class subfolders live one level deeper than what get_file() returns.
    flowers_data = pathlib.Path(extracted_dir) / DATASET_DIRNAME
    print(flowers_data)
    return flowers_data


def build_augmentation():
    """Data augmentation used to generate extra training images: random rotation + random crop."""
    return keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomCrop(HEIGHT, WIDTH),
    ], name="data_augmentation")


def augment_and_save_dataset(flowers_data, output_dir=None, copies_per_image=COPIES_PER_IMAGE):
    """Expand the flower photos dataset on disk.

    For every original image, save a (center-cropped) copy of the original plus
    `copies_per_image` randomly rotated/cropped copies into a new directory that
    mirrors the original class-subfolder structure. Returns the new directory.
    """
    output_dir = pathlib.Path(output_dir) if output_dir else flowers_data.parent / AUGMENTED_DIRNAME
    augmentation = build_augmentation()
    center_crop = layers.CenterCrop(HEIGHT, WIDTH)

    class_dirs = [d for d in flowers_data.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob('*'):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_resized = cv2.resize(image, (LOAD_WIDTH, LOAD_HEIGHT))
            batch = np.expand_dims(image_resized, axis=0)

            original_cropped = center_crop(batch).numpy()[0]
            original_cropped = np.clip(original_cropped, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_class_dir / image_path.name), original_cropped)

            for i in range(copies_per_image):
                augmented = augmentation(batch, training=True).numpy()[0]
                augmented = np.clip(augmented, 0, 255).astype(np.uint8)
                augmented_path = out_class_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                cv2.imwrite(str(augmented_path), augmented)

    print(f"Expanded dataset saved to {output_dir}")
    return output_dir


def get_augmented_data_dir():
    """Path where augment_and_save_dataset() writes the augmented dataset by default.

    Does not create or download anything - the dataset must already have been
    built on disk, e.g. by running this module standalone (`python flower_data.py`).
    """
    keras_cache_dir = pathlib.Path(os.path.expanduser('~')) / '.keras' / 'datasets'
    return keras_cache_dir / DATASET_DIRNAME / AUGMENTED_DIRNAME


def load_datasets(flowers_data):
    """Build the training and validation datasets from a (possibly augmented) flower photos directory."""
    train_set, validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        flowers_data,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=TRAINING_BATCH_SIZE)
    image_cat = train_set.class_names
    print(image_cat)

    return train_set, validation_set, image_cat


def main():
    """Standalone entry point: download the flower photos and build the augmented dataset on disk."""
    flowers_data = download_flowers_data()
    augment_and_save_dataset(flowers_data)


if __name__ == "__main__":
    main()
