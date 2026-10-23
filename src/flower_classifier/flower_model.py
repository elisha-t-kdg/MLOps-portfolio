"""Transfer-learning flower classification model based on MobileNetV2: build, train and predict."""

import pathlib

import cv2
import hydra
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from omegaconf import DictConfig
from tensorflow import keras

from flower_classifier.flower_data import load_datasets


def build_model(model_cfg):
    """Build and compile a transfer-learning model on a frozen MobileNetV2 backbone."""
    imported_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=tuple(model_cfg.input_shape),
        pooling=model_cfg.pooling,
        weights=model_cfg.weights)
    for layer in imported_model.layers:
        layer.trainable = model_cfg.trainable_backbone

    imported_model.summary()

    dnn_model = keras.Sequential()
    dnn_model.add(imported_model)
    # dnn_model.add(Flatten())
    dnn_model.add(Dense(model_cfg.dense_units, activation='relu'))
    dnn_model.add(Dense(model_cfg.num_classes, activation='softmax'))

    dnn_model.compile(optimizer=Adam(learning_rate=model_cfg.learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return dnn_model


def train_model(dnn_model, train_set, validation_set, epochs):
    """Train the model and return the training history."""
    history = dnn_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=epochs
    )
    return history


def predict_image(dnn_model, image_path, image_cat, height, width):
    """Run inference on a single image and print the predicted category."""
    image = cv2.imread(str(image_path))
    image_resized = cv2.resize(image, (width, height))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)

    model_pred = dnn_model.predict(image)
    print(model_pred)
    predicted_class = image_cat[np.argmax(model_pred)]
    print("The predicted category is", predicted_class)
    return predicted_class


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Train the model on the processed dataset and predict one sample image."""
    train_set, validation_set, test_set, image_cat = load_datasets(cfg.data, cfg.seed)
    dnn_model = build_model(cfg.model)
    train_model(dnn_model, train_set, validation_set, cfg.model.epochs)

    sample_image = list((pathlib.Path(cfg.data.processed_dir) / 'test' / 'sunflowers').glob('*'))[1]
    predict_image(dnn_model, sample_image, image_cat, cfg.data.image_height, cfg.data.image_width)


if __name__ == "__main__":
    main()
