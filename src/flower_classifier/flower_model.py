"""Transfer-learning flower classification model based on MobileNetV2: build, train and predict."""

import pathlib

import cv2
import hydra
import mlflow
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Rescaling
from keras.optimizers import Adam
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix
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
    dnn_model.add(keras.Input(shape=tuple(model_cfg.input_shape)))
    # MobileNetV2 was trained on pixels in [-1, 1]; our images are in [0, 255].
    dnn_model.add(Rescaling(1.0 / 127.5, offset=-1.0))
    dnn_model.add(imported_model)
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


def evaluate_model(dnn_model, test_set, image_cat):
    """Evaluate the trained model on the held-out test set.

    Returns the test loss and accuracy, the confusion matrix and a per-class report.
    """
    test_loss, test_accuracy = dnn_model.evaluate(test_set)

    # test_set is not shuffled, so labels and predictions are in the same order.
    y_true = np.concatenate([labels.numpy() for _, labels in test_set])
    y_pred = np.argmax(dnn_model.predict(test_set), axis=1)

    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=image_cat, digits=3)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(report)
    print(matrix)
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix,
        "report": report,
    }


def flatten_config(config, parent_key=""):
    """Flatten a nested config dict into {"model.learning_rate": 0.001, ...} for MLflow."""
    flat = {}
    for key, value in config.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def log_training_history(history):
    """Log the loss and accuracy of every epoch to MLflow, so the curves show up in the UI."""
    for epoch in range(len(history.history["loss"])):
        mlflow.log_metrics(
            {name: values[epoch] for name, values in history.history.items()}, step=epoch
        )


def predict_image(dnn_model, image_path, image_cat, height, width):
    """Run inference on a single image and print the predicted category."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    # OpenCV reads images as BGR, but the model was trained on RGB images (Keras loader).
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    """Train and evaluate the model, and track the run (params + metrics) in MLflow."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(flatten_config(OmegaConf.to_container(cfg, resolve=True)))

        train_set, validation_set, test_set, image_cat = load_datasets(cfg.data, cfg.seed)
        dnn_model = build_model(cfg.model)
        history = train_model(dnn_model, train_set, validation_set, cfg.model.epochs)
        log_training_history(history)

        results = evaluate_model(dnn_model, test_set, image_cat)
        mlflow.log_metrics({
            "test_loss": float(results["test_loss"]),
            "test_accuracy": float(results["test_accuracy"]),
        })

        sample_image = list((pathlib.Path(cfg.data.processed_dir) / 'test' / 'sunflowers').glob('*'))[1]
        predict_image(dnn_model, sample_image, image_cat, cfg.data.image_height, cfg.data.image_width)


if __name__ == "__main__":
    main()
