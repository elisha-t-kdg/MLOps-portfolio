import cv2
import mlflow
import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from flower_classifier.flower_model import (
    build_model,
    evaluate_model,
    flatten_config,
    get_data_version,
    log_training_history,
    plot_confusion_matrix,
    plot_training_history,
    predict_image,
    train_model,
)

CLASSES = ["daisy", "roses", "tulips"]
SIZE = 32


@pytest.fixture
def model_cfg():
    """A small model config: no ImageNet download (weights=None) and tiny images."""
    return OmegaConf.create({
        "input_shape": [SIZE, SIZE, 3],
        "weights": None,
        "pooling": "avg",
        "trainable_backbone": False,
        "dense_units": 8,
        "dropout": 0.3,
        "num_classes": len(CLASSES),
        "learning_rate": 0.001,
    })


@pytest.fixture
def tiny_dataset():
    """A batched tf.data dataset of 6 random images with labels 0, 1, 2, 0, 1, 2."""
    images = np.random.randint(0, 255, (6, SIZE, SIZE, 3)).astype("float32")
    labels = np.array([0, 1, 2, 0, 1, 2])
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(3)


class FakeHistory:
    """Stands in for the History object returned by model.fit()."""

    history = {
        "loss": [1.0, 0.6, 0.5],
        "val_loss": [1.1, 0.8, 0.9],
        "accuracy": [0.5, 0.8, 0.9],
        "val_accuracy": [0.4, 0.7, 0.65],
    }


def test_build_model_output_and_preprocessing(model_cfg):
    model = build_model(model_cfg)
    assert model.output_shape == (None, len(CLASSES))

    # first layer scales pixels from [0, 255] to [-1, 1]
    scaled = model.layers[0](np.array([[0.0, 127.5, 255.0]]))
    np.testing.assert_allclose(np.asarray(scaled), [[-1.0, 0.0, 1.0]])


def test_build_model_freezes_backbone(model_cfg):
    model = build_model(model_cfg)
    backbone = model.layers[1]

    assert not any(layer.trainable for layer in backbone.layers)


def test_train_model_returns_history(model_cfg, tiny_dataset):
    model = build_model(model_cfg)
    history = train_model(model, tiny_dataset, tiny_dataset, epochs=1)

    assert set(history.history) >= {"loss", "accuracy", "val_loss", "val_accuracy"}


def test_evaluate_model_results(model_cfg, tiny_dataset):
    model = build_model(model_cfg)
    results = evaluate_model(model, tiny_dataset, CLASSES)

    assert 0.0 <= results["test_accuracy"] <= 1.0
    assert results["confusion_matrix"].shape == (3, 3)
    assert results["confusion_matrix"].sum() == 6
    assert "roses" in results["report"]


def test_predict_image_returns_known_class(model_cfg, tmp_path):
    image_path = tmp_path / "flower.jpg"
    cv2.imwrite(str(image_path), np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8))
    model = build_model(model_cfg)

    assert predict_image(model, image_path, CLASSES, SIZE, SIZE) in CLASSES


def test_predict_image_rejects_unreadable_file(model_cfg, tmp_path):
    bad_file = tmp_path / "not_an_image.jpg"
    bad_file.write_text("this is not an image")

    with pytest.raises(ValueError, match="Could not read image"):
        predict_image(build_model(model_cfg), bad_file, CLASSES, SIZE, SIZE)


def test_flatten_config():
    nested = {"seed": 1, "model": {"lr": 0.1, "shape": [2, 2]}, "data": {"split": {"train": 0.7}}}

    assert flatten_config(nested) == {
        "seed": 1,
        "model.lr": 0.1,
        "model.shape": [2, 2],
        "data.split.train": 0.7,
    }


def test_get_data_version_reads_dvc_hash(tmp_path):
    (tmp_path / "processed.dvc").write_text("outs:\n- md5: abc123.dir\n  path: processed\n")

    assert get_data_version(tmp_path / "processed") == "abc123.dir"


def test_get_data_version_untracked(tmp_path):
    assert get_data_version(tmp_path / "processed") == "untracked"


def test_plot_training_history():
    fig = plot_training_history(FakeHistory())
    titles = [ax.get_title() for ax in fig.axes]

    assert titles == ["Loss", "Accuracy"]


def test_plot_confusion_matrix():
    fig = plot_confusion_matrix(np.eye(3, dtype=int), CLASSES)

    assert fig.axes[0].get_title() == "Confusion matrix (test set)"


def test_log_training_history(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    with mlflow.start_run() as run:
        log_training_history(FakeHistory())
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_accuracy")

    assert [(m.step, m.value) for m in history] == [(0, 0.4), (1, 0.7), (2, 0.65)]


def test_build_model_has_dropout(model_cfg):
    model = build_model(model_cfg)
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]

    assert [layer.rate for layer in dropout_layers] == [0.3]


def test_train_model_stops_early(model_cfg, tiny_dataset):
    # With learning rate 0 the weights never change, so val_loss never improves:
    # early stopping must stop after 1 + patience epochs instead of running all 10.
    model_cfg.learning_rate = 0.0
    model = build_model(model_cfg)
    history = train_model(model, tiny_dataset, tiny_dataset, epochs=10, patience=2)

    assert len(history.history["loss"]) == 3
