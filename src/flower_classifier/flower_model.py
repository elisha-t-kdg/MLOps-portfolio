import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.optimizers import Adam

from flower_data import HEIGHT, WIDTH, get_augmented_data_dir, load_datasets


def build_model():
    """Build and compile the transfer-learning model based on MobileNetV2 (a much smaller/lighter backbone than ResNet101)."""
    imported_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(180, 180, 3),
        pooling='avg', classes=5,
        weights='imagenet')
    for layer in imported_model.layers:
        layer.trainable = False

    imported_model.summary()

    dnn_model = keras.Sequential()
    dnn_model.add(imported_model)
    # dnn_model.add(Flatten())
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dense(5, activation='softmax'))

    dnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return dnn_model


def train_model(dnn_model, train_set, validation_set, epochs=10):
    """Train the model and return the training history."""
    history = dnn_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=epochs
    )
    return history


def predict_image(dnn_model, image_path, image_cat):
    """Run inference on a single image and print the predicted category."""
    image = cv2.imread(str(image_path))
    image_resized = cv2.resize(image, (HEIGHT, WIDTH))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)

    model_pred = dnn_model.predict(image)
    print(model_pred)
    predicted_class = image_cat[np.argmax(model_pred)]
    print("The predicted category is", predicted_class)
    return predicted_class


def main():
    augmented_data = get_augmented_data_dir()
    train_set, validation_set, image_cat = load_datasets(augmented_data)
    dnn_model = build_model()
    train_model(dnn_model, train_set, validation_set)

    sample_image = list((augmented_data / 'sunflowers').glob('*'))[1]
    predict_image(dnn_model, sample_image, image_cat)


if __name__ == "__main__":
    main()
