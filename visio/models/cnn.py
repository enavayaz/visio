"""
visio.models.cnn
~~~~~~~~~~~~~~~~
CNN-based image classifier with built-in augmentation and training callbacks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


class ImNet:
    """
    A three-block convolutional neural network for image classification.

    Architecture:
        - Blocks 1–2 : Conv2D → BN → Conv2D → BN → MaxPool → Dropout
        - Block  3   : Conv2D → BN → Dropout
        - Head       : Dense(512) → BN → Dropout → Dense(256) → BN → Dropout → Softmax

    Args:
        input_shape  : Shape of a single input image. Default: (28, 28, 1).
        num_classes  : Number of output classes. Default: 10.
        learning_rate: Adam optimizer learning rate. Default: 1e-3.
        batch_size   : Mini-batch size for training. Default: 64.
        epochs       : Maximum number of training epochs. Default: 50.
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=10,
        learning_rate=1e-3,
        batch_size=64,
        epochs=50,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model: models.Sequential | None = None
        self.datagen: ImageDataGenerator | None = None
        self.history = None

    def compile(self) -> None:
        """Build, compile the model and initialise the data augmentation pipeline."""
        self.model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Classifier head
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax"),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )

        print("Model compiled and augmentation ready.")
        self.model.summary()

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        """
        Train the model with augmentation and adaptive callbacks.

        Args:
            X_train: Training images of shape (N, H, W, C).
            y_train: Integer class labels for training set.
            X_val  : Validation images. If None, 10% of X_train is used.
            y_val  : Validation labels. If None, 10% of y_train is used.
        """
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            print("No val set provided — using internal 90/10 split.")

        self.datagen.fit(X_train)

        callbacks = [
            ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ]

        self.history = self.model.fit(
            self.datagen.flow(X_train, y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            steps_per_epoch=len(X_train) // self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )

        loss, acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Final Val Loss: {loss:.4f} | Final Val Accuracy: {acc:.4f}")

    def predict(self, X_test) -> np.ndarray:
        """
        Generate class predictions for the given input.

        Args:
            X_test: Images of shape (N, H, W, C).

        Returns:
            Integer class predictions of shape (N,).
        """
        return np.argmax(self.model.predict(X_test), axis=1)

    def save(self, path="visio_model.keras") -> None:
        """
        Save the trained model to disk.
    
        Args:
            path: File path to save the model. Default: 'visio_model.keras'.
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path="visio_model.keras") -> None:
        """
        Load a previously saved model from disk.
    
        Args:
            path: File path of the saved model.
        """
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
