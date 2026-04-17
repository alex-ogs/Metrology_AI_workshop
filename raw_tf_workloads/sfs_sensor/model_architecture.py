"""
Model for the sfs sensor data.
Uses the same workload script as the other models (other sensors or controllers experiments).
"""
from typing import Tuple
import tensorflow as tf

class model_architectures:
    def __init__(self):
        pass

    @staticmethod
    def build_sfs_cnn_model(input_shape: Tuple[int, int, int], num_outputs: int = 2, flat: bool = False) -> tf.keras.Model:
        """Build a CNN regressor that predicts (x_px, y_px) in resized pixels."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.Flatten() if flat else tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation="relu") if flat else tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_outputs, activation="linear"),
            ]
        )

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse")],
        )
        return model