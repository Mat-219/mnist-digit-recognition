import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ExpDecay(keras.optimizers.schedules.LearningRateSchedule):
    """lr = lrmin + (lrmax - lrmin) * exp(-step / decay_steps)"""

    def __init__(self, lrmin=0.0001, lrmax=0.003, decay_steps=2000.0):
        self.lrmin, self.lrmax, self.decay_steps = lrmin, lrmax, decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.lrmin + (self.lrmax - self.lrmin) * tf.exp(-step / self.decay_steps)

    def get_config(self):
        return {"lrmin": self.lrmin, "lrmax": self.lrmax, "decay_steps": self.decay_steps}


def softmax_1layer():
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ], name="softmax_1layer")


def dense_5layer(dropout=0.0):
    model = keras.Sequential([layers.Input((28, 28, 1)), layers.Flatten()],
                             name="dense_5layer_dropout" if dropout else "dense_5layer")
    for units in (200, 100, 60, 30):
        model.add(layers.Dense(units, activation="relu"))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation="softmax"))
    return model


def conv_5layer(dropout=0.25):
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(4, 5, strides=1, padding="same", activation="relu"),   # 28x28x4
        layers.Conv2D(8, 5, strides=2, padding="same", activation="relu"),   # 14x14x8
        layers.Conv2D(12, 4, strides=2, padding="same", activation="relu"),  # 7x7x12
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ], name="conv_5layer")


# name -> (model builder, optimizer builder)
EXPERIMENTS = {
    "1_softmax": (softmax_1layer, lambda: keras.optimizers.SGD(0.3)),
    "2_dense5_relu": (lambda: dense_5layer(0.0), lambda: keras.optimizers.Adam(0.003)),
    "3_dense5_relu_dropout": (lambda: dense_5layer(0.25), lambda: keras.optimizers.Adam(0.003)),
    "4_conv5_dropout_lrdecay": (conv_5layer, lambda: keras.optimizers.Adam(ExpDecay())),
}