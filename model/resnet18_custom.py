import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    GlobalAveragePooling2D,
    Dense,
    Add
)

# ======================================================
# RESNET BLOCK
# ======================================================
@tf.keras.utils.register_keras_serializable()
class ResnetBlock(Model):
    def __init__(self, channels, down_sample=False, **kwargs):
        super().__init__(**kwargs)

        self.channels = channels
        self.down_sample = down_sample
        self.strides = [2, 1] if down_sample else [1, 1]

        self.conv1 = Conv2D(
            channels,
            3,
            strides=self.strides[0],
            padding="same",
            kernel_initializer="he_normal"
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            channels,
            3,
            strides=self.strides[1],
            padding="same",
            kernel_initializer="he_normal"
        )
        self.bn2 = BatchNormalization()

        if down_sample:
            self.res_conv = Conv2D(
                channels,
                1,
                strides=2,
                padding="same",
                kernel_initializer="he_normal"
            )
            self.res_bn = BatchNormalization()
        else:
            self.res_conv = None
            self.res_bn = None

        self.add = Add()

    def call(self, x, training=False):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.down_sample:
            shortcut = self.res_conv(shortcut)
            shortcut = self.res_bn(shortcut, training=training)

        x = self.add([x, shortcut])
        return tf.nn.relu(x)

    # 🔥 SERIALIZAÇÃO
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "down_sample": self.down_sample
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ======================================================
# RESNET18
# ======================================================
@tf.keras.utils.register_keras_serializable()
class ResNet18(Model):
    def __init__(self, num_classes, input_shape=(160, 160, 3), **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.input_shape_ = input_shape

        self.conv1 = Conv2D(
            64,
            7,
            strides=2,
            padding="same",
            kernel_initializer="he_normal"
        )
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(3, strides=2, padding="same")

        self.res1 = [ResnetBlock(64), ResnetBlock(64)]
        self.res2 = [ResnetBlock(128, True), ResnetBlock(128)]
        self.res3 = [ResnetBlock(256, True), ResnetBlock(256)]
        self.res4 = [ResnetBlock(512, True), ResnetBlock(512)]

        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation="softmax")

        # força build para salvar corretamente
        self.build((None, *input_shape))

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        for block in self.res1 + self.res2 + self.res3 + self.res4:
            x = block(x, training=training)

        x = self.gap(x)
        return self.fc(x)

    # 🔥 SERIALIZAÇÃO
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
