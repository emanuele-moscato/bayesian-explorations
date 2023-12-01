import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Reshape, Conv2DTranspose)


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Sequence of number of filters (for the corresponding
        # convolutional layers in the stack).
        self.filters_list = [32, 64, 128]

        # Input placeholder. Actually not needed if defining
        # the model this way.
        self.model_input = Input(shape=(32, 32, 1))

        # Stack of convolutional layers.
        self.conv_layers = [
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=2,
                activation='relu',
                padding='same'
            )
            for filters in self.filters_list
        ]

        # Flattening layer to reshape the image into a rank-1
        # tensor.
        self.flatten = Flatten()

        # Final dense layer, projecting the intermediate outputs
        # to the 2-dimensional latent space.
        self.final_dense = Dense(units=2)

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.flatten(x)

        x = self.final_dense(x)

        return x


class Decoder(tf.keras.Model):
    def __init__(self, image_reshaping_size):
        super().__init__()

        self.image_reshaping_size = image_reshaping_size

        # Initial dense layer: maps latent vectors into
        # vectors with enough components to be reshaped
        # into images.
        self.initial_dense = Dense(
            units=np.prod(image_reshaping_size)
        )

        # Reshapes a rank-1 vector into a rank-3 tensor
        # representing an image.
        self.reshape = Reshape(image_reshaping_size)

        # List of numbers filters to be used by the stack of
        # transposed convolutional layers. The length of the list
        # corresponds to the number of layers in the stack.
        self.filters_list = [128, 64, 32]

        # Stack of transposed convolutional layers.
        self.conv_transp_layers = [
            Conv2DTranspose(
                filters=filters,
                kernel_size=(3, 3),
                strides=2,
                activation='relu',
                padding='same'
            )
            for filters in self.filters_list
        ]

        # Final convolutional layer, outputting a tensor with
        # the right shape, one channel dimension (grayscale)
        # and pixel intensities normalized in [0, 1] (sigmoid
        # activation).
        self.final_conv = Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=1,
            activation='sigmoid',
            padding='same'
        )

    def get_config(self):
        """
        Generates the config for the model.
        """
        config_extension = {
            'image_reshaping_size': self.image_reshaping_size
        }

        return {**config_extension}

    @classmethod
    def from_config(cls, config):
        """
        Class method that generates a new instance of the model starting from
        a config.
        """
        return cls(**config)

    def call(self, x):
        x = self.initial_dense(x)
        x = self.reshape(x)

        for conv_t_layer in self.conv_transp_layers:
            x = conv_t_layer(x)

        x = self.final_conv(x)

        return x
