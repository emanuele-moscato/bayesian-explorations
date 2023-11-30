import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense


tfd = tfp.distributions


class SampleLayer(tf.keras.layers.Layer):
    """
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """
        Forward pass, returning samples from the multivariate
        Gaussian distributions parametrized by the inputs.
        """
        # Unpack the inputs (assumed to be produced by a
        # `VariationalEncoder` model.
        z_mean, z_log_var = inputs

        z_var = tf.exp(0.5 * z_log_var)

        epsilon_samples = tfd.Normal(loc=[0., 0.], scale=[1., 1.]).sample(tf.shape(z_mean)[0])

        return z_mean + tf.sqrt(z_var) * epsilon_samples


class VariationalEncoder(tf.keras.Model):
    """
    Subclass of Keras `Model` implementing the encoder part of a variational
    autoencoder. The output of a forward pass consists in three objects: the
    `mu` and `sigma` parameters of a multivariate Gaussian distribution on
    latent space and the samples from latent space (using those
    distributions).
    """
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

        # Dense layer outputting the mean parameter for the output
        # Gaussian distribution.
        self.dense_mean = Dense(units=2)

        # Dense layer outputting the variance parameter for the output
        # Gaussian distribution (i.e. the logarithme of the diagonal
        # entries of the diagonal covariance matrix.
        self.dense_variance = Dense(units=2)

        # Layer implementing the sampling from latent space.
        self.sample_layer = SampleLayer()

    def call(self, x):
        """
        Forward pass. The output is a list of tensors, the first one
        corresponding to the mean parameter of the Gaussian distribution on
        latent space, the second one corresponding to (the log of the diagonal
        entries of) its convariant matrix, the second one corresponding to the
        samples from latent space.
        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.flatten(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_variance(x)

        z_samples = self.sample_layer([z_mean, z_log_var])

        return [z_mean, z_log_var, z_samples]
