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

        epsilon_samples = tfd.Normal(
            loc=[0., 0.],
            scale=[1., 1.]
        ).sample(tf.shape(z_mean)[0])

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


class VAE(tf.keras.Model):
    """
    """
    def __init__(self, variational_encoder, decoder, beta=500.):
        """
        """
        super().__init__()

        self.variational_encoder = variational_encoder
        self.decoder = decoder

        # Coefficient of the reconstruction loss term in the loss function.
        self.beta = beta

        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name='reconstruction_loss'
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

    def get_config(self):
        """
        Generates the config for the model (needed for serialization/
        deserialization) and seralizes custom objects (layers, etc.).
        """
        base_config = super().get_config()

        config_extension = {
            'variational_encoder': tf.keras.saving.serialize_keras_object(
                self.variational_encoder
            ),
            'decoder': tf.keras.saving.serialize_keras_object(self.decoder),
            'beta': self.beta,
            # 'decoder_image_reshaping_size': self.decoder.image_reshaping_size
        }

        return {**base_config, **config_extension}

    @classmethod
    def from_config(cls, config):
        """
        Class method (receives the class as the implicit first input just like
        instsance methods receive the instance) returning an instance of the
        class inintialized with the parameters in `config`. Basically,
        generates an instance of the present class from a config.
        """
        variational_encoder = tf.keras.saving.deserialize_keras_object(
            config.pop('variational_encoder')
        )
        decoder = tf.keras.saving.deserialize_keras_object(
            config.pop('decoder')
        )

        return cls(
            variational_encoder=variational_encoder,
            decoder=decoder,
            **config
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, x):
        z_mean, z_log_var, z_samples = self.variational_encoder(x)

        reconstructed_samples = self.decoder(z_samples)

        return z_mean, z_log_var, reconstructed_samples

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstructed_samples = self(x)

            # Mean over trianing batch.
            reconstruction_loss = tf.reduce_mean(
                self.beta
                * tf.keras.losses.binary_crossentropy(
                    x,
                    reconstructed_samples,
                    axis=(1, 2, 3)
                )
            )

            # Mean over training batch.
            kl_loss = tf.reduce_mean(
                # Sum over the dimensions of latent space.
                tf.reduce_sum(
                    - 0.5 * (
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    ),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        # Compute gradient of the total loss w.r.t. the model's weights.
        grad = tape.gradient(total_loss, self.trainable_weights)

        # Gradient descent step.
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))

        # Track losses.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
