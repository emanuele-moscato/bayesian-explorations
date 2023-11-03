import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


tfd = tfp.distributions


class CouplingLayer(tf.keras.layers.Layer):
    """
    Subclass of Keras `Layer` implementing a single coupling layer
    for a RealNVP flow model. The layer outputs two tensors that
    parametrize the affine transformation for the features that
    transform nontrivially. These tensors are obtained via a stack
    of Dense layers.
    """
    def __init__(self, n_masked_dims, n_affine_dims, hidden_layers_dims):
        """
        Parameters
        ----------
        n_affine_dims : int
            Number of dimensions (features) that will be transformed
            via the affine transformation parametrized by the output
            of the layer. Each tensor (scale and translation) outputted
            by the layer will have dimension (batch_size, n_affine_dims).
        hidden_layers_dims : list
            List of integers indicating the number of units of the HIDDEN
            layers. The total number of layers is in the stacks generating
            the output tensors is len(hidden_layers_dims) + 1.
        """
        super().__init__()

        # Compute dimensions.
        self.n_masked_dims = n_masked_dims
        self.n_affine_dims = n_affine_dims
        self.n_total_dims = n_masked_dims + n_affine_dims

        self.n_hidden_layers = len(hidden_layers_dims)

        # Stak of Dense layers parametrizing the scale factor.
        self.s_stack = [
            Dense(units=units, activation='relu')
            for units in hidden_layers_dims
        ]
        self.s_stack.append(
            Dense(units=n_affine_dims)
        )

        # Stak of Dense layers parametrizing the translation factor.
        self.t_stack = [
            Dense(units=units, activation='relu')
            for units in hidden_layers_dims
        ]
        self.t_stack.append(
            Dense(units=n_affine_dims)
        )

    def call(self, z):
        """
        Forward pass for the layer. The output tensors s and t parametrize
        the scale and the translation factors, respectively. The forward pass
        procedes in two steps:
          1. We compute the nontrivial part of s and t, resulting in tensors
             with shape (batch_shape, n_affine_dims). These parametrize the
             affine transformation of the affine dimensions.
          2. We concatenate a trivial part to s and t, corresponding to a
             trivial affine transformation for the masked dimensions. This is
             achieved by introducing zeros for the masked dimensions (both for
             the scale and translation factors, as the scale factor is then
             passed through an exponential).
        """
        # Isolate the masked dimensions: the (nontrivial part of the) scale
        # and translation factors will be functions of these alone.
        s = z[..., :self.n_masked_dims]
        t = z[..., :self.n_masked_dims]

        # Resulting shape: (batch_shape, n_affine_dims).
        for i in range(self.n_hidden_layers + 1):
            s = self.s_stack[i](s)
            t = self.t_stack[i](t)

        # Introduce a trivial scaling factor (unity) for the masked dimension.
        # Resulting shape: (batch_shape, n_masked_dims + n_affine_dims).
        s = tf.concat(
            [tf.zeros(shape=(*s.shape[:-1], self.n_masked_dims)), s],
            axis=1
        )

        # Introduce a trivial translation factor (zero) for the masked
        # dimension.
        # Resulting shape: (batch_shape, n_masked_dims + n_affine_dims).
        t = tf.concat(
            [tf.zeros(shape=(*t.shape[:-1], self.n_masked_dims)), t],
            axis=1
        )

        return [s, t]


class RealNVPBijector(tfp.bijectors.Bijector):
    """
    Subclass of TFP's `Bijector` object implementing the affine bijector
    needed for a RealNVP model. In particular, the affine transformation
    should be trivial (i.e. the identity, i.e. with unit scaling and zero
    translation factor) for the first n_masked_dims dimensions of the input,
    and parametrized by a neural network for the last n_affine_dims dimensions
    of the input.
    """
    def __init__(self, coupling_layer, validate_args=False, name='real_nvp'):
        """
        """
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            is_constant_jacobian=False,
            name=name
        )

        # Coupling layer parametrizing the bijector.
        self.coupling_layer = coupling_layer

    def _forward(self, z):
        """
        """
        s, t = self.coupling_layer(z)

        affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=t),
            tfp.bijectors.Scale(log_scale=s)
        ])

        return affine_bijector.forward(z)

    def _forward_log_det_jacobian(self, z):
        """
        """
        s, t = self.coupling_layer(z)

        affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=t),
            tfp.bijectors.Scale(log_scale=s)
        ])

        return affine_bijector.forward_log_det_jacobian(z)

    def _inverse(self, x):
        """
        """
        # We can pass x (belonging to the target space) to the coupling layer
        # because it only really uses the first n_masked_dims dimensions,
        # for which it holds that x_i = z_i.
        s, t = self.coupling_layer(x)

        inverse_affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Scale(scale=tf.math.exp(-s)),
            tfp.bijectors.Shift(shift=-t)
        ])

        return inverse_affine_bijector.forward(x)

    def _inverse_log_det_jacobian(self, x):
        """
        """
        # We can pass x (belonging to the target space) to the coupling layer
        # because it only really uses the first n_masked_dims dimensions,
        # for which it holds that x_i = z_i.
        s, t = self.coupling_layer(x)

        inverse_affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Scale(tf.math.exp(-s)),
            tfp.bijectors.Shift(shift=-t)
        ])

        return inverse_affine_bijector.forward_log_det_jacobian(x)


class RealNVPLayer(tf.keras.Model):
    """
    Subclass of Keras `Model` implementing a RealNVP flow model.
    """
    def __init__(
            self
        ):
        """
        """
        super().__init__()

        pass

    def call(self, z):
        """
        """
        pass
