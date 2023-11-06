import numpy as np
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


class AffineBijector(tfp.bijectors.Bijector):
    """
    Subclass of TFP's `Bijector` object implementing an affine bijector.

    Note: the fact that this bijector is used in RealNVP model is fully
          encoded in the function passed to the constructor to generate the
          affine parameters. In general, this bijector knows nothing about
          RealNVP models.
    """
    def __init__(
        self,
        scale_and_transl_fn,
        validate_args=False,
        name='affine_bij'
    ):
        """
        Class constructor. The `scale_and_transl_fn` parameter is a callable
        returning the scale and translation (shift) parameters for each point
        it's called upon.
        """
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            is_constant_jacobian=False,
            name=name
        )

        # Callable returning the scale and translation values parametrizing
        # the bijector.
        self.scale_and_transl_fn = scale_and_transl_fn

    def _forward(self, z):
        """
        Forward trasformation.
        """
        s, t = self.scale_and_transl_fn(z)

        affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=t),
            tfp.bijectors.Scale(log_scale=s)
        ])

        return affine_bijector.forward(z)

    def _forward_log_det_jacobian(self, z):
        """
        Log determinant of the forward transformation.
        """
        s, t = self.scale_and_transl_fn(z)

        affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=t),
            tfp.bijectors.Scale(log_scale=s)
        ])

        return affine_bijector.forward_log_det_jacobian(z)

    def _inverse(self, x):
        """
        Inverse transformation.
        """
        # We can pass x (belonging to the target space) to the coupling layer
        # because it only really uses the first n_masked_dims dimensions,
        # for which it holds that x_i = z_i.
        s, t = self.scale_and_transl_fn(x)

        inverse_affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Scale(scale=tf.math.exp(-s)),
            tfp.bijectors.Shift(shift=-t)
        ])

        return inverse_affine_bijector.forward(x)

    def _inverse_log_det_jacobian(self, x):
        """
        Log determinant of the inverse transformation.
        """
        # We can pass x (belonging to the target space) to the coupling layer
        # because it only really uses the first n_masked_dims dimensions,
        # for which it holds that x_i = z_i.
        s, t = self.scale_and_transl_fn(x)

        inverse_affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Scale(tf.math.exp(-s)),
            tfp.bijectors.Shift(shift=-t)
        ])

        return inverse_affine_bijector.forward_log_det_jacobian(x)


class RealNVPLayer(tf.keras.layers.Layer):
    """
    Subclass of Keras' `Layer` implementing a RealNVP block consisting in
    (sequentially):
      1. A permutation of the features of the input.
      2. An affine transformation parametrized by a `CouplingLayer` object.
    """
    def __init__(
        self,
        n_masked_dims,
        n_affine_dims,
        hidden_layers_dims
    ):
        """
        Class constructor. Three basic objects are instantiated (and assigned
        to attributes) and composed:
          * A `CouplingLayer` providing the parametrization for an affine
            bijector for the RealNVP block.
          * An affine bijector.
          * A permutation bijector.
        """
        super().__init__()

        self.n_masked_dims = n_masked_dims
        self.n_affine_dims = n_affine_dims
        self.hidden_layers_dims = hidden_layers_dims

        # Generate a permutation of the feature indices.
        self.feature_permutation = np.random.permutation(
            range(self.n_masked_dims + self.n_affine_dims)
        )

        # Instantiate a permutation bijector.
        self.permutation = tfp.bijectors.Permute(
            permutation=self.feature_permutation, axis=-1
        )

        # Instantiate a RealNVP affine bijector.
        self.coupling_layer = CouplingLayer(
            n_masked_dims=self.n_masked_dims,
            n_affine_dims=self.n_affine_dims,
            hidden_layers_dims=self.hidden_layers_dims
        )

        self.affine_bijector = AffineBijector(
            scale_and_transl_fn=self.coupling_layer
        )

        self.bijector = tfp.bijectors.Chain([
            self.affine_bijector,
            self.permutation
        ])

    def call(self, z):
        """
        Forward pass, consisting in the forward tranformation of the chained
        bijector composed by (in sequential order):
          1. A permutation of the features of the input x.
          2. An affine transformation parametrized by a `CouplingLayer`
             object.
        """
        return self.bijector.forward(z)


class RealNVPModel(tf.keras.Model):
    """
    Subclass of Keras `Model` implementing a RealNVP flow model.
    """
    def __init__(
        self,
        n_masked_dims,
        n_affine_dims,
        n_real_nvp_blocks,
        hidden_layers_dims
    ):
        """
        Parameters
        ----------
        n_masked_dims : int
            Number of masked dimension (left unaltered by the RealNVP affine
            transformations).
        n_affine_dims : int
            Number of dimensions actually transformed by the RealNVP affine
            transformations.
        n_real_nvp_blocks : int
            Number of RealNVP (permutation + affine transformation) blocks.
        hidden_layers_dims : list
            List of integers indicating the number of units of the HIDDEN
            Dense layers in the coupling layers. The total number of Dense
            layers is in the stacks generating the output tensors is
            len(hidden_layers_dims) + 1.
        """
        super().__init__()

        self.n_masked_dims = n_masked_dims
        self.n_affine_dims = n_affine_dims
        self.n_real_nvp_blocks = n_real_nvp_blocks
        self.hidden_layers_dims = hidden_layers_dims

        self.base_distr = tfd.Independent(
                tfd.Normal(
                loc=tf.zeros(shape=self.n_masked_dims + self.n_affine_dims),
                scale=tf.ones(shape=self.n_masked_dims + self.n_affine_dims)
            ),
            reinterpreted_batch_ndims=1
        )

        self.blocks = [
            RealNVPLayer(
                n_masked_dims=self.n_masked_dims,
                n_affine_dims=self.n_affine_dims,
                hidden_layers_dims=self.hidden_layers_dims
            )
            for _ in range(self.n_real_nvp_blocks)
        ]

        self.full_bijector = tfp.bijectors.Chain(
            [block.bijector for block in self.blocks]
        )

        self.transformed_distr = tfd.TransformedDistribution(
            distribution=self.base_distr,
            bijector=self.full_bijector,
        )

    def call(self, z):
        """
        Forward pass, corresponding to the forward transformation of the full
        bijector.
        """
        return self.transformed_distr.bijector.forward(z)
