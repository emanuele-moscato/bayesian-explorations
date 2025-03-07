import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


class NEpochsLogger(tf.keras.callbacks.Callback):
    """
    """
    def __init__(self, n_epochs):
        """
        """
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """
        """
        if epoch % self.n_epochs == 0:
            if 'mse' in logs.keys():
                print(
                    f'Epoch: {epoch} - loss: {logs["loss"]}'
                    f' - mse: {logs["mse"]}'
                )
            else:
                print(f'Epoch: {epoch} - loss: {logs["loss"]}')


def append_to_full_history(training_history, full_history):
    """
    """
    if isinstance(full_history, tf.keras.callbacks.History):
        for key, value in training_history.history.items():
            if key in full_history.history.keys():
                full_history.history[key] += value
            else:
                full_history.history[key] = value
    elif isinstance(full_history, dict):
        for key, value in training_history.history.items():
            if key in full_history.keys():
                full_history[key] += value
            else:
                full_history[key] = value
    else:
        raise ValueError(
            'Input training_history should be of type '
            '{tf.keras.callbacks.History} or dict'
        )

    return full_history


def plot_history(training_history):
    """
    """
    if isinstance(training_history, tf.keras.callbacks.History):
        history = training_history.history
    elif isinstance(training_history, dict):
        history = training_history
    else:
        raise ValueError(
            'Input training_history should be of type '
            '{tf.keras.callbacks.History} or dict'
        )

    for key, values_list in history.items():
        fig = plt.figure(figsize=(14, 6))

        sns.lineplot(
            x=range(len(values_list)),
            y=values_list
        )

        plt.title(f'{key}', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Value')


class MCDropoutModel(tf.keras.Model):
    """
    Given a model with dropout layers (passed to the constructor), this class
    builds an equivalent model with the same exact layers (with trained
    parameters), but in which the dropout layers are always called with the
    `training` option set to True so the sampling happens at inference time as
    well.
    """
    def __init__(self, original_model):
        """
        Class constructor. In requires the original model (with dropout
        layers) as the input.
        """
        super().__init__()

        self.original_model = original_model

    def build(self):
        """
        """
        input = tf.keras.layers.Input(
            shape=self.original_model.input.shape[1:]
        )

        output = self.original_model.layers[0](input)

        for layer in self.original_model.layers[1:]:
            if 'dropout' in layer.name:
                print(f'Dropout layer found: {layer.name}')

                output = layer(output, training=True)
            else:
                output = layer(output)

        return tf.keras.Model(inputs=input, outputs=output)


def get_intermediate_output(x, model, n_layers):
    """
    Given the NN `model`, passes the input `x` through its first `n_layers`
    layers and outputs the result.
    """
    output = x

    for i in range(n_layers):
        output = model.layers[i](output)

    return output
