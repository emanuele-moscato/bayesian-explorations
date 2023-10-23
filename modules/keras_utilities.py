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
            print(f'Epoch: {epoch} - loss: {logs["loss"]} - mse: {logs["mse"]}')


def append_to_full_history(training_history, full_history):
    """
    """
    for key, value in training_history.history.items():
        if key in full_history.keys():
            full_history[key] += value
        else:
            full_history[key] = value

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


def nll(y_true, distr):
    """
    """
    return - distr.log_prob(y_true)
