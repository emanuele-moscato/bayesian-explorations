import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns


tfd = tfp.distributions

sns.set_theme()


def get_divergence_fn(norm_factor):
    """
    Return a callable to be used as divergence function given by the KL
    divergence scaled by the factor `norm_factor` (at the DENOMINATOR).
    """
    return lambda q, p, _: tfd.kl_divergence(q, p) / norm_factor


def plot_prediction_distr(
        x_data,
        y_data,
        model,
        x_plot_min=None,
        x_plot_max=None
    ):
    """
    """
    # Prediction with the model returns probability distributions: sample
    # them.
    # model_samples = model(x_data).sample()

    fig = plt.figure(figsize=(14, 6))

    sns.scatterplot(
        x=x_data[:, 0],
        y=y_data,
        color=sns.color_palette()[3],
        label='Data'
    )

    # Generate new x values to plot for.
    if x_plot_min is None:
        x_plot_min = x_data.numpy().min()
    if x_plot_max is None:
        x_plot_max = x_data.numpy().max()

    x_plot = tf.linspace(
        x_plot_min,
        x_plot_max,
        500
    )

    # Generate predictions (distributions) for the x values to plot.
    pred_plot = model(x_plot)

    # Plot the
    sns.scatterplot(
        x=x_plot,
        y=pred_plot.sample()[:, 0],
        color=sns.color_palette()[0],
        label='Model-generated data'
    )

    # Plot the means of the distributions.
    sns.lineplot(
        x=x_plot.numpy(),
        y=pred_plot.mean()[:, 0].numpy(),
        color=sns.color_palette()[1],
        label='Model distribution mean'
    )

    # Plot the means +/- 2 * [standard deviation] interval of the
    # distributions.
    plt.fill_between(
        x=x_plot.numpy(),
        y1=(pred_plot.mean()[:, 0] - 2. * pred_plot.stddev()[:, 0]).numpy(),
        y2=(pred_plot.mean()[:, 0] + 2. * pred_plot.stddev()[:, 0]).numpy(),
        color=sns.color_palette()[2],
        label='Model 2$\sigma$ interval',
        alpha=.2
    )

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Synthetic data', fontsize=14)
