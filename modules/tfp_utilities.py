import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns


tfd = tfp.distributions

sns.set_theme()


def nll(y_true, distr):
    """
    """
    return - distr.log_prob(y_true)


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
    pred_plot = model(x_plot[:, tf.newaxis])

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


def generate_predictions(model, points, n_distr=5, n_samples=500):
    """
    Generates predictions for the specified points from a Bayesian model.
    For each point, first we sample `n_distr` sets of parameters, then we
    sample the distribution corresponding to each set of parameters
    `n_samples` times.

    In total for each point in `points` we'll have:
      * `n_distr` output Gaussian distributions, the mean and standard
        deviation of which is appended to `means` and `stdevs` respectively.
      * `n_means * n_samples` samples (`n_samples` for each mean).

    If `n_samples` is set to `None`, then the distributions are not sampled at
    all.

    Parameters
    ----------
    points : list
        List of values to predict for (i.e. the x values).
    n_means : int (default: 5)
        Number of distributions to generate for each point.
    n_samples : int (default: 500)
        Number of sampled points (y values) from each distribution.
    """
    points = tf.constant([[p] for p in points])

    # Final shape: (n_means, n_points).
    means = []

    # Final shape: (n_means, n_points).
    stdevs = []

    # Final shape: (n_samples, n_means, n_points).
    samples = []

    for i in range(n_distr):
        distr = model(points)

        means.append(distr.mean()[:, 0].numpy().tolist())
        stdevs.append(distr.stddev()[:, 0].numpy().tolist())

        if n_samples is not None:
            samples.append(distr.sample(n_samples)[:, :, 0].numpy().tolist())

    means = tf.constant(means)
    stdevs = tf.constant(stdevs)
    samples = tf.constant(samples)

    return means, stdevs, samples


def plot_uncertainty(
        x_min,
        x_max,
        n_points,
        model,
        how='means',
        n_samples=None,
        n_distr=50
    ):
    """
    Given a Bayesian model and a set of `n_points` evenly spaced points
    between `x_min` and `x_max`, plots the uncertainties associated to the
    predictions on those points in two different ways:
      * means: for each point, `n_distr` output distributions (predictions)
               are generated and their means are considered. The means for
               each point, along with their average and their +/- 2 * stdev
               interval are plotted.
      * samples: for each point, `n_distr` output distributions (predictions)
                 are generated and a number `n_samples` of samples are drawn
                 from each of them. Then the samples, their mean and their
                 +/- 2 * stdev intereval are plotted.

    Parameters
    ----------
    x_min : float
        Smallest x for which to generate predictions.
    x_max : float
        Biggest x for which to generate predictions.
    n_points : int
        Number of evenly spaced points to generate predictions for (between
        `x_min` and `x_max`, extrema included).
    model : tf.keras.Model
        A Bayesian model. We assume that each inference run on the same input
        values generates a different set of output distributions.
    how : str (either 'means' or 'samples')
        String indicating how to plot the uncertainty.
    n_samples : int (default: None)
        Number of samples to draw from each distribution in case
        `how='samples'`.
    n_distr : int (default: 50)
        Number of output distributions (inference runs) to make for each
        value of x.
    """
    x_interval = tf.linspace(x_min, x_max, n_points)

    if how == 'means':
        means, stdevs, samples = generate_predictions(
            model,
            x_interval.numpy().tolist(),
            n_samples=None,
            n_distr=n_distr
        )

        average_means = tf.reduce_mean(means, axis=0)
        stdev_means = tfp.stats.stddev(means)
        average_stdevs = tf.reduce_mean(stdevs, axis=0)

        fig = plt.figure(figsize=(14, 6))

        sns.lineplot(
            x=x_interval.numpy(),
            y=average_means,
            color=sns.color_palette()[0],
            label='Average of the means'
        )

        sns.scatterplot(
            x=tf.reshape(tf.constant([[x_position.numpy()] * means.shape[0] for x_position in x_interval]), x_interval.shape[0] * means.shape[0]).numpy(),
            y=tf.reshape(tf.transpose(means), means.shape[0] * means.shape[1]),
            color=sns.color_palette()[0],
            label='Means',
            alpha=.5
        )

        plt.fill_between(
            x=x_interval.numpy(),
            y1=average_means - 2. * stdev_means,
            y2=average_means + 2. * stdev_means,
            color=sns.color_palette()[0],
            alpha=.3,
            label='$\pm$ 2 * [st. dev. of the means] interval'
        )

        plt.legend(fontsize=10, loc='upper left')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Predictions (means)', fontsize=14)

    elif how == 'samples':
        if (n_samples is None) or (n_samples == 0):
            raise Exception(
                'n_samples must be a positive integer if working with '
                f'samples (received: {n_samples})'
            )
        means, stdevs, samples = generate_predictions(
            model,
            x_interval.numpy().tolist(),
            n_samples=n_samples,
            n_distr=n_distr
        )

        sample_means = tf.reduce_mean(
            tf.reshape(tf.transpose(samples, perm=[2, 0, 1]), (samples.shape[2], samples.shape[0] * samples.shape[1])),
            axis=1
        )

        sample_stdevs = tfp.stats.stddev(
            tf.reshape(tf.transpose(samples, perm=[2, 0, 1]), (samples.shape[2], samples.shape[0] * samples.shape[1])),
            sample_axis=1
        )

        fig = plt.figure(figsize=(14, 6))

        sns.lineplot(
            x=x_interval.numpy(),
            y=sample_means,
            color=sns.color_palette()[0],
            label='Samples means'
        )

        plt.fill_between(
            x=x_interval.numpy(),
            y1=sample_means - 2. * sample_stdevs,
            y2=sample_means + 2. * sample_stdevs,
            color=sns.color_palette()[0],
            alpha=.3,
            label='$\pm$ 2 * [sample st. dev.] interval'
        )

        sns.scatterplot(
            x=tf.reshape(tf.constant([[x_position.numpy()] * (samples.shape[0] * samples.shape[1]) for x_position in x_interval]), x_interval.shape[0] * samples.shape[0] * samples.shape[1]).numpy(),
            y=tf.reshape(tf.transpose(samples, perm=[2, 0, 1]), samples.shape[0] * samples.shape[1] * samples.shape[2]),
            color=sns.color_palette()[0],
            label='Samples',
            alpha=.3
        )

        plt.legend(fontsize=10, loc='upper left')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Predictions (samples)', fontsize=14)

    else:
        raise NotImplementedError(f'Option how={how} not implemented')


