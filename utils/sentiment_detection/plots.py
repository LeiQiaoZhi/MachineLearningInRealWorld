import os
import typing


def best_fit(token_frequencies_log: typing.List[typing.Tuple[float, float]],
             token_frequencies: typing.List[typing.Tuple[int, int]]) -> typing.List[float]:
    """
    Uses linear least squares regression to calculate a line of best fit for the given log token frequencies. Log
    frequencies of word are weighted by its frequency to avoid distortion in favour of less common words.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: the slope and y-intersect of the best fit
    """

    total_count = sum([x[1] for x in token_frequencies])
    frequencies = [x[1] for x in token_frequencies]

    X = [x[0] for i, x in enumerate(token_frequencies_log)]
    Y = [x[1] for i, x in enumerate(token_frequencies_log)]

    X_weighted = [x[0] * (frequencies[i]) for i, x in enumerate(token_frequencies_log)]
    Y_weighted = [x[1] * (frequencies[i]) for i, x in enumerate(token_frequencies_log)]

    mean_x = sum(X_weighted) / total_count
    mean_y = sum(Y_weighted) / total_count

    n = len(X)

    covariance = 0
    x_variance = 0
    for i in range(n):
        covariance += (X[i] - mean_x) * (Y[i] - mean_y) * frequencies[i]
        x_variance += ((X[i] - mean_x) ** 2) * frequencies[i]
    m = covariance / x_variance
    c = mean_y - (m * mean_x)
    return [m, c]


def clean_plot():
    import matplotlib.pyplot as plt
    """
    Cleans the plot canvas. Call this method between different draw methods (e.g. draw_frequency_ranks and draw_zipf)
    """

    plt.clf()


def chart_plot(data: typing.List[typing.Tuple[float, float]], title: str, x_label: str, y_label: str,label=''):
    import matplotlib.pyplot as plt
    """
    Takes any number of data points in the form of a list of tuples (x and y coordinates), and plots the points and
    corresponding line on a chart and saves it to file.

    @param data: list of data points in the form of tuples, with the first entry being the x and the second entry the y
        coordinate.
    @param title: the title of the plot
    @param x_label: the label of the x-axis
    @param y_label: the label of the y-axis
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    """

    plt.plot([x[0] for x in data], [x[1] for x in data], '-o', markersize=3,label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    directory = 'figures/sentiment_detection/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if label != '':
        plt.legend()
    plt.savefig(os.path.join(directory, f'{title}.png'), dpi=300)
