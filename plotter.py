import matplotlib as mpl
import matplotlib.pyplot as plt


def plot(
    xdata,
    ydata,
    ax=None,
    axis_labels=["x", "y"],
    axis_limits=None,
    title=None,
    **kwargs,
):
    ax = ax or plt.gca()

    if title is not None:
        ax.set(title=title)

    if isinstance(axis_labels, list) == True:
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1])

    if isinstance(axis_limits, list) == True:
        if len(axis_limits) == 2:
            ax.set_xlim(axis_limits[0], axis_limits[1])
        else:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])

    return ax.plot(xdata, ydata, **kwargs)
