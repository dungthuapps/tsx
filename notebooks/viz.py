"""Module to support visualization."""
import pyts 
import pandas as pd
import numpy as np

import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


def plt_legend(fig, per_subplot=False):
    if per_subplot:
        # Plot all legends outside
        [ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) for ax in fig.axes]
        # [ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5)) for ax in axes]

    # Or all legends into one
    else:
        #1 get all handles and labels of the fig.
        #2 use zip() -> group homonegous objects, ex: (Lines, Labels)
        #3 concatenate/flatten lists of lists -> list: 
        #   - ex: [[1,2, 3], [4, 5]] ­-> [1, 2, 3, 2, 3]
        #   - sum([[1,2,3]], []) do a trick 
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(l, []) for l in zip(*lines_labels)]   
        fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

def plt_hide_yticks(fig):
    for ax in fig.axes:
        ax.set_yticks([])

def plt_vlines(fig, n_steps=128, window_size=4, **kwargs):
    starts, ends, n_windows = pyts.utils.segmentation(n_steps, window_size, overlapping=False)
    vlines_pos = [0] + list(ends)   # vlines positions from 0
    for ax in fig.axes:
        ymin, ymax = ax.get_ylim()
        ax.vlines(vlines_pos, ymin=ymin, ymax=ymax, colors='r', ls=':')

# Plot segment-labels
def plt_segment_labels(fig, n_steps, window_size, text_y=-0.10):
    starts, ends, n_windows = pyts.utils.segmentation(n_steps, window_size, overlapping=False)
    ax = fig.axes[-1]
    ax.text(-10, text_y, 'X\':')
    for i, p_x in enumerate(starts):
        # p_x = window_size * (i + 1)
        # text_x = p_x - (window_size / 2) # center
        text_x = p_x
        ax.text(text_x, text_y, r"$w_{%i}$" % i, c='r')

def plt_rescale_to_x(fig, x):
    # Rescale ylim to match originally with instance x
    for i, ax in enumerate(fig.axes):
        ymin = x.iloc[:, i].min()
        ymax = x.iloc[:, i].max()
        ax.set_ylim(ymin, ymax)
        
# Highlight perturbed area
def plt_highlight_perturbed_area(fig, z_prime, n_steps, w_size):
    starts, ends, n_windows = pyts.utils.segmentation(n_steps, w_size, overlapping=False)
    for i, ax in enumerate(fig.axes):
        for s, e, w in zip(starts, ends, range(n_windows)):
            if z_prime[i, w] == 0:
                ax.axvspan(s, e, color='gold', alpha=0.8)

# Hide xlabels
def plt_hide_subplot_xlabel(fig):
    for ax in fig.axes:
        x_axis = ax.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

def plt_hide_subplot_title(fig):
    # Hide title of each subplot
    for ax in fig.axes:
        ax.set_title("")

def plt_x_instance(x_mts, title=""):
    x_mts.plot(subplots=True, legend=False, title=title)

    fig = plt.gcf()
    plt_legend(fig, per_subplot=True)

def plt_x_segmented(x_mts, window_size, title="", label_pos=None):
    x_mts.plot(subplots=True, legend=False, title=title)

    fig = plt.gcf()
    plt_legend(fig, per_subplot=True)

    # Plot segment-lines
    n_steps = len(x_mts.index)
    plt_vlines(fig, n_steps, window_size=window_size)
    plt.subplots_adjust(hspace=.0)

    if not label_pos:
        ymin, ymax = fig.axes[-1].get_ylim()
        label_pos = ymin - (ymax - ymin)
    plt_segment_labels(fig, n_steps, window_size, label_pos)

# Visualization Perturbation - Async and z
def plt_sample_z(x, z, z_prime, independents, window_size, hspace=.1, title="", label_pos=None):
    n_segments, n_steps = z.shape
    z_df = pd.DataFrame(z.T, columns=independents)
    z_df.plot(subplots=True, legend=False, title=title)

    fig = plt.gcf()
    plt_legend(fig, per_subplot=True)

    # Plot segment-lines and labels
    plt_rescale_to_x(fig, x)
    plt_vlines(fig, n_steps, window_size=window_size)

    if not label_pos:
        ymin, ymax = fig.axes[-1].get_ylim()
        label_pos = ymin - (ymax - ymin)
    plt_segment_labels(fig, n_steps, window_size, label_pos)

    # Highlight and rescale ylim to original
    plt_highlight_perturbed_area(fig, z_prime, n_steps, window_size)

    # Adjust subplots
    plt.subplots_adjust(hspace=hspace)


def plt_sample_z_prime(z_prime, ylabels=None, xlabels=None):
    n_variables = z_prime.shape[0]
    n_segments = z_prime.shape[1]

    fig, ax = plt.subplots()
    
    colors = ['gold', 'lightblue']
    cmap = LinearSegmentedColormap.from_list('', colors)
    # cm = 'GnBu'

    mat = ax.imshow(z_prime, cmap=cmap, interpolation='nearest')
    if not xlabels:
        xlabels = [r"$w_{%i}$" % i for i in range(n_segments)]
    if ylabels:
        plt.yticks(range(n_variables), ylabels)
    if xlabels:
        plt.xticks(range(n_segments), xlabels)
    plt.xticks(rotation=30)
    plt.xlabel('X\'')

    # this places 0 or 1 centered in the individual squares
    for y in range(n_variables):
        for x in range(n_segments):
            t = z_prime[y, x]
            ax.annotate(t, xy=(x, y), horizontalalignment='center', verticalalignment='center')

    # Plot legend
    # white as 1, on
    # black as 0, off -> perturbed segment
    legend_elements = [Patch(facecolor=color, edgecolor='w') for color in colors]

    ax.legend(loc='upper left',
              bbox_to_anchor=(1.0, 0.0, 1, 1),
              labels=["0: off", "1: on"],
              handles=legend_elements
              )

def plt_coef(coef, feature_names=None, scaler=None, **kwargs):
    
    # coef = coef.reshape(n_features, -1)
    coef_df = pd.DataFrame(coef.T)  # back to (n_steps, n_cols)
    if feature_names:
        coef_df.columns = feature_names   
    if scaler:
        scaler.fit(coef_df.values)
        coef_df = pd.DataFrame(data=scaler.transform(coef_df.values),
                                columns=coef_df.columns)
    kwargs['kind'] = kwargs.get('kind') or 'bar'
    kwargs['subplots'] = kwargs.get('subplots') or 1
    coef_df.plot(**kwargs)

    fig = plt.gcf()
    fig.axes[int(len(coef_df.columns)/2 + 1)].set_ylabel("Weights")
    plt.xlabel("Features (Windows)")
    plt_legend(fig, per_subplot=True)
    plt.subplots_adjust(hspace=0.2)

    plt_hide_subplot_title(fig)


def plt_x_coef(x, w, c=None, names=None):
    n_features, n_steps = x.shape
    assert x.shape[0] == w.shape[0], \
        f"Shape of x {x.shape} does not match the weights {w.shape}"
    if isinstance(c, list):
        assert n_features == len(c), \
        f"length of c {len(c)} does not match x features{n_features}"
    if isinstance(c, str):
        c = np.repeat(c, n_features)
    if c is None:
        c = np.repeat("Reds", n_features)

    if names is None:
        names = np.repeat(None, n_features)
    
    fig, axes = plt.subplots(n_features,1)
    for i, ax in enumerate(axes):
        _plt_xcoef(ax, x[i], w[i], c[i], names[i])
            # ax.set_ylabel(names[i], 
            #    rotation='horizontal', 
            #    horizontalalignment='right')


def _plt_xcoef(ax, x_ts, w, c=None, name=None):
        # Create coordinates (x, y)
        coords = list(enumerate(x_ts))

        # Line-starts from : 0 -> semi-last (-1)
        starts = coords[:-1]

        # Line-ends from : 1 -> final
        ends = coords[1:]

        # Colors
        if not isinstance(c, str):
            c = "Reds"
        color_map = plt.cm.get_cmap(c)
        color_norm = colors.Normalize()

        # Line intervals
        lines = [(start, end) for start, end in zip(starts, ends)]
        lines_lc = LineCollection(
            lines, 
            # linewidths= 1 + w * 2, 
            array=w, 
            # norm=color_norm, 
            cmap=color_map,
        )

        line = ax.add_collection(lines_lc)
        ax.autoscale()
        # plt.colorbar(line)
        if name is not None:
            # Custom Legend for LineCollection
            handle = Line2D([0, 1], [0, 1], color=lines_lc.cmap(0.5))
            ax.legend([handle], [name], loc='center left', bbox_to_anchor=(1.0, 0.5))
    
