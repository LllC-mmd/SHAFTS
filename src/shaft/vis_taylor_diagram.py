import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.axisartist.floating_axes as mFA
import mpl_toolkits.axisartist.grid_finder as mGF


# ref to: https://gist.github.com/ycopin/3342888
class TaylorDiagram(object):
    """
    Taylor diagram.
        Plot model standard deviation and correlation to reference (data)
        sample in a single-quadrant polar plot, with r=stddev and
        theta=arccos(correlation).
    """

    def __init__(self, std_ref, fig=None, rect=111, label='_', std_min=None, std_max=None, std_range=(0, 1.5),
                 normalized=False, extend=False, std_label_format='%.2f', num_std=6, ylabel_text=None):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
            * std_ref: reference standard deviation to be compared to
            * fig: input Figure or None
            * rect: subplot definition
            * label: reference label
            * std_range: stddev axis extension, in units of *std_ref*
            * extend: extend diagram to negative correlations
        """

        if normalized:
            self.std_ratio = std_ref
        else:
            self.std_ratio = 1.0

        self.std_ref = std_ref / self.std_ratio

        tr = PolarAxes.PolarTransform()

        # ------correlation labels
        r_locs = np.array([0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        # ------extended to negative correlations
        if extend:
            self.theta_max = np.pi
            r_locs = np.concatenate((-r_locs[:0:-1], r_locs))
        else:
            self.theta_max = np.pi / 2.0

        theta_locs = np.arccos(r_locs)
        gl1 = mGF.FixedLocator(theta_locs)
        tf1 = mGF.DictFormatter(dict(zip(theta_locs, map(str, r_locs))))

        # ------std axis extent (in units of reference stddev)
        if std_min is None or std_max is None:
            self.std_min = self.std_ref * std_range[0] / self.std_ratio
            self.std_max = self.std_ref * std_range[1] / self.std_ratio
        else:
            self.std_min = std_min / self.std_ratio
            self.std_max = std_max / self.std_ratio

        std_locs = np.linspace(self.std_min, self.std_max, num_std)
        std_labels = ["{0}".format(std_label_format) % v for v in std_locs]
        tf2 = mGF.DictFormatter(dict(zip(std_locs, std_labels)))

        ghelper = mFA.GridHelperCurveLinear(tr, extremes=(0, self.theta_max, self.std_min, self.std_max),
                                            grid_locator1=gl1, tick_formatter1=tf1, tick_formatter2=tf2)

        if fig is None:
            fig = plt.figure()

        ax = mFA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # ------adjust axes
        # ---------angle axis
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("CC")
        ax.axis["top"].label.set_size("large")

        # ---------X axis
        ax.axis["left"].set_axis_direction("bottom")
        if normalized:
            ax.axis["left"].label.set_text("$\sigma_{\hat{y}}/\sigma_{y}$")
        else:
            ax.axis["left"].label.set_text("\sigma_{\hat{y}}", fontsize="large")
        ax.axis["left"].label.set_size("large")
        # ---------Y axis
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("bottom" if extend else "left")
        if ylabel_text is not None:
            ax.axis["right"].label.set_text(ylabel_text)

        if self.std_min:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # ------add reference point and stddev contour
        l, = self.ax.plot([0], self.std_ref, 'r*', ls='', ms=5, label=label)
        theta = np.linspace(0, self.theta_max)
        r = np.zeros_like(theta) + self.std_ref
        self.ax.plot(theta, r, 'k--', label='_')

        # ------collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """
        Add a grid.
        """
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.std_min, self.std_max), np.linspace(0, self.theta_max))

        # ------calculate RMS difference
        rms = np.sqrt(self.std_ref**2 + rs**2 - 2 * self.std_ref * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


if __name__ == "__main__":
    """Display a Taylor diagram in a separate axis."""

    # Reference dataset
    x = np.linspace(0, 4 * np.pi, 100)
    data = np.sin(x)
    refstd = data.std(ddof=1)  # Reference standard deviation

    # Generate models
    m1 = data + 0.2 * np.random.randn(len(x))  # Model 1
    m2 = 0.8 * data + .1 * np.random.randn(len(x))  # Model 2
    m3 = np.sin(x - np.pi / 10)  # Model 3

    # Compute stddev and correlation coefficient of models
    samples = np.array([[m.std(ddof=1), np.corrcoef(data, m)[0, 1]] for m in (m1, m2, m3)])

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1, xlabel='X', ylabel='Y')
    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, rect=122, label="Reference", std_range=(0.5, 1.5))

    colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))

    ax1.plot(x, data, 'ko', label='Data')
    for i, m in enumerate([m1, m2, m3]):
        ax1.plot(x, m, c=colors[i], label='Model %d' % (i + 1))
    ax1.legend(numpoints=1, prop=dict(size='small'), loc='best')

    # Add the models to Taylor diagram
    for i, (stddev, corrcoef) in enumerate(samples):
        dia.add_sample(stddev, corrcoef,
                       marker='$%d$' % (i + 1), ms=2, ls='',
                       mfc=colors[i], mec=colors[i],
                       label="Model %d" % (i + 1))

    # Add grid
    dia.add_grid()

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize='medium', fmt='%.2f')

    # Add a figure legend
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size='small'), loc='upper right')

    plt.savefig("taylor_test.png", dpi=600)

