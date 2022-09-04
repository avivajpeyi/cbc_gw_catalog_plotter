# +
"""
Module to plot "template_bank.png"
"""
# %reload_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from bilby.gw import conversion
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from tqdm import tqdm
from matplotlib import rcParams


ZORDER = dict(
    qline=3,
    regions=0,
    region_text=4,
    upper_shade=1,
    prior=1,
    filter=2,

)

N = 1000


from matplotlib import rcParams


def set_matplotlib_style_settings(major=7, minor=3, linewidth=1.5, grid=False, mirror=True):
    rcParams["font.size"] = 30
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["axes.labelsize"] = 30
    rcParams["axes.titlesize"] = 30
    rcParams["axes.labelpad"] = 10
    rcParams["axes.linewidth"] = linewidth
    rcParams["axes.edgecolor"] = "black"
    rcParams["xtick.labelsize"] = 25
    rcParams["ytick.labelsize"] = 25
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.size"] = major
    rcParams["xtick.minor.size"] = minor
    rcParams["ytick.major.size"] = major
    rcParams["ytick.minor.size"] = minor
    rcParams["xtick.minor.width"] = linewidth
    rcParams["xtick.major.width"] = linewidth
    rcParams["ytick.minor.width"] = linewidth
    rcParams["ytick.major.width"] = linewidth
    if mirror:
        rcParams["xtick.top"] = True
        rcParams["ytick.right"] = True
    rcParams["axes.grid"] = grid
    rcParams["axes.titlepad"] = 8



import numpy as np
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
            self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.
        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).
        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.
        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.
        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i+1], ys[i+1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system
        """
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)
    See also
    --------
    `LineAnnotation`
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return 

def add_shaded_regions(ax):
    xlim, ylim = ax.get_xlim()[1], ax.get_ylim()[1]

    # Make the shaded region for m2 > m1
    verts = [(0, 0), (0, 1000), (1000,1000)]
    poly = Polygon(verts, facecolor='white', alpha=0.6, zorder=ZORDER['upper_shade'])
    ax.add_patch(poly)

    BNS = dict(min=[1,1], max=[3,3])
    BBH = dict(min=[5,5], max=[95,95])
    IMBH = dict(min=[5, 5], max=[1000,1000])
    NSBH = dict(min=[5,1], max=[1000,3])
    BNS_COL = "tab:blue"
    GAP_COL = "white"
    BBH_COL = "tab:red"
    IMBH_COL = "tab:purple"
    NSBH_COL = "tab:orange"

    alpha = 0.5
    # Make the BNS region

    verts = [
        (BNS['min'][0], BNS['min'][1]),
        (BNS['min'][0], BNS['max'][1]),
        (BNS['max'][0], BNS['max'][1]),
        (BNS['max'][0], BNS['min'][1])
    ]
    poly = Polygon(verts, facecolor=BNS_COL, alpha=alpha, zorder=ZORDER['regions'])
    ax.add_patch(poly)
    ax.annotate("BNS", xy=(BNS['min'][0] + 0.05, BNS['max'][1]), ha='left', va='top', xycoords='data',
                c=BNS_COL, zorder=ZORDER['region_text'])

    # Make the mass-gap region
    verts = [
        (BNS['max'][0], BNS['min'][1]),
        (BNS['max'][0], BNS['max'][1]),
        (BNS['min'][0], BNS['max'][1]),
        (BNS['min'][0], NSBH['min'][0]),
        (BNS['max'][0], NSBH['min'][0]),
        (BNS['max'][0], NSBH['max'][0]),
        (NSBH['min'][0], NSBH['max'][0]),
        (NSBH['min'][0], NSBH['min'][0]),
        (NSBH['max'][0], NSBH['min'][0]),
        (NSBH['max'][0], BNS['max'][1]),
        (NSBH['min'][0], NSBH['max'][1]),
        (NSBH['min'][0], NSBH['min'][1]),
    ]
#     poly = Polygon(verts, facecolor=GAP_COL, alpha=alpha, zorder=ZORDER['regions'])
#     ax.add_patch(poly)
#     ax.annotate("MASS GAP", xy=(BNS['max'][0], ylim), ha='left', va='top', xycoords='data',
#                 c=GAP_COL, zorder=ZORDER['region_text'], rotation=90)

    # Make the NSBH region
    verts = [
        (NSBH['min'][0], NSBH['min'][1]),
        (NSBH['min'][0], NSBH['max'][1]),
        (NSBH['max'][0], NSBH['max'][1]),
        (NSBH['max'][0], NSBH['min'][1])
    ]
    poly = Polygon(verts, facecolor=NSBH_COL, alpha=alpha, zorder=ZORDER['regions'])
    ax.add_patch(poly)
    verts = [
        (NSBH['min'][1], NSBH['min'][0]),
        (NSBH['max'][1], NSBH['min'][0]),
        (NSBH['max'][1], NSBH['max'][0]),
        (NSBH['min'][1], NSBH['max'][0])
    ]
    poly = Polygon(verts, facecolor=NSBH_COL, alpha=alpha, zorder=ZORDER['regions'])
    ax.add_patch(poly)
    ax.annotate("NSBH", xy=(1+ 0.05, 5), ha='left', va='bottom', xycoords='data',
                c=NSBH_COL, zorder=ZORDER['region_text'], rotation=0)

    # Make the BBH region
    m1s = np.linspace(BBH['min'][0], BBH['max'][0], 1000)
    m2s = _get_m2_from_m1_for_imbh_boundary(m1s)
    verts = [
        (BBH['max'][0], BBH['min'][1]),
        (BBH['min'][0], BBH['min'][1]),
        (BBH['min'][0], BBH['max'][1]),
        *zip(m1s, m2s),
        # IMBH boundary curve
    ]
    poly = Polygon(verts, facecolor=BBH_COL, alpha=alpha, zorder=ZORDER['regions'])
    ax.add_patch(poly)
    ax.annotate("BBH", xy=(5.25, 75), ha='left', va='top', xycoords='data',
                c=BBH_COL, zorder=ZORDER['region_text'], rotation=0)

    # Make the IMBH region
    m1s = np.linspace(BBH['max'][0], BBH['min'][0], 2000)
    m2s = _get_m2_from_m1_for_imbh_boundary(m1s)
    verts = [
        (IMBH['max'][0], IMBH['min'][1]),
        (BBH['max'][0], IMBH['min'][1]),
        *zip(m1s, m2s),
        (IMBH['min'][0], BBH['max'][1]),
        (IMBH['min'][0], IMBH['max'][1]),
        (IMBH['max'][0], IMBH['max'][1]),
        # IMBH boundary curve
    ]
    poly = Polygon(verts, facecolor=IMBH_COL, alpha=alpha, zorder=ZORDER['regions'])
    ax.add_patch(poly)
    ax.annotate("IMBH", xy=(5.25, BBH['max'][0]), ha='left', va='bottom', xycoords='data',
                c=IMBH_COL, zorder=ZORDER['region_text'], rotation=0)

def _get_m2_from_m1_for_imbh_boundary(m1):
    return 100 - m1

def add_q_lines(ax):
    m1 = np.array([1.0, 10000.0])
    kwargs = dict(c='k', linestyle='--', zorder=ZORDER['qline'], lw=1.5)
    for q, m1_text in zip([1.0, 1.0 / 10.0, 1.0/100.0], [325, 375, 525]):
        m2 = m1 * q
        line,  = ax.plot(m1, m2, **kwargs)
        xytext=(0, 10)
        if q==1.0/10.0:
            xytext=(0, -25)
        line_annotate( f'$q={q}$', line, m1_text, fontsize='medium', ha='right', xytext=xytext, zorder=ZORDER['qline'])






def adjust_axes(fig, ax):
    # removing the default axis on all sides:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # defining custom minor tick locations:
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(plt.FixedLocator([]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([5, 25, 150, 400]))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(plt.FixedLocator([1,3, 5, 10, 25, 45, 100, 400]))
    ax.tick_params(axis='both',reset=False,which='both',length=8,width=2)

    # Add arrowheads
    ax.plot(1, 1, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize= 15)
    ax.plot(1, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize = 15)

    
def scatter_gw_catalog_points():
    data = pd.read_csv("https://www.gw-openscience.org/eventapi/csv/GWTC/")
    data['m1'] = data["mass_1_source"]
    data['m2'] = data["mass_2_source"]
    plt.scatter(data.m1, data.m2, marker="+", color='k', s=90)



def plot_template_bank():
    m1_range = [1, 600]
    m2_range = [1, 600]

    set_matplotlib_style_settings(major=15, minor=8, linewidth=1.5, grid=False, mirror=False)
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams['hatch.linewidth'] = 2.5

    fig, ax_m1m2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))  # 3, 2

    axis_label_kwargs = dict(fontsize="x-large", labelpad=8)

    # set labels
    ax_m1m2.set_xlabel(r"Mass 1 $[M_{\odot}]$", **axis_label_kwargs)
    ax_m1m2.set_ylabel(r"Mass 2 $[M_{\odot}]$", **axis_label_kwargs)

    # set scales
    ax_m1m2.set_yscale("log")
    ax_m1m2.set_xscale("log")

    # set scale limits
    ax_m1m2.set_xlim(m1_range[0], m1_range[1])
    ax_m1m2.set_ylim(m2_range[0], m2_range[1])

    add_shaded_regions(ax_m1m2)
    add_q_lines(ax_m1m2)
    adjust_axes(fig, ax_m1m2)

    scatter_gw_catalog_points()
    plt.tight_layout()
    fname = "gw_catalog.png"
    plt.savefig(fname, dpi=500)

    

plot_template_bank()


