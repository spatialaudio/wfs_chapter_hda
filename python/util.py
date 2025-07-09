# wfs_chapter_hda
# - git repository https://github.com/spatialaudio/wfs_chapter_hda
# - drafts for the chapters (english, german) on **Wave Field Synthesis** for
# Stefan Weinzierl (ed.): *Handbuch der Audiotechnik*, 2nd ed., Springer, 2025
# https://link.springer.com/book/10.1007/978-3-662-60369-7
# - text and graphics under CC BY 4.0 license https://creativecommons.org/licenses/by/4.0/
# - source code under MIT license https://opensource.org/licenses/MIT
# - Springer has copyright to the final english / german chapters and their layouts
# - we might also find https://git.iem.at/zotter/wfs-basics useful
# - we use violine image from https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Violin.svg/2048px-Violin.svg.png to create picture `python/violin_wfs.png`

# Authors:
# - Frank Schultz, https://orcid.org/0000-0002-3010-0294, https://github.com/fs446
# - Nara Hahn, https://orcid.org/0000-0003-3564-5864, https://github.com/narahahn
# - Sascha Spors, https://orcid.org/0000-0001-7225-9992, https://github.com/spors

import numpy as np

from numpy.lib.scimath import sqrt
from scipy.special import erf

from matplotlib import rcParams
from matplotlib.pyplot import get_cmap
from matplotlib.colors import BoundaryNorm

from util_shelving_cascade import shelving_filter_parameters
from util_shelving_cascade import low_shelving_2nd_cascade
from util_shelving_cascade import check_shelving_filter_validity
from util_shelving_cascade import sosfreqs


def speed_of_sound(c=343):
    return c


def wave_quantities(c, lmb):
    k = 2 * np.pi / lmb  # wave number in rad/m
    f = c / lmb  # temporal frequency in Hz
    tper = 1 / f  # temporal period duration in s
    w = 2 * np.pi * f  # angular temporal frequency in rad/s
    return k, f, tper, w


def audience_plane(nx, ny, xmin, xmax, ymin, ymax):
    xrm, yrm = np.meshgrid(np.linspace(xmin, xmax, nx, endpoint=True),
                           np.linspace(ymin, ymax, ny, endpoint=True),
                           indexing='ij')  # ij indexing is good for us
    xr = np.zeros((nx * ny, 3, 1))  # dim for numpy broadcasting
    xr[:, 0, 0], xr[:, 1, 0] = np.reshape(xrm, (1, -1)), np.reshape(yrm,
                                                                    (1, -1))
    return xrm, yrm, xr


def atf(xr, x, k, w, t):
    # get acoustic transfer function (ATF)
    # xr / x dimension for numpy broadcasting
    r = np.linalg.norm(xr - x, axis=1)  # m, distance between specific x0 and x
    # ATF with ideal point source
    atf_matrix = np.exp(-1j * k * r) / (4 * np.pi * r) * np.exp(+1j * w * t)
    # atf_matrix has proper matrix dim: matrix G * col vector d = col vector p
    return r, atf_matrix


def vec_ps2ss(x, x0):
    xx0 = x - x0
    xx0_length = np.linalg.norm(xx0, axis=1)
    xx0_unit = xx0 / xx0_length
    return xx0, xx0_length, xx0_unit


def driving_function(k, xx0_unit, n, xx0_length, xxr_length):
    g_xx0 = np.exp(-1j * k * xx0_length) / 4 / np.pi / xx0_length
    xx0_n_dot = np.einsum('ijk,ijk->ik', xx0_unit, n)
    max_op = 2 * np.maximum(-xx0_n_dot, 0)
    r_factor = np.sqrt((xxr_length * xx0_length) / (xxr_length + xx0_length))
    f_factor = np.sqrt(2 * np.pi * 1j * k)
    d = np.squeeze(f_factor * r_factor * max_op * g_xx0)
    return d, max_op


def driving_function_no_prefilter(k, xx0_unit, n, xx0_length, xxr_length):
    g_xx0 = np.exp(-1j * k * xx0_length) / 4 / np.pi / xx0_length
    xx0_n_dot = np.einsum('ijk,ijk->ik', xx0_unit, n)
    max_op = 2 * np.maximum(-xx0_n_dot, 0)
    r_factor = np.sqrt((xxr_length * xx0_length) / (xxr_length + xx0_length))
    f_factor = np.sqrt(2 * np.pi)
    d = np.squeeze(f_factor * r_factor * max_op * g_xx0)
    return d, max_op


def synthesize(g, d, dx0, nx, ny):
    return np.reshape(g @ d * dx0, (nx, ny))


# taken from Nara's AES148 shelving filter code
# https://matplotlib.org/stable/tutorials/text/usetex.html
def set_rcparams():
    """Plot helping."""
    rcParams['axes.linewidth'] = 0.5
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.facecolor'] = 'None'
    rcParams['axes.labelcolor'] = 'black'
    #
    rcParams['xtick.color'] = 'black'
    rcParams['xtick.major.size'] = 0
    rcParams['xtick.major.width'] = 1
    rcParams['xtick.minor.size'] = 0
    rcParams['xtick.minor.width'] = 1
    #
    rcParams['ytick.color'] = 'black'
    rcParams['ytick.major.size'] = 0
    rcParams['ytick.major.width'] = 1
    rcParams['ytick.minor.size'] = 0
    rcParams['ytick.minor.width'] = 1
    #
    rcParams['grid.color'] = 'silver'
    rcParams['grid.linestyle'] = '-'
    rcParams['grid.linewidth'] = 0.33
    #
    rcParams['legend.frameon'] = 'False'
    #
    rcParams['font.family'] = 'serif'
    # rcParams['font.sans-serif'] = 'cm'
    rcParams['font.size'] = 8
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'].join([r'\usepackage{amsmath}',
                                         r'\usepackage{gensymb}'])
    rcParams['legend.title_fontsize'] = 8
    #
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05
    rcParams['savefig.facecolor'] = 'white'  # 'lavender'
    rcParams['savefig.dpi'] = 1200


# colormap stuff
def set_cmap_lin():
    # for Re(p) in linear amplitude
    col_tick = np.linspace(-8, +8, 255, endpoint=True)
    cmap_lin = get_cmap('seismic').copy()  # PuOr_r, RdBu_r
    cmap_lin.set_over('C1')  # orange
    cmap_lin.set_under('C9')  # cyan
    norm_lin = BoundaryNorm(col_tick, cmap_lin.N)
    return cmap_lin, norm_lin


def set_cmap_db():
    # for 20log10(|p|) in dB
    col_tick = np.linspace(-18, 18, 13, endpoint=True)
    cmap_db = get_cmap('viridis').copy()
    cmap_db.set_over('C1')  # orange
    cmap_db.set_under('C9')  # cyan
    norm_db = BoundaryNorm(col_tick, cmap_db.N)
    return cmap_db, norm_db


def set_cmap_lin_err():
    # for error Re(p)-Re(pref)
    col_tick = np.linspace(-0.1, +0.1, 255, endpoint=True)
    cmap_lin_err = get_cmap('bwr').copy()
    norm_lin_err = BoundaryNorm(col_tick, cmap_lin_err.N)
    return cmap_lin_err, norm_lin_err


def set_cmap_db_err():
    # for Re(p)-Re(pref) in dB
    col_tick = np.linspace(-36, 6, 15, endpoint=True)
    cmap_db_err = get_cmap('viridis').copy()
    norm_db_err = BoundaryNorm(col_tick, cmap_db_err.N)
    return cmap_db_err, norm_db_err


def set_figure_width(d_in_mm=117):
    return d_in_mm / 25.4


def set_figure_height(d_in_mm=117):
    return (d_in_mm/2) / 25.4


# that width is to be used, height is chosen by us
def set_figure_width_one_col(d_in_mm=85):
    return d_in_mm / 25.4


def set_figure_height_one_col(d_in_mm=85):
    return (d_in_mm/2) / 25.4


def set_figure_width_two_col(d_in_mm=174):
    return d_in_mm / 25.4


def set_figure_height_two_col(d_in_mm=174):
    return (d_in_mm/2) / 25.4


def wfs_pre_filter_impulse_response(fs, Npre=2**14+1, beta_kaiser=9,
                                    fnorm=686, c=343):
    r"""Ideal WFS finite impulse response (FIR) pre-filter.

    Parameters
    ----------
    fs: sampling frequency in Hz.
    Npre: FIR length, should be odd.
    beta_kaiser: beta of kaiser window.
    fnorm: frequency in Hz used to match digital frequency response to analog
    filter H_analog = sqrt(1j*w/c), which is dependent on speed of sound c
    c: speed of sound in m/s

    Returns
    -------
    windowed FIR.

    Notes
    -----
    see Frank Schultz (2016): "Sound field synthesis for line source array
    applications in large-scale sound reinforcement", dissertation,
    University of Rostock, https://doi.org/10.18453/rosdok_id00001765
    equation (2.198), page 66.

    """
    k = np.arange(Npre, dtype='complex') - Npre//2  # Npre odd
    hpre = (sqrt(8 * k) * np.cos(np.pi * k)
            + erf(np.exp(1j * 3/4 * np.pi) * sqrt(np.pi * k))
            - erf(np.exp(1j * 1/4 * np.pi) * sqrt(np.pi * k))) / \
           (4 * sqrt(np.pi) * k ** 1.5)
    hpre[Npre//2] = sqrt(2/9 * np.pi)
    # print(np.max(np.abs(hpre.imag)))
    hpre = hpre.real
    # compensate amplitude from digital domain
    hpre /= sqrt(2*np.pi * fnorm/fs)
    # apply desired amplitude for quasi analog domain
    hpre *= sqrt(2*np.pi * fnorm/c)
    w = np.kaiser(Npre, beta_kaiser)
    return hpre * w


def wfs_pre_filter_shelving_sos(fl, fh, c=343, biquad_per_octave=3):
    r"""WFS pre-filter from analog shelving filter cascade.

    Parameters
    ----------
    fl: lower cut-off frequency in Hz.
    fh: higher cut-off frequency in Hz.
    c: speed of sound in m/s.
    biquad_per_octave: number of cascaded biquads per octave

    Returns
    -------
    sos
    second order sections (one row = 3xb, 3xa coefficients of
    Laplace transfer function)

    Notes
    -----
    we use code in util_shelving_cascade that originally belongs to the project
    Frank Schultz, Nara Hahn, Sascha Spors
    Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
    In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
    http://www.aes.org/e-lib/browse.cfm?elib=20756

    """
    slope = 10*np.log10(2)  # the ideal pre-filter slope
    wl = 2 * np.pi * fl  # lower cut-off frequency in rad/s
    wh = 2 * np.pi * fh  # higher cut-off frequency in rad/s
    BWd = np.log10(wh / wl) / np.log10(2)  # get bandwidth in octaves
    w_ref = sqrt(wl*wh)  # harmonic mean of wl and wh to calc gain

    flag_shelving = check_shelving_filter_validity(biquad_per_octave,
                                                   slope=slope, BWd=BWd)
    print('shelving cascade ok?', flag_shelving)
    num_biquad, Gb, _ = shelving_filter_parameters(biquad_per_octave,
                                                   slope=slope, BWd=BWd)
    sos = low_shelving_2nd_cascade(wh, Gb, num_biquad, biquad_per_octave)
    _, gain_shelving = sosfreqs(sos, worN=w_ref, plot=None)
    gain_shelving = np.abs(gain_shelving)
    gain_ideal = np.abs(sqrt(1j * w_ref/c))
    sos[0][0:3] *= gain_ideal / gain_shelving  # normalize filter such
    # that 3dB/oct slope matches the level of the ideal sqrt(1j w/c) filter
    return sos
