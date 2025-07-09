# wfs_chapter_hda
# - git repository https://github.com/spatialaudio/wfs_chapter_hda
# - drafts for the chapters (english, german) on **Wave Field Synthesis** for
# Stefan Weinzierl (ed.): *Handbuch der Audiotechnik*, 2nd ed., Springer,
# https://link.springer.com/referencework/10.1007/978-3-662-60357-4
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
import sfs
import matplotlib.pyplot as plt
import copy
from matplotlib.pyplot import get_cmap
from scipy.signal import (fftconvolve as conv, freqz, firwin)
from util_local_wfs import (log_frequency, f2w, wfs_25d_pw_sht,
                            RF_Design_Method, soundfield_ir)
from sfs.util import sph2cart, db
from util import set_rcparams, wfs_pre_filter_impulse_response
import sys


def set_figure_width_one_col(d_in_mm=145):
    return d_in_mm / 25.4


def set_figure_height_one_col(d_in_mm=145):
    return d_in_mm / 2.5 / 25.4


def plot_soundfield(ax, grid, p):
    plt.sca(ax)  # set to current axis
    vmin, vmax = -30, 15
    cmap = copy.copy(get_cmap('Blues'))
    cmap.set_over('black')
    cmap.set_under('whitesmoke')

    im = sfs.plot2d.level(
        p, grid, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar_kwargs={'ticks': np.arange(-30, 15+15, 15),
                         'label': r'$\rightarrow$ dB$_\mathrm{rel\,30k}$'},
        xlabel=r'$x_\mathrm{r}$ / m',
        ylabel=r'$y_\mathrm{r}$ / m')
    return im


def plot_loudspeakers(ax, ssd):
    plt.sca(ax)  # set to current axis
    ls = sfs.plot2d.loudspeakers(ssd.x, ssd.n, ssd.a, size=0.3, show_numbers=True)
    return ls


def plot_evaluation_positions(ax, xc, evaluation_positions):
    kw_on = dict(marker='x', color='C0', ms=8)
    kw_off = dict(marker='.', mec='C3', ms=2, mfc='none')
    for x_off in evaluation_positions[1:]:
        x_eval = xc + x_off
        ax.plot(*x_eval[:2], **kw_off)
    ax.plot(*xc[:2], **kw_on)
    return 0


def plot_ir(ax, t, h):
    lines = ax.plot(t[0]*1000, db(h[0]), lw=1)
    ax.grid(True)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-30, 15)
    ax.set_xticks(np.arange(-5, 5 + 1, 1))
    ax.set_yticks(np.arange(-30, +15+3, 3))
    ax.set_xlabel(r'$t$ / ms')
    ax.set_ylabel(r'$\rightarrow$ dB$_\mathrm{rel\,30k}$')
    ax.set_yticklabels(['-30', '', '', '', '', '-15', '', '', '', '', '0',
                         '', '', '', '', '15'])
    return lines


def plot_spectrum(ax, f, H):
    kw_off = dict(color='C3', lw=0.2)

    if language == 'DEU':
        ax.plot(f, f*0, 'C7-', lw=0.75, label='ideale ebene Welle')
    elif language == 'ENG':
        ax.plot(f, f*0, 'C7-', lw=0.75, label='ideal plane wave')

    for H_i in H[1:]:
        ax.plot(f, db(H_i), **kw_off)

    if language == 'DEU':
        lines = ax.plot(f, db(H[0]), 'C0', lw=1, label='2.5D lokale WFS')
    elif language == 'ENG':
        lines = ax.plot(f, db(H[0]), 'C0', lw=1, label='2.5D local WFS')

    ax.plot(f, db(np.sqrt(np.mean(np.abs(H)**2, axis=0))), 'k', lw=0.75)

    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlim(10, 20000)
    ax.set_ylim(-18, 24)
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(['10', '20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
    ax.set_yticks(np.arange(-18, 24 + 6, 6))
    ax.set_xlabel(r'$f$ / Hz')
    ax.set_ylabel(r'$\rightarrow$ dB$_\mathrm{rel\,1}$')
    plt.legend(loc='upper left')
    return lines


def plot_all(results):
    xc = res['xc']
    p = res['p']
    h = res['h']
    H = res['H']
    time = res['time']
    scaling = res['scaling']

    kw_figure = dict(
        constrained_layout=True,
        figsize=(set_figure_width_one_col(), set_figure_height_one_col()))

    fig = plt.figure(**kw_figure)
    gs = fig.add_gridspec(2, 4)

    # synthesized sound field
    ax1 = fig.add_subplot(gs[:, 2:])
    plot_soundfield(ax1, grid, scaling*p)
    plot_loudspeakers(ax1, ssd)
    plot_evaluation_positions(ax1, xc, evaluation_positions)
    # ax1.axis([-0.2, 0.2, -0.2, 0.2])

    # impulse response
    ax2 = fig.add_subplot(gs[0, 0:2])
    plot_ir(ax2, time, scaling*h)

    # frequency responses
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_spectrum(ax3, f, H)

    return fig


def save_plots(fig, refpointname, language='DEU'):
    kw_savefig = dict(bbox_inches='tight', dpi=300)
    figname = 'lwfs25d_circSSD_time_domain_{}_py_{}.png'.format(refpointname, language)
    plt.savefig(figname, **kw_savefig)


# language = 'DEU'  # 'DEU' or 'ENG'
# call with python wfs25d_circSSD_aliasing_time_domain.py language='DEU'
language = sys.argv[1]  # <= This doesn't work in Nara's laptop.

set_rcparams()  # looks nice already

c = 343.
fs = 48000 * 4
Ts = 1/fs
N_os = 1

# Frequencies
fmin, fmax, num_f = 1, 24000, 500
f = log_frequency(fmin, fmax, num_f)
omega = f2w(f)

# Cartesian grid
# dx = 0.001
dx = 0.01
grid = sfs.util.xyz_grid([-2.5, 2.5], [-2.5, 2.5], 0, spacing=dx)

# Secondary sources
num_ssd = 32
R = 2
ssd = sfs.array.circular(num_ssd, R)
x0, n0, a0 = ssd.x, ssd.n, ssd.a
phi0, _, _ = sfs.util.cart2sph(*x0.T)
phi0_deg = np.mod(np.rad2deg(phi0), 360)
secondary_source = sfs.td.secondary_source_point(c)
ssd_weights = np.ones(num_ssd)

# Virtual plane wave
phipw = 0
npw = sfs.util.direction_vector(alpha=phipw)

Npre = 2**14 + 1  # length FIR
fnorm = 686
beta_kaiser = 9
hpre = wfs_pre_filter_impulse_response(fs, Npre=Npre, beta_kaiser=beta_kaiser,
                                       fnorm=fnorm, c=sfs.default.c)
_, Hpre = freqz(hpre, 1, f, fs=fs)
delay_preeq = 2**13/fs
Nbp = 601  # length FIR
cutoff = 15000  # Hz
beta = 0.1102 * (100-8.7)
hbp = firwin(numtaps=Nbp,
             cutoff=cutoff,
             window=('kaiser', beta),
             pass_zero='lowpass',
             scale=True,
             fs=fs)

hpre = np.convolve(hpre, hbp, mode='full')
delay_preeq += 300/fs

scaling = fs / (2*cutoff)


# Target positions
lwfs_params = [
    dict(name='offcenter', xc=np.array([0, 0.75, 0]), max_order=15, max_degree=30, beta=6,
         poly_name='lagrange', poly_order=5, deriv_order=5, sss_order=20),
    dict(name='center', xc=np.array([0, 0, 0]), max_order=15, max_degree=30, beta=6,
         poly_name='lagrange', poly_order=5, deriv_order=5, sss_order=20),
    ]


# evaluation positions relative to the target position
r = 0.3
beta = np.pi/2
evaluation_positions = np.stack([[*sph2cart(alpha, beta, r)] for
                         alpha in np.linspace(0, 2*np.pi, num=8, endpoint=False)])
evaluation_positions = np.insert(evaluation_positions, 0, [0, 0, 0], axis=0)

lwfs_results = []
for par in lwfs_params:
    method_bl = RF_Design_Method(
        dict(name='bandlimit',
             poly_name=par['poly_name'],
             max_degree=par['max_degree'],
             poly_order=par['poly_order'],
             deriv_order=par['deriv_order'],
             beta=par['beta'], fc=fs/2))
    xc = par['xc']
    max_order = par['max_order']
    t0 = np.dot(npw, xc) / c
    sss_order = par['sss_order']
    d_raw = wfs_25d_pw_sht(x0, n0, npw, xc, max_order, c, fs, method_bl, sss_order)
    d_eq = (np.stack([conv(hpre, di) for di in d_raw[0].T], axis=-1),
            d_raw[1], d_raw[2] - delay_preeq)

    # synthesized sound field
    p = sfs.td.synthesize(d_eq, ssd_weights, ssd, secondary_source,
                           grid=grid, observation_time=t0)
    # impulse responses
    time = np.zeros((len(evaluation_positions), 2*fs))
    h = np.zeros((len(evaluation_positions), 2*fs))

    # frequency responses
    H = np.zeros((len(evaluation_positions), num_f), dtype=complex)
    for i, offset in enumerate(evaluation_positions):
        x_eval = xc + offset
        t, ir = soundfield_ir(ssd, x_eval, d_eq, t0, c=None, fs=fs)
        w, tf = freqz(ir, 1, f, fs=fs)
        time[i] = t
        h[i] = ir
        H[i] = tf

    res = dict(name=par['name'], xc=xc, d=d_eq, p=p, time=time, h=h, H=H, scaling=scaling)
    lwfs_results.append(res)


for res in lwfs_results:
    fig = plot_all(res)
    save_plots(fig, res['name'], language=language)
