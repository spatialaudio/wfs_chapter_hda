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

import matplotlib.pyplot as plt
import numpy as np
import sfs  # implemented with version 0.6.2 for correct td.wfs.point25d()
from matplotlib.pyplot import get_cmap
from scipy.signal import firwin
from util import vec_ps2ss, set_rcparams, wfs_pre_filter_impulse_response
from util import speed_of_sound
import sys


def set_figure_width_one_col(d_in_mm=145):
    return d_in_mm / 25.4


def set_figure_height_one_col(d_in_mm=145):
    return d_in_mm/2.5 / 25.4


def plot(drive, sel, sec_src, ti=0):
    p_field = sfs.td.synthesize(drive, sel, array, sec_src, grid=grid,
                                observation_time=ti)

    cmap = get_cmap('Blues').copy()
    # this over clipping actually does not occur in the chosen examples
    cmap.set_over('black')
    # under clip gray to distinguish from Blues' white
    cmap.set_under('whitesmoke')

    sfs.plot2d.level((p_field / Ts) / (2*cutoff), grid,
                     cmap=cmap, vmin=-30, vmax=+15,
                     colorbar_kwargs={'ticks':
                                      np.arange(-30, +15 + 15, 15),
                                      'label':
                                      r'$\rightarrow$ dB$_\mathrm{rel\,30k}$'},
                     xlabel=r'$x_\mathrm{r}$ / m',
                     ylabel=r'$y_\mathrm{r}$ / m')
    plt.plot(array.x[:, 0], array.x[:, 1], 'C7')
    plt.plot(array.x[selection, 0],
             array.x[selection, 1], 'k')


# language = 'ENG'  # 'DEU' or 'ENG'
# call with python wfs25d_circSSD_aliasing_time_domain.py language='DEU'
language = sys.argv[1]


set_rcparams()
sfs.default.c = speed_of_sound()
fs = 192000  # Hz
Ts = 1 / fs  # s

# point source
xs = -5, 0, 0
rs = np.linalg.norm(xs)  # distance from origin
ts = rs / sfs.default.c  # time-of-arrival at origin

# circular loudspeaker array
Nssd = 2**14  # number of loudspeakers
R = 2  # radius
array = sfs.array.circular(Nssd, R)

# sfs prediction plane
grid = sfs.util.xyz_grid([-2.5, 2.5], [-2.5, 2.5], 0, spacing=0.001)

###############################################################################
# same example as the frequency domain,
# so make that same parameters are used
# see wfs25d_circSSD_aliasing.py
# we basically calc xr_ref for later use here:
phi = np.arange(Nssd)/Nssd * 2*np.pi
dx0 = 2*np.pi/Nssd * R
x = np.zeros((1, 3, Nssd))  # dim for numpy broadcasting
x[0, 0, :], x[0, 1, :] = np.cos(phi) * R, np.sin(phi) * R  # put into xy-plane
n = x / np.linalg.norm(x, axis=1)  # outward !!!, unit vector
x0 = np.zeros((1, 3, 1))  # init for broadcasting
x0[0, :, 0] = np.array([-5, 0, 0])  # set up position in xy-plane outside SSD
# vector primary source to secondary sources
xx0, xx0_length, xx0_unit = vec_ps2ss(x, x0)
xr_ref = x + xx0_unit * (np.linalg.norm(x0) - np.linalg.norm(x - x0, axis=1))

plt.plot(x0[0, 0, :], x0[0, 1, :], 'C0x', label='virtuelle Punktquelle')
plt.plot(x[0, 0, :], x[0, 1, :], 'C7o-', label='WFS Array')
plt.plot(xr_ref[0, 0, :], xr_ref[0, 1, :], 'C7o-', label='Referenzkurve')
# region to check amplitude correct frequency response:
Nd = 5
plt.plot(xr_ref[0, 0, Nssd//2-Nd:Nssd//2+Nd+1],
         xr_ref[0, 1, Nssd//2-Nd:Nssd//2+Nd+1],
         'C3o-', ms=3, label='PCS')
plt.axis('equal')
plt.legend()
plt.grid(True)
# plt.savefig('td_soundfield_check_grid.png', dpi=1200)

###############################################################################
# ideal prefilter, cf. fig. 2.5 in dissertation Frank Schultz
Npre = 2**14 + 1  # length FIR
fnorm = 686
beta_kaiser = 9
hpre = wfs_pre_filter_impulse_response(fs, Npre=Npre, beta_kaiser=beta_kaiser,
                                       fnorm=fnorm, c=sfs.default.c)

# lowpass -> serves as bandlimited dirac impulse
# parameters manually tuned such that
# 20 kHz is pass thru, > 22050 Hz has > 100 dB stop
# 2021-06-19: we use more convenient numbers Nbp and cutoff
# 2 * 20000 = 40000 is the max peak of the analog lowpass filter sinc IR
# that's why we use the 4e4 stepsize in yticks below for IR plot
Nbp = 601  # length FIR
cutoff = 15000  # Hz
beta = 0.1102 * (100-8.7)
hbp = firwin(numtaps=Nbp,
             cutoff=cutoff,
             window=('kaiser', beta),
             pass_zero='lowpass',
             scale=True,
             fs=fs)

h = np.convolve(hpre, hbp, mode='full')
print(h.shape)

###############################################################################
toa_origin = ts  # time-of-arrival at origin
toa_pre = (Npre-1) / 2 / fs  # delay of prefilter
toa_bp = (Nbp-1) / 2 / fs  # delay of lowpass filter
toa_origin_pre_bp = toa_origin + toa_pre + toa_bp

x_range = 6  # m
ts_min = toa_origin_pre_bp - x_range / sfs.default.c
ts_max = toa_origin_pre_bp + x_range / sfs.default.c
t = np.linspace(ts_min, ts_max,
                np.int32(np.ceil((ts_max-ts_min) * fs)))

for x_mic in (-1, 0, +1):
    # if x_mic == 0:  # debug
    #    break
    print('x_mic pos', x_mic)

    # open and setup figure
    fig = plt.figure(constrained_layout=True,
                     figsize=(set_figure_width_one_col(),
                              set_figure_height_one_col()))
    gs = fig.add_gridspec(2, 4)

    # get amplitude decay of ideal point source w.r.t. origin
    dBoffs = 20 * np.log10(
        np.linalg.norm(np.array(xs) - np.array([0, 0, 0])) /
        np.linalg.norm(np.array(xs) - np.array([x_mic, 0, 0])))
    print('virtual point src dB decay', dBoffs)

    # the x_mic offset w.r.t. time works since we've chosen points on x-axis:
    ts = toa_origin_pre_bp + x_mic / sfs.default.c

    # driving function
    # proper referencing scheme, such as in freq domain, needs sfs > 0.6.1
    delays, weights, selection, secondary_source = \
        sfs.td.wfs.point_25d(array.x, array.n, xs,
                             xref=np.squeeze(xr_ref).T,
                             c=sfs.default.c)
    weights *= 4 * np.pi * np.linalg.norm(x0)  # normalize 4 pi r
    d = sfs.td.wfs.driving_signals(delays, weights, (h, fs))

    p = np.zeros(np.size(t))
    Np = np.size(p)
    for cnt, val in enumerate(t):
        p[cnt] = sfs.td.synthesize(d, selection, array, secondary_source,
                                   grid=sfs.util.xyz_grid(
                                       x_mic, 0, 0, spacing=False),
                                   observation_time=val)
    print('max dB', 20 * np.log10(np.max(np.abs(p))))
    # zero padding as DFT to DTFT interpolator
    pz = np.append(p, np.zeros(2*Np))
    Npz = np.size(pz)
    fz = np.arange(Npz)/Npz * fs  # frequency vector, Hz
    Pz_level = 20 * np.log10(np.abs(np.fft.fft(pz)))

    # right subfigure = sound field
    f_ax1 = fig.add_subplot(gs[:, 2:])
    plot(d, selection, secondary_source, ti=ts)
    plt.plot(xr_ref[0, 0, Nssd//2-Nd:Nssd//2+Nd+1],
             xr_ref[0, 1, Nssd//2-Nd:Nssd//2+Nd+1],
             'C7-', lw=0.5)
    plt.plot(x_mic, 0, 'x', color='black', ms=8)

    # top left subfigure = impulse response
    f_ax2 = fig.add_subplot(gs[0, 0:2])
    # we should plot over travel time from primary source to receiver probe
    # thus we need to compensate the delays of both FIR filters
    # p / Ts -> we plot the analog domain impulse response
    # rather than digital, to be independent from fs
    # we furthermore plot amplitude normalized to 2 * cutoff in dB
    tmp = cutoff * 2
    plt.plot((t - toa_pre - toa_bp) * 1000,
             20*np.log10(np.abs(p / Ts / tmp)), lw=1)  # time axis in ms
    print(np.max(20*np.log10(np.abs(p / Ts / tmp))))
    plt.xlabel(r'$t$ / ms')
    plt.ylabel(r'$\rightarrow$ dB$_\mathrm{rel\,30k}$')
    plt.xlim(11, 21)
    plt.xticks(np.arange(11, 21 + 1, 1))
    plt.ylim(-30, +15)
    plt.yticks(np.arange(-30, +15+3, 3),
               labels=['-30', '', '', '', '', '-15', '', '', '', '', '0',
                       '', '', '', '', '15'])
    plt.grid(True)

    # bottom left = frequency response
    f_ax3 = fig.add_subplot(gs[1, 0:2])
    # plot amplitude decay of ideal point source as ideal frequency response
    if language == 'DEU':
        plt.semilogx([10, 20000], [dBoffs, dBoffs],
                     'C7-', lw=0.75,
                     label='ideale Punktquelle')
    elif language == 'ENG':
        plt.semilogx([10, 20000], [dBoffs, dBoffs],
                     'C7-', lw=0.75,
                     label='ideal point source')
    # plot frequency response measured with probe
    plt.semilogx(fz, Pz_level, label='2.5D WFS', lw=1)
    plt.xlim(10, 20000)
    plt.ylim(-18, +24)
    plt.xticks(np.array([10, 20, 50, 100, 200, 500,
                         1000, 2000, 5000, 10000, 20000]),
               labels=['10', '20', '50', '100', '200', '500',
                       '1k', '2k', '5k', '10k', '20k'])
    plt.yticks(np.arange(-18, 24 + 6, 6))
    plt.xlabel(r'$f$ / Hz')
    plt.ylabel(r'$\rightarrow$ dB$_\mathrm{rel\,1}$')
    plt.legend(loc='upper left')
    plt.grid(True)

    # render plot
    if language == 'DEU':
        plt.savefig('wfs25d_circSSD_noaliasing_time_domain_x%2.2f_m_py_DEU.png'
                    % x_mic, dpi=1200)
    if language == 'ENG':
        plt.savefig('wfs25d_circSSD_noaliasing_time_domain_x%2.2f_m_py_ENG.png'
                    % x_mic, dpi=1200)
