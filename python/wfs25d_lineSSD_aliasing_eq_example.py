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

# own code taken from https://git.iem.at/zotter/wfs-basics
# version SHA 2829319b as of 2020-10-24

import matplotlib.pyplot as plt
import numpy as np
import sfs
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from util import speed_of_sound, wave_quantities
from util import audience_plane, atf, vec_ps2ss, driving_function, synthesize
from util import driving_function_no_prefilter, wfs_pre_filter_shelving_sos
from util import sosfreqs
from util import set_rcparams
# from util import set_figure_height_one_col
# from util import set_figure_width_two_col, set_figure_height_two_col
from util import set_cmap_lin, set_cmap_db, set_cmap_lin_err, set_cmap_db_err
import sys


def set_figure_width_one_col(d_in_mm=174):
    return d_in_mm / 25.4


# language = 'ENG'  # 'DEU' or 'ENG'
# call with python wfs25d_circSSD_aliasing_time_domain.py language='DEU'
language = sys.argv[1]
print(language)

set_rcparams()
cmap_lin, norm_lin = set_cmap_lin()
cmap_db, norm_db = set_cmap_db()
cmap_lin_err, norm_lin_err = set_cmap_lin_err()
cmap_db_err, norm_db_err = set_cmap_db_err()


# width_in = set_figure_width_one_col()
# height_in = set_figure_height_one_col()
# print(width_in, height_in)

# acoustic stuff --------------------------------------------------------------
lmb = 0.5  # 0.5  # wave length in m
c = speed_of_sound()  # speed of sound in m/s
k, f, T, w = wave_quantities(c, lmb)
print('c = %3.1fm/s, lambda = %4.3fm, k = %4.2frad/m' % (c, lmb, k))
print('f = %3.1fHz, omega = %3.1frad/s, T = %es' % (f, w, T))
t = T * (5/lmb)  # time instance for synthesis

# define circular secondary source distribution (SSD) -------------------------
N = 2 ** 6
phi = np.arange(N) * 2 * np.pi / N
R = 2
dx0 = 2 * np.pi / N * R
print('dx0', dx0)
x = np.zeros((1, 3, N))  # dim for numpy broadcasting
x[0, 0, :], x[0, 1, :] = np.cos(phi) * R, np.sin(phi) * R  # put into xy-plane
n = x / np.linalg.norm(x, axis=1)  # outward !!!, unit vector

# define audience plane -------------------------------------------------------
tmp = 9
Nx, Ny = (2 ** tmp, 2 ** tmp)  # number of points wrt x, y
tmp = 2.5
xmin, xmax = (-tmp, +tmp)  # m
ymin, ymax = (-tmp, +tmp)  # m
Xr, Yr, xr = audience_plane(Nx, Ny, xmin, xmax, ymin, ymax)

# get ATF matrix --------------------------------------------------------------
_, ATF = atf(xr, x, k, w, t)

# setup of primary source = virtual point source ------------------------------
x0 = np.zeros((1, 3, 1))  # init for broadcasting
x0[0, :, 0] = np.array([-5, 0, 0])  # set up position in xy-plane outside SSD

# vector primary source to secondary sources
xx0, xx0_length, xx0_unit = vec_ps2ss(x, x0)
print('min kr', np.min(k * xx0_length))

# setup of reference curve ----------------------------------------------------
# xr_ref = x * 0  # alloc
d_tmp = np.zeros(N, dtype='complex')  # alloc
if True:
    print('xref along a circle with radius R from origin x0')
    # when SSD is origin centered
    xr_ref = x + xx0_unit * (
            np.linalg.norm(x0) - np.linalg.norm(x - x0, axis=1))
    # analytic result, TBD
else:
    print('reference line within SSD in left xy-plane')
    # or: xref along a straight line inside SSD
    # this is hard coded for a virtual point source on negative x-axis
    # Firtha, IEEE 2017, 10.1109/TASLP.2017.2689245, eq. (52)
    xr_ref_line = -0.25
    for i in range(N):
        cosbeta = np.dot(n[0, :, i], [-1, 0, 0])
        # using numerical vector projection to find xr
        xr_ref[0, :, i] = x[0, :, i] + \
            (xx0_unit[0, :, i] * -xx0_length[0, i] *
             (xr_ref_line + R * cosbeta) / (
                x0[0, 0, 0] + R * cosbeta))
        # analytic expression:
        # we do this only for sqrt(>0), note however that the max operator
        # yields d_tmp[i]=0 when sqrt(<0), we do this just to suppress numpy
        # warnings
        if (xr_ref_line + R * cosbeta) / (xr_ref_line - x0[0, 0, 0]) > 0:
            d_tmp[i] = np.sqrt(1j * k / 2 / np.pi) * \
                       np.sqrt((xr_ref_line + R * cosbeta) / (
                               xr_ref_line - x0[0, 0, 0])) * \
                       np.maximum((- R ** 2 + x0[0, 0, 0] * x[0, 0, i]) / R,
                                  0) * \
                       np.exp(-1j * k * xx0_length[0, i]) / (
                               xx0_length[0, i] ** (3 / 2))

xxr_length = np.linalg.norm(x - xr_ref, axis=1)
print('min kr', np.min(k * xxr_length))

# render sound pressure -------------------------------------------------------

# driving function
d, max_op = driving_function(k, xx0_unit, n, xx0_length, xxr_length)
# d=0 if ref line outside SSD (works only if SSD centered around in origin)
# also adapt the max operator since we use it to plot the active part of SSD
# del_src = np.squeeze(np.linalg.norm(xr_ref, axis=1) >= R)
# d[del_src] *= 0
# max_op[0, del_src] *= 0

# sound field synthesis
p = synthesize(ATF, d, dx0, Nx, Ny)

# calc reference field of primary source
r = np.linalg.norm(xr - x0, axis=1)
pref = np.exp(-1j * k * r) / (4 * np.pi * r) * np.exp(+1j * w * t)
pref = np.reshape(pref, (Nx, Ny))

# normalize wrt origin such that the following plots
# work with absolute cbar ticks
p *= 4 * np.pi * np.linalg.norm(x0)
pref *= 4 * np.pi * np.linalg.norm(x0)

# Sound Field Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

surf00 = ax[0].pcolormesh(Xr, Yr, 20 * np.log10(np.abs(p)),
                          shading='nearest', cmap=cmap_db, norm=norm_db)
surf01 = ax[1].pcolormesh(Xr, Yr, np.real(p),
                          shading='nearest', cmap=cmap_lin, norm=norm_lin)

# not needed -> extra figure with colorbars only
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.025)
# fig.colorbar(surf00, cax=cax0, ticks=np.arange(-24, 12 + 3, 3))
# cax0.set_xlabel('dB')

# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.025)
# fig.colorbar(surf01, cax=cax1, ticks=np.arange(-2, 2 + 1, 1))
# cax1.set_xlabel(r'$\Re(p)$')

# ax[0].set_title('2.5D WFS with Circ Array: dB')
# ax[1].set_title('2.5D WFS with Circ Array: lin')

for i in range(2):
    ax[i].plot(x[0, 0, np.squeeze(max_op == 0)],  # non active SSD
               x[0, 1, np.squeeze(max_op == 0)],
               'C7o', ms=1, mew=0.75, mfc='C7', mec='C7')
    ax[i].plot(x[0, 0, np.squeeze(max_op > 0)],  # active SSD
               x[0, 1, np.squeeze(max_op > 0)],
               'C7o-', ms=4, mew=1, mfc='k', mec='w')
    ax[i].plot(xr_ref[0, 0, np.squeeze(max_op > 0)],  # reference line
               xr_ref[0, 1, np.squeeze(max_op > 0)],
               'C7o-.', lw=1, ms=3, mew=0.75, mfc='w', mec='k')
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(ymin, ymax)
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    ax[i].set_xticks(np.arange(-2, 3, 1))
    ax[i].set_yticks(np.arange(-2, 3, 1))
    ax[i].set_xlabel(r'$x_\mathrm{r}$ / m')
ax[0].set_ylabel(r'$y_\mathrm{r}$ / m')
ax[1].set_yticklabels([])
# plt.savefig('test1.png')

# SFS Toolbox Stuff -----------------------------------------------------------
# if we use the sfs toolbox https://github.com/sfstoolbox/sfs-python.git
# checked for https://github.com/sfstoolbox/sfs-python/commit/21553ec9a1fbeddc766bd114c2789137123f7c08  # noqa
# we can implement some one liners:
sfs.default.c = c
grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.01953125)
array = sfs.array.circular(N, R)
d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega=w, x0=array.x, n0=array.n, xs=x0[0, :, 0],
    xref=np.squeeze(xr_ref).T, c=c)
p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)
p *= np.exp(1j * w * t)
normalize_gain = 4 * np.pi * np.linalg.norm(x0)

plt.figure(figsize=(5, 10))
plt.subplot(2, 1, 2)
sfs.plot2d.amplitude(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
plt.subplot(2, 1, 1)
sfs.plot2d.level(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
# plt.savefig('test2.png')

# do NOT touch the stuff above, relevant parts of it shall be the same as
# wfs25d_lineSSD_aliasing.py
# since we want to get the frequency response and EQing for this case

# set_rcparams()

fal = c / (2*dx0)
print('fal', fal, 'Hz')

# frequency response-----------------------------------------------------------
Nf = 2**16
fv = np.linspace(0, 50000, Nf)
fv[0] = 1e-16
p_freq_resp = np.zeros(Nf, dtype='complex')

lmbv = c / fv
kv, fv, Tv, wv = wave_quantities(c, lmbv)

xr = np.zeros((1, 3, 1))  # receiver point in origin

# get p for desired frequencies
for cnt, (k, w) in enumerate(zip(kv, wv)):
    d, max_op = driving_function_no_prefilter(k, xx0_unit, n,
                                              xx0_length, xxr_length)
    _, ATF = atf(xr, x, k, w, t=0)
    p_freq_resp[cnt] = np.squeeze(ATF @ d * dx0)
# normalize to 0 dB level at ref curve:
p_freq_resp *= 4*np.pi*np.linalg.norm(x0)

fl = 75  # Hz, shelving wfs pre-filter lower cut-off
fh = fal  # Hz, shelving wfs pre-filter higher cut-off

sos = wfs_pre_filter_shelving_sos(fl=2**1, fh=2**17, c=c, biquad_per_octave=3)
_, H_ideal = sosfreqs(sos, worN=2*np.pi*fv, plot=None)
sos = wfs_pre_filter_shelving_sos(fl=fl, fh=fh, c=c, biquad_per_octave=9)
_, H_shelving = sosfreqs(sos, worN=2*np.pi*fv, plot=None)

p_freq_resp_ideal = p_freq_resp * H_ideal
p_freq_resp_shelving = p_freq_resp * H_shelving

# do smoothing the manual way:
bw_oct = 1/1
p_freq_resp_ideal_smooth = p_freq_resp_ideal * 1
p_freq_resp_shelving_smooth = p_freq_resp_shelving * 1
for cnt, f_tmp in enumerate(fv):
    if fal < f_tmp < 20000:  # smooth only where we need this
        flc = 2**(-bw_oct / 2) * f_tmp
        fhc = 2**(+bw_oct / 2) * f_tmp
        idx = np.logical_and(fv > flc, fv < fhc)
        p_freq_resp_ideal_smooth[cnt] = \
            np.mean(np.abs(p_freq_resp_ideal[idx]))
        p_freq_resp_shelving_smooth[cnt] =\
            np.mean(np.abs(p_freq_resp_shelving[idx]))

# this is not elegant, since we need log f-axis:
print('dB mean(f>fal, f<20000):')
idx = np.logical_and(fv > fal, fv < 20000)
print(np.mean(20 * np.log10(np.abs(p_freq_resp_ideal_smooth[idx]))))
print(np.mean(20 * np.log10(np.abs(p_freq_resp_shelving_smooth[idx]))))

width_in = set_figure_width_one_col()
height_in = 0.245*width_in
print(width_in*2.54)

fig, ax = plt.subplots(figsize=(width_in, height_in), nrows=1, ncols=2)

if language == 'DEU':
    ax[0].semilogx(fv, 20*np.log10(np.abs(p_freq_resp_ideal)), 'C0', lw=1.75,
                   label='Frequenzgang mit WFS-Vorfilter', zorder=2)
    ax[0].semilogx(fv, 20*np.log10(np.abs(p_freq_resp_shelving)), 'C1', lw=1,
                   label='Frequenzgang mit Shelving-EQ', zorder=5)
    ax[0].semilogx([2**-4 * fl, fl], [-3*4, 0],
                   'C4:', label=r'Flanke Fernfeld', zorder=3)
    ax[0].semilogx([fal, 2**5 * fal], [0, 3*4],
                   'C3--', label=r'Flanke Aliasing', zorder=4)
elif language == 'ENG':
    ax[0].semilogx(fv, 20*np.log10(np.abs(p_freq_resp_ideal)), 'C0', lw=1.75,
                   label='frequency response with pre-filter', zorder=2)
    ax[0].semilogx(fv, 20*np.log10(np.abs(p_freq_resp_shelving)), 'C1', lw=1,
                   label='frequency response with shelving EQ', zorder=5)
    ax[0].semilogx([2**-4 * fl, fl], [-3*4, 0],
                   'C4:', label=r'slope farfield', zorder=3)
    ax[0].semilogx([fal, 2**5 * fal], [0, 3*4],
                   'C3--', label=r'slope aliasing', zorder=4)


ax[1].semilogx([2**-4 * fl, fl], [3*-4, 0], 'C4:')
ax[1].semilogx([fl, fh], [0, 0], 'C7:', lw=1)
ax[1].semilogx([fal, 2**+5 * fal], [0, 3*+4], 'C3--')


if language == 'DEU':
    ax[1].semilogx(fv,
                   20*np.log10(np.abs(np.sqrt(1j*2*np.pi*fv/c))),
                   'C0', lw=1.75,
                   label=r'WFS-Vorfilter $\sqrt{\mathrm{j}\omega/c}$, +3\,dB/Okt.')
    ax[1].semilogx(fv, 20*np.log10(np.abs(H_shelving)), color='C1', lw=1,
                   label='Shelving-EQ')
elif language == 'ENG':
    ax[1].semilogx(fv,
                   20*np.log10(np.abs(np.sqrt(1j*2*np.pi*fv/c))),
                   'C0', lw=1.75,
                   label=r'pre-filter $\sqrt{\mathrm{j}\omega/c}$, +3\,dB/oct')
    ax[1].semilogx(fv, 20*np.log10(np.abs(H_shelving)), color='C1', lw=1,
                   label='shelving EQ')


if False:  # plot smoothed freq resp just for debug/check
    ax[0].semilogx(fv, 20 * np.log10(np.abs(p_freq_resp_ideal_smooth)),
                   'navy', lw=1)
    ax[0].semilogx(fv, 20 * np.log10(np.abs(p_freq_resp_shelving_smooth)),
                   'peru', lw=1)

ax[0].set_ylabel(r'$\rightarrow$ dB')
for i in range(2):
    ax[i].set_xlim(10, 20000)
    ax[i].set_ylim(-18, 24)
    ax[i].set_xticks(np.array([10, 20, 50, 100, 200, 500,
                              1000, 2000, 5000, 10000, 20000]))
    ax[i].set_xticklabels(['10', '20', '50', '100', '200', '500',
                           '1k', '2k', '5k', '10k', '20k'])
    ax[i].set_yticks(np.arange(-18, 24+6, 6))
    # ax[i].set_yticklabels(
    ax[i].grid(True, which='both')
    ax[i].set_xlabel(r'$f$ / Hz')

ax[0].legend(loc='upper left')
ax[1].legend(loc='lower right')

# plt.savefig('wfs25d_lineSSD_aliasing_eq_example.pdf')
if language == 'DEU':
    plt.savefig('wfs25d_lineSSD_aliasing_eq_example_DEU.png', dpi=1200)
elif language == 'ENG':
    plt.savefig('wfs25d_lineSSD_aliasing_eq_example_ENG.png', dpi=1200)
