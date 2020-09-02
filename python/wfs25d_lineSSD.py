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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from util import speed_of_sound, wave_quantities
from util import audience_plane, atf, vec_ps2ss, driving_function, synthesize
from util import set_rcparams
from util import set_figure_width_one_col, set_figure_height_one_col
from util import set_cmap_lin, set_cmap_db, set_cmap_lin_err, set_cmap_db_err


set_rcparams()
cmap_lin, norm_lin = set_cmap_lin()
cmap_db, norm_db = set_cmap_db()
cmap_lin_err, norm_lin_err = set_cmap_lin_err()
cmap_db_err, norm_db_err = set_cmap_db_err()

width_in = set_figure_width_one_col()
height_in = set_figure_height_one_col()
print(width_in, height_in)

# acoustic stuff --------------------------------------------------------------
lmb = 1  # wave length in m
c = speed_of_sound()  # speed of sound in m/s
k, f, T, w = wave_quantities(c, lmb)
print('c = %3.1fm/s, lambda = %3.1fm, k = %4.2frad/m' % (c, lmb, k))
print('f = %3.1fHz, omega = %3.1frad/s, T = %es' % (f, w, T))
t = T / 2  # time instance for synthesis

# define linear secondary source distribution (SSD) ---------------------------
N = 2 ** 7
dx0 = lmb / 8
print('N = %d, dx0 = %f, N*dx0 = %f' % (N, dx0, N*dx0))
x = np.zeros((1, 3, N))  # dim for numpy broadcasting
x[0, 1, :] = (np.arange(N) - N // 2) * dx0
n = np.zeros((1, 3, N))
n[0, 0, :] = -1  # outward !!!, unit vector

# define audience plane -------------------------------------------------------
tmp = 9
Nx, Ny = (2 ** tmp, 2 ** tmp)  # number of points wrt x, y
xmin, xmax = (-0.5, +4.5)  # m
ymin, ymax = (-2.5, +2.5)  # m
Xr, Yr, xr = audience_plane(Nx, Ny, xmin, xmax, ymin, ymax)

# get ATF matrix --------------------------------------------------------------
r, ATF = atf(xr, x, k, w, t)

# setup of primary source = virtual point source ------------------------------
x0 = np.zeros((1, 3, 1))  # init for broadcasting
x0[0, :, 0] = np.array([-5, 0, 0])  # set up position in xy-plane outside SSD

# vector primary source to secondary sources
xx0, xx0_length, xx0_unit = vec_ps2ss(x, x0)
print('min kr', np.min(k * xx0_length))

# setup of reference curve ----------------------------------------------------
dref = 4  # m
R = -x0[0, 0, 0] + dref
if True:  # either
    print('xref along a circle with radius R from origin x0')
    # get stationary phase xr(x|x0) numerically:
    # this works precisely for xr in the listening plane
    # for xr in the virtual source plane the driving function is slightly
    # different from the closed analytical exp
    xr_ref = x0 + xx0_unit * R
    xxr_length = np.linalg.norm(x - xr_ref, axis=1)
    # closed analytical expression, see wfs_basics.tex tutorial:
    d_tmp = np.squeeze(-x0[0, 0, 0] * np.sqrt(1j * k / 2 / np.pi) *
                       np.sqrt(xxr_length / R) *
                       np.exp(-1j * k * xx0_length) /
                       xx0_length ** (3 / 2))
else:  # or
    print('parallel to SSD in distance dref')
    # get stationary phase xr(x|x0) numerically (use intercept theorem):
    tmp = -x0[0, 0, 0] / (-x0[0, 0, 0] + dref)
    xr_ref = x + xx0_unit * (1 - tmp) / tmp * xx0_length
    xxr_length = np.linalg.norm(x - xr_ref, axis=1)
    # closed analytical expression, see wfs_basics.tex tutorial:
    d_tmp = np.squeeze(-x0[0, 0, 0] * np.sqrt(1j * k / 2 / np.pi) *
                       np.sqrt(dref / (dref - x0[0, 0, 0])) *
                       np.exp(-1j * k * xx0_length) /
                       xx0_length ** (3 / 2))

# render sound pressure -------------------------------------------------------

# driving function
d, max_op = driving_function(k, xx0_unit, n, xx0_length, xxr_length)
# d=0 if ref line is behind SSD:
del_src = xr_ref[0, 0, :] <= x[0, 0, 0]  # SSD is on y-axis
d[del_src] *= 0
d_tmp[del_src] *= 0
max_op[0, del_src] *= 0
# check if analytic and numeric approaches are identical
print('np.allclose(d, d_tmp)', np.allclose(d, d_tmp))

# sound field synthesis
p = synthesize(ATF, d, dx0, Nx, Ny)

# calc reference field of primary source
r = np.linalg.norm(xr - x0, axis=1)
pref = np.exp(-1j * k * r) / (4 * np.pi * r) * np.exp(+1j * w * t)
pref = np.reshape(pref, (Nx, Ny))

# normalize wrt origin such that the following plots
# work with absolute cbar ticks
p *= 4 * np.pi * R
pref *= 4 * np.pi * R

# Sound Field Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(width_in, height_in), nrows=1, ncols=2)

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

# ax[0].set_title('2.5D WFS with Linear Array: dB')
# ax[1].set_title('2.5D WFS with Linear Array: lin')

for i in range(2):
    # active SSD
    ax[i].plot(x[0, 0, np.squeeze(max_op > 0)],
               x[0, 1, np.squeeze(max_op > 0)],
               color='black', ls='solid', lw=1)
    # reference line
    ax[i].plot(xr_ref[0, 0, np.squeeze(max_op > 0)],
               xr_ref[0, 1, np.squeeze(max_op > 0)],
               color='black', ls='dotted', lw=0.75)
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(ymin, ymax)
    ax[i].set_aspect('equal')
    ax[i].set_xticks(np.arange(0, 5, 1))
    ax[i].set_yticks(np.arange(-2, 3, 1))
    ax[i].set_xlabel(r'$x_\mathrm{r}$ / m')
ax[0].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.3)
ax[1].grid(True, color='grey', linestyle=(0, (5, 5)), linewidth=0.3)
ax[0].set_ylabel(r'$y_\mathrm{r}$ / m')
ax[1].set_yticklabels([])
plt.savefig('wfs25d_lineSSD.png')
# plt.savefig('wfs25d_lineSSD.pdf')

# Error Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(width_in, 2*height_in), nrows=2, ncols=1)

surf00 = ax[0].pcolormesh(Xr, Yr, 20 * np.log10(np.abs(pref - p)),
                          shading='nearest',
                          cmap=cmap_db_err, norm=norm_db_err)
surf01 = ax[1].pcolormesh(Xr, Yr, np.real(p) - np.real(pref),
                          shading='nearest',
                          cmap=cmap_lin_err, norm=norm_lin_err)

divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.025)
fig.colorbar(surf00, cax=cax0, ticks=np.arange(-36, 6 + 3, 3))
cax0.set_xlabel('dB')

divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.025)
fig.colorbar(surf01, cax=cax1, ticks=np.arange(-0.1, 0.1 + 0.05, 0.05))
cax1.set_xlabel(r'$\Re(p)$')

# ax[0].set_title('2.5D WFS with Circ Array: Error dB')
# ax[1].set_title('2.5D WFS with Circ Array: Error lin')

for i in range(2):
    ax[i].plot(x[0, 0, np.squeeze(max_op > 0)],  # active SSD
               x[0, 1, np.squeeze(max_op > 0)], 'k-')
    ax[i].plot(xr_ref[0, 0, np.squeeze(max_op > 0)],  # reference line
               xr_ref[0, 1, np.squeeze(max_op > 0)], 'C7-.')
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(ymin, ymax)
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    ax[i].set_xticks(np.arange(0, 5, 1))
    ax[i].set_yticks(np.arange(-2, 3, 1))
    ax[i].set_xlabel(r'$x_\mathrm{r}$ / m')
    ax[i].set_ylabel(r'$y_\mathrm{r}$ / m')
plt.savefig('wfs25d_lineSSD_error.png')
# plt.savefig('wfs25d_lineSSD_error.pdf')


# SFS Toolbox Stuff -----------------------------------------------------------
# if we use the sfs toolbox https://github.com/sfstoolbox/sfs-python.git
# checked for https://github.com/sfstoolbox/sfs-python/commit/21553ec9a1fbeddc766bd114c2789137123f7c08 # noqa
# we can implement some one liners:
sfs.default.c = c
grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.01953125)
array = sfs.array.linear(N, dx0, orientation=[1, 0, 0])
d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega=w, x0=array.x, n0=array.n, xs=x0[0, :, 0],
    xref=np.squeeze(xr_ref).T, c=c)
p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)
p *= np.exp(1j * w * t)
normalize_gain = 4 * np.pi * R

plt.figure(figsize=(width_in, 2*height_in))
plt.subplot(2, 1, 2)
sfs.plot2d.amplitude(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
plt.subplot(2, 1, 1)
sfs.plot2d.level(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
plt.savefig('wfs25d_lineSSD_sfstoolbox.png')
