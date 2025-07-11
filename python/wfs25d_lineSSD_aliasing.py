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

# own code taken from https://git.iem.at/zotter/wfs-basics
# version SHA 2829319b as of 2020-10-24

import matplotlib.pyplot as plt
import numpy as np
import sfs
# from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# acoustic stuff --------------------------------------------------------------
c = speed_of_sound()  # speed of sound in m/s
lmb = 1/3
k, f, T, w = wave_quantities(c, lmb)
print('c = %3.1fm/s, lambda = %4.3fm, k = %4.2frad/m' % (c, lmb, k))
print('f = %3.1fHz, omega = %3.1frad/s, T = %es' % (f, w, T))
t = T * (104/lmb)  # + T/2  # time instance for synthesis

wc = w/c

# define linear secondary source distribution (SSD) ---------------------------
dx0 = 1/(np.sin(60*np.pi/180)*wc / 2/np.pi)
print('dx0', dx0)
N = int(4/dx0)+1
print('N = %d, dx0 = %f, N*dx0 = %f' % (N, dx0, N*dx0))
x = np.zeros((1, 3, N))  # dim for numpy broadcasting
x[0, 1, :] = (np.arange(N) - N // 2) * dx0
n = np.zeros((1, 3, N))
n[0, 0, :] = -1  # outward !!!, unit vector

L = N * dx0
print('L', L, 'k L', k*L)


ky = np.array([0., 1., 2., 3., 4.]) * 2*np.pi/dx0
print('wc', wc)
print('my_dky', ky)
print('my_dky within visible region', ky[ky < wc])
print('radiation angle in deg', np.arcsin(ky[ky < wc] / wc) * 180/np.pi)

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
x0[0, :, 0] = np.array([-100, 0, 0])  # set up position in xy-plane outside SSD

# vector primary source to secondary sources
xx0, xx0_length, xx0_unit = vec_ps2ss(x, x0)
print('min kr', np.min(k * xx0_length))

# setup of reference curve ----------------------------------------------------
dref = 4  # m
R = -x0[0, 0, 0] + dref
if False:  # either
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
width_in = set_figure_width_one_col()
height_in = set_figure_height_one_col()

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
    # correct SSD length is N * dx0:
    ax[i].plot([x[0, 0, 0], x[0, 0, -1]],
               [x[0, 1, 0]-dx0/2, x[0, 1, -1]+dx0/2], 'dimgray', lw=0.25)
    # active SSD monopoles
    ax[i].plot(x[0, 0, np.squeeze(max_op > 0)],
               x[0, 1, np.squeeze(max_op > 0)],
               'o', color='dimgray', ms=2.5, mew=0.5, mfc='k', mec='w')
    # reference line
    ax[i].plot(xr_ref[0, 0, np.squeeze(max_op > 0)],
               xr_ref[0, 1, np.squeeze(max_op > 0)],
               'o', color='dimgray', ms=2.5, mew=0.5, mfc='w', mec='k')
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(ymin, ymax)
    ax[i].set_aspect('equal')
    ax[i].set_xticks(np.arange(0, 5, 1))
    ax[i].set_yticks(np.arange(-2, 3, 1))
    ax[i].set_xlabel(r'$x_\mathrm{r}$ / m')
ax[0].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.25)
ax[1].grid(True, color='grey', linestyle=(0, (5, 5)), linewidth=0.25)
ax[0].set_ylabel(r'$y_\mathrm{r}$ / m')
ax[1].set_yticklabels([])

plt.savefig('wfs25d_lineSSD_aliasing.png')

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

plt.subplot(1, 2, 1)
sfs.plot2d.amplitude(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
plt.subplot(1, 2, 2)
sfs.plot2d.level(p * normalize_gain, grid)
sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.05)
plt.savefig('wfs25d_lineSSD_aliasing_sfstoolbox.png')

# farfield radiation pattern---------------------------------------------------
phi = np.linspace(-90, 90, 2 ** 11, endpoint=True) * np.pi/180
Nf = np.size(phi)
R = 1e7
xr = np.zeros((Nf, 3, 1))  # init for broadcasting
xr[:, 0, 0] = np.cos(phi) * R
xr[:, 1, 0] = np.sin(phi) * R
r, ATF = atf(xr, x, k, w, t)
p = ATF @ d * dx0
p /= np.max(np.abs(p))

width_in = set_figure_width_one_col()
height_in = set_figure_width_one_col()
plt.figure(figsize=(width_in, height_in))
plt.figure(figsize=(width_in, height_in))
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(phi, 20*np.log10(np.abs(p)))
ax.set_rmax(0)
ax.set_rmin(-36)
ax.set_rgrids([-36, -30, -24, -18, -12, 0])
ax.set_thetagrids(np.arange(-90, 90+15, 15))
ax.set_rlabel_position(-22.5)
ax.set_thetamin(-90)
ax.set_thetamax(+90)
ax.grid(True)
ax.text(-97*np.pi/180, -5, 'dB')
plt.savefig('wfs25d_lineSSD_aliasing_FRP.png')
np.savez('wfs25d_lineSSD_aliasing_FRP_data', phi=phi, p=p)
