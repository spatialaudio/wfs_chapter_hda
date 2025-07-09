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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from util import speed_of_sound, wave_quantities
from util import audience_plane, atf, synthesize
from util import set_rcparams, set_figure_width_one_col
from util import set_cmap_lin, set_cmap_db, set_cmap_lin_err, set_cmap_db_err


set_rcparams()
cmap_lin, norm_lin = set_cmap_lin()
cmap_db, norm_db = set_cmap_db()
cmap_lin_err, norm_lin_err = set_cmap_lin_err()
cmap_db_err, norm_db_err = set_cmap_db_err()

width_in = set_figure_width_one_col()
height_in = set_figure_width_one_col()
print(width_in, height_in)

# acoustic stuff --------------------------------------------------------------
lmb = 1  # wave length in m
c = speed_of_sound()  # speed of sound in m/s
k, f, T, w = wave_quantities(c, lmb)
print('c = %3.1fm/s, lambda = %3.1fm, k = %4.2frad/m' % (c, lmb, k))
print('f = %3.1fHz, omega = %3.1frad/s, T = %es' % (f, w, T))
t = 0  # time instance for synthesis

# define linear secondary source distribution (SSD) ---------------------------
x = np.zeros((1, 3, 1))  # dim for numpy broadcasting
x[0, :, 0] = np.array([0, 0, 0])
n = np.zeros((1, 3, 1))
n[0, 0, :] = +1  # dipole main lobe, x
n[0, 1, :] = +0  # dipole main lobe, y

# define audience plane -------------------------------------------------------
tmp = 9
Nx, Ny = (2 ** tmp, 2 ** tmp)  # number of points wrt x, y
xmin, xmax = (-2.5, +2.5)  # m
ymin, ymax = (-2.5, +2.5)  # m
Xr, Yr, xr = audience_plane(Nx, Ny, xmin, xmax, ymin, ymax)

# get ATF matrix --------------------------------------------------------------
r, ATF = atf(xr, x, k, w, t)

# get projection onto n vector
xxr = x - xr
xxr_length = np.expand_dims(np.linalg.norm(xxr, axis=1), 2)
xxr_unit = xxr / xxr_length
inner_n_xxr_unit = np.einsum('ijk,ijk->ik', n, xxr_unit)

# monopole
Gm = ATF

# dipole (normalized, i.e. we omit (jw/c+1/r) term), BUT: we must use minus
# Gd = - inner_n_xxr_unit * ATF
# we should not use the above 'farfield' dipol,
# but rather use full term and normalize by (1j*w/c) to get similar
# level between mono- and dipole in the farfield
tmp = np.squeeze(xxr_length, axis=2)
Gd = - inner_n_xxr_unit * (1j*w/c + 1/tmp) * ATF / (w/c)


# sound field
pm = synthesize(Gm, np.array([1]), 1, Nx, Ny)
pm *= 4*np.pi

pd = synthesize(Gd, np.array([1]), 1, Nx, Ny)
pd *= 4*np.pi

# Sound Field Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(width_in, height_in), nrows=2, ncols=2)

surf00 = ax[0, 0].pcolormesh(Xr, Yr, 20 * np.log10(np.abs(pm)),
                             shading='nearest', cmap=cmap_db, norm=norm_db)
surf10 = ax[1, 0].pcolormesh(Xr, Yr, 20 * np.log10(np.abs(pd)),
                             shading='nearest', cmap=cmap_db, norm=norm_db)
surf01 = ax[0, 1].pcolormesh(Xr, Yr, np.real(pm),
                             shading='nearest', cmap=cmap_lin, norm=norm_lin)
surf11 = ax[1, 1].pcolormesh(Xr, Yr, np.real(pd),
                             shading='nearest', cmap=cmap_lin, norm=norm_lin)

divider00 = make_axes_locatable(ax[0, 0])
divider10 = make_axes_locatable(ax[1, 0])
divider01 = make_axes_locatable(ax[0, 1])
divider11 = make_axes_locatable(ax[1, 1])

# not needed -> extra figure with colorbars only
# cax00 = divider00.append_axes("right", size="5%", pad=0.025)
# fig.colorbar(surf00, cax=cax00, ticks=np.arange(-24, 12 + 3, 3))
# cax00.set_xlabel('dB')
# cax11 = divider11.append_axes("right", size="5%", pad=0.025)
# fig.colorbar(surf11, cax=cax11, ticks=np.arange(-2, 2 + 1, 1))
# cax11.set_xlabel(r'$\Re(p)$')

# # ax[0].set_title('2.5D WFS with Linear Array: dB')
# # ax[1].set_title('2.5D WFS with Linear Array: lin')
#
for i in range(2):
    for ii in range(2):
        ax[i, ii].set_xlim(xmin, xmax)
        ax[i, ii].set_ylim(ymin, ymax)
        ax[i, ii].set_aspect('equal')
        ax[i, ii].set_xticks(np.arange(-2, 3, 1))
        ax[i, ii].set_yticks(np.arange(-2, 3, 1))
        if i == 1:
            ax[i, ii].set_xlabel(r'$x$ / m')
        if ii == 0:
            ax[i, ii].set_ylabel(r'$y$ / m')

ax[0, 0].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.3)
ax[1, 0].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.3)
ax[0, 1].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.3)
ax[1, 1].grid(True, color='silver', linestyle=(0, (5, 5)), linewidth=0.3, zorder=3)
ax[0, 1].set_yticklabels([])
ax[1, 1].set_yticklabels([])
ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])

col_vec = 'black'

ax[1, 1].arrow(0, 0, 1, 0, head_width=0.25, head_length=0.25,
               length_includes_head=True,
               fc=col_vec, ec=col_vec, lw=0.5, zorder=2)
ax[1, 1].text(0.15, -0.35, r'$\mathbf{n}_{\mathbf{x}}$', color=col_vec)

ax[1, 1].plot(1.5*1.025, 1.85*1.025, 'o', ms=1.25, color='k')
ax[1, 1].text(1.5*1.075, 1.85*1.075, r'$\mathbf{x}_\mathrm{r}$', color=col_vec)
ax[1, 1].arrow(0, 0, 1.5, 1.85, head_width=0.25, head_length=0.25,
               length_includes_head=True,
               fc=col_vec, ec=col_vec, lw=0.5)
ax[1, 1].text(0.25, 0.2, r'$\mathbf{x}_\mathrm{r}-\mathbf{x}$',
              rotation=np.arctan2(1.85, 1.5)*180/np.pi,
              rotation_mode='default',
              color=col_vec)

ax[1, 1].plot(0, 0, 'o', ms=1.25, color='k')
ax[1, 1].text(-0.15, 0.15, r'$\mathbf{x}$', color=col_vec)

plt.savefig('monopole_dipole.png')
