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

import matplotlib.pyplot as plt
import numpy as np
from util import speed_of_sound, wave_quantities
from util import audience_plane, atf, synthesize
from util import set_rcparams, set_figure_width_one_col
# from util import set_cmap_lin, set_cmap_db, set_cmap_lin_err, set_cmap_db_err
from matplotlib.pyplot import get_cmap
from matplotlib.colors import BoundaryNorm
import sys


# language = 'ENG'  # 'DEU' or 'ENG'
# call with python wfs25d_circSSD_aliasing_time_domain.py language='DEU'
language = sys.argv[1]


set_rcparams()
# cmap_lin, norm_lin = set_cmap_lin()
# cmap_db, norm_db = set_cmap_db()
# cmap_lin_err, norm_lin_err = set_cmap_lin_err()
# cmap_db_err, norm_db_err = set_cmap_db_err()

col_tick = np.linspace(-3, +3, 255, endpoint=True)
cmap_lin = get_cmap('seismic_r')  # PuOr_r
norm_lin = BoundaryNorm(col_tick, cmap_lin.N)

width_in = set_figure_width_one_col()
height_in = width_in/2
print(width_in, height_in)

# acoustic stuff --------------------------------------------------------------
lmb = 0.95  # wave length in m
c = speed_of_sound()  # speed of sound in m/s
k, f, T, w = wave_quantities(c, lmb)
print('c = %3.1fm/s, lambda = %3.1fm, k = %4.2frad/m' % (c, lmb, k))
print('f = %3.1fHz, omega = %3.1frad/s, T = %es' % (f, w, T))
t = T/np.pi  # time instance for synthesis

# define linear secondary source distribution (SSD) ---------------------------
x = np.zeros((1, 3, 1))  # dim for numpy broadcasting
x[0, :, 0] = np.array([0, 0, 0])
n = np.zeros((1, 3, 1))
n[0, 0, :] = +1  # dipole main lobe, x
n[0, 1, :] = +0  # dipole main lobe, y

extent_val = 4

# define audience plane -------------------------------------------------------
tmp = 9
Nx, Ny = (2 ** tmp, 2 ** tmp)  # number of points wrt x, y
xmin, xmax = (-extent_val, extent_val)  # m
ymin, ymax = (-extent_val, +extent_val)  # m
Xr, Yr, xr = audience_plane(Nx, Ny, xmin, xmax, ymin, ymax)

# get ATF matrix --------------------------------------------------------------
r, ATF_primary = atf(xr, x, k, w, t)
_, ATF_synthesis = atf(xr, x, k, w, t)

for cnt, val in enumerate(xr):
    if np.linalg.norm(val) < 1:
        ATF_primary[cnt, 0] = 0

for cnt, val in enumerate(xr):
    if val[1] < 1.5:
        ATF_synthesis[cnt, 0] = 0

# sound field
p_primary = synthesize(ATF_primary, np.array([1]), 1, Nx, Ny)
# *2 for nicer plot, we don't refer to an actual colorbar for this plot
p_primary *= 4*np.pi * 2

p_synthesis = synthesize(ATF_synthesis, np.array([1]), 1, Nx, Ny)
# *2 for nicer plot, we don't refer to an actual colorbar for this plot
p_synthesis *= 4*np.pi * 2


fig, ax = plt.subplots(figsize=(width_in, height_in),
                       nrows=1, ncols=2, frameon=False)

im = plt.imread('violin.png')
extent_violine = -1, 1, -1, 1  # normalized extent for instrument

# primary source with monopole / dipole microphones
extent = -extent_val, extent_val, -extent_val, extent_val
ax[0].imshow(np.real(p_primary), cmap=cmap_lin, norm=norm_lin,
             alpha=1, extent=extent)

ax[0].axvline(x=1.5, ymin=-1, ymax=4, color='C7', lw=0.3)
for i in np.arange(-3.9725, 1, 0.5):
    ax[0].text(1.5-0.14, i, r'$\oplus$', fontsize=6)
    ax[0].text(1.5+0, i+0.25, r'$\ominus$', fontsize=6)
    ax[0].text(1.5-0.27, i+0.25, r'$\oplus$', fontsize=6)
# overlay violine
ax[0].imshow(im, extent=extent_violine)


ax[1].imshow(np.real(p_synthesis), cmap=cmap_lin, norm=norm_lin,
             alpha=1, extent=extent)

ax[1].axvline(x=1.5, ymin=-1, ymax=4, color='C7', lw=0.3)
for i in np.arange(-3.9725, 1, 0.5):
    ax[1].text(1.5-0.14, i+0.25, r'$\oplus$', fontsize=6)
    ax[1].text(1.5+0, i, r'$\oplus$', fontsize=6)
    ax[1].text(1.5-0.27, i, r'$\ominus$', fontsize=6)

ax[0].set_xlim(-1, 4)
ax[0].set_ylim(-4, 1)
ax[1].set_xlim(-1, 4)
ax[1].set_ylim(-4, 1)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])

if language == 'DEU':
    ax[0].text(0.9, -2.8, 'Mikrofon-Wand', fontsize=5, rotation=90)
    ax[1].text(0.9, -2.8, 'Lautsprecher-Wand', fontsize=5, rotation=90)
    ax[1].text(-0.9, -1.6, r'kein Schalldruck', fontsize=5, rotation=60)
    ax[0].text(-0.9, -3.9, r'Aufnahme', fontsize=5, rotation=0)
    ax[1].text(-0.9, -3.9, r'Wiedergabe', fontsize=5, rotation=0)
    # ax[0].set_title('Aufnahme')
    # ax[1].set_title('Wiedergabe')
elif language == 'ENG':
    ax[0].text(0.9, -2.8, 'microphone wall', fontsize=5, rotation=90)
    ax[1].text(0.9, -2.8, 'loudspeaker wall', fontsize=5, rotation=90)
    ax[1].text(-0.9, -1.6, r'no sound', fontsize=5, rotation=60)
    ax[0].text(-0.9, -3.9, r'recording', fontsize=5, rotation=0)
    ax[1].text(-0.9, -3.9, r'playback', fontsize=5, rotation=0)
    # ax[0].set_title('Aufnahme')
    # ax[1].set_title('Wiedergabe')

plt.savefig('violin_wfs_'+language+'.png')
