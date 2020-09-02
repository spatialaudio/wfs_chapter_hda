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

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from util import set_rcparams
# from util import set_figure_width_one_col
from util import set_cmap_lin, set_cmap_db

set_rcparams()

width_in = 85 / 25.4  # this leads to too large font
width_in = 95 / 25.4  # trial/error solution :-(

cmap_lin, norm_lin = set_cmap_lin()
cmap_db, norm_db = set_cmap_db()

fig, ax = plt.subplots(figsize=(width_in, 0.35/1.7*width_in), nrows=2, ncols=1)
pad = -0.275

ax[0].plot([0, 1], [0, 0])
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("bottom", size="100%", pad=pad)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm_db, cmap=cmap_db),
             ticks=np.arange(-18, 18 + 6, 6),
             cax=cax0, orientation='horizontal')  # , label='dB')
cax0.set_ylabel(r'$\mathrm{dB}$')
cax0.xaxis.tick_top()
# cax0.tick_params(axis='x', labelrotation=30)
cax0.tick_params(axis='x', length=1)

ax[1].plot([0, 1], [0, 0])
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("bottom", size="100%", pad=pad)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm_lin, cmap=cmap_lin),
             ticks=np.array([-8, -4, 0, 4, 8]),
             cax=cax1, orientation='horizontal')  # , label=r'$\Re(p)$')
cax1.set_ylabel(r'$\Re\{p\}$')
# cax1.tick_params(axis='x', labelrotation=-30)
cax1.tick_params(axis='x', length=1)

for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.savefig('plot_colorbar.png')
