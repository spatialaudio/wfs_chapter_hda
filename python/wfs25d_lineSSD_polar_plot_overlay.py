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
from util import set_rcparams
# from util import set_figure_width_one_col

set_rcparams()


width_in = 85 / 25.4
height_in = 85 / 25.4 # 40 / 25.4

# plt.figure(figsize=(width_in, height_in))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(width_in, height_in),
                       subplot_kw={'projection': 'polar'})

data = np.load('wfs25d_lineSSD_truncation_FRP_data.npz')
phi = data['phi']
p = data['p']
ax[0].plot(phi, 20*np.log10(np.abs(p)), 'C0', lw=0.75)

data = np.load('wfs25d_lineSSD_aliasing_FRP_data.npz')
phi = data['phi']
p = data['p']
ax[1].plot(phi, 20*np.log10(np.abs(p)), 'C1', lw=0.75)

for i in range(2):
    ax[i].set_rmax(0)
    ax[i].set_rmin(-36)
    ax[i].set_thetagrids(np.arange(-90, 90+15, 15))
    ax[i].set_rgrids([-36, -30, -24, -18, -12, -6, 0],
                     labels=['dB', '', '-24', '', '-12', '', '0'])
    ax[i].set_thetamin(-90)
    ax[i].set_thetamax(+90)
    ax[i].grid(True)


fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)

plt.savefig('wfs25d_lineSSD_polar_plot_overlay.png')
