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
#
#!/bin/sh
# python check_wfs_pre_filter_impulse_response.py  # not used in chapter
python violin_wfs.py 'DEU'  # fig.2
python violin_wfs.py 'ENG'
python plot_colorbar.py  # fig. 3
python monopole_dipole.py  # fig. 4
python wfs25d_lineSSD.py  # fig. 9
python wfs25d_circSSD.py  # fig. 10
python wfs25d_lineSSD_truncation.py  # fig. 11
python wfs25d_lineSSD_aliasing.py  # fig. 13
python wfs25d_circSSD_aliasing.py  # fig. 14
# this will produce high CPU load and run some hours
# better run manually overnight:
# python wfs25d_circSSD_noaliasing_time_domain.py 'DEU'  # fig. 15
# python wfs25d_circSSD_noaliasing_time_domain.py 'ENG'
python wfs25d_circSSD_aliasing_time_domain.py 'DEU'  # fig. 16...18
python wfs25d_circSSD_aliasing_time_domain.py 'ENG'
python wfs25d_lineSSD_aliasing_eq_example.py 'DEU'  # fig. 19
python wfs25d_lineSSD_aliasing_eq_example.py 'ENG'
python lwfs25d_circSSD_time_domain.py 'DEU'  # fig. 20 & 21
python lwfs25d_circSSD_time_domain.py 'ENG'
# make the polar plot the last script to ensure that valid *.npz data is used
# so this must in any case come after calling
# wfs25d_lineSSD_truncation.py and wfs25d_lineSSD_aliasing.py
python wfs25d_lineSSD_polar_plot_overlay.py  # fig. 12
