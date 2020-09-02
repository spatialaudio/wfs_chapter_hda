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
#
read -p "conda env is mywfs?! [Enter] start..."
#
rm -r wfs_chapter_hda_release_DEU
rm -r wfs_chapter_hda_release_DEU.zip
mkdir wfs_chapter_hda_release_DEU
mkdir wfs_chapter_hda_release_DEU/graphics_DEU
mkdir wfs_chapter_hda_release_DEU/fotos
mkdir wfs_chapter_hda_release_DEU/latex
mkdir wfs_chapter_hda_release_DEU/python
#
cd python
./make-all-figures.sh
cd ..
#
cp -p README.md wfs_chapter_hda_release_DEU/
cp -p macro_DEU.sty wfs_chapter_hda_release_DEU/
cp -p latex/*_DEU.tex wfs_chapter_hda_release_DEU/latex
cp -p latex/*.bib wfs_chapter_hda_release_DEU/latex
#
cp -p graphics_DEU/khi_geometry.pdf wfs_chapter_hda_release_DEU/graphics_DEU
cp -p graphics_DEU/spa_3d.pdf wfs_chapter_hda_release_DEU/graphics_DEU
cp -p graphics_DEU/spa_25d.pdf wfs_chapter_hda_release_DEU/graphics_DEU
cp -p graphics_DEU/WFS_Blockdiagramm.eps wfs_chapter_hda_release_DEU/graphics_DEU
cp -p graphics_DEU/WFS_Blockdiagramm.png wfs_chapter_hda_release_DEU/graphics_DEU
#
cp -p fotos/WFS_Array_UniRostockH8_2014.jpg wfs_chapter_hda_release_DEU/fotos
#
cp -p python/monopole_dipole.png wfs_chapter_hda_release_DEU/python
cp -p python/plot_colorbar.png wfs_chapter_hda_release_DEU/python
cp -p python/violin_wfs_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x-1.00_m_py_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x0.00_m_py_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x1.00_m_py_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD_aliasing.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD_noaliasing_time_domain_x0.00_m_py_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_circSSD.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_lineSSD_aliasing_eq_example_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_lineSSD_aliasing.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_lineSSD_polar_plot_overlay.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_lineSSD_truncation.png wfs_chapter_hda_release_DEU/python
cp -p python/wfs25d_lineSSD.png wfs_chapter_hda_release_DEU/python
cp -p python/lwfs25d_circSSD_time_domain_center_py_DEU.png wfs_chapter_hda_release_DEU/python
cp -p python/lwfs25d_circSSD_time_domain_offcenter_py_DEU.png wfs_chapter_hda_release_DEU/python
#
cd wfs_chapter_hda_release_DEU/latex/
#
rm wfs_manuscript_DEU.aux
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_DEU.tex
bibtex Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_DEU.aux
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_DEU.tex
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_DEU.tex
#
cd ..
cd ..
zip -r wfs_chapter_hda_release_DEU.zip wfs_chapter_hda_release_DEU/
