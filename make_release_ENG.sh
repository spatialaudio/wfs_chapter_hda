# wfs_chapter_hda
# git repository https://github.com/spatialaudio/wfs_chapter_hda
# source code under MIT license https://opensource.org/licenses/MIT
#
# Authors:
# Frank Schultz
# https://orcid.org/0000-0002-3010-0294, https://github.com/fs446
# Nara Hahn
# https://orcid.org/0000-0003-3564-5864, https://github.com/narahahn
# Sascha Spors
# https://orcid.org/0000-0001-7225-9992, https://github.com/spors
#
# 'uv' is recommended for Python, packaging and environment handling
# see # https://docs.astral.sh/uv/
# get environment by `uv sync` called in root folder of this repository
# the `pyproject.toml` contains all dependencies
# activate the environment by `source .venv/bin/activate`
#
#!/bin/sh
#
read -p "env wfs-chapter-hda is active?! [Enter] start..."
#
rm -r wfs_chapter_hda_release_ENG
rm -r wfs_chapter_hda_release_ENG.zip
mkdir wfs_chapter_hda_release_ENG
mkdir wfs_chapter_hda_release_ENG/graphics_ENG
mkdir wfs_chapter_hda_release_ENG/fotos
mkdir wfs_chapter_hda_release_ENG/latex
mkdir wfs_chapter_hda_release_ENG/python
#
cd python
./make-all-figures.sh
cd ..
#
cp -p README.md wfs_chapter_hda_release_ENG/
cp -p uv.lock wfs_chapter_hda_release_ENG/
cp -p pyproject.toml wfs_chapter_hda_release_ENG/
cp -p .venv/pyvenv.cfg wfs_chapter_hda_release_ENG/
cp -p macro_ENG.sty wfs_chapter_hda_release_ENG/
cp -p latex/*_ENG.tex wfs_chapter_hda_release_ENG/latex
cp -p latex/*.bib wfs_chapter_hda_release_ENG/latex
#
cp -p graphics_ENG/khi_geometry.pdf wfs_chapter_hda_release_ENG/graphics_ENG
cp -p graphics_ENG/spa_3d.pdf wfs_chapter_hda_release_ENG/graphics_ENG
cp -p graphics_ENG/spa_25d.pdf wfs_chapter_hda_release_ENG/graphics_ENG
cp -p graphics_ENG/WFS_Blockdiagramm.png wfs_chapter_hda_release_ENG/graphics_ENG
#
cp -p fotos/WFS_Array_UniRostockH8_2014.jpg wfs_chapter_hda_release_ENG/fotos
#
cp -p python/monopole_dipole.png wfs_chapter_hda_release_ENG/python
cp -p python/plot_colorbar.png wfs_chapter_hda_release_ENG/python
cp -p python/violin_wfs_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x-1.00_m_py_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x0.00_m_py_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD_aliasing_time_domain_x1.00_m_py_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD_aliasing.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD_noaliasing_time_domain_x0.00_m_py_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_circSSD.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_lineSSD_aliasing_eq_example_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_lineSSD_aliasing.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_lineSSD_polar_plot_overlay.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_lineSSD_truncation.png wfs_chapter_hda_release_ENG/python
cp -p python/wfs25d_lineSSD.png wfs_chapter_hda_release_ENG/python
cp -p python/lwfs25d_circSSD_time_domain_center_py_ENG.png wfs_chapter_hda_release_ENG/python
cp -p python/lwfs25d_circSSD_time_domain_offcenter_py_ENG.png wfs_chapter_hda_release_ENG/python
#
cd wfs_chapter_hda_release_ENG/latex/
#
rm wfs_manuscript_ENG.aux
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
bibtex Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
makeindex Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
pdflatex -shell-escape Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_ENG
#
cd ..
cd ..
zip -r wfs_chapter_hda_release_ENG.zip wfs_chapter_hda_release_ENG/
