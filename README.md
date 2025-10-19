# wfs_chapter_hda
- Repository https://github.com/spatialaudio/wfs_chapter_hda
- Authors' versions for the chapters (English, German) on **Wave Field Synthesis** in
  - Stefan Weinzierl (editor): *Handbuch der Audiotechnik*, 2nd GER edition, Springer, 2025
  - Stefan Weinzierl (editor): *Handbook of Audio Technology*, 1st ENG edition, Springer, TBA
  - https://link.springer.com/book/10.1007/978-3-662-60369-7

![WFS_Blockdiagramm](graphics_ENG/WFS_Blockdiagramm.png)
*2.5D WFS signal flow for rendering a virtual source.*

![wfs25d_lineSSD_aliasing_eq_example_ENG](python/wfs25d_lineSSD_aliasing_eq_example_ENG.png)
*2.5D WFS pre-filtering.*

![wfs25d_circSSD_aliasing_time_domain_x0.00_m_py_ENG](python/wfs25d_circSSD_aliasing_time_domain_x0.00_m_py_ENG.png)
*2.5D WFS of a virtual point source: acoustic impulse response (left, top), acoustic transfer function (left, bottom) at probe point (x=0, y=0); and wave front snapshot (right) due to excitation with a low-pass filtered impulse.*

![wfs25d_circSSD_aliasing](python/wfs25d_circSSD_aliasing.png)
*2.5D WFS of a virtual point source. Single-frequency soundfield: level (left), snap shot of instantaneous sound pressure (right). Colors as indicated below.*
![wfs25d_circSSD_aliasing](python/plot_colorbar.png)


- Rendered PDF files available at
  - [GER](latex/Schultz_2023_WFS_Chapter_Weinzierl_HdA2nd_IEEE_DEU_b615499.pdf)
  - [ENG]() TBD
- Licenses
  - text and graphics under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
  - source code under [MIT license](https://opensource.org/licenses/MIT)
  - publisher Springer has copyright to their finally edited and authors' approved English / German chapters and their layouts
  - we use the violine image from https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Violin.svg/2048px-Violin.svg.png
  to create the files `python/violin_wfs_ENG.png` and `python/violin_wfs_DEU.png`
  - we use the photo `fotos/WFS_Array_UniRostockH8_2014.jpg` CC BY 4.0 Matthias Geier & Sascha Spors
  - all other graphics (as pdf, png, eps, svn, ipe) in this repository are CC BY 4.0 Frank Schultz & Nara Hahn
- Reference implementation
  - the reference implementation uses and is double checked against the [sfs](https://github.com/sfstoolbox/sfs-python/releases/tag/0.6.3) (>= 0.6.3) toolbox
  - Python environment install is straightforward with `uv sync`  using the provided `pyproject.toml`
- Recommended additional resources
  - [The complete basics of wave field synthesis in a nutshell](https://git.iem.at/zotter/wfs-basics)
  - [Jupyter notebook on 2.5D WFS referencing scheme examples](https://sfs-python.readthedocs.io/en/latest/examples/wfs-referencing.html)
- Authors
  - Frank Schultz, https://orcid.org/0000-0002-3010-0294, https://github.com/fs446
  - Nara Hahn, https://orcid.org/0000-0003-3564-5864, https://github.com/narahahn
  - Sascha Spors, https://orcid.org/0000-0001-7225-9992, https://github.com/spors
