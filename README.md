# wfs_chapter_hda
- git repository https://github.com/spatialaudio/wfs_chapter_hda
- drafts for the chapters (english, german) on **Wave Field Synthesis** for
Stefan Weinzierl (ed.): *Handbuch der Audiotechnik*, 2nd ed., Springer, 2025
https://link.springer.com/book/10.1007/978-3-662-60369-7
- text and graphics under CC BY 4.0 license https://creativecommons.org/licenses/by/4.0/
- source code under MIT license https://opensource.org/licenses/MIT
- Springer has copyright to the final english / german chapters and their layouts
- we might also find https://git.iem.at/zotter/wfs-basics useful
- we use violine image from https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Violin.svg/2048px-Violin.svg.png to create picture `python/violin_wfs.png`
- we use the photo WFS_Array_UniRostockH8_2014.jpg CC BY 4.0 Matthias Geier & Sascha Spors
- all other graphics (as pdf, png, eps, ipe) in this repository are CC BY 4.0 Frank Schultz & Nara Hahn
- the latest release 0.6.2 of the [sfs-python toolbox](https://github.com/sfstoolbox/sfs-python) is somewhat outdated, unfortunately
- however the [commit #c060b93](https://github.com/sfstoolbox/sfs-python/commit/c060b93e9c38bd5e9819a2b7e73fbcccd8419d96) in master branch from 2025-07-08 was tested with very recent python, numpy, scipy, matplotlib, thus
```
git clone https://github.com/sfstoolbox/sfs-python.git
cd sfs-python
python3 -m pip install --user -e .
```
is currently recommended to work with the code, more precisely, the parts of the code that use the sfs-toolbox

Authors:
- Frank Schultz, https://orcid.org/0000-0002-3010-0294, https://github.com/fs446
- Nara Hahn, https://orcid.org/0000-0003-3564-5864, https://github.com/narahahn
- Sascha Spors, https://orcid.org/0000-0001-7225-9992, https://github.com/spors
