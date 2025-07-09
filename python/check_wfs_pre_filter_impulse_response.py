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

from scipy.signal import freqz

from util import set_rcparams
from util import speed_of_sound
from util import wfs_pre_filter_impulse_response, wfs_pre_filter_shelving_sos
from util_shelving_cascade import sosfreqs

set_rcparams()

c = speed_of_sound()
fs = 48000
f_ideal = np.arange(fs/2)
k = 2*np.pi*f_ideal/c

H_ideal = np.sqrt(1j*k)  # ideal WFS pre-filter

# ideal WFS pre-filter -> FIR filter from analytic iDTFT adapted to filter in
# analog domain
N_pre = 2**16 + 1
h_fir = wfs_pre_filter_impulse_response(fs, Npre=N_pre, beta_kaiser=9,
                                        fnorm=1000, c=c)
# check FFT spectrum and DTFT (freqz)
H_fft = np.fft.fft(h_fir)
f_fft = np.arange(N_pre)/N_pre * fs
f_freqz, H_freqz = freqz(b=h_fir, a=1, worN=N_pre,
                         whole=True, fs=fs)

# shelving filter cascade
fl = 2**1
fh = 2**17
sos = wfs_pre_filter_shelving_sos(fl=fl, fh=fh, c=c, biquad_per_octave=3)
_, H_shelving = sosfreqs(sos, worN=2*np.pi*f_ideal, plot=None)


fl = 75
fh = 873.4423276883216
sos = wfs_pre_filter_shelving_sos(fl=fl, fh=fh, c=c, biquad_per_octave=9)
_, H_shelving = sosfreqs(sos, worN=2*np.pi*f_ideal, plot=None)


# plot
plt.figure()
plt.semilogx(f_ideal, 20*np.log10(np.abs(H_ideal)), 'C0',
             lw=3, label='analog')
plt.semilogx(f_fft, 20*np.log10(np.abs(H_fft)), 'C1:',
             lw=2, label='FIR FFT')
plt.semilogx(f_freqz, 20*np.log10(np.abs(H_freqz)), 'C3--',
             lw=1, label='FIR freqz')
plt.semilogx(f_ideal, 20*np.log10(np.abs(H_shelving)), 'C2',
             lw=1, label='Shelving Laplace')
plt.xlabel('f / Hz')
plt.ylabel('level / dB')
plt.title('ideal WFS prefilter, analog vs. FIR')
plt.grid(True)
plt.legend()
plt.savefig('check_wfs_pre_filter_impulse_response.png')
