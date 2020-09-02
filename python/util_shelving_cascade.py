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

"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth.

the functions below originate from the original repository
https://github.com/spatialaudio/aes148-shelving-filter/releases/tag/v1.0
provided with MIT License

code belongs originally to the project
Frank Schultz, Nara Hahn, Sascha Spors
Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

"""
import numpy as np
from scipy.signal import tf2sos, freqs


def halfpadloss_shelving_filter_num_den_coeff(G):
    """Half-pad-loss polynomial coefficients for 1st/2nd order shelving filter.

    - see type III in
    long-url: https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_desig/audiofilter.ipynb  # noqa

    - see Sec. 3.2 in https://doi.org/10.3390/app6050129

    """
    sign = np.sign(G)  # amplify/boost (1) or attenuate/cut (-1)
    g = 10**(np.abs(G) / 20)  # linear gain
    n1, n2 = g**(sign / 4), g**(sign / 2)  # numerator coeff
    d1, d2 = 1 / n1, 1 / n2  # denominator coeff
    return n1, n2, d1, d2


def normalized_low_shelving_2nd_coeff(G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """See low_shelving_2nd_coeff() for omega=1."""
    n1, n2, d1, d2 = halfpadloss_shelving_filter_num_den_coeff(G)
    b, a = np.array([1, n1 / Q, n2]), np.array([1, d1 / Q, d2])
    return b, a


def low_shelving_2nd_coeff(omega=1, G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """Half-pad-loss/mid-level low shelving filter 2nd order.

    Parameters
    ----------
    omega : angular frequency in rad/s at half-pad-loss/mid-level
    G : level in dB (G/2 at omega)
    Q : pole/zero quality, Q>0.5
    Returns
    -------
                                          b[0] s^2 + b[1] s^1 + b[2] s^0
    b,a : coefficients for Laplace H(s) = ------------------------------
                                          a[0] s^2 + a[1] s^1 + a[2] s^0
    with s = j omega

    see halfpadloss_shelving_filter_num_den_coeff() for references

    """
    b, a = normalized_low_shelving_2nd_coeff(G=G, Q=Q)
    scale = omega**np.arange(-2., 1.)  # powers in the Laplace domain
    return b * scale, a * scale


def db(x, *, power=False):
    """Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def shelving_slope_parameters(slope=None, BWd=None, Gd=None):
    """Compute the third parameter from the given two.

    Parameters
    ----------
    slope : float, optional
        Desired shelving slope in decibel per octave.
    BWd : float, optional
        Desired bandwidth of the slope in octave.
    Gd : float, optional
        Desired gain of the stop band in decibel.

    """
    if slope == 0:
        raise ValueError("`slope` should be nonzero.")
    if slope and BWd is not None:
        Gd = -BWd * slope
    elif BWd and Gd is not None:
        slope = -Gd / BWd
    elif Gd and slope is not None:
        if Gd * slope > 1:
            raise ValueError("`Gd` and `slope` cannot have the same sign.")
        else:
            BWd = np.abs(Gd / slope)
    else:
        print('At lest two parameters need to be specified.')
    return slope, BWd, Gd


def shelving_filter_parameters(biquad_per_octave, **kwargs):
    """Parameters for shelving filter design.

    Parameters
    ----------
    biquad_per_octave : float
        Number of biquad filters per octave.

    Returns
    -------
    num_biquad : int
        Number of biquad filters.
    Gb : float
        Gain of each biquad filter in decibel.
    G : float
        Gain of overall (concatenated) filters in decibel. This might differ
        from what is returned by `shelving_parameters`.

    """
    slope, BWd, Gd = shelving_slope_parameters(**kwargs)
    num_biquad = int(np.ceil(BWd * biquad_per_octave))
    Gb = -slope / biquad_per_octave
    G = Gb * num_biquad
    return num_biquad, Gb, G


def check_shelving_filter_validity(biquad_per_octave, **kwargs):
    """Level, slope, bandwidth validity for shelving filter cascade.

    Parameters
    ----------
    biquad_per_octave : float
        Number of biquad filters per octave.

    see shelving_slope_parameters(), shelving_filter_parameters()

    Returns
    -------
    flag = [Boolean, Boolean, Boolean]

    if all True then intended parameter triplet holds, if not all True
    deviations from desired response occur

    """
    flag = [True, True, True]
    slope, BWd, Gd = shelving_slope_parameters(**kwargs)
    _, Gb, G = shelving_filter_parameters(biquad_per_octave, **kwargs)

    # BWd < 1 octave generally fails
    if BWd <= 1:
        flag[0] = False

    # BWd * biquad_per_octave needs to be integer
    flag[1] = float(BWd * biquad_per_octave).is_integer()

    # biquad_per_octave must be large enough
    # for slope < 12.04 dB at least one biquad per ocatve is required
    tmp = slope / (20*np.log10(4))
    if tmp > 1.:
        if biquad_per_octave < tmp:
            flag[2] = False
    else:
        if biquad_per_octave < 1:
            flag[2] = False
    return flag


def low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave,
                             Q=1/np.sqrt(2)):
    """Low shelving filter design using cascaded biquad filters.

    Parameters
    ----------
    w0 : float
        Cut-off frequency in radian per second.
    Gb : float
        Gain of each biquad filter in decibel.
    num_biquad : int
        Number of biquad filters.
    biquad_per_octave: int
        Number of biquad filters per octave.
    Q : float, optional
        Quality factor of each biquad filter.

    Returns
    -------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.

    """
    sos = np.zeros((num_biquad, 6))
    for m in range(num_biquad):
        wm = w0 * 2**(-(m + 0.5) / biquad_per_octave)
        b, a = low_shelving_2nd_coeff(omega=wm, G=Gb, Q=Q)
        sos[m] = tf2sos(b, a)
    return sos


def sosfreqs(sos, worN=200, plot=None):
    """Compute the frequency response of an analog filter in SOS format.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations).  If a single
        integer, then compute at that many frequencies.  Otherwise, compute the
        response at the angular frequencies (e.g. rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    """
    h = 1.
    for row in sos:
        w, rowh = freqs(row[:3], row[3:], worN=worN, plot=plot)
        h *= rowh
    return w, h
