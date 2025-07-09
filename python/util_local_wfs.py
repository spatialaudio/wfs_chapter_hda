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

# local WFS method presented in
# Nara Hahn, Frank Schultz, Sascha Spors (2022): "Cylindrical Radial Filter Design
# With Application to Local Wave Field Synthesis" J. Aud. Eng. Soc. 70(6):510-525,
# June 2022

# code's main author: Nara Hahn

import sfs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import (eval_legendre as legendre, sph_harm,
                           legendre as poly_legendre, \
                           eval_chebyt as chebyt, erf, comb, gamma, binom,
                           jv as besselj, spherical_jn as sphbesselj)
from scipy.signal import fftconvolve as conv, freqz, lfilter, filtfilt,\
                         butter, unit_impulse, remez
from scipy.signal.windows import kaiser
from scipy.interpolate import interp1d
from sfs.util import db


def _max_order_circular_harmonics(N, max_order):
    """Compute order of 2D HOA."""
    return N // 2 if max_order is None else max_order


def wfs_window(phi, phin=0):
    """Spatial window for WFS driving functions"""
    a = np.cos(phi - phin)
    a[a < 0] = 0
    return a


def fourier_series_wfs_window(max_order, phin=0):
    orders = np.arange(-max_order, max_order + 1)
    idx_ones = np.abs(orders) == 1
    idx_even = (orders % 2) == 0

    Am = np.zeros_like(orders, dtype=complex)
    Am[idx_ones] = 1 / 4
    Am[idx_even] = (-1)**(orders[idx_even]/2)/np.pi/(1-orders[idx_even]**2)
    Am *= np.exp(-1j * orders * phin)
    return orders, Am


def xyplane_sht_coeff(max_degree, max_order):
    """Square of associated Legendre polynomial evaluated in the xy-plane."""
    Anm = []
    Anm.append([1])
    for n in range(1, max_degree + 1):
        mmax = np.min([n, max_order])
        Am = np.zeros(2 * mmax + 1)
        for m in range(-mmax + np.mod(n + mmax, 2), mmax, 2):
            Am[m] = Anm[n-1][m+1] * (n-m-1) / (n-m)
        Am[mmax] = Anm[n-1][mmax-1] * (n+mmax-1) / (n+mmax)
        Anm.append(Am)
    return Anm


def modal_window_max_re(max_order, normalize=False):
    Wn = np.array([legendre(n, np.cos(np.deg2rad(137.9) / (max_order + 1.51)))
                   for n in range(max_order + 1)])
    if normalize:
        Wn *= np.sqrt(1 / np.sum([(2 * n + 1)
                      * Wn[n]**2 / 4 / np.pi for n in range(max_order + 1)]))
#    Wn *= (max_order + 1) / np.sum(Wn)
    return Wn


def modal_window_rectangle(max_order):
    return np.ones(max_order + 1)


def modal_window_kaiser(max_order, beta=0):
    return righthalf_kaiser(max_order+1, beta)
#    return kaiser(2*max_order+1, beta)[max_order:]


def righthalf_kaiser(n, beta=0):
    return kaiser(2*n-1, beta)[n-1:]


def wfs_25d_pw_sht(x0, n0, npw, xc=None, max_order=None,
                   c=343, fs=48000, rf_design=None, sss_order=20):
    """
    2.5D local WFS driving functions for a virtual plane wave.

    Parameters
    ----------
    x0 : (3,), array_like
        Sequence of secondary source positions.
    n0 : (3,), array_like
        Sequence of secondary source orientations.
    npw : (3,), array_like
        Unit vector (propagation direction) of plane wave.
    xc : (3,), array_like
        Expansion center
    max_order : int
        Maximum order 'm' of spherical harmonics
    c : float
        Speed of sound.
    fs : float
        Sampling frequency
    rf_design : RF_Design_Method
        Radial filter design method.
    sss_order : int
        Fourier series order of the secondary source selection window.

    Returns
    -------
    driving_functions :
        Delayed signal

    """
    num_ssd = x0.shape[0]
    if max_order is None:
        max_order = _max_order_circular_harmonics(num_ssd, max_order)
    if xc is None:
        xc = [0, 0, 0]

    phi0, theta0, r0 = sfs.util.cart2sph(*(x0 - xc).T)
    phin0, thetan0, _ = sfs.util.cart2sph(*n0.T)
    phipw, thetapw, _ = sfs.util.cart2sph(*npw)
    t_shift = np.dot(npw, xc) / c

    T = r0.max() / c
    if rf_design.name == 'bandlimit':
        T += (rf_design.poly_order+1)/2/fs
    t_offset = -T
    L = 2 * int(np.ceil(T * fs)) + 1
    t = np.arange(L) / fs - T

    # modal spectrum
    S_max_degree, S_max_order = rf_design.max_degree, max_order
    S_orders = np.arange(-S_max_order, S_max_order + 1)
    Sm = np.exp(-1j * S_orders * phipw)
    B_max_order = sss_order
    D_max_degree = S_max_degree + B_max_order
    D_max_order = S_max_order + B_max_order
    D_orders = np.arange(-D_max_order, D_max_order + 1)
    B_orders, Bm_zeroph = fourier_series_wfs_window(B_max_order, phin=0)

    Knm = xyplane_sht_coeff(D_max_degree, D_max_order)

    d = np.zeros((L, num_ssd), dtype='complex')
    for nls in range(num_ssd):
        Tn = r0[nls] / c
        if rf_design.name == 'bandlimit':
            Tn += (rf_design.poly_order+1)/2/fs
        tn = t[np.abs(t) < Tn]
        Ln = len(tn)
        ni = int(np.round((T - Tn) * fs))  # index of the first nonzero sample
        nf = ni + Ln

        Bm = Bm_zeroph * np.exp(-1j*B_orders*phin0[nls])
        Dm = np.convolve(Sm, Bm)

        h_sph = she_td_radial_functions_pw(D_max_degree, tn, r0[nls]/c, fs,
                                           dc_match=False, method=rf_design)
        dn = np.zeros(Ln, dtype=complex)
        for i, m in enumerate(D_orders):
            window_length = D_max_degree - np.abs(m) + 1
            modal_window = righthalf_kaiser(window_length, rf_design.beta)
            for n in range(np.abs(m), D_max_degree+1):
                if (n+m) % 2 == 0:
                    dn += ((2*n+1) * Knm[n][m] * Dm[i]
                           * modal_window[n-np.abs(m)]
                           * h_sph[n]
                           * np.exp(1j*m*phi0[nls]))
        d[ni:nf, nls] += (8 * np.pi * r0[nls])**0.5 * dn

    return sfs.util.as_delayed_signal((np.real(d), fs, t_offset + t_shift))


def lagrange_filter(filter_order, dfrac):
    """
    Design Lagrange interpolation filter.

    Parameters
    ----------
        dfrac : float
        Fractional delay
        filter_order : int
        Order of the FIR filter

    Returns
    -------
        h : (filter_order+1,) array_like
        FIR filter coefficients
    """
    h = np.ones(filter_order + 1)
    for k in range(filter_order + 1):
        for m in range(filter_order + 1):
            if m == k:
                continue
            h[k] *= (dfrac - m) / (k - m)
    return h


def soundfield_ir(ssd, xr, d, t_offset, c=None, fs=44100):
    """
    Impulse response of a synthesized sound field at a position.

    Parameters
    ----------
    ssd :
        Secondary source data.
    xr :
        Receiver position.
    d :
        Driving signals.
    time_offset :

    c :
        Speed of sound.
    fs :
        Sampling frequency.

    Returns
    -------
    t :
    h :
    """

    if c is None:
        c = sfs.default.c
    Ts = 1 / fs
    x0, _, a0 = ssd
    N = np.shape(x0)[0]
    _, _, r = sfs.util.cart2sph(*(xr - x0).T)
    delay = r / c
    filter_order = 20
    delay_offset = 10.5
    delay_int = (np.floor(delay / Ts) - delay_offset).astype('int')
    delay_frac = delay / Ts - delay_int

    Ls = int(2 * fs)
    h = np.zeros(Ls)
    for nls in range(N):
        hfd = lagrange_filter(filter_order, delay_frac[nls])
        htemp = conv(d[0][:, nls], hfd)
        ni = delay_int[nls]
        h[ni:ni + len(htemp)] += a0[nls] / r[nls] * htemp
    t = np.linspace(0, Ls * Ts, num=Ls, endpoint=False) + d[2]
    return t, 1 / 4 / np.pi * h


def wfs_prefilter(order, fl, fc, fu, fs, c=343):
    fd = np.logspace(np.log10(1), np.log10(fl), num=10, endpoint=True)
    Hd = np.ones_like(fd)

    fd = np.insert(fd, 0, 0)
    Hd = np.insert(Hd, 0, 1)

    fd = np.append(fd, fc)
    Hd = np.append(Hd, 2)

    fh = np.logspace(np.log10(fu), np.log10(fs / 2), num=10, endpoint=True)
    fd = np.append(fd, fh)
    Hd = np.append(Hd, 10**np.log10(fh / fc * 2))

    interp = interp1d(fd, Hd, 'quadratic')

    Lf = 2 * order + 1
    num_f = order + 1
    f = fs / 2 * np.arange(num_f) / num_f

    H = 2 * np.pi * interp(f) / c
    h = np.roll(np.fft.irfft(H, Lf), order)
#    h *= 1 / np.linalg.norm(h)

    if False:
        plt.figure()
        plt.semilogx(f, db(H))
        plt.semilogx(fd, db(Hd), '.')
    return h, (order - 1) / fs


def _wfs_prefilter_fir(dim, N, fl, fu, fs, c):
    """Create pre-equalization filter for WFS.
    Rising slope with 3dB/oct ('2.5D') or 6dB/oct ('2D' and '3D').
    Constant magnitude below fl and above fu.
    Type 1 linear phase FIR filter of order N.
    Simple design via "frequency sampling method".
    Parameters
    ----------
    dim : str
        Dimensionality, must be '2D', '2.5D' or '3D'.
    N : int
        Filter order, shall be even.
    fl : int
        Lower corner frequency in Hertz.
    fu : int
        Upper corner frequency in Hertz.
        (Should be around spatial aliasing limit.)
    fs : int
        Sampling frequency in Hertz.
    c : float
        Speed of sound.
    Returns
    -------
    h : (N+1,) numpy.ndarray
        Filter taps.
    delay : float
        Pre-delay in seconds.
    """
    if N % 2:
        raise ValueError('N must be an even int.')

    bins = int(N / 2 + 1)
    delta_f = fs / (2 * bins - 1)
    f = np.arange(bins) * delta_f
    if dim == '2D' or dim == '3D':
        alpha = 1
    elif dim == '2.5D':
        alpha = 0.5
    desired = np.power(2 * np.pi * f / c, alpha)
    low_shelf = np.power(2 * np.pi * fl / c, alpha)
    high_shelf = np.power(2 * np.pi * fu / c, alpha)
    desired = np.clip(desired, low_shelf, high_shelf)
    desired /= desired[-1]

    h = np.fft.irfft(desired, 2*bins - 1)
    h = np.roll(h, bins - 1)
#    h = h / np.sqrt(np.sum(abs(h)**2))  # normalize energy

    E = np.pi * fs / 2 / c
    h *= np.sqrt(E) / np.linalg.norm(h)

    delay = (bins - 1) / fs
    return h, delay


def _wfs_prefilter_iir(fc, G, fs):
    """[Zoelzer 2008] Table 5.5"""
    wc = 2 * np.pi * fc
    K = np.tan(wc / fs / 2)
    V = 10**(-G / 20)
    den = 1 + np.sqrt(2) * K + K**2
    A = np.array([V + np.sqrt(2 * V) * K + K**2,
                  2 * (K**2 - V),
                  V - np.sqrt(2 * V) * K + K**2]) / den
    B = np.array([1,
                  2 * (K**2 - 1),
                  1 - np.sqrt(2) * K + K**2]) / den
    return B, A


def phase_filter_quaterpi(order):
    n = np.arange(-order, order + 1)
    a = np.zeros(2 * order + 1)
    odd = n % 2 == 1
    a[odd] = -np.sqrt(2) / np.pi / n[odd]
    a[n == 0] = 1 / np.sqrt(2)
    return a


def phase_filter_halfpi(order):
    n = np.arange(-order, order + 1)
    a = np.zeros(2 * order + 1)
    odd = n % 2 == 1
    a[odd] = -1 / np.pi / n[odd]
    return a


def discrete_ir_constant_phase(n, phase_angle):
    idx_zero = (n == 0)
    idx_odd = ~idx_zero * (n % 2 == 1)
    h = np.zeros(len(n))
    h[idx_zero] = np.cos(phase_angle)
    h[idx_odd] = -2 / np.pi / n[idx_odd] * np.sin(phase_angle)
    return h


def periodic_constant_phase_shifter_ir(Ndft, phase_angle):
    n = np.arange(Ndft)
    h = np.zeros(Ndft)

    if Ndft % 2 == 0:
        n_odd = n[n % 2 == 1]
        h[n % 2 == 1] = 2 / Ndft / np.tan(np.pi * n_odd / Ndft)
    elif Ndft % 2 == 1:
        n_odd = n[n % 2 == 1]
        n_even_nonzero = n[(n % 2 == 0) & (n != 0)]
        h[n % 2 == 1] = 1 / Ndft / np.tan(np.pi * n_odd / 2 / Ndft)
        h[(n % 2 == 0) & (n != 0)] = \
            1 / Ndft / np.tan(np.pi * (n_even_nonzero + Ndft) / 2 / Ndft)
    h *= -np.sin(phase_angle)
    h[0] = np.cos(phase_angle)
    return h


def constant_phase_shifter(filter_order, phase_angle, beta=0, frac_delay=0):
    filter_length = filter_order + 1
    n = np.arange(-filter_order / 2, filter_order / 2 + 1) - frac_delay
    h = discrete_ir_constant_phase(n, phase_angle)
    h *= kaiser(filter_length, beta=beta)
    return n, h


def acausal_filter(x, h, idx_zero=None):
    """FIR filtering with group delay compensation.

    The convolution result is truncated to the same length as the input.
    The first sample of the output is determined by center.

    Parameters
    ----------
    x : array_like
        Input signal
    h : array_like
        FIR coefficients
    center : float, optional
        Index of the 0-th coefficient

    """
    Nx = len(x)
    Nh = len(h)
    if idx_zero is None:
        idx_zero = Nh // 2
    return conv(x, h)[idx_zero:idx_zero + Nx]


def set_rcParams():
    rcParams['axes.linewidth'] = 0.5
    rcParams['axes.edgecolor'] = 'k'
    rcParams['axes.facecolor'] = 'None'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.labelsize'] = 14
    #rcParams['text.usetex'] = True
    #rcParams['font.family'] = 'serif'
    #rcParams['font.sans-serif'] = 'Times New Roman'
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = 'Times New Roman'
    rcParams['font.weight'] = 'normal'
    rcParams['font.size'] = 14
    #rcParams['figure.figsize'] = [12, 4]
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



def dir_data():
    return './'


def bspline_poly(n, type='blim', T=1):
    denom = np.math.factorial(n)
    if type == 'blim':
        poly = [np.poly(T * (k - (n+1)/2) * np.ones(n))
                * binom(n+1, k) * (-1)**k
                for k in range(n+1)]
        poly = [np.sum(poly[:k+1], axis=0) / denom / T**n
                for k in range(n+1)]
        if n == 0:
            poly = [poly]
    elif type == 'blep':
        poly = [np.poly(T * (k - (n+1)/2) * np.ones(n+1))
                * binom(n+1, k) * (-1)**k / (n+1)
                for k in range(n+1)]
        poly = [np.sum(poly[:k+1], axis=0) / denom / T**(n+1)
                for k in range(n+1)]
    return np.array(poly)


def bspline_blim(n, t, tau, T=1, fs=44100):
    poly = bspline_poly(n, type='blim', T=T*fs)
    blim = np.zeros_like(t)
    for k in range(n+1):
        ti, tf = (k - (n+1)/2) / fs, (k + 1 - (n+1)/2) / fs
        idx = (ti < (t - tau)) & ((t - tau) <= tf)
        blim[idx] = np.polyval(poly[k], (t[idx] - tau) * fs)
    return blim


def bspline_blep(n, t, tau, T=1, fs=44100):
    poly = bspline_poly(n, type='blep', T=T*fs)
    blep = np.zeros_like(t)
    blep[t > tau] = 1
    for k in range(n+1):
        ti, tf = (k - (n+1)/2) / fs, (k + 1 - (n+1)/2) / fs
        idx = (ti < (t - tau)) & ((t - tau) <= tf)
        blep[idx] = np.polyval(poly[k], (t[idx] - tau) * fs)
    return blep


#def lagrange_poly(n, type='blim', T=1):
#    m = np.arange(n+1)
#    if type == 'blim':
#        poly = np.zeros((n+1, n+1))
#        for k in range(n+1):
#            poly[k] = np.poly(k - np.delete(m, k))
#            poly[k] *= 1 / np.polyval(poly[k], 0)
#        return poly
#    elif type == 'blep':
#        poly = lagrange_poly(n, type='blim', T=T)
#        const = 0
#        ipoly = np.zeros((n+1, n+2))
#        for k in range(n+1):
#            ni, nf = T * (k - (n+1)/2), T * (k + 1 - (n+1)/2)
#            ipoly[k] = np.polyint(poly[k], m=1)
#            const += -np.polyval(ipoly[k], ni)
#            ipoly[k, -1] += const
#            const = np.polyval(ipoly[k], nf)
#        return ipoly


def lagrange_poly(n, type='blim', T=1):
    m = np.arange(n+1)
    if type == 'blim':
        poly = []
        for k in range(n+1):
            pk = np.poly(k - np.delete(m, k))
            pk *= 1 / np.polyval(pk, 0)
            poly.append(np.poly1d(pk))
        return poly
    elif type == 'blep':
        poly = lagrange_poly(n, type='blim', T=T)
        const = 0
        ipoly = np.zeros((n+1, n+2))
        for k in range(n+1):
            ni, nf = (k - (n+1)/2) * T, (k + 1 - (n+1)/2) * T
            ipoly[k] = np.polyint(poly[k], m=1)
            const += -np.polyval(ipoly[k], ni)
            ipoly[k, -1] += const
            const = np.polyval(ipoly[k], nf)
        return ipoly


def lagrange_blim(n, t, tau, T=1, fs=44100):
    poly = lagrange_poly(n, type='blim', T=T*fs)
    blim = np.zeros_like(t)
    for k in range(n+1):
        ti, tf = (k - (n+1)/2) / fs, (k + 1 - (n+1)/2) / fs
        idx = (ti < (t - tau)) & ((t - tau) <= tf)
        blim[idx] = np.polyval(poly[k], (t[idx] - tau) * fs)
    return blim


def lagrange_blep(n, t, tau, T=1, fs=44100):
    poly = lagrange_poly(n, type='blep', T=T*fs)
    blep = np.zeros_like(t)
    blep[t > tau] = 1
    for k in range(n+1):
        ti, tf = (k - (n+1)/2) / fs, (k + 1 - (n+1)/2) / fs
        idx = (ti < (t - tau)) & ((t - tau) <= tf)
        blep[idx] =\
            np.polyval(poly[k], (t[idx] - tau) * fs)
    return blep


def step(x):
    y = np.zeros_like(x)
    y[x > 0] = 1
    y[x == 0] = 0.5
    return y


def nmse(href, h, axis=-1, order=2, normalize=True):
    num = np.mean(np.abs(href-h)**order, axis=axis)
    den = np.mean(np.abs(href)**order, axis=axis)
    return (num/den)**(1/order) if normalize else num**(1/order)
#    return (np.sum(np.abs(href - h)**order, axis=axis)
#            / np.sum(np.abs(href)**order, axis=axis))**(1/order)


def nse(href, h, axis=-1, order=2, normalize=True):
    num = np.sum(np.abs(href-h)**order, axis=axis)
    den = np.sum(np.abs(href)**order, axis=axis)
    return (num/den)**(1/order) if normalize else num**(1/order)
#    return (np.sum(np.abs(href - h)**order, axis=axis)
#            / np.sum(np.abs(href)**order, axis=axis))**(1/order)


def log_frequency(fmin, fmax, num_f, endpoint=True):
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_f,
                       endpoint=endpoint)


def f2w(f):
    return 2 * np.pi * f


def w2k(w, c=343):
    return w / c


def f2k(f, c=343):
    return w2k(f2w(f), c=c)


def hz2khz(f):
    return f/1000.


def s2ms(t):
    return t*1000.


def time_axis(fs, tmax, tmin=None, t0=0):
    if tmin is None:
        tmin = -tmax
#    if (t0 > tmax) or (t0 < tmin):
#        raise ValueError("t0 is not in [tmin, tmax].")
    L1 = int(np.floor((tmax - t0) * fs))
    L2 = int(np.floor((t0 - tmin) * fs))
    t1 = t0 + np.arange(L1) / fs
    t2 = t0 - np.arange(1, L2) / fs
    return np.concatenate((t2[::-1], t1))


def she_radial_td_step(max_order, t, roverc, tshift=0):
    tau = t - tshift
    x = tau / roverc
    amplitude = 0.5 / roverc
    h = np.zeros((max_order+1, len(t)))
    h[0] = step(x + 1) - step(x - 1)
    for n in range(1, max_order+1):
        h[n] = ((2*n-1) * x * h[n-1] - (n-1) * h[n-2]) / n
    return amplitude * h


def she_td_radial_functions(max_degree, t, rc, fs, method=None, blep_order=0, fc=None):
    if fc is None:
        fc = fs/2

    h = np.zeros((max_degree+1, len(t)))
    amplitude = 1/2/fs/rc
    crt = t/rc
    if method is None or method == 'step':
        h[0] = step(t + rc) - step(t - rc)
    elif method == 'lagrange':
        h[0] = (lagrange_blep(blep_order, t, -rc, T=1/fs, fs=fs)
                - lagrange_blep(blep_order, t, rc, T=1/fs, fs=fs))
    elif method == 'bspline':
        h[0] = (bspline_blep(blep_order, t, -rc, T=0.5/fc, fs=fs)
                - bspline_blep(blep_order, t, rc, T=0.5/fc, fs=fs))

    delta = (blep_order+1)/2/fs
    idx = (t > -rc-delta) & (t < rc+delta)
    for n in range(1, max_degree+1):  # recurrence relation
        h[n, idx] = ((2*n-1) * crt[idx] * h[n-1, idx]
                     - (n-1) * h[n-2, idx]) / n
    h[:, idx] *= amplitude
    return h


class RF_Design_Method:
    def __init__(self, iterable=[]):
#        self.name = name
        self.__update(iterable)

    def __update(self, iterable):
        for key in iterable:
            self.__setattr__(key, iterable[key])
    pass


def che_td_radial_functions_pw(max_order, t, rc, fs, method=None):
#                               kw_blep=None, max_degree=None, beta=0):
    h = np.zeros((max_order+1, len(t)))
    amplitude = 1/np.pi/rc/fs
    crt = t/rc
    if method is None:
        idx = (np.abs(t) < rc)
        for m in range(max_order+1):
            h[m, idx] = chebyt(m, crt[idx])
        h[:, idx] *= amplitude/np.sqrt(1-crt[idx]**2)
    elif method.name == 'she':
        max_degree = method.max_degree
        h_sph = she_td_radial_functions_pw(max_degree, t, rc, fs)
        Knm = xyplane_sht_coeff(max_degree, max_order)
        for m in range(max_order+1):
            window_length = max_degree - np.abs(m) + 1
            modal_window = righthalf_kaiser(window_length, method.beta)
            for n in range(np.abs(m), max_degree+1):
                if (n+m) % 2 == 0:
                    h[m] += ((2*n+1) * Knm[n][m] * h_sph[n]
                             * modal_window[n-np.abs(m)])
    elif method.name == 'bandlimit':
        max_degree = method.max_degree
        h_sph = she_td_radial_functions_pw(max_degree, t, rc, fs,
                                           dc_match=False,
                                           method=method)
        idx = np.where(h_sph[0])[0]
        Knm = xyplane_sht_coeff(max_degree, max_order)
        for m in range(max_order+1):
            window_length = max_degree - np.abs(m) + 1
            modal_window = righthalf_kaiser(window_length, method.beta)
            for n in range(np.abs(m), max_degree+1):
                if (n+m) % 2 == 0:
                    h[m] += ((2*n+1) * Knm[n][m] * h_sph[n]
                             * modal_window[n-np.abs(m)])
        idx_pos = t > 0
        idx_neg = t < 0
        for m in range(max_order+1):
            h[m, idx_pos] = h[m, idx_neg][::-1] * (-1)**m
    return h


#def she_td_radial_functions_pw(max_degree, t, rc, fs, dc_match=False,
#                               method=None, kw_blep={}):
#
#    amplitude = 1/2/rc/fs
#    h = np.zeros((max_degree+1, len(t)))
#    idx_nonzero = (np.abs(t) < rc)
#    u = t/rc
#    for n in range(max_degree+1):
#        h[n, idx_nonzero] = legendre(n, u[idx_nonzero])
#
#    # kw_blep = dict(polynomial, poly_order, deriv_order, fc)
#    if method == 'integrate_blep':
#        polynomial = kw_blep['polynomial']
#        poly_order = kw_blep['poly_order']
#        deriv_order = kw_blep['deriv_order']
#        fc = kw_blep['fc']
#
#        T = 0.5/fc
#        num_interval = poly_order + 1
#        delta = (num_interval/2) * T / rc
##        delta = 0
#        r_left, r_right = derivative_discontinuity(max_degree, deriv_order, delta)
#        poly = smoothing_poly(polynomial, poly_order, deriv_order, residual=True)
#
#        ti, tf = -rc - (num_interval/2)*T, -rc + (num_interval/2)*T
#        idx = (t > ti) & (t < tf)
#        scaling_factor = 1
#        for m in range(deriv_order):
#            if m >= 1:
#                scaling_factor *= m
#            res = eval_piecewise_poly(poly[m], (t[idx]+rc)/T) * 1 * (T/rc)**m
#            for n in range(max_degree+1):
#                h[n, idx] += r_left[n, m] * res
#        ti, tf = rc - (num_interval/2)*T, rc + (num_interval/2)*T
#        idx = (t > ti) & (t < tf)
#        scaling_factor = 1
#        for m in range(deriv_order):
#            if m >= 1:
#                scaling_factor *= m
#            res = eval_piecewise_poly(poly[m], (t[idx]-rc)/T) * 1 * (T/rc)**m
#            for n in range(max_degree+1):
#                h[n, idx] += r_right[n, m] * res
#        idx_nonzero = (t > -rc-(num_interval/2)*T) & (t < rc+(num_interval/2)*T)
#
#    h[:, idx_nonzero] *= amplitude
#
#    if dc_match is True:
#        for n in range(1, max_degree+1):
#            h[n, idx_nonzero] -= np.mean(h[n, idx_nonzero])
#    return h


def integrate_piecewise_poly(poly, m=1, continuous=True):
    """Integrate picewise polynomial.

    Parameters
    ----------
    poly : list of numpy.poly1d
        Coefficients of piecewise polynomials.
    m : int
        Integeration order
    continuous : bool
        Continuous polynomials returned if True.

    Return
    ------
    ipoly : list of numpy.poly1d
        Coefficients of integrated piecewise polynomials.
    """
    if m == 1:
        ipoly = [np.polyint(p, m=1) for p in poly]
        if continuous:
            const = 0
            num_interval = len(poly)
            ni = -num_interval/2
            nf = ni + 1
            for k, ip in enumerate(ipoly):
                const -= np.polyval(ip, ni)
                ipoly[k][0] += const
                const = np.polyval(ip, nf)
                ni += 1
                nf += 1
    elif m > 1:
        for i in range(m):
            ipoly = integrate_piecewise_poly(ipoly, m=1, continuous=continuous)
    return ipoly


def bessel_poly(n, reverse=False):
    a = np.zeros(n+1)
    a[0] = 1
    for k in range(1, n+1):
        a[k] = (n+k)*(n-k+1)/k/2 * a[k-1]
    return (np.poly1d(a) if reverse else np.poly1d(a[::-1]))


def blep_antiderivative(poly_name, poly_order, max_integrate, residual=False):
    """Higher-oder derivatives of BLEP functions.

    Parameters
    ----------
    poly_name : string
        Polynomial name {'lagrange', 'bspline'}
    poly_order : int
        Polynomial order
    max_integrate : int
        Maximum anti-derivative order
    residual : bool
        Polynomial of the residual functions are returned if True.
    """

    if poly_name == 'lagrange':
        poly_blim = lagrange_poly(poly_order, type='blim', T=1)
    elif poly_name == 'bspline':
        poly_blim = bspline_poly(poly_order, type='blim', T=1)

    poly = []
    poly.append(integrate_piecewise_poly(poly_blim))  # BLEP function
    for k in range(1, max_integrate+1):
        poly.append(integrate_piecewise_poly(poly[k-1], m=1))
    if residual:
        m0 = int((poly_order+1)/2)
        scaling_factor = 1  # computing 1/k!
        for k in range(max_integrate+1):
            for m in range(m0, len(poly[k])):
                poly[k][m][k] -= scaling_factor
            scaling_factor *= 1/(k+1)

        # alternative using "numpy.polysub"
#        poly_disc = np.poly1d([1])
#        for k in range(max_integrate+1):
#                poly[k][m] = np.polysub(poly[k][m], poly_disc)
#            poly_disc = np.polyint(poly_disc)
    return poly


def she_td_radial_functions_pw(max_sh_order, t, rc, fs, dc_match=False,
                               method=None):
#                               method=None, kw_blep={}):

    amplitude = 1/2/rc/fs
    h = np.zeros((max_sh_order+1, len(t)))
    u = t/rc  # normalized argument

    for n in range(max_sh_order+1):
        idx_nonzero = (np.abs(t) < rc)
        h[n, idx_nonzero] = legendre(n, u[idx_nonzero])

    if method is not None:
        if method.name == 'recurrence':
            poly_name = method.poly_name
            poly_order = method.poly_order
            fc = method.fc
    #        poly_name = kw_blep['poly_name']
    #        poly_order = kw_blep['poly_order']
    #        fc = kw_blep['fc']
            T = 0.5/fc
            num_interval = poly_order + 1
            tau = num_interval / 2 * T  # half length of piece-wise polynomial

            poly = blep_antiderivative(poly_name, poly_order, 0, residual=True)

            idx_onset = (t+rc > -tau) & (t+rc < tau)
            idx_offset = (t-rc > -tau) & (t-rc < tau)
            residual_onset = eval_piecewise_poly(poly[0], (t[idx_onset]+rc)/T)
            residual_offset = eval_piecewise_poly(poly[0], (t[idx_offset]-rc)/T)
            h[0, idx_onset] += residual_onset
            h[0, idx_offset] += - residual_offset

            idx_nonzero = (t+rc > -tau) & (t-rc < tau)
            for n in range(1, max_sh_order+1):  # recurrence relation
                h[n, idx_nonzero] = ((2*n-1) * u[idx_nonzero] * h[n-1, idx_nonzero]
                                     - (n-1) * h[n-2, idx_nonzero]) / n
        elif method.name == 'bandlimit':
            poly_name = method.poly_name
            poly_order = method.poly_order
            deriv_order = method.deriv_order
            fc = method.fc
    #        poly_name = kw_blep['poly_name']
    #        poly_order = kw_blep['poly_order']
    #        deriv_order = kw_blep['deriv_order']
    #        fc = kw_blep['fc']
            T = 0.5/fc
            num_interval = poly_order + 1
            tau = num_interval / 2 * T  # half length of piece-wise polynomial

            poly = blep_antiderivative(poly_name, poly_order,
                                       deriv_order, residual=True)

            idx_onset = (t+rc > -tau) & (t+rc < tau)
            idx_offset = (t-rc > -tau) & (t-rc < tau)
            residuals_onset = [eval_piecewise_poly(poly[k], (t[idx_onset]+rc)/T)
                               * (T/rc)**k for k in range(deriv_order+1)]
            residuals_offset = [eval_piecewise_poly(poly[k], (t[idx_offset]-rc)/T)
                                * (T/rc)**k for k in range(deriv_order+1)]

            for n in range(max_sh_order+1):
                beta = bessel_poly(n, reverse=True).coef
                max_deriv_order = np.min([n, deriv_order])
                for k in range(max_deriv_order+1):
                    h[n, idx_onset] += (-1)**(n-k) * beta[k] * residuals_onset[k]
                    h[n, idx_offset] += -beta[k] * residuals_offset[k]

            idx_nonzero = (t+rc > -tau) & (t-rc < tau)

#            idx_pos = t > 0
#            idx_neg = t < 0
#            for m in range(max_sh_order+1):
#                h[m, idx_pos] = h[m, idx_neg][::-1] * (-1)**m

    h[:, idx_nonzero] *= amplitude

    if dc_match is True:
        for n in range(1, max_sh_order+1):
            h[n, idx_nonzero] -= np.mean(h[n, idx_nonzero])
    return h


def che_td_radial_functions(max_order, t, rc, fs, method=None, blep_order=0,
                            max_degree=None, beta=0):
    h = np.zeros((max_order+1, len(t)))
    amplitude = 1/np.pi/rc/fs
    crt = t/rc
    if method is None:
        idx = (np.abs(t) < rc)
        for m in range(max_order+1):
            h[m, idx] = chebyt(m, crt[idx])
        h[:, idx] *= amplitude/np.sqrt(1-crt[idx]**2)
    else:
        if max_degree is None:
            max_degree = max_order
        h_sph = she_td_radial_functions(max_degree, t, rc, fs, method, blep_order)
        h_sph *= kaiser(2*max_degree+1, beta=beta)[max_degree:][:, np.newaxis]
        idx = np.where(h_sph[0])[0]
        Knm = xyplane_sht_coeff(max_degree, max_order)
        for m in range(max_order+1):
            for n in range(np.abs(m), max_degree+1):
                if (n+m) % 2 == 0:
                    h[m] += (2*n+1) * Knm[n][m] * h_sph[n]
    return h


def che_td_radial_function(m, t, rc):
    amplitude = 1/np.pi/rc
    crt = t/rc
    idx = (np.abs(crt) < 1)
    h = np.zeros_like(t)
    h[idx] = chebyt(m, crt[idx])
    h[idx] *= amplitude / np.sqrt(1-crt[idx]**2)
    return h


def she_td_radial_function(n, t, rc):
    amplitude = 0.5/rc
    crt = t/rc
    idx = (np.abs(crt) < 1)
    h = np.zeros_like(t)
    h[idx] = legendre(n, crt[idx])
    return amplitude * h


def che_fd_radial_functions(max_order, f, rc):
#    H = np.zeros((max_order+1, len(f)), dtype=complex)
#    omega = f2w(f)
#    for m in range(max_order+1):
#        H[m] = 1j**-m * besselj(m, omega*rc)
    return np.stack([che_fd_radial_function(m, f, rc)
                     for m in range(max_order+1)])


def che_fd_radial_function(m, f, rc):
    return 1j**-m * besselj(m, f2w(f)*rc)


def she_fd_radial_function(n, f, rc):
    return 1j**-n * sphbesselj(n, f2w(f)*rc)


def discontinuity(t, t0=0, order=0):
    y = np.zeros_like(t)
    idx = (t > t0)
    y[idx] = (t[idx]-t0)**order
    return y


def integrate_poly(poly, m=1):
    if m == 1:
        num_interval, poly_order = poly.shape
        ipoly = np.zeros((num_interval, poly_order+m))
        for k, p in enumerate(poly):
            ipoly[k] = np.polyint(p, m=m)
        const = 0
        for k, ip in enumerate(ipoly):
            ni, nf = (k-num_interval/2), (k-num_interval/2+1)
            const -= np.polyval(ip, ni)
            ipoly[k, -1] += const
            const = np.polyval(ip, nf)
    elif m > 1:
        ipoly = poly
        for i in range(m):
            ipoly = integrate_poly(ipoly, m=1)
    return ipoly


def eval_piecewise_poly(poly, x):
    num_interval = len(poly)
    y = np.zeros_like(x)
    for k, p in enumerate(poly):
        xi = k - num_interval/2
        xf = xi + 1
        idx = (x >= xi) & (x <= xf)
        y[idx] = np.polyval(p, x[idx])
    return y


def factorial(n):
    if (n == 0 or n == 1):
        return 1
    elif n >= 2:
        return n * factorial(n-1)
    else:
        return None


def derivative_discontinuity(max_degree, deriv_order, delta=0):
    """Discontinuities of the radial functions ant their derivatives.
    """
    left_discontinuity = np.zeros((max_degree+1, deriv_order+1))
    right_discontinuity = np.zeros((max_degree+1, deriv_order+1))
    for n in range(max_degree+1):
        p = poly_legendre(n)
        for m in range(deriv_order+1):
            if True:
                left_discontinuity[n, m] = np.polyval(p, -1)
                right_discontinuity[n, m] = -np.polyval(p, 1)
            else:
                left_discontinuity[n, m] = np.polyval(p, -1+delta)
                right_discontinuity[n, m] = -np.polyval(p, 1-delta)
            p = np.polyder(p, m=1)
    return left_discontinuity, right_discontinuity


def smoothing_poly(polynomial, poly_degree, max_integrate, residual=False):
    poly = []
    if polynomial == 'lagrange':
        poly.append(lagrange_poly(poly_degree, type='blep', T=1))
    elif polynomial == 'bspline':
        poly.append(bspline_poly(poly_degree, type='blep', T=1))
    for m in range(1, max_integrate+1):
        poly.append(integrate_poly(poly[m-1], m=1))
    if residual:
        n0 = int((poly_degree+1)/2)
        scaling_factor = 1
        for m in range(max_integrate+1):
            poly[m][n0:, -m-1] -= scaling_factor
            scaling_factor *= 1/(m+1)
    return poly
