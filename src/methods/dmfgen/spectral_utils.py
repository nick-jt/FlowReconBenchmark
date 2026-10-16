"""Shell-averaged spectra for non-periodic sub-cubes.

The JHU cutouts are sub-blocks of a larger DNS, so they are NOT periodic.
A plain FFT of such a block leaks the edge discontinuity across all
wavenumbers and produces a broadband floor: measured on our 125^3 cutout the
raw spectrum falls only 2.4x from k=32 to the Nyquist wavenumber, and a
deliberately smooth Gaussian-process draw -- whose true energy at k=62 is
e^-1700, i.e. exactly zero -- registers the same floor as the data. Any band
ratio computed above the crossover is then a ratio of two leakage floors and
tends to one regardless of the model.

Applying a separable Hann window before the transform restores the expected
decay (5.8e5x over the same range) at the cost of a mild low-k bias, and is
the estimator used for every spectral number we report. Each field's own mean
is removed before the window; otherwise the window spreads that mean into the
lowest shells. Fields that ARE periodic (the 2D Kolmogorov box) need no window and are
passed window=False.
"""

import functools

import numpy as np

_WINDOW_CACHE = {}


def _hann(n, ndim):
    if (n, ndim) not in _WINDOW_CACHE:
        W = functools.reduce(np.multiply.outer, [np.hanning(n)] * ndim)
        _WINDOW_CACHE[(n, ndim)] = (W / np.sqrt((W ** 2).mean())).astype(np.float64)
    return _WINDOW_CACHE[(n, ndim)]


def _taper(field, window):
    """float64 copy of a field, mean-removed and Hann-windowed if window."""
    g = np.asarray(field, dtype=np.float64)
    return (g - g.mean()) * _hann(g.shape[0], g.ndim) if window else g


def _shells(n, ndim):
    """|k| on the n^ndim FFT grid and its nearest-integer shell index."""
    kk = np.fft.fftfreq(n) * n
    kmag = np.sqrt(sum(k ** 2 for k in np.meshgrid(*[kk] * ndim, indexing="ij")))
    return kmag, np.round(kmag).astype(int)


def _amp(kmag, k_power):
    """|k|^(k_power/2) per mode (0 at k=0), so |amp * F|^2 carries the weight |k|^k_power."""
    if not k_power:
        return 1.0
    with np.errstate(divide="ignore"):
        return np.where(kmag > 0, kmag ** (0.5 * k_power), 0.0)


def shell_spectrum(field, kmax=None, window=True, k_power=0):
    """Shell-summed energy spectrum E(k), k = 1..kmax, of a field on a square
    or cubic grid. k_power weights each mode by |k|^k_power: -2 turns a 2D
    vorticity field into its kinetic energy spectrum."""
    g = _taper(field, window)
    n = g.shape[0]
    kmag, kb = _shells(n, g.ndim)
    f = np.abs(np.fft.fftn(g) * _amp(kmag, k_power)) ** 2
    spec = np.bincount(kb.ravel(), weights=f.ravel())
    kmax = kmax or (n // 2)
    return spec[1:kmax + 1]


def reliable_kmax(truth_spec, floor_ratio=10.0):
    """Largest k where the truth still sits a factor floor_ratio above its tail."""
    tail = np.median(truth_spec[-5:])
    ok = np.nonzero(truth_spec > floor_ratio * max(tail, 1e-300))[0]
    return int(ok[-1] + 1) if len(ok) else len(truth_spec)


def band_ratio(sample_spec, truth_spec, lo, hi):
    return float(sample_spec[lo - 1:hi].sum() / truth_spec[lo - 1:hi].sum())


def band_corr(samples, truths, bands, window=True, k_power=0):
    """Band-limited correlation of samples with truths, pooled over channels.

    samples, truths: sequences of square or cubic fields, one per channel. bands:
    {name: (lo, hi)} on the integer shells of shell_spectrum; k_power as there.
    Returns {name: correlation}.
    """
    t0 = np.asarray(truths[0])
    n = t0.shape[0]
    kmag, kb = _shells(n, t0.ndim)
    masks = {b: (kb >= lo) & (kb <= hi) for b, (lo, hi) in bands.items()}
    A = _amp(kmag, k_power)
    acc = {b: np.zeros(3) for b in bands}            # cross, |s|^2, |t|^2
    for s, t in zip(samples, truths):
        Fs = np.fft.fftn(_taper(s, window)) * A
        Ft = np.fft.fftn(_taper(t, window)) * A
        for b, m in masks.items():
            acc[b] += [np.real(np.sum(Fs[m] * np.conj(Ft[m]))), np.sum(np.abs(Fs[m]) ** 2),
                       np.sum(np.abs(Ft[m]) ** 2)]
    return {b: float(a[0] / np.sqrt(a[1] * a[2])) for b, a in acc.items()}


def band_metrics(samples, truth, bands, window=True, k_power=0):
    """Band energy and band correlation of ensemble members against the truth.

    samples: [K, C, *grid] members; truth: [C, *grid] (square or cubic grid).
    Spectra are summed and correlations pooled over the C channels. Returns, per
    band name: "energy", the band energy ratio of single members to the truth,
    averaged over members; "corr", the band correlation, averaged over members;
    "energy_of_mean", the band energy ratio of the ensemble mean.
    """
    spec = lambda f: sum(shell_spectrum(c, window=window, k_power=k_power) for c in f)
    s_true = spec(truth)
    s_mean = spec(np.mean(samples, axis=0))
    s_members = [spec(s) for s in samples]
    corr = [band_corr(s, truth, bands, window=window, k_power=k_power) for s in samples]
    return {b: {"energy": float(np.mean([band_ratio(s, s_true, lo, hi) for s in s_members])),
                "corr": float(np.mean([c[b] for c in corr])),
                "energy_of_mean": band_ratio(s_mean, s_true, lo, hi)}
            for b, (lo, hi) in bands.items()}
