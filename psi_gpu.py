"""
GPU-ready (CuPy-compatible), fully vectorized version of occulter_edge_integral,
for evaluating many (a, b, c) triples at once over an arbitrary-shaped array.

Replaces scipy.integrate.quad's adaptive Gauss-Kronrod with a FIXED-ORDER
Gauss-Legendre rule. This is safe here (not a generic hack) because the
rotated-ray construction, once restricted to x >= 0 with negative x handled
by reflection (see psi_p.py), makes the denominator D(s) strictly positive
for every s >= 0 and every gamma != 0 -- there is no near-singular feature
for adaptivity to chase. The fixed order needed (default 48) was checked
against the adaptive scipy reference across x from 0 to 1e5 and matches to
machine precision until the intrinsic float64 phase-rounding floor at large
x (the same floor the adaptive version has -- not a quadrature-order issue).

Works transparently on CPU (numpy) or GPU (cupy): pass numpy arrays for CPU,
cupy arrays for GPU. No other code changes needed.

Usage:
    import numpy as np
    a = np.array([...])   # any shape
    b = np.array([...])   # broadcastable with a
    c = np.array([...])   # broadcastable with a, b
    I = occulter_edge_integral_batch(a, b, c)   # complex array, broadcast shape

    # on GPU:
    import cupy as cp
    a, b, c = cp.asarray(a), cp.asarray(b), cp.asarray(c)
    I = occulter_edge_integral_batch(a, b, c)   # computed entirely on GPU
"""
import numpy as np

try:
    import cupy as _cp
    def get_array_module(x):
        return _cp.get_array_module(x)
except ImportError:
    def get_array_module(x):
        return np

SQRT2 = 2.0 ** 0.5
_L = 46.0          # envelope cutoff: exp(-46) ~ 1e-20
_ORDER = 48        # fixed Gauss-Legendre order (see module docstring)

_gl_cache = {}      # order -> (nodes, weights) as plain numpy, host side


def _gl_nodes(order):
    if order not in _gl_cache:
        _gl_cache[order] = np.polynomial.legendre.leggauss(order)
    return _gl_cache[order]


def _smax(x, xp):
    """Numerically stable S_max(x) (rationalized form, see psi_p.py)."""
    return 2.0 * _L / (SQRT2 * x + xp.sqrt(2.0 * x * x + 4.0 * _L))


def _phi_batch(x, gamma, order, xp):
    """
    Phi_{i*gamma}(x) for x >= 0, fully vectorized, fixed-order Gauss-Legendre.

    x, gamma: arrays already on module xp, broadcastable, x >= 0 everywhere.
    Returns a complex array of the broadcast shape.
    """
    xi, wi = _gl_nodes(order)
    xi = xp.asarray(xi, dtype=float)
    wi = xp.asarray(wi, dtype=float)

    x = x[..., None]                                          # (..., 1)
    gamma = xp.broadcast_to(gamma, x.shape[:-1])[..., None]   # (..., 1)

    Smax = _smax(x[..., 0], xp)[..., None]       # (..., 1)
    s = 0.5 * Smax * (xi + 1.0)                  # (..., order)

    A = xp.exp(-s * s - SQRT2 * x * s)
    D = s * s + SQRT2 * (x - gamma) * s + x * x + gamma * gamma
    u = x + s / SQRT2
    w = s / SQRT2 - gamma
    ph = x * x + SQRT2 * x * s + xp.pi / 4.0
    cph, sph = xp.cos(ph), xp.sin(ph)

    integ_re = (A / D) * (u * cph + w * sph)
    integ_im = (A / D) * (u * sph - w * cph)

    Re = 0.5 * Smax[..., 0] * xp.sum(wi * integ_re, axis=-1)
    Im = 0.5 * Smax[..., 0] * xp.sum(wi * integ_im, axis=-1)
    return Re + 1j * Im


def Phi_batch(x, gamma, order=_ORDER):
    """Phi_{i*gamma}(x), any x >= 0, batched over broadcastable arrays."""
    xp = get_array_module(x)
    x = xp.asarray(x, dtype=float)
    gamma = xp.asarray(gamma, dtype=float)
    gamma = xp.broadcast_to(gamma, x.shape)
    print('x.shape, gamma.shape',x.shape, gamma.shape)
    return _phi_batch(x, gamma, order, xp)


def Psi_batch(x, gamma, order=_ORDER):
    """
    Psi_gamma(x) = int_0^x e^{it^2}/(t - i*gamma) dt, ANY real x, batched.

    Negative x is handled by reflection (t -> -t sends gamma -> -gamma),
    exactly as in the scalar version -- no branch-point crossing, no
    Heaviside step, no adaptive bridging needed anywhere.
    """
    xp = get_array_module(x)
    x = xp.asarray(x, dtype=float)
    gamma = xp.broadcast_to(xp.asarray(gamma, dtype=float), x.shape)

    neg = x < 0
    x_eff = xp.where(neg, -x, x)
    gamma_eff = xp.where(neg, -gamma, gamma)

    return _phi_batch(xp.zeros_like(x_eff), gamma_eff, order, xp) \
         - _phi_batch(x_eff, gamma_eff, order, xp)


def G_batch(gamma, a, b, order=_ORDER):
    """int_a^b e^{ix^2}/(x - i*gamma) dx, batched, any real a, b, gamma != 0."""
    return Psi_batch(b, gamma, order) - Psi_batch(a, gamma, order)


def occulter_edge_integral_batch(a, b, c, order=_ORDER):
    """
    int_a^b e^{ix^2}/(x^2+c^2) dx, fully batched/vectorized, GPU-ready.

    a, b, c: arrays, any shape, broadcastable to each other.
    Returns: complex array of the broadcast shape.
    """
    xp = get_array_module(a)
    a = xp.asarray(a, dtype=float)
    b = xp.asarray(b, dtype=float)
    c = xp.asarray(c, dtype=float)
    return (G_batch(c, a, b, order) - G_batch(-c, a, b, order)) / (2j * c)