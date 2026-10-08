"""
GPU-ready (CuPy / NumPy) fixed-order evaluation of the occulter-edge integral

        I(a, b, c) = int_a^b exp(i x^2) / (x^2 + c^2) dx ,      c > 0,

accurate from small (c << 1) to large (c^2 ~ 1e4 and beyond) Fresnel numbers.

Method (no adaptivity, no data-dependent branching -> vectorises on a GPU)
-------------------------------------------------------------------------
Split the interval at +-A0 (A0 = 4 where c < C0, else 0):

 * Outer pieces |x| >= A0  (the whole range when c >= C0):
       T(x) = int_x^inf e^{iy^2}/(y^2+c^2) dy      (x >= 0, tail integral)
   rotated onto the 45-degree ray y = x + s e^{i pi/4} (the poles +-ic are
   never enclosed for x >= 0), which gives a Gaussian-damped integrand
       P = x^2 + c^2 + sqrt2 x s,  Q = s^2 + sqrt2 x s,  phi = x^2 + sqrt2 x s + pi/4
       T = int_0^inf e^{-Q} (cos phi + i sin phi)(P - iQ)/(P^2 + Q^2) ds.
   P >= x^2 + c^2 >= max(A0^2, c^2): never near-singular.  The integrand
   is even in x, so negative pieces use T(|x|).

 * Inner piece |x| <= A0 (only when c < C0, where the Lorentzian is sharp):
       e^{ix^2} = e^{-ic^2} e^{iq},  q = x^2 + c^2,
       I_in = e^{-ic^2} [ (1/c)(atan(b/c) - atan(a/c)) + int h(q) dx ],
       h(q) = (e^{iq} - 1)/q = i e^{iq/2} sin(q/2)/(q/2)   (entire, smooth).
   The singular part is exact; the remainder is Gauss-Legendre.

Peak memory: O(N_points) per call (quadrature nodes are looped, not stacked).
Call free_gpu_memory() every so often when looping over many edges with CuPy.
"""
import numpy as np

try:
    import cupy as _cp

    def get_array_module(x):
        return _cp.get_array_module(x)

    def free_gpu_memory():
        """Return CuPy's pooled-but-unused memory to the driver (no-op on CPU)."""
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()

except ImportError:
    def get_array_module(x):
        return np

    def free_gpu_memory():
        pass

SQRT2 = 2.0 ** 0.5
_L = 46.0            # envelope cut-off exp(-46) ~ 1e-20
_ORDER = 48          # Gauss-Legendre nodes, rotated-ray tails
_INNER_ORDER = 64    # Gauss-Legendre nodes, smooth inner remainder
_A0 = 4.0
_C0 = 1.5

_gl_cache = {}


def _gl(order):
    if order not in _gl_cache:
        _gl_cache[order] = np.polynomial.legendre.leggauss(order)
    return _gl_cache[order]


def _smax(x, xp):
    # rationalised form: no cancellation at large x
    return 2.0 * _L / (SQRT2 * x + xp.sqrt(2.0 * x * x + 4.0 * _L))


def _tail(x, c2, order, xp):
    """T(x) = int_x^inf exp(i y^2)/(y^2 + c2) dy for x >= 0 (arrays, same shape)."""
    xi, wi = _gl(order)
    half = 0.5 * _smax(x, xp)
    x2 = x * x
    sx = SQRT2 * x
    P0 = x2 + c2
    Re = xp.zeros_like(x)
    Im = xp.zeros_like(x)
    for k in range(order):
        s = half * (xi[k] + 1.0)
        Q = s * s + sx * s
        P = P0 + sx * s
        phi = x2 + sx * s + xp.pi / 4.0
        f = xp.exp(-Q) * wi[k] / (P * P + Q * Q)
        Re += f * (P * xp.cos(phi) + Q * xp.sin(phi))
        Im += f * (P * xp.sin(phi) - Q * xp.cos(phi))
        del s, Q, P, phi, f
    out = half * (Re + 1j * Im)
    del Re, Im, half, x2, sx, P0
    return out


def _inner(ai, bi, c, c2, order, xp):
    """int_{ai}^{bi} e^{ix^2}/(x^2+c^2) dx via the arctan + smooth-remainder split."""
    xi, wi = _gl(order)
    L = bi - ai
    mid = 0.5 * (ai + bi)
    hl = 0.5 * L
    Re = xp.zeros_like(ai)
    Im = xp.zeros_like(ai)
    for k in range(order):
        x = mid + hl * xi[k]
        th = 0.5 * (x * x + c2)
        sinc = xp.sinc(th / xp.pi)          # sin(th)/th, stable at th -> 0
        Re += wi[k] * (-xp.sin(th) * sinc)  # h = i e^{i th} sinc(th)
        Im += wi[k] * (xp.cos(th) * sinc)
        del x, th, sinc
    rem = hl * (Re + 1j * Im)
    diff = xp.arctan2(c * L, c2 + ai * bi)  # atan(b/c) - atan(a/c), no cancellation
    out = (xp.cos(c2) - 1j * xp.sin(c2)) * (diff / c + rem)
    del Re, Im, rem, diff, L, mid, hl
    return out


def occulter_edge_integral(a, b, c, order=_ORDER, inner_order=_INNER_ORDER,
                           A0=_A0, C0=_C0):
    """I(a,b,c) = int_a^b e^{ix^2}/(x^2+c^2) dx, elementwise on broadcastable
    arrays a, b, c (c > 0).  Returns a complex array of the broadcast shape."""
    xp = get_array_module(a)
    a, b, c = xp.broadcast_arrays(xp.asarray(a, dtype=float),
                                  xp.asarray(b, dtype=float),
                                  xp.abs(xp.asarray(c, dtype=float)))
    c2 = c * c
    A = xp.where(c < C0, A0, 0.0)           # split point; 0 -> no inner piece

    ai = xp.minimum(xp.maximum(a, -A), A)
    bi = xp.minimum(xp.maximum(b, -A), A)
    out = _inner(ai, bi, c, c2, inner_order, xp)
    del ai, bi

    out += _tail(xp.maximum(a, A), c2, order, xp) - _tail(xp.maximum(b, A), c2, order, xp)
    out += _tail(-xp.minimum(b, -A), c2, order, xp) - _tail(-xp.minimum(a, -A), c2, order, xp)
    return out


def occulter_total_amplitude(a_edges, b_edges, c_edges, free_every=20, **kw):
    """Sum over edges (outer Python loop), O(N_points) memory.
    a_edges, b_edges, c_edges: sequences of per-edge (N_points,) arrays."""
    xp = get_array_module(a_edges[0])
    total = xp.zeros(xp.asarray(a_edges[0]).shape, dtype=complex)
    for i, (a, b, c) in enumerate(zip(a_edges, b_edges, c_edges)):
        total += occulter_edge_integral(a, b, c, **kw)
        if free_every and (i + 1) % free_every == 0:
            free_gpu_memory()
    if free_every:
        free_gpu_memory()
    return total
