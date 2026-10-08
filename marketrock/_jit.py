"""Numba if installed, plain Python otherwise (identical results, just slower)."""
try:  # pragma: no cover - import-time branch
    from numba import njit as _njit

    def jit(fn):
        return _njit(cache=True, nogil=True)(fn)

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    def jit(fn):
        return fn

    HAVE_NUMBA = False
