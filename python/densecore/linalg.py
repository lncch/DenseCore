"""torch.linalg (real dtypes): NumPy LAPACK forward in float64, analytic backward passes.

Each differentiable result is a custom autograd node (autograd.Function) whose backward is
the standard formula for that decomposition. Multi-output ops (qr, eigh, svd, lu, slogdet)
make one node per differentiable output; the VJP is linear in the output gradients, so the
contributions add up exactly.
"""
import collections
import math

import numpy as np

from .tensor import Tensor

_FLOAT = ("float32",)
_returns = {name: collections.namedtuple(name, fields) for name, fields in [
    ("slogdet", "sign logabsdet"), ("qr", "Q R"), ("eigh", "eigenvalues eigenvectors"),
    ("svd", "U S Vh"), ("lu", "P L U"), ("lu_factor", "LU pivots"), ("inv_ex", "inverse info"),
    ("solve_ex", "result info"), ("cholesky_ex", "L info"), ("lu_factor_ex", "LU pivots info"),
    ("triangular_solve", "solution cloned_coefficient"), ("legacy_svd", "U S V"),
    ("lstsq", "solution residuals rank singular_values")]}


def _np(t):
    return t._array().astype(np.float64) if isinstance(t, Tensor) else np.asarray(t, np.float64)


def _mT(x):
    return np.swapaxes(x, -1, -2)


def _custom(inputs, value, vjp):
    """A float32 result with a NumPy backward: vjp(upstream) -> one gradient (or None) per input."""
    from .function import Function
    value = np.asarray(value)
    tracked = [t for t in inputs if isinstance(t, Tensor) and t.dtype == "float32"]
    if len(tracked) != len(inputs) or not any(t.requires_grad for t in inputs):
        return Tensor(value.astype(np.float32))

    def forward(ctx, *tensors):
        return Tensor(value.astype(np.float32))

    def backward(ctx, upstream):
        grads = vjp(upstream._array().astype(np.float64))
        return tuple(None if g is None else Tensor(np.asarray(g).astype(np.float32)) for g in grads)

    node = type("LinalgFunction", (Function,), {"forward": staticmethod(forward), "backward": staticmethod(backward)})
    return node.apply(*inputs)


def _sum_to(gradient, shape):
    """Reduce a broadcast gradient back to an input's shape."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(0)
    for axis, n in enumerate(shape):
        if n == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis, keepdims=True)
    return gradient


def _square(a, name):
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise RuntimeError(f"{name}: A must be a batch of square matrices")


# ---- inverses and determinants -------------------------------------------------------

def inv(A):
    a = _np(A)
    _square(a, "linalg.inv")
    y = np.linalg.inv(a)
    return _custom([A], y, lambda g: [-_mT(y) @ g @ _mT(y)])


def inv_ex(A, *, check_errors=False):
    return _returns["inv_ex"](inv(A), Tensor(np.zeros(_np(A).shape[:-2], np.int32), dtype="int32"))


def _adjugate_transpose(a):
    """det(A) * inv(A)^T, computed from the SVD so singular matrices work."""
    u, s, vh = np.linalg.svd(a)
    n = a.shape[-1]
    if n == 0:
        return np.zeros_like(a)
    cofactor = np.stack([np.prod(np.delete(s, i, axis=-1), axis=-1) for i in range(n)], -1)
    sign = np.linalg.det(u) * np.linalg.det(vh)
    adj = _mT(vh) @ (cofactor[..., :, None] * _mT(u))       # adj(A)
    return _mT(adj) * sign[..., None, None]


def det(A):
    a = _np(A)
    _square(a, "linalg.det")
    return _custom([A], np.linalg.det(a), lambda g: [g[..., None, None] * _adjugate_transpose(a)])


def slogdet(A):
    a = _np(A)
    _square(a, "linalg.slogdet")
    sign, logabs = np.linalg.slogdet(a)
    with np.errstate(all="ignore"):
        logabs_node = _custom([A], logabs, lambda g: [g[..., None, None] * _mT(np.linalg.inv(a))])
    sign_node = _custom([A], sign, lambda g: [np.zeros_like(a)])  # on the graph, zero gradient
    return _returns["slogdet"](sign_node, logabs_node)


def logdet(A):
    a = _np(A)
    sign, logabs = np.linalg.slogdet(a)
    value = np.where(sign > 0, logabs, np.where(sign == 0, -np.inf, np.nan))
    with np.errstate(all="ignore"):
        return _custom([A], value, lambda g: [g[..., None, None] * _mT(np.linalg.inv(a))])


# ---- solves ------------------------------------------------------------------------------

def _vector_rhs(a, b):
    return b.ndim == 1 or (b.ndim >= 2 and b.shape == a.shape[:-1])


def solve(A, B, *, left=True):
    a, b = _np(A), _np(B)
    _square(a, "linalg.solve")
    if not left:
        # X A = B  <=>  A^T X^T = B^T
        out = solve(A.mT, B.mT if B.dim() > 1 else B, left=True)
        return out.mT if B.dim() > 1 else out
    vector = _vector_rhs(a, b)
    bb = b[..., None] if vector else b
    x = np.linalg.solve(np.broadcast_to(a, np.broadcast_shapes(a.shape[:-2], bb.shape[:-2]) + a.shape[-2:]), bb)

    def vjp(g):
        gg = g[..., None] if vector else g
        gb = np.linalg.solve(np.broadcast_to(_mT(a), x.shape[:-2] + a.shape[-2:]), gg)
        ga = -gb @ _mT(x)
        return [_sum_to(ga, a.shape), _sum_to(gb[..., 0] if vector else gb, b.shape)]
    return _custom([A, B], x[..., 0] if vector else x, vjp)


def solve_ex(A, B, *, left=True, check_errors=False):
    out = solve(A, B, left=left)
    return _returns["solve_ex"](out, Tensor(np.zeros(_np(A).shape[:-2], np.int32), dtype="int32"))


def _triangle(a, upper, unitriangular):
    t = np.triu(a) if upper else np.tril(a)
    if unitriangular:
        n = a.shape[-1]
        t = t * (1 - np.eye(n)) + np.eye(n)
    return t


def solve_triangular(A, B, *, upper, left=True, unitriangular=False):
    a, b = _np(A), _np(B)
    t = _triangle(a, upper, unitriangular)
    if left:
        x = np.linalg.solve(np.broadcast_to(t, np.broadcast_shapes(t.shape[:-2], b.shape[:-2]) + t.shape[-2:]), b)
    else:
        x = _mT(np.linalg.solve(np.broadcast_to(_mT(t), np.broadcast_shapes(t.shape[:-2], b.shape[:-2]) + t.shape[-2:]), _mT(b)))
    mask = np.triu(np.ones(a.shape[-2:])) if upper else np.tril(np.ones(a.shape[-2:]))
    if unitriangular:
        mask = mask * (1 - np.eye(a.shape[-1]))

    def vjp(g):
        if left:
            gb = np.linalg.solve(np.broadcast_to(_mT(t), x.shape[:-2] + t.shape[-2:]), g)
            ga = -gb @ _mT(x)
        else:
            gb = _mT(np.linalg.solve(np.broadcast_to(t, x.shape[:-2] + t.shape[-2:]), _mT(g)))
            ga = -_mT(x) @ gb
        return [_sum_to(ga * mask, a.shape), _sum_to(gb, b.shape)]
    return _custom([A, B], x, vjp)


def triangular_solve(b, A, upper=True, transpose=False, unitriangular=False):
    coefficient = A.mT if transpose else A
    out = solve_triangular(coefficient, b, upper=upper != transpose, unitriangular=unitriangular)
    return _returns["triangular_solve"](out, A.clone())


# ---- Cholesky --------------------------------------------------------------------------

def cholesky(A, *, upper=False):
    a = _np(A)
    _square(a, "linalg.cholesky")
    lower = np.linalg.cholesky(a)

    def vjp(g):
        gl = _mT(g) if upper else g
        phi = _mT(lower) @ gl
        phi = np.tril(phi)
        idx = np.arange(phi.shape[-1])
        phi[..., idx, idx] *= 0.5
        # L^{-T} phi L^{-1}, then symmetrize (PyTorch's cholesky_backward).
        tmp = np.linalg.solve(_mT(lower), phi)
        ga = _mT(np.linalg.solve(_mT(lower), _mT(tmp)))
        return [0.5 * (ga + _mT(ga))]
    return _custom([A], _mT(lower) if upper else lower, vjp)


def cholesky_ex(A, *, upper=False, check_errors=False):
    return _returns["cholesky_ex"](cholesky(A, upper=upper), Tensor(np.zeros(_np(A).shape[:-2], np.int32), dtype="int32"))


def cholesky_inverse(L, upper=False):
    product = L.mT @ L if upper else L @ L.mT
    return inv(product)


def cholesky_solve(B, L, upper=False):
    product = L.mT @ L if upper else L @ L.mT
    return solve(product, B)


# ---- QR ----------------------------------------------------------------------------------

def _copyltu(m):
    return np.tril(m, -1) + _mT(np.tril(m, -1)) + np.tril(np.triu(m))  # mirror the strict lower triangle


def _qr_backward(q, r, gq, gr):
    """Liao et al.: for m >= n, gA = (gQ + Q copyltu(M)) R^{-T}, M = R gR^T - gQ^T Q."""
    m, n = q.shape[-2], r.shape[-1]
    if m >= n:
        mm = r @ _mT(gr) - _mT(gq) @ q
        b = gq + q @ _copyltu(mm)
        return _mT(np.linalg.solve(r, _mT(b)))  # b R^{-T}
    # Wide case: A = Q [R1 R2]; X = A[:, :m] = Q R1, Y = A[:, m:].
    r1, gr1, gr2 = r[..., :, :m], gr[..., :, :m], gr[..., :, m:]
    y = q @ r[..., :, m:]
    gx = _qr_backward(q, r1, gq + y @ _mT(gr2), gr1)
    gy = q @ gr2
    return np.concatenate([gx, gy], -1)


def qr(A, mode="reduced"):
    a = _np(A)
    q, r = np.linalg.qr(a, mode="complete" if mode == "complete" else "reduced")
    if mode == "r":
        return _returns["qr"](Tensor(np.zeros((0,), np.float32)), _custom([A], r, lambda g: [None]))
    k = min(a.shape[-2], a.shape[-1])
    qq, rr = (q[..., :, :k], r[..., :k, :]) if mode == "complete" else (q, r)
    q_node = _custom([A], q, lambda g: [_qr_backward(qq, rr, g[..., :, :k], np.zeros_like(rr))])
    r_node = _custom([A], r, lambda g: [_qr_backward(qq, rr, np.zeros_like(qq), g[..., :k, :])])
    return _returns["qr"](q_node, r_node)


def geqrf(input):
    a = _np(input)
    h, tau = np.linalg.qr(a, mode="raw")
    return Tensor(_mT(h).astype(np.float32)), Tensor(tau.astype(np.float32))


# ---- symmetric eigendecomposition ------------------------------------------------------

def eigh(A, UPLO="L"):
    a = _np(A)
    _square(a, "linalg.eigh")
    sym = np.tril(a) + _mT(np.tril(a, -1)) if UPLO == "L" else np.triu(a) + _mT(np.triu(a, 1))
    values, vectors = np.linalg.eigh(sym)
    with np.errstate(all="ignore"):
        gap = values[..., None, :] - values[..., :, None]
        f = np.where(np.eye(values.shape[-1], dtype=bool), 0.0, 1.0 / gap)

    def values_vjp(g):
        return [vectors @ (g[..., :, None] * _mT(vectors))]

    def vectors_vjp(g):
        inner = _mT(vectors) @ g
        inner = 0.5 * (inner - _mT(inner))  # PyTorch projects onto the tangent space
        return [vectors @ (f * inner) @ _mT(vectors)]
    return _returns["eigh"](_custom([A], values, values_vjp), _custom([A], vectors, vectors_vjp))


def eigvalsh(A, UPLO="L"):
    return eigh(A, UPLO).eigenvalues


# ---- SVD -----------------------------------------------------------------------------------

def _svd_backward(u, s, vh, gu, gs, gvh):
    """Reduced-SVD backward (Townsend's formula, real case)."""
    v, gv = _mT(vh), None if gvh is None else _mT(gvh)
    k = s.shape[-1]
    eye = np.eye(k, dtype=bool)
    with np.errstate(all="ignore"):
        s2 = s * s
        f = np.where(eye, 0.0, 1.0 / (s2[..., None, :] - s2[..., :, None]))
        inv_s = np.where(s != 0, 1.0 / s, 0.0)
    smat = s[..., None, :] * np.eye(k)
    middle = np.zeros(s.shape[:-1] + (k, k))
    out = np.zeros(u.shape[:-1] + (v.shape[-2],))
    if gs is not None:
        middle = middle + gs[..., None, :] * np.eye(k)
    if gu is not None:
        j = _mT(u) @ gu
        middle = middle + (f * (j - _mT(j))) @ smat
        projector = np.eye(u.shape[-2]) - u @ _mT(u)
        out = out + projector @ gu @ (inv_s[..., :, None] * vh)
    if gv is not None:
        k2 = _mT(v) @ gv
        middle = middle + smat @ (f * (k2 - _mT(k2)))
        projector = np.eye(v.shape[-2]) - v @ _mT(v)
        out = out + (u * inv_s[..., None, :]) @ _mT(gv) @ projector
    return u @ middle @ vh + out


def svd(A, full_matrices=True, *, driver=None):
    a = _np(A)
    u, s, vh = np.linalg.svd(a, full_matrices=full_matrices)
    k = s.shape[-1]
    uk, vhk = u[..., :, :k], vh[..., :k, :]
    u_node = _custom([A], u, lambda g: [_svd_backward(uk, s, vhk, g[..., :, :k], None, None)])
    s_node = _custom([A], s, lambda g: [_svd_backward(uk, s, vhk, None, g, None)])
    vh_node = _custom([A], vh, lambda g: [_svd_backward(uk, s, vhk, None, None, g[..., :k, :])])
    return _returns["svd"](u_node, s_node, vh_node)


def svdvals(A, *, driver=None):
    return svd(A, full_matrices=False).S


def legacy_svd(input, some=True, compute_uv=True):
    u, s, vh = svd(input, full_matrices=not some)
    return _returns["legacy_svd"](u, s, vh.mT)


# ---- LU --------------------------------------------------------------------------------------

def _lu_numpy(a):
    """Partial-pivoting LU, batched: returns (LU packed, pivots 1-based, P, L, U)."""
    a = np.array(a, np.float64)
    *batch, m, n = a.shape
    k = min(m, n)
    lu = a.copy()
    pivots = np.zeros(tuple(batch) + (k,), np.int32)
    perm = np.broadcast_to(np.arange(m), tuple(batch) + (m,)).copy()
    for index in np.ndindex(*batch) if batch else [()]:
        block = lu[index]
        order = perm[index]
        for j in range(k):
            p = j + int(np.argmax(np.abs(block[j:, j])))
            pivots[index + (j,)] = p + 1
            if p != j:
                block[[j, p]] = block[[p, j]]
                order[[j, p]] = order[[p, j]]
            if block[j, j] != 0:
                block[j + 1:, j] /= block[j, j]
                block[j + 1:, j + 1:] -= np.outer(block[j + 1:, j], block[j, j + 1:])
    lower = np.tril(lu[..., :, :k], -1) + np.eye(m, k)
    upper = np.triu(lu[..., :k, :])
    p_matrix = np.zeros(tuple(batch) + (m, m))
    for index in np.ndindex(*batch) if batch else [()]:
        p_matrix[index][perm[index], np.arange(m)] = 1.0
    return lu, pivots, p_matrix, lower, upper


def _lu_square_backward(lower, upper, gl, gu):
    """P^T gA for square LU: L^{-T} (tril(L^T gL, -1) + triu(gU U^T)) U^{-T}."""
    phi = np.tril(_mT(lower) @ gl, -1) + np.triu(gu @ _mT(upper))
    left = np.linalg.solve(_mT(lower), phi)
    return _mT(np.linalg.solve(upper, _mT(left)))


def _lu_backward(p, lower, upper, gl, gu):
    """LU backward for square, wide and tall matrices (A = P L U)."""
    gl = np.zeros_like(lower) if gl is None else gl
    gu = np.zeros_like(upper) if gu is None else gu
    m, k, n = lower.shape[-2], lower.shape[-1], upper.shape[-1]
    if m == n:
        return p @ _lu_square_backward(lower, upper, gl, gu)
    if m < n:
        # A = P L [U1 U2]; U2 = L^{-1} P^T A2 adds -L^{-T} gU2 U2^T to gL.
        u1, u2, gu1, gu2 = upper[..., :, :m], upper[..., :, m:], gu[..., :, :m], gu[..., :, m:]
        solved = np.linalg.solve(_mT(lower), gu2)              # L^{-T} gU2
        g1 = _lu_square_backward(lower, u1, gl - solved @ _mT(u2), gu1)
        return p @ np.concatenate([g1, solved], -1)
    # Tall: P^T A = [L1; L2] U; L2 = B2 U^{-1} adds -L2^T gL2 U^{-T} to gU.
    l1, l2, gl1, gl2 = lower[..., :n, :], lower[..., n:, :], gl[..., :n, :], gl[..., n:, :]
    gb2 = _mT(np.linalg.solve(upper, _mT(gl2)))                 # gL2 U^{-T}
    g1 = _lu_square_backward(l1, upper, gl1, gu - _mT(l2) @ gb2)
    return p @ np.concatenate([g1, gb2], -2)


def lu(A, *, pivot=True):
    a = _np(A)
    _, _, p, lower, upper = _lu_numpy(a)
    l_node = _custom([A], lower, lambda g: [_lu_backward(p, lower, upper, g, None)])
    u_node = _custom([A], upper, lambda g: [_lu_backward(p, lower, upper, None, g)])
    return _returns["lu"](Tensor(p.astype(np.float32)), l_node, u_node)


def lu_factor(A, *, pivot=True):
    a = _np(A)
    packed, pivots, p, lower, upper = _lu_numpy(a)
    k = min(a.shape[-2:])

    def vjp(g):
        return [_lu_backward(p, lower, upper, np.tril(g[..., :, :k], -1), np.triu(g[..., :k, :]))]
    return _returns["lu_factor"](_custom([A], packed, vjp), Tensor(pivots, dtype="int32"))


def lu_factor_ex(A, *, pivot=True, check_errors=False):
    packed, pivots = lu_factor(A, pivot=pivot)
    return _returns["lu_factor_ex"](packed, pivots, Tensor(np.zeros(_np(A).shape[:-2], np.int32), dtype="int32"))


def _permutation_from_pivots(pivots, m):
    piv = np.asarray(pivots._array() if isinstance(pivots, Tensor) else pivots, np.int64) - 1
    *batch, k = piv.shape
    p = np.zeros(tuple(batch) + (m, m))
    for index in np.ndindex(*batch) if batch else [()]:
        order = np.arange(m)
        for j in range(k):
            order[[j, piv[index + (j,)]]] = order[[piv[index + (j,)], j]]
        p[index][order, np.arange(m)] = 1.0
    return p


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    m, n = LU_data.shape[-2:]
    k = min(m, n)
    keep_lower = np.tril(np.ones((m, k), bool), -1)
    lower = LU_data[..., :, :k].masked_fill(~keep_lower, 0.0) + Tensor(np.eye(m, k, dtype=np.float32))
    upper = LU_data[..., :k, :].masked_fill(~np.triu(np.ones((k, n), bool)), 0.0)
    p = Tensor(_permutation_from_pivots(LU_pivots, m).astype(np.float32))
    return _returns["lu"](p, lower, upper)


def lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    p, lower, upper = lu_unpack(LU, pivots)
    a = p @ lower @ upper
    a = a.mT if adjoint else a
    return solve(a, B, left=left)


def lu_solve_legacy(b, LU_data, LU_pivots):
    return lu_solve(LU_data, LU_pivots, b)


# ---- pseudo-inverse, least squares, rank ------------------------------------------------

def _default_rtol(a):
    return np.finfo(np.float32).eps * max(a.shape[-2:])


def pinv(A, *, atol=None, rtol=None, hermitian=False):
    """Moore-Penrose inverse with PyTorch's cutoffs; the backward uses the direct formula
    (no singular-value gaps), so singular inputs work:
    gA = -P^T g P^T + (I - A P) g^T P P^T + P^T P g^T (I - P A)."""
    a = _np(A)
    if hermitian:  # PyTorch reads only the lower triangle
        a = np.tril(a) + _mT(np.tril(a, -1))
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if atol is None and rtol is None:
        rtol = _default_rtol(a)
    atol = 0.0 if atol is None else _np(atol)
    rtol = 0.0 if rtol is None else _np(rtol)
    largest = s.max(axis=-1, initial=0.0) if s.size else np.zeros(a.shape[:-2])
    cutoff = np.maximum(atol, rtol * largest)[..., None] if s.size else 0.0
    inv_s = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
    p = _mT(vh) @ (inv_s[..., :, None] * _mT(u))

    original = _np(A)  # PyTorch's pinv backward uses the input as given, both triangles

    def vjp(g):
        m, n = a.shape[-2:]
        pt, gt = _mT(p), _mT(g)
        left = np.eye(m) - original @ p
        right = np.eye(n) - p @ original
        return [-pt @ g @ pt + left @ gt @ p @ pt + pt @ p @ gt @ right]
    return _custom([A], p, vjp)


def pinverse(input, rcond=1e-15):
    return pinv(input, rtol=rcond)


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, tol=None):
    a = _np(A)
    s = np.linalg.svd(a, compute_uv=False)
    if tol is not None:
        atol = tol
    if atol is None and rtol is None:
        rtol = _default_rtol(a)
    elif rtol is None:
        rtol = 0.0
    atol = 0.0 if atol is None else atol
    cutoff = np.maximum(_np(atol) if isinstance(atol, Tensor) else atol,
                        (_np(rtol) if isinstance(rtol, Tensor) else rtol) * s.max(axis=-1, initial=0.0))
    return Tensor((s > np.asarray(cutoff)[..., None]).sum(-1), dtype="int64")


def lstsq(A, B, rcond=None, *, driver=None):
    """Solution through pinv (so it is differentiable); the other outputs follow PyTorch's
    per-driver shapes (CPU default driver 'gelsy')."""
    driver = driver or "gelsy"
    a, b = _np(A), _np(B)
    vector = b.ndim == a.ndim - 1
    rhs = B.unsqueeze(-1) if vector else B
    solution = pinv(A, rtol=rcond if rcond is not None else _default_rtol(a)) @ rhs
    m, n = a.shape[-2:]
    batch = np.broadcast_shapes(a.shape[:-2], (rhs.shape[:-2]))
    rank = int(np.linalg.matrix_rank(a.reshape(-1, m, n)[0])) if a.size else 0
    empty = A.reshape(A.numel())[0:0].reshape(*batch, 0) if A.numel() else Tensor(np.zeros(batch + (0,), np.float32))
    if driver != "gelsy" and m > n and rank == n:
        residual = A @ solution - rhs
        residuals = (residual * residual).sum(-2)
    else:
        residuals = empty
    ranks = matrix_rank(A) if driver != "gels" else Tensor(np.zeros((0,), np.int64), dtype="int64")
    singular = svdvals(A) if driver in ("gelsd", "gelss") else empty
    return _returns["lstsq"](solution.squeeze(-1) if vector else solution, residuals, ranks, singular)


# ---- norms -------------------------------------------------------------------------------

def vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None):
    source = x if dtype is None else x.to(dtype)
    return source.norm(ord, dim, keepdim)


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None):
    x = A if dtype is None else A.to(dtype)
    rank = len(x.shape)
    rows, cols = (d % rank for d in dim)
    moved = x.movedim((rows, cols), (-2, -1))
    if ord == "fro":
        out = (moved * moved).sum((-2, -1)).sqrt()
    elif ord == "nuc":
        out = svdvals(moved).sum(-1)
    elif ord in (2, -2):
        values = svdvals(moved)
        out = values.amax(-1) if ord == 2 else values.amin(-1)
    elif ord in (1, -1):
        sums = moved.abs().sum(-2)
        out = sums.amax(-1) if ord == 1 else sums.amin(-1)
    elif ord in (float("inf"), float("-inf")):
        sums = moved.abs().sum(-1)
        out = sums.amax(-1) if ord > 0 else sums.amin(-1)
    else:
        raise ValueError(f"unsupported matrix norm order {ord!r}")
    if keepdim:
        for axis in sorted((rows, cols)):
            out = out.unsqueeze(axis)
    return out


def norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    if isinstance(dim, (tuple, list)) and len(dim) == 2:
        return matrix_norm(A, "fro" if ord is None else ord, dim, keepdim, dtype=dtype)
    if dim is None and ord is not None and len(A.shape) == 2:
        return matrix_norm(A, ord, (-2, -1), keepdim, dtype=dtype)
    return vector_norm(A, 2 if ord is None else ord, dim, keepdim, dtype=dtype)


def cond(A, p=None):
    if p in (None, 2, -2):
        values = svdvals(A)
        big, small = values.amax(-1), values.amin(-1)
        return big / small if p in (None, 2) else small / big
    return matrix_norm(A, p) * matrix_norm(inv(A), p)


# ---- matrix functions and products ---------------------------------------------------------

def matrix_power(A, n):
    from . import eye
    if n < 0:
        return matrix_power(inv(A), -n)
    size = A.shape[-1]
    result = None
    base = A
    while n:
        if n & 1:
            result = base if result is None else result @ base
        n >>= 1
        if n:
            base = base @ base
    if result is None:
        return A * 0.0 + eye(size)  # identity, still on A's graph (zero gradient) like PyTorch
    return result


_PADE13 = [64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
           129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
           40840800., 960960., 16380., 182., 1.]


def _expm(a):
    """Matrix exponential, Pade 13 with scaling and squaring (Higham 2005), batched."""
    a = np.asarray(a, np.float64)
    n = a.shape[-1]
    norm = np.abs(a).sum(-2).max(-1, initial=0.0)
    squarings = np.maximum(0, np.ceil(np.log2(np.maximum(norm, 1e-300) / 5.371920351148152))).astype(int)
    scaled = a / (2.0 ** squarings)[..., None, None]
    ident = np.broadcast_to(np.eye(n), a.shape)
    b = _PADE13
    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = scaled @ (a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2) + b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * ident)
    v = a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2) + b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * ident
    result = np.linalg.solve(v - u, v + u)
    for index in np.ndindex(*a.shape[:-2]) if a.ndim > 2 else [()]:
        for _ in range(int(squarings[index]) if np.ndim(squarings) else int(squarings)):
            result[index] = result[index] @ result[index]
    return result


def matrix_exp(A):
    a = _np(A)

    def vjp(g):
        # d/dA <g, exp(A)>: the upper-right block of exp([[A^T, g], [0, A^T]]).
        n = a.shape[-1]
        block = np.zeros(a.shape[:-2] + (2 * n, 2 * n))
        block[..., :n, :n] = _mT(a)
        block[..., n:, n:] = _mT(a)
        block[..., :n, n:] = g
        return [_expm(block)[..., :n, n:]]
    return _custom([A], _expm(a), vjp)


def matrix_sqrth(A):
    if A.numel() == 0:
        return A.clone()
    values, vectors = eigh(A)
    return vectors @ (values.sqrt().unsqueeze(-1) * vectors.mT)


def polar(A):
    u, s, vh = svd(A, full_matrices=False)
    return u @ vh, vh.mT @ (s.unsqueeze(-1) * vh)


def multi_dot(tensors):
    tensors = list(tensors)
    first_vector, last_vector = len(tensors[0].shape) == 1, len(tensors[-1].shape) == 1
    if first_vector:
        tensors[0] = tensors[0].unsqueeze(0)
    if last_vector:
        tensors[-1] = tensors[-1].unsqueeze(-1)
    out = tensors[0]
    for t in tensors[1:]:
        out = out @ t
    if first_vector:
        out = out.squeeze(0)
    if last_vector:
        out = out.squeeze(-1)
    return out


def vecdot(x, y, *, dim=-1):
    return (x * y).sum(dim)


def cross(input, other, *, dim=-1):
    return input.cross(other, dim)


def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    return A.diagonal(offset, dim1, dim2)


def vander(x, N=None):
    from . import stack
    x = x.reshape(1) if not x.shape else x
    count = x.shape[-1] if N is None else N
    if count == 0:
        return x.unsqueeze(-1)[..., 0:0]
    columns, power = [x * 0.0 + 1.0], None
    for _ in range(1, count):
        power = x if power is None else power * x
        columns.append(power)
    return stack(columns, -1)


def tensorinv(A, ind=2):
    left = int(np.prod(A.shape[:ind]))
    return inv(A.reshape(left, -1)).reshape(*A.shape[ind:], *A.shape[:ind])


def tensorsolve(A, B, dims=None):
    if dims is not None:
        rank = len(A.shape)
        dims = [d % rank for d in dims]
        A = A.movedim(dims, list(range(rank - len(dims), rank)))
    rest = A.shape[len(B.shape):]
    solution = solve(A.reshape(B.numel(), -1), B.reshape(-1))
    return solution.reshape(*rest)


def _reflector_product(A, tau, columns):
    """H_1 H_2 ... H_k with H_i = I - tau_i v_i v_i^T, v_i = [0, ..., 1, A[i+1:, i]]."""
    from . import eye
    m = A.shape[-2]
    k = tau.shape[-1]
    lower = np.tril(np.ones((m, A.shape[-1]), bool), -1)
    vectors = A.masked_fill(~lower, 0.0) + eye(m, A.shape[-1])
    q = eye(m).expand(*A.shape[:-2], m, m) * 1.0
    for i in range(k):
        v = vectors[..., :, i:i + 1]
        q = q - (q @ v) @ v.mT * tau[..., i:i + 1].unsqueeze(-1)
    return q[..., :, :columns]


def householder_product(A, tau):
    return _reflector_product(A, tau, A.shape[-1])


def ormqr(input, tau, other, left=True, transpose=False):
    q = _reflector_product(input, tau, input.shape[-2])
    q = q.mT if transpose else q
    return q @ other if left else other @ q


# ---- general (non-symmetric) eigendecomposition: complex results, no gradients ----------

def eig(A):
    a = _np(A)
    values, vectors = np.linalg.eig(a)
    dtype = "complex128" if A.dtype == "float64" else "complex64"
    return collections.namedtuple("eig", "eigenvalues eigenvectors")(Tensor(values, dtype=dtype), Tensor(vectors, dtype=dtype))


def eigvals(A):
    return eig(A).eigenvalues
