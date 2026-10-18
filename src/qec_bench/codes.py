"""CSS codes as parity-check matrix pairs, for code-capacity simulations.

Conventions: H_X rows are X-type stabilizers and H_Z rows are Z-type stabilizers.
An X error e is detected by the Z checks (syndrome H_Z e), and an X-type logical
is a vector in ker(H_Z) that is not in rowspace(H_X).
"""

from dataclasses import dataclass

import numpy as np
from ldpc import BpOsdDecoder


def gf2_rref(m: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2), and the pivot column of each nonzero row."""
    a = (np.array(m, dtype=np.uint8) % 2).copy()
    pivots = []
    row = 0
    for col in range(a.shape[1]):
        if row == a.shape[0]:
            break
        hits = np.nonzero(a[row:, col])[0]
        if len(hits) == 0:
            continue
        pivot = row + hits[0]
        a[[row, pivot]] = a[[pivot, row]]
        others = np.nonzero(a[:, col])[0]
        others = others[others != row]
        a[others] ^= a[row]
        pivots.append(col)
        row += 1
    return a[:row], pivots


def gf2_rank(m: np.ndarray) -> int:
    return len(gf2_rref(m)[1])


def gf2_nullspace(m: np.ndarray) -> np.ndarray:
    """Basis (as rows) of {v : m v = 0 mod 2}."""
    rref, pivots = gf2_rref(m)
    n = m.shape[1]
    free = [c for c in range(n) if c not in set(pivots)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        # Each pivot variable equals the sum of the free variables in its row.
        for r, p in enumerate(pivots):
            basis[i, p] = rref[r, f]
    return basis


def in_rowspace(v: np.ndarray, m: np.ndarray) -> bool:
    return gf2_rank(np.vstack([m, v])) == gf2_rank(m)


def gf2_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a square invertible matrix over GF(2)."""
    k = m.shape[0]
    rref, pivots = gf2_rref(np.hstack([m % 2, np.eye(k, dtype=np.uint8)]))
    if pivots[:k] != list(range(k)) or len(pivots) < k:
        raise ValueError("matrix is singular over GF(2)")
    return rref[:k, k:]


@dataclass(frozen=True)
class CSSCode:
    name: str
    hx: np.ndarray
    hz: np.ndarray

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def k(self) -> int:
        """Logical qubits from GF(2) ranks: k = n - rank H_X - rank H_Z."""
        return self.n - gf2_rank(self.hx) - gf2_rank(self.hz)


def rotated_surface_code(d: int) -> CSSCode:
    """Rotated planar surface code on a d x d grid of data qubits (index r*d + c).

    A plaquette sits at every grid corner (i, j), 0 <= i, j <= d, and touches the
    up-to-4 data qubits around it. Its type is set by the parity of i + j. Bulk
    plaquettes are weight 4. On the top/bottom edges only X-type weight-2
    plaquettes are kept, and on the left/right edges only Z-type ones, giving
    (d^2 - 1)/2 checks of each type.
    """
    hx_rows, hz_rows = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qubits = [
                r * d + c for r in (i - 1, i) for c in (j - 1, j) if 0 <= r < d and 0 <= c < d
            ]
            is_x = (i + j) % 2 == 0
            on_top_bottom = i in (0, d) and 0 < j < d
            on_left_right = j in (0, d) and 0 < i < d
            bulk = 0 < i < d and 0 < j < d
            if not (bulk or (on_top_bottom and is_x) or (on_left_right and not is_x)):
                continue
            row = np.zeros(d * d, dtype=np.uint8)
            row[qubits] = 1
            (hx_rows if is_x else hz_rows).append(row)
    return CSSCode(f"surface_d{d}", np.array(hx_rows), np.array(hz_rows))


def _cyclic_shift(size: int) -> np.ndarray:
    return np.roll(np.eye(size, dtype=np.uint8), 1, axis=1)


def bivariate_bicycle_code(
    l: int,  # noqa: E741 -- the paper's symbol (ell); renaming would change the keyword API
    m: int,
    a_terms: list[tuple[int, int]],
    b_terms: list[tuple[int, int]],
    name: str,
) -> CSSCode:
    """Bivariate bicycle code, Bravyi et al., arXiv:2308.07915 (Nature 627, 778 (2024)).

    Sec. 4, Eq. (1): x = S_l (x) I_m, y = I_l (x) S_m, with A and B sums of monomials.
    Sec. 4, Eq. (2): H_X = [A | B], H_Z = [B^T | A^T].
    Each term (a, b) stands for the monomial x^a y^b.
    """
    x = np.kron(_cyclic_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _cyclic_shift(m))

    def poly(terms: list[tuple[int, int]]) -> np.ndarray:
        total = np.zeros((l * m, l * m), dtype=np.int64)
        for a, b in terms:
            total += np.linalg.matrix_power(x.astype(np.int64), a) @ np.linalg.matrix_power(
                y.astype(np.int64), b
            )
        return (total % 2).astype(np.uint8)

    a_mat, b_mat = poly(a_terms), poly(b_terms)
    hx = np.hstack([a_mat, b_mat])
    hz = np.hstack([b_mat.T, a_mat.T])
    return CSSCode(name, hx, hz)


def gross_code() -> CSSCode:
    """[[144,12,12]] code, Bravyi et al. Table 3: l=12, m=6, A = x^3+y+y^2, B = y^3+x+x^2."""
    return bivariate_bicycle_code(
        12, 6, a_terms=[(3, 0), (0, 1), (0, 2)], b_terms=[(0, 3), (1, 0), (2, 0)], name="gross"
    )


def _independent_cosets(kernel: np.ndarray, stabilizers: np.ndarray, k: int) -> np.ndarray:
    """Pick k kernel vectors that are independent modulo rowspace(stabilizers)."""
    chosen = []
    current = stabilizers
    rank = gf2_rank(current)
    for v in kernel:
        trial = np.vstack([current, v])
        if gf2_rank(trial) > rank:
            chosen.append(v)
            current, rank = trial, rank + 1
            if len(chosen) == k:
                break
    return np.array(chosen, dtype=np.uint8)


def logical_operators(code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
    """(lx, lz), k rows each, with lx @ lz.T = I (mod 2).

    lz spans ker(H_X) modulo rowspace(H_Z): Z operators that commute with every
    X check but are not stabilizers. lx is the same with X and Z swapped.
    """
    k = code.k
    lz = _independent_cosets(gf2_nullspace(code.hx), code.hz, k)
    lx = _independent_cosets(gf2_nullspace(code.hz), code.hx, k)
    pairing = (lx.astype(np.int64) @ lz.T.astype(np.int64)) % 2
    lx = (gf2_inverse(pairing.astype(np.uint8)).astype(np.int64) @ lx) % 2
    return lx.astype(np.uint8), lz


def find_low_weight_x_logical(
    code: CSSCode, target: int, seed: int = 0, attempts: int = 200, osd_order: int = 10
) -> np.ndarray | None:
    """Search for an X-type logical of weight <= target. Upper-bound witness only.

    Each attempt picks a random nonzero combination z of the Z logicals and asks
    BP-OSD-CS for a low-weight v with H_Z v = 0 and z . v = 1. Such a v commutes
    with every Z check and anticommutes with a Z logical, so it is a nontrivial
    X logical. Random channel priors vary which low-weight solution OSD reaches.
    """
    rng = np.random.default_rng(seed)
    _lx, lz = logical_operators(code)
    syndrome = np.zeros(code.hz.shape[0] + 1, dtype=np.uint8)
    syndrome[-1] = 1
    best = None
    for _ in range(attempts):
        coeffs = rng.integers(0, 2, size=len(lz))
        if not coeffs.any():
            coeffs[rng.integers(len(lz))] = 1
        z = (coeffs @ lz) % 2
        decoder = BpOsdDecoder(
            np.vstack([code.hz, z]).astype(np.uint8),
            channel_probs=list(rng.uniform(0.01, 0.1, size=code.n)),
            bp_method="minimum_sum",
            max_iter=100,
            osd_method="osd_cs",
            osd_order=osd_order,
        )
        v = decoder.decode(syndrome).astype(np.uint8)
        if best is None or v.sum() < best.sum():
            best = v
        if best.sum() <= target:
            return best
    return None
