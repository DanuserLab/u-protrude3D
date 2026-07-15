import numpy as np
import igl


def local_max_peaks_sparse(V, F, values, n_hops=1, strict=True):
    """Detect local maxima on a mesh via sparse adjacency traversal.

    For each vertex v, checks whether values[v] exceeds the maximum over its
    k-hop neighbourhood (excluding v itself).

    Parameters
    ----------
    V : (N, 3) float array – vertex positions
    F : (M, 3) int array – face indices
    values : (N,) float array – scalar field
    n_hops : int – neighbourhood radius in graph hops
    strict : bool – if True use >, else use >=

    Returns
    -------
    is_peak : (N,) bool array
    """
    values = np.asarray(values, dtype=float)
    n = V.shape[0]
    A = igl.adjacency_matrix(F).astype(bool)

    A_hop = A.copy()
    for _ in range(n_hops - 1):
        A_hop = (A_hop @ A).astype(bool)
    A_hop = A_hop.tocsr()

    neighbour_max = np.full(n, -np.inf)
    for v in range(n):
        start, end = A_hop.indptr[v], A_hop.indptr[v + 1]
        nbrs = A_hop.indices[start:end]
        if len(nbrs):
            neighbour_max[v] = values[nbrs].max()

    return values > neighbour_max if strict else values >= neighbour_max


def suppress_nearby_peaks(V, F, is_peak, values, suppression_hops=3):
    """Non-maximum suppression: among nearby peaks keep only the highest.

    Parameters
    ----------
    V : (N, 3) float array
    F : (M, 3) int array
    is_peak : (N,) bool array – initial peak mask
    values : (N,) float array
    suppression_hops : int – suppression radius in hops

    Returns
    -------
    suppressed : (N,) bool array – thinned peak mask
    """
    adj = igl.adjacency_list(F)
    peak_indices = np.where(is_peak)[0]
    suppressed = is_peak.copy()

    def _khop_set(v, hops):
        frontier = set(adj[v])
        visited = set(frontier)
        for _ in range(hops - 1):
            nxt = {w for u in frontier for w in adj[u] if w not in visited}
            visited |= nxt
            frontier = nxt
        return visited

    for v in np.argsort(values[peak_indices]):
        vi = peak_indices[v]
        if not suppressed[vi]:
            continue
        for u in _khop_set(vi, suppression_hops):
            if suppressed[u] and u != vi and values[u] < values[vi]:
                suppressed[u] = False

    return suppressed
