import numpy as np

from modules.utils import apply_exclusion_zone


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = np.array(matrix_profile['mp'], dtype=float, copy=True)
    mpi = np.asarray(matrix_profile['mpi'], dtype=int)
    excl_zone = matrix_profile.get('excl_zone')
    if excl_zone is None:
        excl_zone = int(np.ceil(matrix_profile['m'] / 2))

    for _ in range(top_k):
        if not np.any(np.isfinite(mp)):
            break
        left_idx = int(np.nanargmin(mp))
        dist = float(mp[left_idx])
        if not np.isfinite(dist):
            break
        right_idx = int(mpi[left_idx])
        motifs_idx.append((left_idx, right_idx))
        motifs_dist.append(dist)
        # remove trivial matches around both motif occurrences
        apply_exclusion_zone(mp, left_idx, excl_zone, np.inf)
        if 0 <= right_idx < len(mp):
            apply_exclusion_zone(mp, right_idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist,
    }
