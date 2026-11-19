import numpy as np

from modules.utils import apply_exclusion_zone


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = np.array(matrix_profile['mp'], dtype=float, copy=True)
    mpi = np.asarray(matrix_profile['mpi'], dtype=int)
    excl_zone = matrix_profile.get('excl_zone')
    if excl_zone is None:
        excl_zone = int(np.ceil(matrix_profile['m'] / 2))

    for _ in range(top_k):
        if not np.any(np.isfinite(mp)):
            break
        idx = int(np.nanargmax(mp))
        dist = float(mp[idx])
        if not np.isfinite(dist):
            break
        nn_idx = int(mpi[idx])
        discords_idx.append(idx)
        discords_dist.append(dist)
        discords_nn_idx.append(nn_idx)
        apply_exclusion_zone(mp, idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx,
    }
