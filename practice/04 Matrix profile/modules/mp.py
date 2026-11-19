import numpy as np

import stumpy
from stumpy import config


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """
    ts1 = np.asarray(ts1, dtype=float).reshape(-1)

    if exclusion_zone is not None and exclusion_zone > 0:
        # excl_zone = ceil(m / DENOM)  →  DENOM ≈ m / excl_zone
        config.STUMPY_EXCL_ZONE_DENOM = max(m / float(exclusion_zone), 1e-12)
    else:
        config.STUMPY_EXCL_ZONE_DENOM = 4

    if ts2 is None:
        mp = stumpy.stump(ts1, m)
    else:
        ts2 = np.asarray(ts2, dtype=float).reshape(-1)
        # AB-join: nearest neighbors of ts1 windows live in ts2
        mp = stumpy.stump(ts1, m, ts2, ignore_trivial=False)

    return {
        'mp': np.asarray(mp[:, 0], dtype=float),
        'mpi': np.asarray(mp[:, 1], dtype=int),
        'm': m,
        'excl_zone': exclusion_zone,
        'data': {'ts1': ts1, 'ts2': ts2},
    }
