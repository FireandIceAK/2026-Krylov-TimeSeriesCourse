import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = ts1.shape[0]
    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)
    sigma1 = np.sqrt(np.mean(ts1 ** 2) - mu1 ** 2)
    sigma2 = np.sqrt(np.mean(ts2 ** 2) - mu2 ** 2)
    dot = np.dot(ts1, ts2)
    corr = (dot - n * mu1 * mu2) / (n * sigma1 * sigma2)
    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - corr)))

    return norm_ed_dist


@njit(cache=True)
def _dtw_sakoe_chiba(ts1: np.ndarray, ts2: np.ndarray, r: float) -> float:
    """DTW со стоимостью (разность)^2 и окном Сакоэ–Чиба ширины r * длина."""

    n = ts1.shape[0]
    m = ts2.shape[0]
    longest = n if n > m else m
    radius = abs(n - m)
    window = int(np.floor(r * longest))
    if window > radius:
        radius = window

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = i - radius
        if j_start < 1:
            j_start = 1
        j_end = i + radius
        if j_end > m:
            j_end = m
        for j in range(j_start, j_end + 1):
            diff = ts1[i - 1] - ts2[j - 1]
            local = diff * diff
            best = cost[i - 1, j]
            if cost[i, j - 1] < best:
                best = cost[i, j - 1]
            if cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
            cost[i, j] = local + best

    return cost[n, m]


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    ts1 = np.ascontiguousarray(ts1, dtype=np.float64)
    ts2 = np.ascontiguousarray(ts2, dtype=np.float64)
    dtw_dist = _dtw_sakoe_chiba(ts1, ts2, float(r))

    return dtw_dist
