import numpy as np


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

    n = ts1.shape[0]
    m = ts2.shape[0]
    # r — доля длины ряда: ширина окна Сакоэ–Чиба. r = 1 не ограничивает путь.
    radius = max(abs(n - m), int(np.floor(r * max(n, m))))

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            local = (ts1[i - 1] - ts2[j - 1]) ** 2
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    dtw_dist = cost[n, m]

    return dtw_dist
