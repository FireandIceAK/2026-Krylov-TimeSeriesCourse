import numpy as np

def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: z-normalize or not subsequences before computing distances

    Returns
    -------
    dist_profile: distance profile between query and the subsequences of the time series
    """

    ts = np.asarray(ts, dtype=np.float64).ravel()
    query = np.asarray(query, dtype=np.float64).ravel()

    n = len(ts)
    m = len(query)
    N = n - m + 1

    dist_profile = np.empty(N, dtype=np.float64)
    # Считаем все окна. Чанки не меняют полный перебор, только объём памяти.
    chunk = 8192

    if is_normalize:
        mu_q = np.mean(query)
        sigma_q = np.sqrt(np.mean(query ** 2) - mu_q ** 2)
        for start in range(0, N, chunk):
            stop = min(N, start + chunk)
            windows = np.lib.stride_tricks.sliding_window_view(ts[start:stop + m - 1], m)
            mu_w = windows.mean(axis=1)
            sigma_w = np.sqrt(np.mean(windows ** 2, axis=1) - mu_w ** 2)
            dot = windows @ query
            corr = (dot - m * mu_w * mu_q) / (m * sigma_w * sigma_q)
            dist_profile[start:stop] = np.sqrt(np.abs(2 * m * (1.0 - corr)))
    else:
        for start in range(0, N, chunk):
            stop = min(N, start + chunk)
            windows = np.lib.stride_tricks.sliding_window_view(ts[start:stop + m - 1], m)
            dist_profile[start:stop] = np.sqrt(np.sum((windows - query) ** 2, axis=1))

    return dist_profile
