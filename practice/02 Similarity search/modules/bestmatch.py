import numpy as np
import math
import copy

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


@njit(cache=True)
def _lb_kim_fl(subs1: np.ndarray, subs2: np.ndarray) -> float:
    first = subs1[0] - subs2[0]
    last = subs1[-1] - subs2[-1]
    return first * first + last * last


@njit(cache=True)
def _lb_keogh_envelope(subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
    m = subs2.shape[0]
    radius = int(np.floor(r * m))
    lb = 0.0
    for i in range(subs1.shape[0]):
        start = i - radius
        if start < 0:
            start = 0
        stop = i + radius + 1
        if stop > m:
            stop = m
        upper = subs2[start]
        lower = subs2[start]
        for k in range(start + 1, stop):
            value = subs2[k]
            if value > upper:
                upper = value
            elif value < lower:
                lower = value
        point = subs1[i]
        if point > upper:
            diff = point - upper
            lb += diff * diff
        elif point < lower:
            diff = lower - point
            lb += diff * diff
    return lb


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using the naive algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of the naive algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf

        bestmatch = {
            'index' : [],
            'distance' : []
        }

        prepared_query = np.ascontiguousarray(
            z_normalize(query) if self.is_normalize else query, dtype=np.float64
        )
        if self.is_normalize:
            mean = ts_data.mean(axis=1, keepdims=True)
            std = ts_data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            candidates = np.ascontiguousarray((ts_data - mean) / std, dtype=np.float64)
        else:
            candidates = np.ascontiguousarray(ts_data, dtype=np.float64)
        for i in range(N):
            dist_profile[i] = DTW_distance(prepared_query, candidates[i], self.r)

        found = topK_match(dist_profile, excl_zone, self.topK)
        bestmatch['index'] = found['indices']
        bestmatch['distance'] = found['distances']
        bestmatch['indices'] = found['indices']
        bestmatch['distances'] = found['distances']

        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        # LB_KimFL: путь DTW обязан совместить первые и последние точки.
        lb_Kim = _lb_kim_fl(
            np.ascontiguousarray(subs1, dtype=np.float64),
            np.ascontiguousarray(subs2, dtype=np.float64),
        )

        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        # Конверт строится по subs2, точки subs1 вне конверта дают квадрат отклонения.
        lb_Keogh = _lb_keogh_envelope(
            np.ascontiguousarray(subs1, dtype=np.float64),
            np.ascontiguousarray(subs2, dtype=np.float64),
            float(r),
        )

        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf
        
        bestmatch = {
            'index' : [],
            'distance' : []
        }

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

        prepared_query = np.ascontiguousarray(
            z_normalize(query) if self.is_normalize else query, dtype=np.float64
        )
        if self.is_normalize:
            mean = ts_data.mean(axis=1, keepdims=True)
            std = ts_data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            candidates = np.ascontiguousarray((ts_data - mean) / std, dtype=np.float64)
        else:
            candidates = np.ascontiguousarray(ts_data, dtype=np.float64)
        exact_pairs = []

        def current_kth() -> float:
            if len(exact_pairs) < self.topK:
                return np.inf
            remaining = sorted(exact_pairs)
            chosen = []
            for distance, index in remaining:
                if any(abs(index - kept) <= excl_zone for kept in chosen):
                    continue
                chosen.append(index)
                if len(chosen) == self.topK:
                    return distance
            return np.inf

        bsf = np.inf
        radius = float(self.r)
        for i in range(N):
            candidate = candidates[i]

            lb_kim = _lb_kim_fl(prepared_query, candidate)
            if lb_kim >= bsf:
                self.lb_Kim_num += 1
                continue

            # LB_Keogh по конверту запроса: точки кандидата не должны выходить за него.
            lb_qc = _lb_keogh_envelope(candidate, prepared_query, radius)
            if lb_qc >= bsf:
                self.lb_KeoghQC_num += 1
                continue

            # Обратный конверт: точки запроса против конверта кандидата.
            lb_cq = _lb_keogh_envelope(prepared_query, candidate, radius)
            if lb_cq >= bsf:
                self.lb_KeoghCQ_num += 1
                continue

            distance = DTW_distance(prepared_query, candidate, self.r)
            dist_profile[i] = distance
            exact_pairs.append((distance, i))
            self.not_pruned_num += 1
            if distance < bsf:
                bsf = current_kth()

        found = topK_match(dist_profile, excl_zone, self.topK)
        bestmatch['index'] = found['indices']
        bestmatch['distance'] = found['distances']
        bestmatch['indices'] = found['indices']
        bestmatch['distances'] = found['distances']

        return bestmatch
