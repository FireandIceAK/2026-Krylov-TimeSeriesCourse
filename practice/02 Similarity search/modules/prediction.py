import numpy as np
import math

from modules.bestmatch import UCR_DTW, topK_match
import mass_ts as mts


default_match_alg_params = {
    'UCR-DTW': {
        'topK': 3,
        'r': 0.05,
        'excl_zone_frac': 1,
        'is_normalize': True
    },
    'MASS': {
        'topK': 3,
        'excl_zone_frac': 1
    }
}


class BestMatchPredictor:
    """
    Predictor based on best match algorithm
    """

    def __init__(self, h: int = 1, match_alg: str = 'UCR-DTW', match_alg_params: dict | None = None, aggr_func: str = 'average') -> None:
        """ 
        Constructor of class BestMatchPredictor

        Parameters
        ----------    
        h: prediction horizon
        match_algorithm: name of the best match algorithm
        match_algorithm_params: input parameters for the best match algorithm
        aggr_func: aggregate function
        """

        self.h: int = h
        self.match_alg: str = match_alg
        self.match_alg_params: dict | None = default_match_alg_params[match_alg].copy()
        if match_alg_params is not None:
            self.match_alg_params.update(match_alg_params)
        self.agg_func: str = aggr_func


    def _calculate_predict_values(self, topK_subs_predict_values: np.array) -> np.ndarray:
        """
        Calculate the future values of the time series using the aggregate function

        Parameters
        ----------
        topK_subs_predict_values: values of time series, which are located after topK subsequences

        Returns
        -------
        predict_values: prediction values
        """

        match self.agg_func:
            case 'average':
                predict_values = np.mean(topK_subs_predict_values, axis=0).round()
            case 'median':
                predict_values = np.median(topK_subs_predict_values, axis=0).round()
            case _:
                raise NotImplementedError
        
        return predict_values


    def predict(self, ts: np.ndarray, query: np.ndarray) -> np.array:
        """
        Predict time series at future horizon
        
        Parameters
        ----------
        ts: time series
        query: query, shorter than time series

        Returns
        -------
        predict_values: prediction values
        """

        predict_values = np.zeros((self.h,))
        query_len = len(query)
        params = self.match_alg_params

        # Ищем только в переданном ряде. h точек после позднего совпадения
        # лежат уже в запросе, поэтому продолжение читаем из склейки ряда и запроса.
        if self.match_alg == 'UCR-DTW':
            finder = UCR_DTW(params['excl_zone_frac'], params['topK'], params['is_normalize'], params['r'])
            found = finder.perform(ts, query)
            indices = found['indices']
        elif self.match_alg == 'MASS':
            # mass2 совпадает с полным перебором z-нормализованной ED и имеет нужную длину.
            # mass() в mass-ts 0.1.4 короче на один элемент.
            distance_profile = np.real(mts.mass2(ts, query)).astype(np.float64)
            excl_zone = math.ceil(query_len * params['excl_zone_frac'])
            indices = topK_match(distance_profile, excl_zone, params['topK'])['indices']
        else:
            raise NotImplementedError(self.match_alg)

        extended = np.concatenate([np.asarray(ts, dtype=np.float64), np.asarray(query, dtype=np.float64)])
        futures = []
        for idx in indices:
            start = int(idx) + query_len
            stop = start + self.h
            if stop <= len(extended):
                futures.append(extended[start:stop])

        if futures:
            predict_values = self._calculate_predict_values(np.vstack(futures))

        return predict_values
