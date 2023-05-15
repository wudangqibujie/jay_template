from abc import ABC, abstractmethod
from typing import Any, Generator, List, Mapping, Optional, Sequence, Union
import numpy as np
from tsp.dataset.timeseries import JayTimeSeries

class BaseTransformer(ABC):
    def __init__(self,
                 name='BaseTransformer',
                 n_jobs=1,
                 verbose=False,
                 parallel_params=None,
                 mask_components=False,):
        self._fixed_params = vars(self).copy()
        if parallel_params and not isinstance(parallel_params, Sequence):
            parallel_params = self._fixed_params.keys()
        elif not parallel_params:
            parallel_params = tuple()
        self._parallel_params = parallel_params
        self._mask_components = mask_components
        self._name = name
        self._n_jobs = n_jobs
        self._verbose = verbose

    def set_verbosr(self, value):
        self._verbose = value

    def set_n_jobs(self, value):
        self._n_jobs = value

    @staticmethod
    @abstractmethod
    def tf_transform(series, params):
        pass

    @staticmethod
    def apply_component_mask(series, component_mask, return_ts=False):
        if component_mask is None:
            masked = series.copy() if return_ts else series.all_values()
        else:
            masked = series.all_values(copy=False)[:, component_mask, :]
            if return_ts:
                pass


    def transform(self,
                  series,
                  *args,
                  component_mask=None,
                  **kwargs):
        desc = f"Transforming {self._name}..."
        if isinstance(series, JayTimeSeries):
            input_series = [series]
            data = [series]
        else:
            input_series = series
            data = series

        if self._mask_components:
            pass


