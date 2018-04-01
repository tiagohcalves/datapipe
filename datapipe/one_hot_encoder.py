"""
One Hot Encoder class.

Thanks to André Costa (https://github.com/andrecosta90) for his contribution.
"""

from operator import itemgetter

import pandas as pd


class Encoder():
    """
    Write CLASS docstring here.
    """

    def __init__(self, _map=None, value_counts=None, missing=-999, variables=None,
                 numerical_vars=None, variables_after_preprocess=None):
        """
        Write docstring here.
        """
        self._map = _map or {}
        self.value_counts = value_counts or {}
        self.missing = missing
        self.variables = variables
        self.numerical_vars = numerical_vars or []
        self.variables_after_preprocess = variables_after_preprocess

    def fit_transform(self, data_frame, limit = 100, with_frequency = True, ignored_variables=None):
        """
        Write docstring here.
        """
        self.fit(data_frame, limit, ignored_variables=ignored_variables)
        data_frame = self.transform(data_frame, with_frequency)
        self.variables_after_preprocess = list(data_frame)
        return data_frame

    def fit(self, data_frame, limit = 100, ignored_variables=None):
        """
        Write docstring here.
        """
        data_frame = data_frame.copy()
        ignored_variables = ignored_variables or {}
        self.variables = [x for x in list(data_frame) if x not in ignored_variables]
        data_frame[self.variables] = data_frame[self.variables].fillna(self.missing)

        for variable in self.variables:
            dtype = data_frame[variable].dtypes

            _type = 'NUMERICAL'
            if dtype == 'object' or dtype == 'str' or dtype == 'unicode':
                _type = 'CATEGORICAL'

            if _type == 'CATEGORICAL':
                self._map[variable] = {}
                dict_freq = data_frame[variable].value_counts().to_dict()
                freq = sorted(dict_freq.items(), key=itemgetter(1), reverse=True)
                self.value_counts[variable] = dict_freq

                for item in enumerate(freq):
                    idx = item[0]
                    label, _ = item[1]
                    self._map[variable][label] = label
                    if idx >= limit:
                        self._map[variable][label] = 'others'
            else:
                self.numerical_vars.append(variable)

    def transform(self, data_frame, with_frequency = True):
        """
        Write docstring here.
        """
        data_frame = data_frame.copy()
        data_frame[self.variables] = data_frame[self.variables].fillna(self.missing)
        for variable in self._map:
            unique = data_frame[variable].unique()
            for label in unique:
                if label not in self._map[variable]:
                    self._map[variable][label] = 'others'
                
            if with_frequency:
                data_frame['%s_freq' % variable] = data_frame[variable]
                data_frame['%s_freq' % variable] = data_frame['%s_freq' % variable].map(self.value_counts[variable])
                data_frame['%s_freq' % variable] = data_frame['%s_freq' % variable].fillna(0)
            
            data_frame[variable] = data_frame[variable].map(self._map[variable])
            ohe = pd.get_dummies(data_frame[variable], prefix=variable)

            data_frame = data_frame.drop(variable, axis=1)
            data_frame = pd.concat([data_frame, ohe], axis=1)

        # adicionar colunas que faltam para completar o variables_after_preprocess
        keys_data_frame = set(data_frame)
        if self.variables_after_preprocess:
            for var in self.variables_after_preprocess:
                if var not in keys_data_frame:
                    data_frame[var] = 0
            data_frame = data_frame[self.variables_after_preprocess]

        for variable in self.numerical_vars:
            data_frame['%s_missing' % variable] = data_frame[variable]
            data_frame['%s_missing' % variable] = data_frame['%s_missing' % variable].apply(
                lambda x: 1. if x == -999 else 0.)
            data_frame[variable] = data_frame[variable].apply(lambda x: x if x != -999 else 0.)

        return data_frame