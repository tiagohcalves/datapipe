"""
DataPipe extends the pandas DataFrame object in order to provide
a base object to run pipelines. It should support every method in pandas
Dataframe and provide a few more utilities.
"""

# Author: Tiago Alves

import os
import gzip
import pickle as pk
import pandas as pd

from data_pipeline.exceptions import InvalidDataTypeException


def wrap_all(decorator):
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def register_call(method):
    def _wrapper(self, *args, **kwargs):
        name = method.__name__
        if not name.startswith("__"):
            self.pipeline.append({
                name: (args, kwargs)
            })
        reval = method(self, *args, **kwargs)
        return reval

    return _wrapper


@wrap_all(register_call)
class DataPipe:
    """DataPipe extends the pandas DataFrame object in order to provide
    a base object to run pipelines. 
    
    It should support every method in pandas Dataframe and provide a few 
    more utility functions. All actions are based on inplace transformations
    that can be broadcast and stacked.

    Parameters
    ----------
    data: numpy ndarray, dict, or DataFrame. Optional
        DataFrame that holds the data used in the pipeline. If a DataFrame is
        not provided, it must be possible to build a DataFrame from the data.

    Attributes
    ----------
    df: pandas.DataFrame
        DataFrame that holds the data used in the pipeline.

    Examples
    -------- 
    """
    def __init__(self, data=None, **kwargs):
        """
        Initiates the DataPipe. If data is provided, it should be a DataFrame
        or numpy ndarray or dict that can have a DataFrame created from.
        
        kwargs are any necessary pandas arguments to load the data.
        
        Examples
        -------- 
        import pandas as pd
        
        df = pd.DataFrame()
        data = DataPipe(df)
        
        ---
        import numpy as np
        
        raw_data = np.random.randint(0, 100, 10)
        data = DataPipe(raw_data, columns=["random"])
        """

        self.pipeline = []
        if data is not None:
            if type(data) is pd.DataFrame:
                self._df = data
            else:
                try:
                    self.df = pd.DataFrame(data, **kwargs)
                except Exception as e:
                    raise InvalidDataTypeException(
                            e, "Could not create DataFrame from data")

    @staticmethod
    def load(filename, **kwargs):
        """Loads the datapipe from a file. 
        
        The file may be a datapipe file (.dtp), text file (.csv, .txt, .json,
        etc) or binary (.xlsx, .hdf, etc). This method tries to infer the file
        type by the extension and calls the suitable read function from pandas.
        
        If the type cannot be inferred, it defaults to csv.
        
        If every load fails, it recommended to use the ``__init___`` method
        from a pandas object.
        
        Returns
        --------
        data: DataPipe
            DataPipe created from given file.
        """
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == ".dtp":
            with gzip.open(filename, "rb") as input_file:
                return pk.load(input_file)
        elif file_extension == ".json":
            df = pd.read_json(filename, **kwargs)
        elif file_extension == ".html":
            df = pd.read_html(filename, **kwargs)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(filename, **kwargs)
        elif file_extension == ".hdf":
            df = pd.read_html(filename, **kwargs)
        elif file_extension == ".pk":
            df = pd.read_pickle(filename, **kwargs)
        else:
            df = pd.read_csv(filename, **kwargs)

        return DataPipe(df)

    def save(self, filename):
        """Saves the datapipe to a proper object (.dtp)
        """
        with gzip.open(filename + ".dtp", "rb") as output_file:
            pk.dump(self, output_file)

    def as_datapipe(self, func):
        return DataPipe(func(self))

    def cast_types(self, type_map: dict):
        """
        Casts column to given types
        :param type_map: Map of {column name -> data type}
        :return: DataPipe with types cast
        """
        for column_name, data_type in type_map.items():
            self._df[column_name] = self._df[column_name].astype(data_type)
        return self

    def set_index(self, columns: list):
        """Defines a set of columns as index of the DataFrame
        :param columns: column label or list of column labels
        :return: DataPipe with index set
        """
        self._df.set_index(keys=columns)
        return self

    def drop(self, columns: list):
        self._df = self._df.drop_columns(columns, axis=1)
        return self

    def keep(self, columns: list):
        if len(columns) == 0:
            return self
        self._df = self._df[columns]
        return self

    def keep_numerics(self):
        self._df = self._df.select_dtypes(include=pd.np.numeric)
        return self

    def fill_null(self, columns=None, value="mean"):
        if columns is None:
            columns = self._df.select_dtypes(include=pd.np.numeric).columns

        for col in columns:
            self._fill_column(col, value)

        return self

    def select(self, query: str):
        self._df = self._df.query(query)
        return self

    def _fill_column(self, column, value="mean"):
        if type(value) is str:
            if value == "mean":
                mean = self._df[column].mean()
                self._df[column] = self._df[column].fillna(mean)
            elif value == "meadian":
                median = self._df[column].median()
                self._df[column] = self._df[column].fillna(median)
            else:
                self._df[column] = self._df[column].fillna(method=value)
        else:
            self._df[column] = self._df[column].fillna(value)

    def remove_outliers(self, columns=None, threshold: float = 2.0):
        if columns is None:
            columns = self._df.select_dtypes(include=pd.np.numeric).columns

        return self

    def drop_sparse(self, threshold: float = 0.05):
        return self

    def drop_duplicates(self, key: str = ""):
        return self

    def normalize(self, columns=None, norm: str = "l2"):
        if columns is None:
            columns = self._df.select_dtypes(include=pd.np.numeric).columns
        return self

    def anonymize(self, columns: list, keys=None):
        if keys is None:
            keys = {}
        return self

    def set_one_hot(self, columns=None, limit: int = 100, with_frequency: bool = True):
        if columns is None:
            columns = self._df.select_dtypes(include="object").columns

        return self

    def sample(self, size: float = 0.1, seed: int = 0):
        return self

    def split_train_test(self, by: str = "", size: float = 0.8, seed: int = 0):
        pass

    def creat_folds(self, n_folds: int = 5, stratified: bool = True,
                    seed: int = 0):
        pass

####

if __name__ == '__main__':
    import numpy as np

    dp = DataPipe(np.random.randint(0, 100, 100).reshape((10, 10)))
    print(type(dp))
    print(dp)
    dp = dp.as_datapipe(lambda x: x.transpose())
    print(type(dp))
    print(dp)
    dp = dp.transpose()
    print(type(dp))
    print(dp)