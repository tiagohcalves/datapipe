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

from sklearn.preprocessing import normalize

from exceptions import InvalidDataTypeException
from one_hot_encoder import Encoder


exclude_decore_methods = set(["load"])
compressed_extensions = (['.gz', '.bz2', '.zip', '.xz'])

def wrap_all(decorator):
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)) and attr not in exclude_decore_methods:
                if not attr.startswith("_"):
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def register_call(method):
    def _wrapper(self, *args, **kwargs):
        name = method.__name__
        self._pipeline.append(
            (name, args, kwargs)
        )
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
    def __init__(self, data=None, verbose: bool = True, **kwargs):
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

        self._pipeline = []
        self._anon_keys = {}
        self._one_hot_encoder = Encoder()
        
        self._verbose = verbose
        
        if data is not None:
            if type(data) is pd.DataFrame:
                self._df = data
            else:
                try:
                    self.df = pd.DataFrame(data, **kwargs)
                except Exception as e:
                    raise InvalidDataTypeException(
                            e, "Could not create DataFrame from data")
                    
    #########################
    # Private Methods
    #########################
    
    def __getattr__(self, name):
        """
        Handle missing method call by trying to call directly the pandas dataframe.
        This lets the DataPipe to be used as an extension to the dataframe.
        It will not record the call from on the pipeline
        """
        def _missing(*args, **kwargs):
            retval = getattr(self._df, name)
            if callable(retval):
                retval = retval(*args, **kwargs)
            return retval
        return _missing
    
    def __str__(self):
        return self._df.head().__str__()
    
    def __repr__(self):
        return self._df.head().__repr__()

    ########################
    # Public methods
    ########################
    
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
        fname, file_extension = os.path.splitext(filename)
        file_extension = file_extension.lower() 
        compression = None
        
        if file_extension in compressed_extensions:
            compression = file_extension.replace(".", "")
            if compression == "gz":
                compression = "gzip"
                
            file_extension = os.path.splitext(fname)[1].lower()
        
        if file_extension == ".dtp":
            with gzip.open(filename, "rb") as input_file:
                return pk.load(input_file)
        elif file_extension == ".json":
            df = pd.read_json(filename, compression=compression, **kwargs)
        elif file_extension == ".html":
            df = pd.read_html(filename, **kwargs)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(filename, **kwargs)
        elif file_extension == ".hdf":
            df = pd.read_html(filename, **kwargs)
        elif file_extension == ".pk":
            df = pd.read_pickle(filename, **kwargs)
        else:
            df = pd.read_csv(filename, compression=compression, **kwargs)

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
        self._df = self._df.set_index(keys=columns)
        return self
        
    def select(self, query: str):
        """
        Performs a query on the DataFrame
        """
        self._df = self._df.query(query)
        return self

    def sample(self, size: float = 0.1, seed: int = 0, inplace=False):
        """
        Get a random sample from the DataFrame
        CAUTION: If inplace=True, it will override the current DataFrame
        """
        if 0.0 < size < 1.0:
            n = None
            frac = size
        else:
            n = size
            frac = None
            
        sample_df = self._df.sample(n, frac)
        if inplace:
            self._df = sample_df
            return self
        else:
            new_dp = DataPipe(sample_df)
            new_dp._pipeline = self._pipeline
            
            self._pipeline = self._pipeline[:-1]
            
            return new_dp

    def drop(self, columns: list):
        """
        Drop specified columns
        """
        self._df = self._df.drop(columns, axis=1)
        return self

    def keep(self, columns: list):
        """
        Drop columns not specified
        """
        if len(columns) == 0:
            return self
        self._df = self._df[columns]
        return self

    def keep_numerics(self):
        """
        Drop columns that are not numeric
        """
        self._df = self._df.select_dtypes(include=pd.np.number)
        return self

    def drop_sparse(self, threshold: float = 0.05):
        """
        Drop sparse columns
        """
        n_rows = self._df.shape[0]
        cols_to_drop = []
        for column in list(self._df):
            filled_perc = column.count() / n_rows
            if filled_perc < threshold:
                cols_to_drop.append(column)
        
        self._df = self._df.drop(cols_to_drop, axis=1)
        return self

    def drop_duplicates(self, key: str = "",  keep='first'):
        """
        Drop duplicated rows
        """
        if key == self._df.index.name:
            self._df = self._df[~self._df.index.duplicated(keep=keep)]
        elif key != "":
            self._df = self._df[~self._df[key].duplicated(keep=keep)]
        else:
            self._df = self._df.drop_duplicates(keep=keep)
            
        return self

    def fill_null(self, columns=None, value="mean"):
        """
        Fills NaN with a given value/method
        """
        if columns is None:
            columns = list(self._df.select_dtypes(include=pd.np.number))

        for col in columns:
            self._fill_column(col, value)

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

    def remove_outliers(self, columns=None, threshold: float = 2.0, fill_value = "mean"):
        if columns is None:
            columns = list(self._df.select_dtypes(include=pd.np.number))
        if type(columns) is str:
            columns = [columns]

        for column in columns:
            col = self._df[column]
            std_limit = threshold * col.std()
            dist_to_mean = (col - col.mean()).abs()
            
            value = fill_value
            if type(fill_value) is str:
                if value == "mean":
                    value = col.mean()
                elif value == "meadian":
                    value = self._df[column].median()
                
            self._df.loc[dist_to_mean > std_limit, column] = value
            
        return self

    def normalize(self, columns=None, axis: int = 0, norm: str = "l2"):
        """
        Scale columns between 0 and 1
        """
        if columns is None:
            columns = list(self._df.select_dtypes(include=pd.np.number))
        
        if type(columns) is str:
            columns = [columns]
        
        self._df[columns] = normalize(self._df[columns], norm=norm, axis=0)
        
        return self

    def anonymize(self, columns: list, keys=None, update=True, missing=-1):
        """
        Anonymize columns with sequential anonimization
        """
        if keys is None:
            keys = self._anon_keys
            
        for column in columns:
            self._df[column] = self._df[column].apply(
                    lambda x: self._get_anon_key(x, keys, update, missing)
                )
            
        return self

    def _get_anon_key(self, value, keys, update, missing):
        if value not in keys:
            if update:
                value_key = len(keys)
                keys[value] = value_key
            else:
                return missing
             
        return keys[value]
        

    def set_one_hot(self, columns=None, limit: int = 100, with_frequency: bool = True, 
                    keep_columns: bool = False, update=True):
        if columns is None:
            columns = list(self._df.select_dtypes(include="object"))            
            if columns is None:
                return self
            
            for column in columns:
                # Do not automatically encode columns with too many unique values.
                # It's probably a mistake.
                if self._df[column].nunique() > 2 * limit:
                    columns.remove(column)
        
        if update:
            one_hot_columns = self._one_hot_encoder.fit_transform(self._df[columns], limit, with_frequency)
        else:
            one_hot_columns = self._one_hot_encoder.transform(self._df[columns], with_frequency)
        
        if not keep_columns:
            self._df = self._df.drop(columns, axis=1)
            
        self._df = pd.concat([self._df, one_hot_columns], axis = 1)
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