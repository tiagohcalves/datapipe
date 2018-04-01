"""
DataPipe wraps the pandas DataFrame object in order to provide
a base object to run pipelines. It should also support methods in pandas
Dataframe and provide a few more utilities.
"""

# Author: Tiago Alves

import os
import gzip
import copy
import pickle as pk
import pandas as pd

from sklearn.preprocessing import normalize

from exceptions import InvalidDataTypeException
from one_hot_encoder import Encoder
from type_check import get_type


exclude_decore_methods = set(["load", "print"])
compressed_extensions = set(['.gz', '.bz2', '.zip', '.xz'])

############################
# Decorators
############################

def wrap_all(decorator):
    """
    Applies a decorator to all methods in a class.
    Ignores methods listed to exclusion and begining with '_'
    """
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr not in exclude_decore_methods:
                if not attr.startswith("_"):
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def register_call(method):
    """
    Decorator to record all method calls made to the object 
    and create checkpoints.
    """
    def _wrapper(self, *args, **kwargs):
        name = method.__name__
        self._pipeline.append(
            (name, args, kwargs)
        )
        reval = method(self, *args, **kwargs)
        return reval

    return _wrapper


############################
# Main Class
############################

@wrap_all(register_call)
class DataPipe:
    """DataPipe wraps the pandas DataFrame object in order to provide
    a base object to run pipelines. It should also support methods in pandas
    Dataframe and provide a few more utilities.

    Parameters
    ----------
    data: numpy ndarray, dict, or DataFrame. Optional
        DataFrame that holds the data used in the pipeline. If a DataFrame is
        not provided, it must be possible to build a DataFrame from the data.

    verbose: boolean
        Print state messages while processing

    parent_pipe: DataPipe
        If provided, will inherit all attributes from the parent DataPipe
        
    force_types: boolean
        If should automatically cast inferred types. It may slow down the
        construction of the object.
    
    kwargs: keyword arguments
        Any necessary arguments to construct the Data Frame from the given
        data and with the pandas DataFrame() call
    
    Attributes
    ----------
    df: pandas.DataFrame
        DataFrame that holds the data used in the pipeline.
        
    pipeline: list
        Sequence of calls made to the object methods.
        
    anon_keys: dict
        Map of keys and values anonymized. Can be used to de-anonymize the
        data.
    
    one_hot_encoder: Encoder
        Auxiliar object used to create and mantain one hot encodings.
    
    verbose: boolean
        Flag that indicates if should print state messages.
        
    column_type_map: dict
        Holds the inferred type of each column.

    Examples
    -------- 
    """
    def __init__(self, data=None, verbose: bool = True, parent_pipe = None,
                 force_types: bool = True, **kwargs):
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
        
        raw_data = np.random.randint(0, 100)
        data = DataPipe(raw_data, columns=["random"])
        """
        if data is not None:
            if type(data) is pd.DataFrame:
                self._df = data
            else:
                try:
                    self.df = pd.DataFrame(data, **kwargs)
                except Exception as e:
                    raise InvalidDataTypeException(
                            e, "Could not create DataFrame from data")
        
        if parent_pipe is not None:
            self._pipeline = copy.deepcopy(parent_pipe._pipeline)
            self._anon_keys = copy.deepcopy(parent_pipe._anon_keys)
            self._one_hot_encoder = copy.deepcopy(parent_pipe._one_hot_encoder)
            self._verbose = parent_pipe._verbose
            self._column_type_map = copy.deepcopy(parent_pipe._column_type_map)
        else:
            self._pipeline = []
            self._anon_keys = {}
            self._one_hot_encoder = Encoder()
            self._verbose = verbose
            self._check_types(force_types)
    
                
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

    def _check_types(self, force_types:bool = True):
        self._column_type_map = {
            "empty": [],
            "numeric": [],
            "date": [],
            "string": []
        }
                    
        for column in self._df:
            col_type = get_type(self._df[column])
            if col_type == "empty":
                self._column_type_map["empty"].append(column)
            elif col_type == "datetime":
                if force_types:
                    self._df[column] = pd.to_datetime(self._df[column])
                self._column_type_map["date"].append(column)
            elif col_type == "string":
                self._column_type_map["string"].append(column)
            else:
                self._column_type_map["numeric"].append(column)

    ########################
    # Public methods
    ########################
    
    @staticmethod
    def load(filename, **kwargs) -> 'DataPipe':
        """Loads the datapipe from a file. 
        
        The file may be a datapipe file (.dtp), text file (.csv, .txt, .json,
        etc) or binary (.xlsx, .hdf, etc). This method tries to infer the file
        type by the extension and calls the suitable read function from pandas.
        
        The file may also be compressed with the following extensions:
            '.gz', '.bz2', '.zip' and '.xz'
        
        If the type cannot be inferred, it defaults to csv.
        
        If every load fails, it is recommended to use the ``__init__`` method
        and provide a pandas object.
        
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
        """Saves the datapipe to a compressed object (.dtp)
        """
        with gzip.open(filename + ".dtp", "rb") as output_file:
            pk.dump(self, output_file)

    def transform(self, func):
        """
        Applies the given function to the underlying dataframe
        """
        self._df = func(self._df)
        return self

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
        self._check_types(False)
        return self
        
    def select(self, query: str):
        """
        Performs a query on the DataFrame. THe query syntax is the same 
        accepted by the pandas `.query()` method.
        """
        self._df = self._df.query(query)
        return self

    def sample(self, size: float = 0.1, seed: int = 0, inplace=False):
        """
        Get a random sample from the DataFrame.
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
            new_dp = DataPipe(sample_df, parent_pipe=self)
            new_dp._pipeline = self._pipeline
            
            self._pipeline = self._pipeline[:-1]
            
            return new_dp

    def drop(self, columns: list):
        """
        Drop specified columns
        """
        self._df = self._df.drop(columns, axis=1)
        self._check_types(False)
        return self

    def keep(self, columns: list):
        """
        Drop columns not specified
        """
        if len(columns) == 0:
            return self
        self._df = self._df[columns]
        self._check_types(False)
        return self

    def keep_numerics(self):
        """
        Drop columns that are not numeric
        """
        self._df = self._df[self._column_type_map["numeric"]]
        self._check_types(False)
        return self

    def drop_sparse(self, threshold: float = 0.05):
        """
        Drop sparse columns.
        
        :param threshold: maximum percentage of values in each column to be
        considered sparse.
        """
        n_rows = self._df.shape[0]
        cols_to_drop = []
        for column in list(self._df):
            filled_perc = self._df[column].count() / n_rows
            if filled_perc < threshold:
                cols_to_drop.append(column)
        
        self._df = self._df.drop(cols_to_drop, axis=1)
        self._check_types(False)
        return self

    def drop_duplicates(self, key: str = "",  keep='first'):
        """
        Drop duplicated rows.
        
        :param key: Name of the column to perform duplicated check. If not 
        provided, will consider full rows.
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
        
        :param columns: column or list of columns to fill. If None, 
        will fill all numeric columns.
        :param value: If a number is given, will replace nulls with the number.
        If a method is specified (e.g., mean), will use the method instead.
        """
        if columns is None:
            columns = self._column_type_map["numeric"]
        elif type(columns) is str:
            columns = [columns]
        
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
        """
        Replace outliers with a given value
        
        :param columns: column or list of columns to search for outliers.
        :param threshold: the number of times the standard deviation to consider
        a value an outlier.
        :param fill_value: If a number is given, will replace the outliers with
        the number. Otherwise it should be either the mean or the median.
        """
        if columns is None:
            columns = self._df[self._column_type_map["numeric"]]
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
        Scale columns between 0 and 1.
        
        :param columns: column name or list of columns to normalize
        :param axis: 0 to normalize columns, 1 to normalize rows
        :param norm: The norm to use (l1, l2 or max)
        """
        if columns is None:
            columns = self._column_type_map["numeric"]
        
        if type(columns) is str:
            columns = [columns]
        
        self._df[columns] = normalize(self._df[columns], norm=norm, axis=axis)
        
        return self

    def anonymize(self, columns: list, keys=None, update=True, missing=-1):
        """
        Anonymize columns with sequential anonimization.
        
        :param columns: column name or list of columns to anonymize.
        :param keys: map with value -> anon value to anonymize the columns
        :param update: if True, will update the keys with new values
        :param missing: if update is False, then set the missing keys with
        this value.
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
        """
        Creates one-hot encoding for the values in the columns.
        
        :param columns: column name or list of columns to anonymize
        :param limit: max number of values to encoded. If there are more values 
        than the limit, will encode the most frequents.
        :param with_frequency: if True will create a column with the frequency
        of each value
        :param keep_columns: if True will keep the original columns
        :param update: if True will create new columns of new values
        """
        if columns is None:
            columns = self._column_type_map["string"]            
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
        """
        Splits the DataPipe into train datapipe and test datapipe.
        
        :param by: if provided, will sort the dataframe by this value to perform
        the split. Useful for temporal splitting
        :param size: percentual size of the train set. The test set size will 
        contain (1 - size) * number_of_rows items.
        :param seed: random seed used to shuffle the dataframe. Useful for
        reproducibility.
        :returns: Train DataPipe and test DataPipe.
        """
        if by != "":
            self._df = self._df.sort_values(by)
        else:
            self._df = self._df.sample(frac=1, random_state = seed)
        
        n_rows = self._df.shape[0]
        train_size = int(n_rows * size)
            
        train_df = self._df.iloc[:train_size]
        test_df = self._df.iloc[train_size:n_rows]
        
        return DataPipe(train_df, parent_pipe=self), DataPipe(test_df, parent_pipe=self)
            

    def creat_folds(self, n_folds: int = 5, stratified: bool = True,
                    seed: int = 0):
        pass
    
    def print(self, line_width = 60, with_args:bool = True):
        """
        Prints the methods called to the datapipe.
        
        :params line_width: A integer specifing how many characters in each line
        or a list with three values, one for method name, one for args and one for
        kwargs.
        :params with_args: if True prints the arguments of the methods too.
        """
        if type(line_width) is int:
            width = int(line_width / 3)
            widths = [width] * 3
        else:
            widths = line_width
        
        header = ["Method Name", "Args", "Kwargs"]
        
        pipeline_str = "_" * (sum(widths)-1) + "|\n"
        for i, header_str in enumerate(header):
            pipeline_str += header_str + (' ' * (widths[i] - len(header_str)-1)) + "|"
        pipeline_str += "\n" + ("_" * (sum(widths)-1)) + "|"
        
        for pipe_tuple in self._pipeline:
            pipeline_str += "\n"
            for i, info in enumerate(pipe_tuple):
                info = str(info)
                if len(info) > widths[i]:
                    info = info[:widths[i]]
                pipeline_str += info + (' ' * (widths[i] - len(info)-1)) + "|"
        
        pipeline_str += "\n"
        print(pipeline_str)