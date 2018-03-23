"""
DataPipe extends the pandas DataFrame object in order to provide
a base object to run pipelines. It should support every method in pandas
Dataframe and provide a few more utilities.
"""

# Author: Tiago Alves

import pandas as pd

from data_pipeline.exceptions import InvalidDataTypeException


class DataPipe(pd.DataFrame):
    """DataPipe extends the pandas DataFrame object in order to provide
    a base object to run pipelines. 
    
    It should support every method in pandas Dataframe and provide a few 
    more utility functions. All actions are based on inplace transformations
    that can be broadcasted and stacked.

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
    
    def __init__(self, data, **kwargs):
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
        if data is not None:
            if type(data) is pd.DataFrame:
                self.df = data
            else:
                try:
                    self.df = pd.DataFrame(data, **kwargs)
                except Exception as e:
                    raise InvalidDataTypeException(
                            e, "Could not create DataFrame from data")

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
        pass
    
    def save(self, filename):
        """Saves the datapipe to a proper object (.dpt)
        """
        pass
        
    def update(self, func):
        pass
    
    def cast_types(self, type_map: dict):
        pass
        
    def set_index(self, columns: list):
        pass
    
    def drop(self, columns: list):
        pass
    
    def keep(self, columns: list):
        pass
    
    def keep_numerics(self. columns: list):
        pass
    
    def fill_null(self, value = "mean"):
        pass
    
    def remove_outliers(self, columns: list = [], threshold: float = 2.0):
        pass
    
    def drop_sparse(self, threshold: float = 0.05):
        pass
    
    def drop_duplicates(self, key: str = ""):
        pass
    
    def normalize(self, columns: list = [], norm: str = "l2"):
        pass
    
    def set_one_hot(self, columns: list = [], limit: int = 100, 
                    with_frequency: bool = True):
        pass
    
    def sample(self, size: float = 0.1, seed: int = 0):
        pass
    
    def split_train_test(self, by: str = "", size: 0.8, seed: int = 0):
        pass
    
    def creat_folds(self, n_folds: int = 5, stratified: bool = True, 
                    seed: int = 0):
        pass
    
    def anonymize(self, columns: list, keys: dict = {}):
        pass
        
        