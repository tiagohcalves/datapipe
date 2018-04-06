"""
Functions to infer the correct type of each column
"""
import pandas as pd

boolean_str_types = ["y", "yes", "s", "sim", "n", "no", "não", "true", 
                     "verdadeiro", "false", "falso"]

def check_int(series):
    """
    Checks series of int type
    """
    if series.nunique() == 2:
        # Possibly boolean
        unique = series.unique()
        if (unique[0] == 0 or unique[0] == 1) and (unique[1] == 0 or unique[1] == 1):
#            return "bool"
            return "int"
    # only remove the type specification
    return "int"


def check_float(series, has_null, n):
    """
    Infer correct type of series considered as float
    """
    if series.nunique() == 2:
        # Possibly boolean
        unique = series.unique()
        if (unique[0] == 0.0 or unique[0] == 1.0) and (unique[1] == 0.0 or unique[1] == 1.0):
#            return "bool"
            return "float"
    # check if the column is actually a nullable int
    all_integer = series.sample(n).apply(lambda x: x.is_integer()).all()
    if all_integer:
        if has_null:
            return "nullable int"
        else:
            return "int"
    else:
        return "float"
    

def check_object(series, n):
    """
    Infer correct type of series considered as object
    """
    try:
        pd.to_datetime(series.sample(n))
        return "datetime"
    except ValueError:
        pass
    if series.nunique() == 2:
        # Possibly boolean
        unique = series.unique()
        if unique[0].lower() in boolean_str_types and unique[1].lower() in boolean_str_types:
            return "bool"
    # no other type was found, must be a string
    return "string"


def get_type(series):
    # Get original type
    col_type = series.dtype
    # Remove null
    has_null = series.isnull().values.any()
    series = series.dropna()
    # Number of samples to be used
    n = min(100, len(series))
    
    if n == 0:
        # If the column is empty, there is no data type
        return "empty"
    elif col_type in ["int64", "uint8"]:
        return check_int(series)
    elif col_type == "float64":
        return check_float(series, has_null, n)
    else:
        return check_object(series, n)