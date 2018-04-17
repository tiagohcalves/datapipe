# Data Pipe ML
Pipeline API to manipulate dataframes for machine learning.

Data Pipe is a framework that wraps Pandas Data Frames to provide a more fluid method to manipulate data. 

Basic concepts:
- Every operation is performed in place. The Data Pipe object keeps one and only one reference to a pandas Data Frame that is constantly updated. 
- ‎Every operation returns a reference to self, which allows chaining methods fluidly. 
- Every method called is recorded internally to provide improved reproducibility and understanding of the preparation pipeline. The exception is the "load" method.
- ‎Data Pipe calls of unimplemented methods default to the internal Data Frame object. This allows quickly accessing some methods, such as shape and head, but please be aware that those calls are not recorded and do not return Data Pipe objects. If it's necessary to use an unimplemented function, please use the Update method to keep manipulating the Data Pipe. 

## Installation

You can install DataPipeML directly from PyPI:

`pip install datapipeml`

Or from source:

```
git clone https://github.com/tiagohcalves/datapipe.git
cd datapipe
pip install .
```
### Dependencies

DataPipeML has the following requirements:

* [Pandas](https://github.com/pandas-dev/pandas): 0.22 or higher
* [Sklearn](http://scikit-learn.org/stable/): 0.19.1 or higher

Older versions might work but are untested.

### Testing

To run the unit tests, we recommend [Nose](http://nose.readthedocs.io/en/latest/). Just run:

```
cd datapipe/datapipeml/tests/
nosetests test_pipeline.py
```

..........................
----------------------------------------------------------------------
Ran 26 tests in 0.237s

OK

## Example

### Full pipeline with time split
```
>>> from datapipeml import DataPipe

>>> X, y = (
>>>     DataPipe.load("data/kiva_loans_sample.csv.gz")
>>>             .anonymize("id")
>>>             .set_index("id")
>>>             .drop("tags")
>>>             .drop_sparse()
>>>             .drop_duplicates()
>>>             .fill_null()
>>>             .remove_outliers()
>>>             .normalize()
>>>             .set_one_hot()
>>>             .split_train_test(by="date")
>>>     )

Anonymizing id
No sparse columns to drop
Found 0 duplicated rows
Fillings columns ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Removing outliers from ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Normalizing ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Encoding columns ['activity', 'sector', 'country_code', 'country', 'currency', 'repayment_interval']
        
>>> X.keep_numerics()
>>> y.keep_numerics()

Dropping columns {'region', 'posted_time', 'date', 'funded_time', 'borrower_genders', 'disbursed_time', 'use'}
Dropping columns {'region', 'posted_time', 'date', 'funded_time', 'borrower_genders', 'disbursed_time', 'use'}

>>> print(X.summary())
___________________________________________________________|
Method Name        |Args               |Kwargs             |
___________________________________________________________|
anonymize          |('id',)            |{}                 |
set_index          |('id',)            |{}                 |
drop               |('tags',)          |{}                 |
drop_sparse        |()                 |{}                 |
drop_duplicates    |()                 |{}                 |
fill_null          |()                 |{}                 |
remove_outliers    |()                 |{}                 |
normalize          |()                 |{}                 |
set_one_hot        |()                 |{}                 |
split_train_test   |()                 |{'by': 'date'}     |
keep_numerics      |()                 |{}                 |
___________________________________________________________|
```

### Create target column and stratified folds
```
>>> folds = (
>>>     DataPipe.load("data/kiva_loans_sample.csv.gz")
>>>             .set_index("id")
>>>             .drop_duplicates()
>>>             .fill_null()
>>>             .remove_outliers()
>>>             .normalize()
>>>             .set_one_hot()
>>>             .create_column("high_loan", lambda x: 1 if x["loan_amount"] > 2000 else 0)
>>>             .keep_numerics()
>>>             .create_folds(stratify_by="high_loan")
>>>     )
        
Found 0 duplicated rows
Fillings columns ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Removing outliers from ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Normalizing ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
One-hot encoding columns ['activity', 'sector', 'country_code', 'country', 'currency', 'borrower_genders', 'repayment_interval']
Creating column high_loan
Dropping columns {'tags', 'funded_time', 'disbursed_time', 'region', 'use', 'posted_time', 'date'}
```

## Additional Features

### Checkpoint

When instantiating a new DataPipe object, or through the method `enable_checkpoint`, one can specify the path to save a copy of the data before each function called. This allows to inspect the data for each step of the pipeline, and to create backups, but keep in mind that this is a costly function, both in execution time (dumping to disk is slow) and in space required (a new file is created for each function called). However, through the methods `enable_checkpoint` and `disable_checkpoint`, this feature can be activated only for critial steps in the pipeline.

### Access to underlying DataFrame

Every function not implemented in the DataPipe class will be forward to the underlying DataFrame object. This means that one can enjoy all methods present on the pandas DataFrame, but beware with the return type, since it will probably broke the pipeline execution. For example, `dp.head(100)` will execute the `head` function on the DataFrame and display the corresponding result. For functions that returns new instances of DataFrame (e.g., the `transpose` function), we recommend the `transform` function, as it is applied to the underlying DataFrame and keeps the DataPipe reference. If any need is not fulfilled by the methods above, it is also possible to access the DataFrame through the `._df` property. 

## List of methods

Here are listed all implemented methods. Full documentation is available in the source script of DataPipe.

```
__init__(self, data=None, verbose: bool = True, parent_pipe = None, force_types: bool = True, checkpoint: str = None, **kwargs)

load(filename, **kwargs)
save(self, filename)
transform(self, func)
cast_types(self, type_map: dict)
set_index(self, columns: list)
select(self, query: str)
sample(self, size: float = 0.1, seed: int = 0, inplace=False)
drop(self, columns: list)
keep(self, columns: list)
keep_numerics(self)
drop_sparse(self, threshold: float = 0.05)
drop_duplicates(self, key: str = "",  keep='first')
fill_null(self, columns=None, value="mean")
remove_outliers(self, columns=None, threshold: float = 2.0, fill_value = "mean")
normalize(self, columns=None, axis: int = 0, norm: str = "l2")
anonymize(self, columns, keys=None, update=True, missing=-1)
set_one_hot(self, columns=None, limit: int = 100, with_frequency: bool = True, keep_columns: bool = False, update=True)
create_column(self, column_name: str, func)
split_train_test(self, by: str = "", size: float = 0.8, seed: int = 0)
create_folds(self, n_folds: int = 5, stratify_by: str = "", seed: int = 0, return_iterator: bool = True)
summary(self, line_width = 60, with_args:bool = True)
disable_checkpoint(self)
enable_checkpoint(self, path: str = "")
```
