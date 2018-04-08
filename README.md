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

>>> X, y = DataPipe.load("data/kiva_loans_sample.csv.gz")\
>>>         .anonymize("id")\
>>>         .set_index("id")\
>>>         .drop("tags")\
>>>         .drop_sparse()\
>>>         .drop_duplicates()\
>>>         .fill_null()\
>>>         .remove_outliers()\
>>>         .normalize()\
>>>         .set_one_hot()\
>>>         .split_train_test(by="date")

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
>>> folds = DataPipe.load("data/kiva_loans_sample.csv.gz")\
>>>         .set_index("id")\
>>>         .drop_duplicates()\
>>>         .fill_null()\
>>>         .remove_outliers()\
>>>         .normalize()\
>>>         .set_one_hot()\
>>>         .create_column("high_loan", lambda x: 1 if x["loan_amount"] > 2000 else 0)\
>>>         .keep_numerics()\
>>>         .create_folds(stratify_by="high_loan")
        
Found 0 duplicated rows
Fillings columns ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Removing outliers from ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
Normalizing ['funded_amount', 'loan_amount', 'partner_id', 'term_in_months', 'lender_count']
One-hot encoding columns ['activity', 'sector', 'country_code', 'country', 'currency', 'borrower_genders', 'repayment_interval']
Creating column high_loan
Dropping columns {'tags', 'funded_time', 'disbursed_time', 'region', 'use', 'posted_time', 'date'}
```
