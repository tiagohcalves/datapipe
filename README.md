# Data Pipe ML
Pipeline API to manipulate dataframes for machine learning.

Data Pipe is a framework that wraps Pandas Data Frames to provide a more fluid method to manipulate data. 

Basic concepts:
- Every operation is performed in place. The Data Pipe object keeps one and only one reference to a pandas Data Frame that is constantly updated. 
- ‎Every operation returns a reference to self, which allows chaining methods fluidly. 
- Every method called is recorded internally to provide improved reproducibility and understanding of the preparation pipeline. The exception is the "load" method.
- ‎Data Pipe calls of unimplemented methods default to the internal Data Frame object. This allows quickly accessing some methods, such as shape and head, but please be aware that those calls are not recorded and do not return Data Pipe objects. If it's necessary to use an unimplemented function, please use the Update method to keep manipulating the Data Pipe. 

## Example

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
Dropping columns {'region', 'posted_time', 'date', 'funded_time', 'borrower_genders', 'disbursed_time', 'use'}
Dropping columns {'region', 'posted_time', 'date', 'funded_time', 'borrower_genders', 'disbursed_time', 'use'}
        
>>> X.keep_numerics()
>>> y.keep_numerics()

>>> X.print()
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
