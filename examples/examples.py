"""
Examples with DataPipe
"""

from datapipeml import DataPipe

X, y = DataPipe.load("data/kiva_loans_sample.csv.gz")\
        .anonymize("id")\
        .set_index("id")\
        .drop("tags")\
        .drop_sparse()\
        .drop_duplicates()\
        .fill_null()\
        .remove_outliers()\
        .normalize()\
        .set_one_hot()\
        .split_train_test(by="date")
        
X.keep_numerics()
y.keep_numerics()

X.print()