"""
Examples with DataPipe
"""

from datapipeml import DataPipe

###################################################
# Full pipeline with time split

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


###################################################
# Create target column and create stratified folds

folds = DataPipe.load("data/kiva_loans_sample.csv.gz")\
        .set_index("id")\
        .drop_duplicates()\
        .fill_null()\
        .remove_outliers()\
        .normalize()\
        .set_one_hot()\
        .create_column("high_loan", lambda x: 1 if x["loan_amount"] > 2000 else 0)\
        .keep_numerics()\
        .create_folds(stratify_by="high_loan")