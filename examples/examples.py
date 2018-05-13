"""
Examples with DataPipe
"""

from datapipeml import DataPipe

###################################################
# Full pipeline with time split

train_dp, test_dp = (
    DataPipe.load("data/kiva_loans_sample.csv.gz")
            .anonymize("id")
            .set_index("id")
            .drop("tags")
            .drop_sparse()
            .drop_duplicates()
            .fill_null()
            .remove_outliers()
            .normalize()
            .set_one_hot()
            .split_train_test(by="date")
    )
        
train_dp.keep_numerics()
test_dp.keep_numerics()

print(train_dp.summary())


###################################################
# Create target column and create stratified folds

folds = (
    DataPipe.load("data/kiva_loans_sample.csv.gz")
            .set_index("id")
            .drop_duplicates()
            .fill_null()
            .remove_outliers()
            .normalize()
            .set_one_hot()
            .create_column("high_loan", lambda x: 1 if x["loan_amount"] > 2000 else 0)
            .keep_numerics()
            .create_folds(stratify_by="high_loan")
    )
