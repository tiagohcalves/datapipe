"""
Test suit for the DataPipe class
"""

import os
import pandas as pd

from unittest import TestCase
from datapipeml import DataPipe

def dp_instance():
    df = pd.DataFrame(pd.np.random.rand(10, 2), columns=['A', 'B'])
    dp = DataPipe(df)
    return dp 

def dp_instance_with_types():
    data = pd.np.array([
        [1] * 10,
        [1.0] * 10,
        ["a"] * 10,
        [pd.np.nan] * 10,
        ["01/01/01"] * 10
    ])
        
    dp = DataPipe(data.T, columns=["int", "float", "str", "empty", "date"])
    dp._df["empty"] = pd.np.nan
    return dp

def assertListEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

class TestDataPipe(TestCase):
    def test_init(self):
        df = pd.DataFrame(pd.np.random.rand(10, 2))
        dp = DataPipe(df)
        
        self.assertEqual(dp._df.shape, df.shape)
        
    def test_init_numpy(self):
        df = pd.DataFrame(pd.np.random.rand(10, 2))
        dp = DataPipe(df.values)
        
        self.assertEqual(dp._df.shape, df.shape)
        
    def test_init_args(self):
        df = pd.DataFrame(pd.np.random.rand(10, 2))
        index = list(range(10, 0, -1))
        dp = DataPipe(df.values, columns=["A", "B"], index=index)
        
        self.assertEqual(dp._df.shape, df.shape)
        self.assertTrue(assertListEqual(dp._df.columns, ["A", "B"]))
        self.assertTrue(assertListEqual(dp._df.index.values, index))
        
    def test_check_types(self):
        dp = dp_instance_with_types()
        dp._check_types()
        
        self.assertTrue(assertListEqual(dp._column_type_map["date"], ["date"]))
        self.assertTrue(assertListEqual(dp._column_type_map["empty"], ["empty"]))
        self.assertTrue(assertListEqual(dp._column_type_map["numeric"], ["int", "float"]))
        self.assertTrue(assertListEqual(dp._column_type_map["string"], ["str"]))
    
    def test_save_load(self):
        dp = dp_instance()
        
        dp.save("tmp")
        loaded_dp = DataPipe.load("tmp.dtp")
        
        self.assertEquals(dp.shape(), loaded_dp.shape())
    
    def test_transform(self):
        dp = dp_instance()
        dp.transform(lambda df: df.transpose())
        
        self.assertEquals(dp.shape(), (2, 10))
    
    def test_cast_types(self):
        dp = dp_instance_with_types()
        dp.cast_types({"int": float})
        
        self.assertTrue(dp._df["int"].dtype == 'float64')
    
    def test_set_index(self):
        dp = dp_instance()
        dp.set_index("A")
        
        self.assertTrue(dp._df.index.name == "A")
    
    def test_select(self):
        dp = dp_instance()
        dp._df["A"] = list(range(10))
        
        dp.select("A > 5")
        
        self.assertTrue(dp.shape()[0] == 4)
        
    
    def test_sample(self):
        dp = dp_instance()
        
        sample = dp.sample(0.5)
        self.assertTrue(sample.shape()[0] == 5)
        
        sample = dp.sample(4)
        self.assertTrue(sample.shape()[0] == 4)
        
        dp.sample(0.5, inplace=True)
        self.assertTrue(dp.shape()[0] == 5)
        
        
    def test_drop(self):
        dp = dp_instance()
        dp.drop("A")
        
        self.assertTrue("A" not in dp._df.columns)
    
    def test_keep(self):        
        dp = dp_instance()
        dp.keep("A")
        
        self.assertTrue(assertListEqual(list(dp._df), ["A"]))
    
    def test_keep_numerics(self):
        dp = dp_instance_with_types()
        dp.keep_numerics()
        
        self.assertTrue(assertListEqual(list(dp._df), ["int", "float"]))
    
    def test_drop_sparse(self):
        dp = dp_instance()
        
        dp._df["A"] = pd.np.nan
        dp.drop_sparse()
        
        self.assertTrue(assertListEqual(list(dp._df), ["B"]))
        
    def test_drop_duplicates(self):
        dp = dp_instance_with_types()
        dp.drop_duplicates()
        
        self.assertEqual(dp.shape()[0], 1)
    
    def test_fill_null(self):
        dp = dp_instance_with_types()
        dp.fill_null(value=5)
        
        self.assertEqual(dp._df["empty"].mean(), 5)
    
    def test_remove_outliers(self):
        dp = dp_instance()
        dp._df.loc[0, "A"] = 1000
        
        dp.remove_outliers(fill_value=0.5)
        
        self.assertTrue(dp._df["A"].max() < 1)
    
    def test_normalize(self):
        dp = dp_instance()
        dp._df["A"] *= 100
        
        dp.normalize()
        
        self.assertTrue(dp._df["A"].min() > 0)
        self.assertTrue(dp._df["A"].max() < 1)
    
    def test_anonymize(self):
        dp = dp_instance()
        dp.anonymize("A")
        
        self.assertTrue(assertListEqual(
            list(dp._df["A"]),
            list(range(10))
        ))
    
    def test_set_one_hot(self):
        dp = dp_instance_with_types()
        dp.set_one_hot()
        
        self.assertTrue("str_a" in dp._df)
        self.assertEqual(dp._df["str_a"].max(), 1)
        self.assertTrue("str_freq" in dp._df)
    
    def test_create_column(self):
        dp = dp_instance_with_types()
        dp.create_column("5", lambda x: 5*x["int"])
        
        self.assertEqual(dp._df["5"].mean(), 5)
    
    def test_split_train_test(self):
        dp = dp_instance()
        
        train, test = dp.split_train_test(size = 0.6)
        
        self.assertEqual(train.shape()[0], 6)
        self.assertEqual(test.shape()[0], 4)
    
    def test_create_folds(self):
        dp = dp_instance()
        
        folds = dp.create_folds(return_iterator = False)
        
        self.assertEqual(len(folds), 5)
        for train_dp, test_dp in folds:
            self.assertEqual(len(train_dp._df), 8)
            self.assertEqual(len(test_dp._df), 2)
        
        #TODO test stratification
    
    def test_summary(self):
        dp = dp_instance()
        dp.normalize("A").fill_null(value=3)
        
        expected_result = "___________________________________________________________|\nMethod Name        |Args               |Kwargs             |\n___________________________________________________________|\nnormalize          |('A',)             |{}                 |\nfill_null          |()                 |{'value': 3}       |\n___________________________________________________________|"
        self.assertEqual(dp.summary(), expected_result)    

    def test_pipeline(self):
        dp = dp_instance()
        dp.normalize("A").fill_null(value=3)
        
        self.assertTrue(dp._pipeline[0][0] == "normalize")
        self.assertTrue(dp._pipeline[1][0] == "fill_null")
    
    def test_checkpoint(self):
        if os.path.exists("_ckp_3.dtp"):
            os.remove("_ckp_3.dtp")
        
        dp = dp_instance()
        dp.normalize("A").fill_null(value=3).enable_checkpoint()\
            .drop_duplicates().disable_checkpoint()\
            .drop_sparse()
        
        self.assertTrue(os.path.exists("_ckp_3.dtp"))
        self.assertFalse(os.path.exists("_ckp_4.dtp"))