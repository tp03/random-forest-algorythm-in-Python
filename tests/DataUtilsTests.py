"""
Filename: DataUtilsTests.py
Author: Tomasz Zalewski, Antoni Kowalski
Description: Testy jednostkowe dla funkcji DataUtils.
"""
import unittest
import os
import sys
sys.path.insert(0, '../')
from DataUtils import dataToList, remove_ans, divide_data

class TestDataUtils(unittest.TestCase):
    def test_dataToList(self):
        data = ["a,b,c\n", "1,2,3\n"]
        result = dataToList(data)
        self.assertEqual(result, [["a", "b", "c\n"], ["1", "2", "3\n"]])

    def test_remove_ans(self):
        data = [["a", "b", "yes"], ["c", "d", "no"]]
        answers, attributes = remove_ans(data, attributes_columns=False)
        self.assertEqual(answers, ["yes", "no"])
        self.assertEqual(attributes, [["a", "b"], ["c", "d"]])

    def test_divide_data(self):
        with open("test_data.data", "w") as f:
            f.write("1\n2\n3\n4\n5\n")
        
        train, test = divide_data("test_data.data", seed=42, train_ratio=0.6, randomise=True)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        os.remove("test_data.data")

if __name__ == "__main__":
    unittest.main()