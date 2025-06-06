"""
Filename: RandomForestTests.py
Author: Tomasz Zalewski, Antoni Kowalski
Description: Testy jednostkowe dla klasy RandomForest.
"""
import unittest
import sys
sys.path.insert(0, '../')
from RouletteForest import RandomForest
from unittest.mock import MagicMock

class TestRandomForest(unittest.TestCase):
    def test_bootstrap(self):
        data = ["1\n", "2\n", "3\n"]
        forest = RandomForest(5, data, "entropy")
        self.assertEqual(len(forest.tree_list), 5)

    def test_voting(self):
        tree1 = MagicMock()
        tree1.classify.return_value = "A"
        
        tree2 = MagicMock()
        tree2.classify.return_value = "B"
        
        tree3 = MagicMock()
        tree3.classify.return_value = "A"
        
        forest = RandomForest(0, [], "entropy")
        forest.tree_list = [tree1, tree2, tree3]
        
        self.assertEqual(forest.vote(["test"]), "A")

if __name__ == "__main__":
    unittest.main()