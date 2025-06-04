import unittest
import sys
sys.path.insert(0, '../')
from ID3 import ID3

class TestID3(unittest.TestCase):
    def test_entropy_pure(self):
        id3 = ID3([], "entropy")
        self.assertAlmostEqual(id3.entropy(['yes', 'yes', 'yes']), 0.0)

    def test_entropy_mixed(self):
        id3 = ID3([], "entropy")
        entropy = id3.entropy(["yes", "no", "no"])
        self.assertAlmostEqual(entropy, 0.64, delta=0.01)

    def test_gini_pure(self):
        id3 = ID3([], "gini")
        self.assertEqual(id3.gini(["yes", "yes"]), 0.0)

    def test_gini_mixed(self):
        id3 = ID3([], "gini")
        gini = id3.gini(["yes", "no", "yes"])
        self.assertAlmostEqual(gini, 0.44444, delta=0.001)    

    def test_build_tree(self):
        data = ["0,0,no\n", "0,1,yes\n", "1,0,no\n", "1,1,yes\n"]
        
        class DeterministicID3(ID3):
            def __init__(self, input_data, classification_type):
                super().__init__(input_data, classification_type)

            def randomize(self, gains):
                return gains
            
            def roulette(self, gains):
                return max(gains, key=gains.get)
        
        id3 = DeterministicID3(data, "entropy")
        root = id3.root
        
        test_cases = [
            (["0", "0"], "no\n"),
            (["1", "0"], "no\n"),
            (["1", "1"], "yes\n"),
            (["0", "1"], "yes\n")
        ]

        for attributes, expected in test_cases:
            result = id3.classify(attributes, root)
            print(f"Input: {attributes} -> Output: {result} | Expected: {expected}")
            assert result == expected

    def test_build_tree2(self):
        data = ["0,0,0,no\n", "0,1,1,yes\n", "1,0,0,no\n", "1,1,0,yes\n", "0,0,1,no\n", "0,1,0,no\n", "1,1,1,no\n"]
        
        class DeterministicID3(ID3):
            def __init__(self, input_data, classification_type):
                super().__init__(input_data, classification_type)

            def randomize(self, gains):
                return gains
            
            def roulette(self, gains):
                return max(gains, key=gains.get)
        
        id3 = DeterministicID3(data, "entropy")
        root = id3.root
        
        test_cases = [
            (["0", "0", "0"], "no\n"),
            (["1", "0", "0"], "no\n"),
            (["1", "1", "0"], "yes\n"),
            (["0", "0", "1"], "no\n"),
            (["0", "1", "0"], "no\n"),
            (["1", "1", "1"], "no\n"),
            (["0", "1", "1"], "yes\n")
        ]

        for attributes, expected in test_cases:
            result = id3.classify(attributes, root)
            print(f"Input: {attributes} -> Output: {result} | Expected: {expected}")
            assert result == expected        

if __name__ == "__main__":
    unittest.main()