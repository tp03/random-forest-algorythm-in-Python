import unittest
import sys
sys.path.insert(0, '../')
from TreeNode import TreeNode

class TestTreeNode(unittest.TestCase):
    def test_node_initialization(self):
        node = TreeNode(attribute_index=0, isLeaf=True, answer="yes")
        self.assertEqual(node.attribute_index, 0)
        self.assertTrue(node.isLeaf)
        self.assertEqual(node.answer, "yes")
        self.assertEqual(node.children, {})

    def test_add_child(self):
        parent = TreeNode(attribute_index=0)
        child = TreeNode(isLeaf=True, answer="yes")
        parent.add_child("high", child)
        self.assertIn("high", parent.children)
        self.assertEqual(parent.children["high"], child)