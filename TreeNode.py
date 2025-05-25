class TreeNode:
    def __init__(self, attribute_index=None, isLeaf=False, answer=None):
        self.attribute_index = attribute_index
        self.isLeaf = isLeaf
        self.answer = answer
        self.children = {}

    def add_child(self, attribute_value, child_node):
        self.children[attribute_value] = child_node