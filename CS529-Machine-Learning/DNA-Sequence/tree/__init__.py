from typing import Dict

#This class is used to generate the Nodes
class Node(object):
    def __init__(self, attribute: int = -1, children: Dict[int, 'Node'] = None):
        self.attr = attribute
        self.children = {} if children is None else children

    def __repr__(self):
        return f'<Node><Attribute value={self.attr}/><Children>{[c for c in self.children]}</Children></Node>'

    def __str__(self):
        return f'Node{self.attr}'
    #Child Node
    def add_child(self, k: int, v: 'Node'):
        self.children[k] = v

#This class is used for generating the Leaf Node
class LeafNode(Node):
    def __init__(self, label: int):
        super().__init__()
        self.label = label

    def __repr__(self):
        return f'<Node><Label value={self.label}/></Node>'

    def __str__(self):
        return f'Leaf: {self.label}'
