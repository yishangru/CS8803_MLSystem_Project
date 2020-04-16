from enum import Enum, auto

"""
For time limit, implement following for simplicity 
    1. flat (inplace/not inplace)
    2. data_clamp (inplace/ not inplace)
    3. detach (inplace/ not inplace)
    4. gram matrix
    5. add
"""

class Transform(Enum):
    FlatTransform = auto()
    DataClampTransform = auto()
    DetachTransform = auto()
    AddTransform = auto()
    GetGramMatrix = auto()

class TransformWrapper(object):
    def __init__(self, name: str):
        super(TransformWrapper, self).__init__()
        self.name = name


class FlatTransform(TransformWrapper):
    def __init__(self, name: str):
        super(FlatTransform, self).__init__(name)


class DataClampTransform(TransformWrapper):
    def __init__(self, name: str):
        super(DataClampTransform, self).__init__(name)


class DetachTransform(TransformWrapper):
    def __init__(self, name: str):
        super(DetachTransform, self).__init__(name)


class AddTransform(TransformWrapper):
    def __init__(self, name: str):
        super(AddTransform, self).__init__(name)


class GetGramMatrix(TransformWrapper):
    def __init__(self, name: str):
        super(GetGramMatrix, self).__init__(name)