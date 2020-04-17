from enum import Enum, auto
"""
The output node id for the model saver and the prediction result of the input data.
User can choose what to save in the model generation, by linking the corresponding
node to the output node.

load trained model, load input, choose mode (eval or train), save model, path
"""

class Monitor(Enum):
    MonitorFinal = auto()


class MonitorWrapper(object):
    def __init__(self, name):
        super(MonitorWrapper, self).__init__()
        self.name = name

    def get_description(self):
        raise NotImplementedError


class MonitorFinal(MonitorWrapper):
    def __init__(self, name):
        super(MonitorFinal, self).__init__(name)