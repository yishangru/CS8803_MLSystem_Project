from enum import Enum, auto
"""
The output node id for the model saver and the prediction result of the input data.
User can choose what to save in the model generation, by linking the corresponding
node to the output node.

load trained model, load input, choose mode (eval or train), save model, path
"""

class MonitorType(Enum):
    MonitorFinal = auto()
    MonitorSaver = auto()


class MonitorWrapper(object):
    def __init__(self, name):
        super(MonitorWrapper, self).__init__()
        self.name = name

    @staticmethod
    def get_description(self):
        raise NotImplementedError


class MonitorFinal(MonitorWrapper):
    def __init__(self, name):
        super(MonitorFinal, self).__init__(name)

    def save_model(self, *layer):
        raise NotImplementedError


class MonitorSaver(MonitorWrapper):
    def __init__(self, name):
        super(MonitorSaver, self).__init__(name)

    def save_output(self, input_to_save, *following_processing):
        raise NotImplementedError