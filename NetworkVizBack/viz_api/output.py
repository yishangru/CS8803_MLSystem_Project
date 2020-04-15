"""
The output node id for the model saver and the prediction result of the input data.
User can choose what to save in the model generation, by linking the corresponding
node to the output node.

Saving Condition checker:
    1. certain number of epoch reaching
    2. end of training
    3. higher accuracy (check accuracy for each epoch)

1. Model Saver
2. Data Saver (certain tensor in model)
- image: use torchvision.utils (provide tensor) - https://pytorch.org/docs/stable/torchvision/utils.html;
- prediction:
"""

class ConditionCheckerWrapper(object):
    def __init__(self, name: str):
        super(ConditionCheckerWrapper, self).__init__()
        self.name = name

    def add_condition(self):
        pass

    def check_condition(self):
        pass