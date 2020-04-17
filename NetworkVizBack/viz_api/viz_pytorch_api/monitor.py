import torch
from viz_api import monitor

class MonitorFinal_Torch(monitor.MonitorFinal):
    def __init__(self, epochs: int, model_save_path: str, device_name: str, name: str="MonitorFinal_Torch"):
        super(MonitorFinal_Torch, self).__init__(name)
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.device = torch.device(device_name)

    def get_description(self):
        return "Monitor for model"