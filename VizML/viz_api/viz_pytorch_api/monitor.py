import os
import torch
from torchvision.utils import save_image
from viz_api import monitor
from viz_api import input
from viz_api.viz_pytorch_api import input as input_torch
from viz_api import tensor

class MonitorFinal_Torch(monitor.MonitorFinal):
    def __init__(self, epochs: int, model_save_path: str, device_name: str, name: str="MonitorFinal_Torch"):
        super(MonitorFinal_Torch, self).__init__(name)
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.device = torch.device(device_name)

    def save_model(self, *layers): # not implement yet
        for layer in list(layers):
            pass

    @staticmethod
    def get_description():
        return "Monitor for model"


"""
To do:
    1. Change the MonitorSaver to different class for support different requirements (tensor, img, img data set)
    2. Store more parameter passing when initialization class 
"""
class MonitorSaver_Torch(monitor.MonitorSaver):
    def __init__(self, model_save_path: str, name: str="MonitorSaver_Torch"):
        super(MonitorSaver_Torch, self).__init__(name)
        self.model_save_path = model_save_path

    # can save input, constant, tensor --- add following post processing (Not support for following process right now)
    def save_output(self, input_to_save, *following_processing):
        def save_tensor(tensor_to_save):
            linked_tensor = tensor_to_save.get_linked_tensor()
            torch.save(obj=linked_tensor,
                       f=os.path.join(self.model_save_path, tensor_to_save.name + ".pt"))

        if isinstance(input_to_save, input.InputWrapper):

            if isinstance(input_to_save, input_torch.ImageLoader_Torch): # single image
                linked_image_tensor = input_to_save.get_loaded_tensor()
                save_image(tensor=linked_image_tensor.get_linked_tensor(),
                           fp=os.path.join(self.model_save_path, linked_image_tensor.name + ".jpg"), normalize=True)
            elif isinstance(input_to_save, input_torch.ConstantLoader_Torch): # constant tensor
                linked_constant_tensor = input_to_save.get_loaded_tensor()
                save_tensor(linked_constant_tensor)
            elif isinstance(input_to_save, input_torch.RandomLoader_Torch): # random tensor
                linked_random_tensor = input_to_save.get_loaded_tensor()
                save_tensor(linked_random_tensor)
            elif isinstance(input_to_save, input_torch.TensorLoader_Torch): # loaded tensor
                linked_load_tensor = input_to_save.get_loaded_tensor()
                save_tensor(linked_load_tensor)

        elif isinstance(input_to_save, input.ConstantWrapper):
            if isinstance(input_to_save, input_torch.ImageConstant_Torch):  # single image
                linked_image_tensor = input_to_save.get_saved_tensor()
                save_image(tensor=linked_image_tensor.get_linked_tensor(),
                           fp=os.path.join(self.model_save_path, linked_image_tensor.name + ".jpg"), normalize=True)
            elif isinstance(input_to_save, input_torch.ConstantConstant_Torch):  # constant tensor
                linked_constant_tensor = input_to_save.get_saved_tensor()
                save_tensor(linked_constant_tensor)
            elif isinstance(input_to_save, input_torch.RandomConstant_Torch):  # random tensor
                linked_random_tensor = input_to_save.get_saved_tensor()
                save_tensor(linked_random_tensor)
            elif isinstance(input_to_save, input_torch.TensorConstant_Torch):  # loaded tensor
                linked_load_tensor = input_to_save.get_saved_tensor()
                save_tensor(linked_load_tensor)

        elif isinstance(input_to_save, tensor.TensorWrapper):
            save_tensor(input_to_save)

    @staticmethod
    def get_description():
        return "Monitor for variable saver"


# --------------------- test saver --------------------- #
def test_monitor_saver():
    monitor_saver = MonitorSaver_Torch("./")

    # --------------------- test input --------------------- #
    def test_save_img_loader():
        imageHolder = input_torch.ImageLoader_Torch(image_path="../../static/img/boat.jpg", imsize=600,
                                        device=torch.device("cuda:0"))
        monitor_saver.save_output(imageHolder)

    def test_tensor_loader():
        tensorHolder_input = input_torch.TensorLoader_Torch(tensor_path="test.pt", device=torch.device("cuda:0"))
        monitor_saver.save_output(tensorHolder_input)
        tensorHolder_output = input_torch.TensorLoader_Torch(tensor_path=tensorHolder_input.get_loaded_tensor().name + ".pt", device=torch.device("cuda:0"))
        print(tensorHolder_input.get_loaded_tensor().get_linked_tensor().size(),
              tensorHolder_output.get_loaded_tensor().get_linked_tensor().size())
        print(torch.eq(tensorHolder_input.get_loaded_tensor().get_linked_tensor(),
                       tensorHolder_output.get_loaded_tensor().get_linked_tensor()))

    def test_random_loader():
        randomHolder = input_torch.RandomLoader_Torch([10, 20, 30, 40], device=torch.device("cuda:0"))
        monitor_saver.save_output(randomHolder)
        tensorHolder = input_torch.TensorLoader_Torch(tensor_path=randomHolder.get_loaded_tensor().name + ".pt", device=torch.device("cuda:0"))
        print(tensorHolder.get_loaded_tensor().get_linked_tensor().size(),
              randomHolder.get_loaded_tensor().get_linked_tensor().size())
        print(torch.eq(randomHolder.get_loaded_tensor().get_linked_tensor(),
                       tensorHolder.get_loaded_tensor().get_linked_tensor()))

    def test_const_loader():
        tensorConst = input_torch.ConstantLoader_Torch([5, 5, 5, 5], value=5, device=torch.device("cuda:0"))
        monitor_saver.save_output(tensorConst)
        tensorHolder = input_torch.TensorLoader_Torch(tensor_path=tensorConst.get_loaded_tensor().name + ".pt", device=torch.device("cuda:0"))
        print(tensorConst.get_loaded_tensor().get_linked_tensor().size(),
              tensorHolder.get_loaded_tensor().get_linked_tensor().size())
        print(torch.eq(tensorHolder.get_loaded_tensor().get_linked_tensor(),
                       tensorHolder.get_loaded_tensor().get_linked_tensor()))

    # --------------------- test const --------------------- #
    def test_save_img_const():
        imageHolder = input_torch.ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=600,
                                                    device=torch.device("cuda:0"))
        monitor_saver.save_output(imageHolder)

    def test_save_tensor_const():
        tensorHolder_input = input_torch.TensorConstant_Torch(tensor_path="test.pt", device=torch.device("cuda:0"))
        monitor_saver.save_output(tensorHolder_input)
        tensorHolder_output = input_torch.TensorConstant_Torch(tensor_path=tensorHolder_input.get_saved_tensor().name + ".pt", device=torch.device("cuda:0"))
        print(tensorHolder_input.get_saved_tensor().get_linked_tensor().size(),
              tensorHolder_output.get_saved_tensor().get_linked_tensor().size())
        print(torch.eq(tensorHolder_input.get_saved_tensor().get_linked_tensor(),
                       tensorHolder_output.get_saved_tensor().get_linked_tensor()))

    def test_save_random_const():
        randomHolder = input_torch.RandomConstant_Torch([10, 20, 30, 40], device=torch.device("cuda:0"))
        monitor_saver.save_output(randomHolder)
        tensorHolder = input_torch.TensorConstant_Torch(tensor_path=randomHolder.get_saved_tensor().name + ".pt",
                                                      device=torch.device("cuda:0"))
        print(tensorHolder.get_saved_tensor().get_linked_tensor().size(),
              randomHolder.get_saved_tensor().get_linked_tensor().size())
        print(torch.eq(randomHolder.get_saved_tensor().get_linked_tensor(),
                       tensorHolder.get_saved_tensor().get_linked_tensor()))

    def test_save_constant_const():
        tensorConst = input_torch.ConstantConstant_Torch([5, 5, 5, 5], value=5, device=torch.device("cuda:0"))
        monitor_saver.save_output(tensorConst)
        tensorHolder = input_torch.TensorConstant_Torch(tensor_path=tensorConst.get_saved_tensor().name + ".pt",
                                                      device=torch.device("cuda:0"))
        print(tensorConst.get_saved_tensor().get_linked_tensor().size(),
              tensorHolder.get_saved_tensor().get_linked_tensor().size())
        print(torch.eq(tensorHolder.get_saved_tensor().get_linked_tensor(),
                       tensorHolder.get_saved_tensor().get_linked_tensor()))

    test = torch.ones(2, 2)
    torch.save(test, "test.pt")
    test_save_img_loader()
    test_random_loader()
    test_const_loader()
    test_tensor_loader()
    test_save_img_const()
    test_save_random_const()
    test_save_constant_const()
    test_save_tensor_const()

#test_monitor_saver()