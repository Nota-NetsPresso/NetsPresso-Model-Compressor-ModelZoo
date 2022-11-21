# Model Zoo for the NetsPresso Compression Toolkit
## Provided Model Descriptions
* ### ImageNet Pretrained Models - /models/torch/torchvision [(link)](https://pytorch.org/vision/0.8/models.html#classification)
  * Following models are now available on [NetsPresso Compression Toolkit](https://compression.netspresso.ai/).
    |       Model       |  FD | Pruning |   ONNX & torch Version    |
    |:-----------------:|:---:|:-------:|:-------------:|
    |      Alexnet      | Yes |   Yes   | 1.10.x        |
    |       VGG16       | Yes |   Yes   | 1.10.x        |
    |      ResNet18     | Yes |   Yes   | 1.10.x        |
    |   SqueezeNet1_0   | Yes |   Yes   | 1.10.x        |
    |    DenseNet161    | Yes |   Yes   | 1.10.x        |
    |     MobileNet_v2    | Yes |   Yes   | 1.10.x        |
    |    MobileNet_v3_large    | Yes |   Yes   | 1.10.x        |
    |    MobileNet_v3_small    | Yes |   Yes   | 1.10.x        |
    |     Wide ResNet50_2     | Yes |   Yes   | 1.10.x        |
    |    MNASNet1_0    | Yes |   Yes   | 1.10.x        |
    |    EfficientNet_b0    | Yes |   Yes   | 1.10.x        |
    |    ResNext50_32x4d    | Yes |   Yes   | 1.10.x        |
    |    RegNet_y_400mf    | Yes |   Yes   | 1.10.x        |
    | RegNet_x_400mf | Yes |   Yes   | 1.10.x        |
* ### Subset of COCO Pretrained models - /models/torch/torchvision [(link)](https://pytorch.org/vision/0.8/models.html#semantic-segmentation)
  * Following models are now available on [NetsPresso Compression Toolkit](https://compression.netspresso.ai/).
    |       Model       |  FD | Pruning |   ONNX & torch Version    |
    |:-----------------:|:---:|:-------:|:-------------:|
    | FCN_ResNet101 | Yes | Yes | 1.10.x        |



## Conversion of PyTorch into GraphModule (torch.fx.GraphModule)

* ### Convert to GraphModule (example script)

```python
import torch.fx
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)
torch.save(traced_model, "resnet18.pt")
```



* ### Inference test of GraphModule

```python
import torch
import torch.fx
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)

# input size is needed to be choosen
input_shape = (1, 3, 224, 224)
random_input = torch.Tensor(np.random.randn(*input_shape))

with torch.no_grad():
    original_output = model(random_input)
    traced_output = traced_model(random_input)

assert torch.allclose(original_output, traced_output), "inference result is not equal!"
```



* ### Load GraphModule
```python
import torch

model = torch.load("resnet18.pt")

```


* ### fx Reference

  - https://pytorch.org/docs/stable/fx.html




## Conversion of PyTorch into ONNX

* ### Input shape of the model should be specified to convert to ONNX.
* ### The ordering of the dimensions in the inputs is **(batch_size, channels, height, width)**.

### **Case 1 : torchvision**

* ### To call onnx.export
  * Model (Required)
  * Input value (Required)
  * Output value (Optional)



* ### Example script
```python
from torchvision.models import resnet18
import torch
from torch.onnx import TrainingMode

input_tensor = torch.rand(torch.Size([1, 3, 224, 224]))
model = resnet18(pretrained=True)
dummy_output = model(input_tensor)
torch.onnx.export(model, input_tensor, "resnet18.onnx", verbose=True, example_outputs=dummy_output, training=TrainingMode.TRAINING)
```


### **Case 2 : custom model**

* ### Example script
```python
# 1. define a simple model
import torch
from torch.nn import functional as F
from torch import nn
from torch.onnx import TrainingMode

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(54 * 54 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 54 * 54 * 50) # [batch_size, 50, 4, 4]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. train
# put your training code

# 3. convert torch to onnx
simple_model= Net()
input_tensor = torch.rand(torch.Size([1, 3, 224, 224]))
dummy_output = simple_model(input_tensor)
torch.onnx.export(simple_model, input_tensor, "simple_model.onnx", verbose=True, example_outputs=dummy_output, training=TrainingMode.TRAINING)
```





## Q&A or Discussion

https://github.com/Nota-NetsPresso/Discussion/discussions/categories/feedbacks-model-compressor
