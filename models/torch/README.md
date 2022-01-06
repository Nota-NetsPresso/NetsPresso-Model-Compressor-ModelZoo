# Conversion of PyTorch into ONNX
* Input shape of the model should be specified to convert to ONNX.
* The ordering of the dimensions in the inputs is **(batch_size, channels, height, width)**.

## Case 1 : torchvision

### **To call onnx.export**
* Model (Required)
* Input value (Required)
* Output value (Optional)



### **Example script**
```python
from torchvision.models import resnet18
import torch

input_tensor = torch.rand(torch.Size([1, 3, 224, 224]))
model = resnet18(pretrained=True)
dummy_output = model(rand_input)
torch.onnx.export(model, input_tensor, "resnet18.onnx", verbose=True, example_outputs=dummy_output)
```


## Case 2 : custom model

### **Example script**
```python
# 1. define a simple model
import torch
from torch.nn import functional as F
from torch import nn

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
torch.onnx.export(simple_model, input_tensor, "simple_model.onnx", verbose=True, example_outputs=dummy_output)
```
