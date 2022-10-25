## GraphModule(torch.fx.GraphModule) Conversion Guide


> GraphModule is the Graph Representation format of torch.fx.



### Sample Models

> TODO: add some models from S3 bucket



### How To Convert to GraphModule

#### 1. Convert to GraphModule

```python
import torch.fx
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)
torch.save(traced_model, "resnet18.pt")
```



#### 2. Inference test of GraphModule

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





### 3. References

- https://pytorch.org/docs/stable/fx.html



### 4. Q&A or Discussion

- https://github.com/Nota-NetsPresso/Discussion/discussions/categories/feedbacks-model-compressor