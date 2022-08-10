## Best Practice - Pytorch
### CIFAR100 models
The original CIFAR100 models are from [here](https://github.com/chenyaofo/pytorch-cifar-models). 
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice Link|
|:---:|:---|:---|:---|:---|:---|:---:|
|VGG16|P (0.4)|71.89 (-2.11)|433.00 (1.45x)|5.16 (2.97x)|24.52 (2.91x)|[link](./cifar100_models/VGG16.ipynb)|
|VGG16|P (0.7)|67.67 (-6.23)|212.74 (2.96x)|1.25 (12.19x)|11.34 (6.32x)|[link](./cifar100_models/VGG16.ipynb)|
|MobileNetV2|P (0.5)|73.42 (-0.87)|111.96 (1.60x)|0.79 (2.96x)|24.50 (1.89x)|[link](./cifar100_models/MobileNetV2.ipynb)|
|RepVGG|P (0.5)|74.80 (-1.64)|1637.71 (1.04x)|10.62 (1.22x)|113.35 (2.19x)|[link](./cifar100_models/RepVGG.ipynb)|
|RepVGG|P (0.75)|70.19 (-6.25)|725.71 (2.36x)|3.00 (4.32x)|51.69 (4.80x)|[link](./cifar100_models/RepVGG.ipynb)|

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
