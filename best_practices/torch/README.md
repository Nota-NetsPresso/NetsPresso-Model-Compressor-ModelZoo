## Best Practice - Pytorch
### CIFAR100 models
The original CIFAR100 models are from [here](https://github.com/chenyaofo/pytorch-cifar-models). 
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice|
|:---:|:---|:---|:---|:---|:---|:---:|
|VGG16|Original|74.00|629.20|15.30|71.65|
|VGG16|P (0.4)|71.89 (-2.11)|433.00 (1.45x)|5.16 (2.97x)|24.52 (2.91x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/VGG16.ipynb)|
|VGG16|P (0.7)|67.67 (-6.23)|212.74 (2.96x)|1.25 (12.19x)|11.34 (6.32x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/VGG16.ipynb)|
|MobileNetV2|Original|74.29|179.46|2.33|46.26|
|MobileNetV2|P (0.5)|73.42 (-0.87)|111.96 (1.60x)|0.79 (2.96x)|24.50 (1.89x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/MobileNetV2.ipynb)|
|RepVGG|Original|76.44|1709.31|12.94|248.10|
|RepVGG|P (0.5)|74.80 (-1.64)|1637.71 (1.04x)|10.62 (1.22x)|113.35 (2.19x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/RepVGG.ipynb)|
|RepVGG|P (0.75)|70.19 (-6.25)|725.71 (2.36x)|3.00 (4.32x)|51.69 (4.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/RepVGG.ipynb)|

### COCO models
|Model|Type|mAP (0.5:0.95) (%)|FLOPs (G)|Params (M)|Latency (ms)|Model Size (MB)|Best Practice|
|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|YOLOX|Original|49.7|156.01|54.21|12239.46|207.37||
|YOLOX|Compressed-1|48.36 (-1.34)|101.80 (1.53x)|19.96 (2.7x)|8502.72 (1.44x)|76.61 (2.7x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/coco_models/YOLOX/YOLOX.ipynb)|
|YOLOX|Compressed-2|42.95 (-6.65)|38.61 (4.04x)|4.93 (11.0x)|4235.37 (2.89x)|19.17 (10.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/coco_models/YOLOX/YOLOX.ipynb)|

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
