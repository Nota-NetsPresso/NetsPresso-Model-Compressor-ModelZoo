## Workflow

## Tensorflow
### CIFAR100 models
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice|
|:---:|:---|:---|:---|:---|:---|:---:|
|VGG19|Original|72.28|796.79|20.09|189.31|
|VGG19|FD (0.0) &rarr; P (0.4)|71.13 (-1.15)|132.20 (6.03x)|1.17 (17.13x)|12.85 (14.73x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/VGG19.ipynb)|
|ResNet50|Original|78.03|2596.06|23.71|450.14|
|ResNet50|P (0.5) &rarr; FD (0.2)|76.92 (-1.11)|613.43 (4.23x)|2.64 (8.99x)|130.39 (3.45x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/ResNet50.ipynb)|
|MobileNetV1|Original|66.68|92.90|3.31|35.61|
|MobileNetV1|FD(0.0) &rarr; P(0.5)| 66.32 (-0.36)|26.09 (3.56x)| 0.53 (6.24x)|3.66 (9.73x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/MobileNetV1.ipynb)|

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).

### PASCAL VOC models
|Model|NPTK Process|mAP (0.5) (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice|
|:---:|:---|:---|:---|:---|:---|:---:|
|YOLOv4|Original|82.22|61871.82|262.90|1324.42|
|YOLOv4|FD(0.2) &rarr; P(0.6)|87.23 (+5.01)|11459.69 (5.4x)|2.75 (7.49x)|639.12 (2.16x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/pascal_voc_models/YOLOv4.ipynb)|
|YOLOv4|P (0.8)|87.91 (+5.69)|14437.96 (4.29x)|10.71 (6.1x)|631.90 (2.10x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/pascal_voc_models/YOLOv4.ipynb)|

The model's latency is measured using a Intel Xeon CPU (2.00GHz x86_64).

- [tensorflow-keras](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/)
  - [cifar100_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/cifar100_models)
  - [pascal_voc_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/pascal_voc_models)


## Pytorch
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

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  

- [torch](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/)
  - [cifar100_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/cifar100_models)
