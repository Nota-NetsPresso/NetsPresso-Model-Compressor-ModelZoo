## Best Practice - Tensorflow
### CIFAR100 models
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|.ipynb|Colab|
|:---:|:---|:---|:---|:---|:---|:---:|---|
|VGG19|FD (0.0) &rarr; P (0.4)|71.13 (-1.15)|132.20 (6.03x)|1.17 (17.13x)|12.85 (14.73x)|[link](./cifar100_models/VGG19.ipynb)|[![](https://colab.research.google.com/assets/colab-badge.svg)]()|
|ResNet50|P (0.5) &rarr; FD (0.2)|76.92 (-1.11)|613.43 (4.23x)|2.64 (8.99x)|130.39 (3.45x)|[link](./cifar100_models/ResNet50.ipynb)|[![](https://colab.research.google.com/assets/colab-badge.svg)]()|
|MobileNetV1|FD(0.0) &rarr; P(0.5)| 66.32 (-0.36)|26.09 (3.56x)| 0.53 (6.24x)|3.66 (9.73x)|[link](./cifar100_models/MobileNetV1.ipynb)|[![](https://colab.research.google.com/assets/colab-badge.svg)]()|

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).

### PASCAL VOC models
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|.ipynb|Colab|
|:---:|:---|:---|:---|:---|:---|:---:|---|
|YOLOv4|FD(0.2) &rarr; P(0.6)|87.23 (+5.01)|11459.69 (5.4x)|2.75 (7.49x)||[link](./pascal_voc_models/YOLOv4.ipynb)|[![](https://colab.research.google.com/assets/colab-badge.svg)]()|
|YOLOv4|P (0.8)|87.91 (+5.69)|14437.96 (4.29x)|10.71 (6.1x)||[link](./pascal_voc_models/TF_YOLOv4_0_8.ipynb)|[![](https://colab.research.google.com/assets/colab-badge.svg)]()|
