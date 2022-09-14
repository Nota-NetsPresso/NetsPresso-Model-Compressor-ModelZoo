## Best Practice - Tensorflow
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
