## Best Practice - Tensorflow
### CIFAR100 models
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice Link|
|:---:|:---|:---|:---|:---|:---|:---:|
|VGG19|FD (0.0) &rarr; P (0.4)|71.13 (-1.15)|132.20 (6.03x)|1.17 (17.13x)|12.85 (14.73x)|[link](./CIFAR100_models/TF_VGG19.ipynb)|
|ResNet50|P (0.5) &rarr; FD (0.2)|76.92 (-1.11)|613.43 (4.23x)|2.64 (8.99x)|130.39 (3.45x)|[link](./CIFAR100_models/TF_ResNet50.ipynb)|
|MobileNetV1|FD(0.0) &rarr; P(0.5)| 66.32 (-0.36)|26.09 (3.56x)| 0.53 (6.24x)|3.66 (9.73x)|[link](./CIFAR100_models/TF_MobileNetV1.ipynb)|

### PASCAL VOC models
|Model|NPTK Process|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Best Practice Link|
|:---:|:---|:---|:---|:---|:---|:---:|
|YOLOv4|FD(0.2) &rarr; P(0.6)|87.23 (+5.01)|11459.69 (5.4x)|2.75 (7.49x)||[link](./PASCAL_VOC_models/TF_YOLOv4.ipynb)|
|YOLOv4|P (0.8)|87.91 (+5.69)|14437.96 (4.29x)|10.71 (6.1x)||[link](./PASCAL_VOC_models/TF_YOLOv4_0_8.ipynb)|

