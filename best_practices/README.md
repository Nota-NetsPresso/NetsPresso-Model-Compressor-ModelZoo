<div align=right>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNota-NetsPresso%2FNetsPresso-Model-Compressor-ModelZoo&count_bg=%23327EEA&title_bg=%23555555&icon=&icon_color=%231ABAFD&title=hits&edge_flat=false"/></a>
</div>

## Workflow

  <p align="center">
    <img src="/imgs/workflow-MC.png" alt="Workflow">
  </p>


## Tensorflow
### CIFAR100 models
|Model|Type|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Model Size (MB)|Best Practice|
|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|VGG19|Original|72.28|796.79|20.09|189.31|78.69|
|VGG19|Compressed|71.13 (-1.15)|132.20 (6.03x)|1.17 (17.13x)|12.85 (14.73x)|4.98 (15.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/VGG19.ipynb)|
|VGG19|Compressed (Adv.)|71.14 (-1.14)|100.09 (7.96x)|0.66 (30.38x)|4.5 (42.06x)|5.68 (13.85x)||
|ResNet50|Original|78.03|2596.06|23.71|450.14|93.31|
|ResNet50|Compressed|76.92 (-1.11)|613.43 (4.23x)|2.64 (8.99x)|130.39 (3.45x)|9.83 (9.49x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/ResNet50.ipynb)|
|ResNet50|Compressed (Adv.)|76.63 (-1.4)|224.70 (11.55x)|2.17 (10.91x)|48.37 (9.31x)|18.35 (5.09x)||
|MobileNetV1|Original|66.68|92.90|3.31|35.61|13.28|
|MobileNetV1|Compressed| 66.32 (-0.36)|26.09 (3.56x)| 0.53 (6.24x)|3.66 (9.73x)|2.38 (5.58x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/cifar100_models/MobileNetV1.ipynb)|
|MobileNetV1|Compressed (Adv.)|66.11 (-0.57)|17.90 (5.19x)|0.35 (9.35x)|2.08 (17.12x)|3.3 (4.02x)||

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).
Options: FP32, 

### PASCAL VOC models
|Model|Type|mAP (0.5) (%)|FLOPs (M)|Params (M)|Latency (ms)|Model Size (MB)|Best Practice|
|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|YOLOv4|Original|82.22|61871.82|65.32|1324.42|262.90||
|YOLOv4|Compressed-1|87.23 (+5.01)|11459.69 (5.4x)|10.59 (6.17x)|639.12 (2.16x)|44.12 (5.96x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/pascal_voc_models/YOLOv4.ipynb)|
|YOLOv4|Compressed-2|87.91 (+5.69)|14437.96 (4.29x)|10.71 (6.1x)|631.90 (2.10x)|44.36 (5.93x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/tensorflow/pascal_voc_models/YOLOv4.ipynb)|

The model's latency is measured using a Intel Xeon CPU (2.00GHz x86_64).

- [tensorflow-keras](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/)
  - [cifar100_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/cifar100_models)
  - [pascal_voc_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/tensorflow/pascal_voc_models)


## Pytorch
### CIFAR100 models
The original CIFAR100 models are from [here](https://github.com/chenyaofo/pytorch-cifar-models). 
|Model|Type|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Model Size (MB)|Best Practice|
|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|VGG16|Original|74.00|629.20|15.30|71.65|59.91|
|VGG16|Compressed-1|71.89 (-2.11)|433.00 (1.45x)|5.16 (2.97x)|24.52 (2.91x)|20.19 (2.97x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/VGG16.ipynb)|
|VGG16|Compressed-2|67.67 (-6.23)|212.74 (2.96x)|1.25 (12.19x)|11.34 (6.32x)|4.90 (12.23x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/VGG16.ipynb)|
|MobileNetV2|Original|74.29|179.46|2.33|46.26|9.36|
|MobileNetV2|Compressed|73.42 (-0.87)|111.96 (1.60x)|0.79 (2.96x)|24.50 (1.89x)|3.33 (2.81x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/MobileNetV2.ipynb)|
|RepVGG|Original|76.44|1709.31|12.94|248.10|51.09|
|RepVGG|Compressed-1|74.80 (-1.64)|1637.71 (1.04x)|10.62 (1.22x)|113.35 (2.19x)|41.74 (1.22x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/RepVGG.ipynb)|
|RepVGG|Compressed-2|70.19 (-6.25)|725.71 (2.36x)|3.00 (4.32x)|51.69 (4.80x)|11.62 (4.40x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/cifar100_models/RepVGG.ipynb)|

### COCO models
|Model|Type|mAP (0.5:0.95) (%)|FLOPs (G)|Params (M)|Latency (ms)|Model Size (MB)|Best Practice|
|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|YOLOX|Original|49.7|156.01|54.21|12239.46|207.37||
|YOLOX|Compressed-1|48.36 (-1.34)|101.80 (1.53x)|19.96 (2.7x)|8502.72 (1.44x)|76.61 (2.7x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/coco_models/YOLOX/YOLOX.ipynb)|
|YOLOX|Compressed-2|42.95 (-6.65)|38.61 (4.04x)|4.93 (11.0x)|4235.37 (2.89x)|19.17 (10.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/torch/coco_models/YOLOX/YOLOX.ipynb)|


The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, ONNX runtime
- [torch](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/)
  - [cifar100_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/cifar100_models)
  - [coco models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/coco_models)
