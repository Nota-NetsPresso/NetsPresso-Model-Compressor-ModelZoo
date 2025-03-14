<!--
Please notice to the manager when you change the contents of the tables.
-->
# Best Practices
<div align=right>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNota-NetsPresso%2FNetsPresso-Model-Compressor-ModelZoo&count_bg=%23327EEA&title_bg=%23555555&icon=&icon_color=%231ABAFD&title=hits&edge_flat=false"/></a>
</div>

## Classification
### TF-Keras
The original CIFAR100 models are from [here](https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/blob/main/models/tensorflow/cifar100.md). 
|Model|Type|Dataset|Top-1 Accuracy<br> (%)|FLOPs<br> (M)|Params<br> (M)|Latency<br> (ms)|Model Size<br> (MB)|Best<br> Practice|
|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|VGG19|Original|CIFAR100|72.28|796.79|20.09|189.31|78.69|
|VGG19|Compressed|CIFAR100|71.13 (-1.15)|132.20 (6.03x)|1.17 (17.13x)|12.85 (14.73x)|4.98 (15.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/tf_keras/vgg19_cifar100.ipynb)|
|VGG19|Compressed (Adv.)|CIFAR100|71.14 (-1.14)|100.09 (7.96x)|0.66 (30.38x)|4.5 (42.06x)|5.68 (13.85x)||
|ResNet50|Original|CIFAR100|78.03|2596.06|23.71|450.14|93.31|
|ResNet50|Compressed|CIFAR100|76.92 (-1.11)|613.43 (4.23x)|2.64 (8.99x)|130.39 (3.45x)|9.83 (9.49x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/tf_keras/resnet50_cifar100.ipynb)|
|ResNet50|Compressed (Adv.)|CIFAR100|76.63 (-1.4)|224.70 (11.55x)|2.17 (10.91x)|48.37 (9.31x)|18.35 (5.09x)||
|MobileNetV1|Original|CIFAR100|66.68|92.90|3.31|35.61|13.28|
|MobileNetV1|Compressed|CIFAR100| 66.32 (-0.36)|26.09 (3.56x)| 0.53 (6.24x)|3.66 (9.73x)|2.38 (5.58x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices//classification/tf_keras/mobilenetv1_cifar100.ipynb)|
|MobileNetV1|Compressed (Adv.)|CIFAR100|66.11 (-0.57)|17.90 (5.19x)|0.35 (9.35x)|2.08 (17.12x)|3.3 (4.02x)||

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, TFlite

### PyTorch
|Model|Type|Dataset|Top-1 Accuracy<br> (%)|FLOPs<br> (M)|Params<br> (M)|Latency<br> (ms)|Model Size<br> (MB)|Best<br> Practice|
|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---:|
|VGG16|Original|CIFAR100|74.00|629.76|15.30|71.65|59.65|
|VGG16|Compressed-1|CIFAR100|72.22 (-1.78)|431.84 (1.46x)|5.16 (2.96x)|24.52 (2.91x)|20.22 (2.95x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/vgg16_cifar100.ipynb)|
|VGG16|Compressed-2|CIFAR100|68.01 (-5.99)|213.06 (2.96x)|1.25 (12.26x)|11.34 (6.32x)|4.93 (12.10x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/vgg16_cifar100.ipynb)|
|MobileNetV2|Original|CIFAR100|74.29|189.30|2.35|46.26|8.98|
|MobileNetV2|Compressed|CIFAR100|73.68 (-0.61)|119.09 (1.59x)|0.82 (2.88x)|24.50 (1.89x)|3.38 (2.66x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/mobilenetv2_cifar100.ipynb)|
|RepVGG|Original|CIFAR100|76.44|1715.70|12.94|248.10|50.33|
|RepVGG|Compressed-1|CIFAR100|74.92 (-1.52)|1644.88 (1.04x)|10.64 (1.22x)|113.35 (2.19x)|41.81 (1.20x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/repvgg_cifar100.ipynb)|
|RepVGG|Compressed-2|CIFAR100|69.84 (-4.60)|721.77 (2.38x)|2.95 (4.39x)|51.69 (4.80x)|11.71 (4.30x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/repvgg_cifar100.ipynb)|
|ViT|Original|CIFAR100|94.42|33725.76|85.80|1396.53|327.43||
|ViT|Compressed|CIFAR100|93.30 (-1.12)|14804.95 (2.28x)|37.78 (2.27x)|737.11 (1.89x)|144.32 (2.27x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/classification/torch/vit/vit_cifar100.ipynb)|

The original CIFAR100 models are from [here](https://github.com/chenyaofo/pytorch-cifar-models). 

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, ONNX runtime

## Object Detection
### TF-Keras
|Model|Type|Dataset|mAP<br> (0.5)<br> (%)|mAP<br> (0.5:0.95)(%)|FLOPs<br> (M)|Params<br> (M)|Latency<br> (ms)|Model Size<br> (MB)|Best Practice|
|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|
|[YOLOv4](https://github.com/david8862/keras-YOLOv3-model-set)|Original|PASCAL VOC|82.22|-|61871.82|65.32|64318.70|262.90||
|YOLOv4|Compressed-1|PASCAL VOC|87.23 (+5.01)|-|11459.69 (5.4x)|10.59 (6.17x)|28651.70 (2.16x)|44.12 (5.96x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/tf_keras/yolov4_voc.ipynb)|
|YOLOv4|Compressed-2|PASCAL VOC|87.91 (+5.69)|-|14442.96 (4.28x)|10.71 (6.1x)|28976.40 (2.14x)|44.36 (5.93x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/tf_keras/yolov4_voc.ipynb)|

＊YOLOv4 model with EfficientNet B1 based backbone.

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, TFlite

### PyTorch
|Model|Type|Dataset|mAP<br> (0.5)<br> (%)|mAP<br> (0.5:0.95)(%)|FLOPs<br> (M)|Params<br> (M)|Latency<br> (ms)|Model Size<br> (MB)|Best Practice|
|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|
|[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|Original|COCO|68.0|49.7|156006.20|54.21|12239.46|207.37||
|YOLOX|Compressed-1|COCO|67.16 (-0.84)|48.64 (-1.06)|101804.06 (1.53x)|19.96 (2.7x)|8502.72 (1.44x)|76.61 (2.7x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/torch/yolox_coco/YOLOX.ipynb)|
|YOLOX|Compressed-2|COCO|61.43 (-6.57)|43.47 (-5.47)|38607.03 (4.04x)|4.93 (11.0x)|4235.37 (2.89x)|19.17 (10.80x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/torch/yolox_coco/YOLOX.ipynb)|
|[YOLOv7](https://github.com/WongKinYiu/yolov7)|Original|PASCAL VOC|89.6|71.3|104739.64|37.27|5464.59|146.33||
|YOLOv7|Compressed-1|PASCAL VOC|88.4 (-1.2)|69.6 (-1.7)|77859.81 (1.35x)|17.12 (2.18x)|3855.95 (1.42x)|67.45 (2.00x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/torch/yolov7_voc/YOLOv7.ipynb)|
|YOLOv7|Compressed-2|PASCAL VOC|85.2 (-4.4)|63.6 (-7.7)|21878.87 (4.79x)|2.55 (14.60x)|2041.65 (2.68x)|10.61 (13.79x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/blob/main/best_practices/object_detection/torch/yolov7_voc/YOLOv7.ipynb)|

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, ONNX runtime

## Semantic Segmentation
### PyTorch
|Model|Type|Dataset|mIoU<br> (%)|Global<br> Correct<br> (%)|FLOPs<br> (M)|Params<br> (M)|Latency<br> (ms)|Model<br> Size<br> (MB)|Best Practice|
|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|
|[FCN ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html)|Original|COCO|60.5|91.4|306554.91|35.32|13167.17|135.09||
|FCN ResNet50|Compressed-1|COCO|59.6 (-0.9)|91.4 (-0.0)|156106.03 (1.96x)|17.58 (2.01x)|6438.06 (2.04x)|67.34 (2.01x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/blob/main/best_practices/semantic_segmentation/torch/fcn_resnet50_coco/fcn_resnet50.ipynb)|
|FCN ResNet50|Compressed-2|COCO|54.7 (-5.8)|90.7 (-0.7)|45826.66 (x6.68)|4.84 (7.31x)|2147.92 (6.13x)|18.70 (7.22x)|[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/blob/main/best_practices/semantic_segmentation/torch/fcn_resnet50_coco/fcn_resnet50.ipynb)|
* We used a subset of COCO dataset to fine-tuning FCN ResNet50. You can check more details of dataset [here](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html).

The model's latency is measured using a Raspberry Pi 4B (1.5GHz ARM Cortex).  
Options: FP32, ONNX runtime
<!-- - [torch](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/)
  - [cifar100_models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/cifar100_models)
  - [coco models](https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo/tree/main/best_practices/torch/coco_models) -->
