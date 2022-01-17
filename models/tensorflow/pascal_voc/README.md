## Original Model

The original model used the model provided by https://github.com/david8862/keras-YOLOv3-model-set.



## Simple Compression 1 

Stage 1: Filter Decomposition (Tucker Decomposition)

![recommendation_fd_0.2](./images/simple_fd_02.png?style=centerme)

Stage 2: Pruning (L2 Norm)

![recommendation_pruning_0.6](./images/simple_p_06.png?style=centerme)


## Simple Compression 2

Stage 1: Pruning (L2 Norm)

![recommendation_pruning_0.8](./images/simple_p_08.png?style=centerme)


## Compressed Results of [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)

|        Network        |      Type      |  mAP@IoU=0.5  |    FLOPs (M)     |  Params (M)   | Model Size (MB) |              Download Link               |
| :-------------------: | :------------: | :-----------: | :--------------: | :-----------: | :-------------: | :--------------------------------------: |
| YOLOv4_EfficientNetB1 |    Original    |     82.22     |     61871.82     |     65.42     |     262.90      | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_original.h5) |
|                       | NPTK-Simple  1 | 87.23 (+5.01) | 11459.69 (5.4x)  | 10.66 (6.14x) |  44.12 (5.96x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_fd_p.h5) |
|                       | NPTK-Simple 2  | 87.91 (+5.69) | 14437.96 (4.29x) | 10.79 (6.06x) |  44.36 (5.93x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_p.h5) |
