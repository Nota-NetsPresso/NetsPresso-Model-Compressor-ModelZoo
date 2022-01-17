<<<<<<< HEAD
# Best Practice for Object Detection
=======
## Compressed Results of [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)

|        Network        |      Type      |  mAP@IoU=0.5  |    FLOPs (M)     |  Params (M)   | Model Size (MB) |              Download Link               |
| :-------------------: | :------------: | :-----------: | :--------------: | :-----------: | :-------------: | :--------------------------------------: |
| YOLOv4_EfficientNetB1 |    Original    |     82.22     |     61871.82     |     65.42     |     262.90      | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_original.h5) |
|                       | NPTK-Simple  1 | 87.23 (+5.01) | 11459.69 (5.4x)  | 10.66 (6.14x) |  44.12 (5.96x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_fd_p.h5) |
|                       | NPTK-Simple 2  | 87.91 (+5.69) | 14437.96 (4.29x) | 10.79 (6.06x) |  44.36 (5.93x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_p.h5) |

## Original Model
>>>>>>> 36211bd3087a1fbfc5e9af816fd024276e7d2376

## 1. Compressed Results of YOLOv4* on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)

|     Type      |  mAP@IoU=0.5  |    FLOPs (M)     |  Params (M)   | Model Size (MB) |              Download Link               |
| :-----------: | :-----------: | :--------------: | :-----------: | :-------------: | :--------------------------------------: |
|   Original    |     82.22     |     61871.82     |     65.32     |     262.90      | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_original.h5) |
| NPTK-Simple 1 | 87.23 (+5.01) | 11459.69 (5.4x)  | 10.59 (6.17x) |  44.12 (5.96x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_fd_p.h5) |
| NPTK-Simple 2 | 87.91 (+5.69) | 14437.96 (4.29x) | 10.71 (6.1x)  |  44.36 (5.93x)  | [Link](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/keras/pascal_voc/yolov4_efficientnetb1_p.h5) |

<<<<<<< HEAD
＊YOLOv4 model with EfficientNet B1 based backbone.

## 2. Original Model
=======
## Simple Compression 1 
>>>>>>> 36211bd3087a1fbfc5e9af816fd024276e7d2376

- The original model used the model provided by [keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set).



## 3. Model Compression

- ### NPTK-Simple 1

  - #### Stage 1: Filter Decomposition (Tucker Decomposition)

    - Recommendation Ratio : **0.2**
      - Except : block1a_se_reduce ~ block7b_add, conv2d_37, conv2d_29, conv2d_21

<<<<<<< HEAD
  - #### Stage 2: Pruning (L2 Norm)

    - Recommendation Ratio : **0.6**
      - Except : block1a_se_reduce ~ block7b_add, top_conv, conv2d_36_tucker_2, conv2d_28_tucker_2 , conv2d_20_tucker_2


- ### NPTK-Simple 2

  - #### Stage 1: Pruning (L2 Norm)

    - Recommendation Ratio : **0.8**
      - Except : block1a_se_reduce ~ block7b_add, top_conv, conv2d_36, conv2d_28, conv2d_20
=======
![recommendation_pruning_0.8](./images/simple_p_08.png?style=centerme)
>>>>>>> 36211bd3087a1fbfc5e9af816fd024276e7d2376
