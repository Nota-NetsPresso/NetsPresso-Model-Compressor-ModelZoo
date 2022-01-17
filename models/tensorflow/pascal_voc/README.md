## Original Model

The original model used the model provided by https://github.com/david8862/keras-YOLOv3-model-set.



## Simple Compression 1 

Stage 1: Filter Decomposition (Tucker Decomposition)



Stage 2: Pruning (L2 Norm)



## Simple Compression 2

Stage 1: Pruning (L2 Norm)



## Compressed Results of [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)

|        Network        |      Type      |  mAP@IoU=0.5  |    FLOPs (M)     |  Params (M)   | Model Size (MB) |
| :-------------------: | :------------: | :-----------: | :--------------: | :-----------: | :-------------: |
| YOLOv4_EfficientNetB1 |    Original    |     82.22     |     61871.82     |     65.42     |     262.90      |
|                       | NPTK-Simple  1 | 87.23 (+5.01) | 11459.69 (5.4x)  | 10.59 (6.17x) |  44.12 (5.96x)  |
|                       | NPTK-Simple 2  | 87.91 (+5.69) | 14437.96 (4.29x) | 10.79 (6.06x) |  44.36 (5.93x)  |
