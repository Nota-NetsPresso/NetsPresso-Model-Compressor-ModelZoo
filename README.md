
# Model Zoo for the NetsPresso Compression Toolkit

## Workflow

  <p align="center">
    <img src="/imgs/compression_workflow.png" alt="Workflow">
  </p>

## Purpose of the Model Zoo

* Providing the model zoo to help the beginner to try the [NetsPresso Compression Toolkit\*](https://compression.netspresso.ai/).
* Verifying the NetsPresso Compression Toolkit.
  * Training file, Original model and its compressed model are provided.


## Supported Deeplearning Frameworks

* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0), [2.4.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.4.0), [2.5.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.5.0)
* [PyTorch\*](https://pytorch.org/) version will be launched on Feb.


## Detail Workflow of the NetsPresso Compression Toolkit

### Preparing the pre-trained model

  ```shell
  $ git clone https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo.git
  $ pip install -r requirements.txt
  ```

  #### For the Web User (Public model only)
    <a target="_blank" href="/imgs/web_user_1.png">
      <img src="/imgs/web_user_1.png" alt="web_user">
    </a>

### Compress the model by using [NetsPresso Compression Toolkit](https://compression.netspresso.ai/)

  #### Please check and follow the [NetsPresso Documentation](https://docs.netspresso.ai/docs)

### Performance regain
  * #### The compression process might lead to performance deterioration. Therefore an additional training process is necessary for the performance regain (Especially the pruning process).
  * #### For given CIFAR100 models
    ```shell
    $ python train.py --model_path ./models/cifar100/vgg19.h5 --save_path ./cifar100_vgg19 --learning_rate 0.01 --batch_size 128 --epochs 100
    ```
    ```
      python train.py -h
      usage: train.py [-h] --model_path MODEL_PATH --save_path SAVE_PATH
                      [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                      [--epochs EPOCHS]

      optional arguments:
        -h, --help            show this help message and exit
        --model_path MODEL_PATH
                              input model path, default=models/cifar100/vgg19.h5
        --save_path SAVE_PATH
                              saved model path, default=./
        --learning_rate LEARNING_RATE
                              Initial learning rate, default=0.01
        --batch_size BATCH_SIZE
                              Batch size for train, default=128
        --epochs EPOCHS       
    ```

## Model Description

* ### Compressed results of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (/models/cifar100)

  |    Network   |     Type    |    Acc (%)    |    FLOPs (M)    |   Params (M)  | Model Size (MB) |
  |:------------:|:-----------:|:-------------:|:---------------:|:-------------:|:---------------:|
  |     VGG19    |   Original  |     72.28     |      796.79     |     20.09     |      80.57      |
  |              | Nota-Simple | 71.13 (-1.15) |  132.20 (6.03x) | 1.17 (17.13x) |  5.00 (16.13x)  |
  |              | Nota-Search | 71.14 (-1.14) |  100.09 (7.96x) | 0.66 (30.38x) |  2.96 (27.23x)  |
  |   ResNet50   |   Original  |     78.03     |     2596.06     |     23.71     |      95.55      |
  |              | Nota-Simple | 76.92 (-1.11) |  613.43 (4.23x) |  2.64 (8.99x) |   11.51 (8.3x)  |
  |              | Nota-Search |  76.63 (-1.4) | 224.70 (11.55x) | 2.17 (10.91x) |  9.54 (10.02x)  |
  | MobileNet V1 |   Original  |     66.68     |      92.90      |      3.31     |      13.59      |
  |              | Nota-Simple | 66.32 (-0.36) |  26.09 (3.56x)  |  0.53 (6.24x) |   2.52 (5.4x)   |
  |              | Nota-Search | 66.11 (-0.57) |  17.90 (5.19x)  |  0.35 (9.35x) |   1.78 (7.66x)  |

* ### ImageNet Pretrained Model (/models/keras-applications)
  * #### Follow models are available now on [NetsPresso Compression Toolkit\*](https://compression.netspresso.ai/).
    | Model             | Support | Version       |
    | ----------------- | ------- | ------------- |
    | VGG16             | Yes     | 2.2.x - 2.5.x |
    | VGG19             | Yes     | 2.2.x - 2.5.x |
    | ResNet50          | Yes     | 2.2.x - 2.5.x |
    | ResNet101         | Yes     | 2.2.x - 2.5.x |
    | ResNet152         | Yes     | 2.2.x - 2.5.x |
    | ResNet50V2        | Yes     | 2.2.x - 2.5.x |
    | ResNet101V2       | Yes     | 2.2.x - 2.5.x |
    | ResNet152V2       | Yes     | 2.2.x - 2.5.x |
    | InceptionV3       | Yes     | 2.2.x - 2.5.x |
    | MobileNet         | Yes     | 2.2.x - 2.5.x |
    | MobileNetV2       | Yes     | 2.2.x - 2.5.x |
    | DenseNet121       | Yes     | 2.2.x - 2.5.x |
    | DenseNet169       | Yes     | 2.2.x - 2.5.x |
    | DenseNet201       | Yes     | 2.2.x - 2.5.x |
    | EfficientNetB1-B7 | Yes     | 2.3.x - 2.5.x |
    | Xception          | WIP     |               |
    | InceptionResNetV2 | WIP     |               |
    | NASNet            | WIP     |               |

<!-- * [Installation](#Installation)
* [How to Use](#How-to-Use)
* [Available Models](#Available-Models)

## Installation

```shell
$ git clone https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo.git
$ pip install -r requirements.txt
```

## How to Use

### Example
```shell
$ python train.py --model_path ./models/cifar100/vgg19.h5 --save_path ./cifar100_vgg19 --learning_rate 0.01 --batch_size 128 --epochs 100
```

### Description
```
python train.py -h
usage: train.py [-h] --model_path MODEL_PATH --save_path SAVE_PATH
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        input model path, default=models/cifar100/vgg19.h5
  --save_path SAVE_PATH
                        saved model path, default=./
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.01
  --batch_size BATCH_SIZE
                        Batch size for train, default=128
  --epochs EPOCHS       
                        Total training epochs, default=100
```



## Available Models

| Dataset  |   Network   | Acc (%) | FLOPs (M) | Params (M) | Model Size (MB) |
| :------: | :---------: | :-----: | :-------: | :--------: | :-------------: |
| CIFAR100 |    VGG19    |  72.28  |  796.79   |   20.09    |      80.57      |
| CIFAR100 |  ResNet50   |  78.03  |  2596.06  |   23.71    |      95.55      |
| CIFAR100 | MobileNetV1 |  66.68  |   92.90   |    3.31    |      13.59      |
 -->
