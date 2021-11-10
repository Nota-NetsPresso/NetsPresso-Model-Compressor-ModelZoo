
# Model Zoo for the NetsPresso Compression Toolkit

## Workflow
  <p align="center">
    <img src="/imgs/compression_workflow.png" alt="Workflow">
  </p>

## Purpose of the Model Zoo
* Verifying the NetsPresso Compression Toolkit.
  * Training file, Original model and its compressed model are provided.
* Providing the model zoo to help the beginner to try the [NetsPresso Compression Toolkit\*](https://compression.netspresso.ai/).
  * For the people 


## Supported Deeplearning Frameworks
* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0), [2.4.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.4.0), [2.5.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.5.0)
* [PyTorch\*](https://pytorch.org/) version will be launched on Feb.

## Installation
```shell
$ git clone https://github.com/Nota-NetsPresso/NetsPresso-CompressionToolkit-ModelZoo.git
$ pip install -r requirements.txt
```
### For the Web User (Download Model)
  <a target="_blank" href="/imgs/web_user_1.png">
    <img src="/imgs/web_user_1.png" alt="Workflow">
  </a>


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
