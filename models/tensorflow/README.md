# Model Zoo for the NetsPresso Model Compressor

<div align=right>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNota-NetsPresso%2FNetsPresso-CompressionToolkit-ModelZoo%2Fblob%2Fmain%2FREADME.md&count_bg=%23368EEB&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

## Detail Workflow of the NetsPresso Model Compressor

### Step 1. Preparing the Pre-trained Model
  ```shell
  $ git clone https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo.git
  $ pip install -r requirements.txt
  ```

  For the web user
  <a target="_blank" href="/imgs/web_user_1.png">
    <img src="/imgs/web_user_1.png" alt="web_user">
  </a>

### Step 2. Compress the Model by Using [NetsPresso Model Compressor](https://compression.netspresso.ai/)

  * For more detail please refer to the [NetsPresso Documentation](https://docs.netspresso.ai/docs).


### Step 3. Performance Regain
  * The compression process might lead to performance deterioration. Therefore an additional training process is necessary for the performance regain (Especially the pruning process).
  * For given CIFAR100 models
    ```shell
    $ python train.py --model_path ./models/tensorflow/cifar100/vgg19.h5 --save_path ./cifar100_vgg19 --learning_rate 0.01 --batch_size 128 --epochs 100
    ```
    ```
      python train.py -h
      usage: train.py [-h] --model_path MODEL_PATH --save_path SAVE_PATH
                      [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                      [--epochs EPOCHS]
    
      optional arguments:
        -h, --help            show this help message and exit
        --model_path MODEL_PATH
                              input model path, default=models/tensorflow/cifar100/vgg19.h5
        --save_path SAVE_PATH
                              saved model path, default=./
        --learning_rate LEARNING_RATE
                              Initial learning rate, default=0.01
        --batch_size BATCH_SIZE
                              Batch size for train, default=128
        --epochs EPOCHS       
                              Total training epochs, default=100
    ```

## Provided Model Descriptions

* ### Compressed Results of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) - /models/tensorflow/cifar100

  <!-- |    Network   |     Type    |    Acc (%)    |    FLOPs (M)    |   Params (M)  | Model Size (MB) |
  |:------------:|:-----------:|:-------------:|:---------------:|:-------------:|:---------------:|
  |     VGG19    |   Original  |     72.28     |      796.79     |     20.09     |      80.57      |
  |              | Nota-Simple | 71.13 (-1.15) |  132.20 (6.03x) | 1.17 (17.13x) |  5.00 (16.13x)  |
  |              | Nota-Search | 71.14 (-1.14) |  100.09 (7.96x) | 0.66 (30.38x) |  2.96 (27.23x)  |
  |   ResNet50   |   Original  |     78.03     |     2596.06     |     23.71     |      95.55      |
  |              | Nota-Simple | 76.92 (-1.11) |  613.43 (4.23x) |  2.64 (8.99x) |   11.51 (8.3x)  |
  |              | Nota-Search |  76.63 (-1.4) | 224.70 (11.55x) | 2.17 (10.91x) |  9.54 (10.02x)  |
  | MobileNet V1 |   Original  |     66.68     |      92.90      |      3.31     |      13.59      |
  |              | Nota-Simple | 66.32 (-0.36) |  26.09 (3.56x)  |  0.53 (6.24x) |   2.52 (5.4x)   |
  |              | Nota-Search | 66.11 (-0.57) |  17.90 (5.19x)  |  0.35 (9.35x) |   1.78 (7.66x)  | -->

We provide the compressed results of following models in the [Best Practices](https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/tree/main/best_practices#tf-keras).

* ### ImageNet Pretrained Models - /models/tensorflow/keras-applications
  * Following models are now available on [NetsPresso Model Compressor](https://compression.netspresso.ai/).
    |       Model       |  FD | Pruning |    Version    |
    |:-----------------:|:---:|:-------:|:-------------:|
    |       VGG16       | Yes |   Yes   | 2.2.x - 2.5.x |
    |       VGG19       | Yes |   Yes   | 2.2.x - 2.5.x |
    |      ResNet50     | Yes |   Yes   | 2.2.x - 2.5.x |
    |     ResNet101     | Yes |   Yes   | 2.2.x - 2.5.x |
    |     ResNet152     | Yes |   Yes   | 2.2.x - 2.5.x |
    |     ResNet50V2    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    ResNet101V2    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    ResNet152V2    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    InceptionV3    | Yes |   Yes   | 2.2.x - 2.5.x |
    |     MobileNet     | Yes |   Yes   | 2.2.x - 2.5.x |
    |    MobileNetV2    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    DenseNet121    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    DenseNet169    | Yes |   Yes   | 2.2.x - 2.5.x |
    |    DenseNet201    | Yes |   Yes   | 2.2.x - 2.5.x |
    | EfficientNetB1-B7 | Yes |   Yes   | 2.3.x - 2.5.x |
    |      Xception     | Yes |   Yes   | 2.3.x - 2.5.x |
    | InceptionResNetV2 | Yes |   WIP   | 2.3.x - 2.5.x |
    |       NASNet      | Yes |   WIP   | 2.3.x - 2.5.x |
